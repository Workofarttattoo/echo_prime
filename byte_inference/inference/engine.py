"""
Universal Inference Engine for Non-Tokenized Models

Supports:
- Byte-level transformers
- Mamba state space models
- Custom architectures

Features:
- Streaming generation
- Batched inference
- KV-caching
- Temperature/top-p/top-k sampling
- Multiple decoding strategies
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import time
from pathlib import Path
import sys

# Add models to path
sys.path.append(str(Path(__file__).parent.parent / "models"))

from byte_level_transformer import ByteLevelTransformer, create_byte_model
from mamba_byte_model import MambaByteModel, create_mamba_model


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_tokens: List[int] = None
    stream: bool = False

    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = [0, 10]  # null, newline


class ByteInferenceEngine:
    """
    Universal inference engine for byte-level models
    """

    def __init__(
        self,
        model_type: str = "transformer",
        model_size: str = "small",
        device: str = None,
        dtype: torch.dtype = torch.float16,
        compile_model: bool = False
    ):
        """
        Initialize inference engine

        Args:
            model_type: "transformer" or "mamba"
            model_size: "tiny", "small", "medium", "large", "xl"
            device: Device to run on (auto-detect if None)
            dtype: Model dtype for efficiency
            compile_model: Use torch.compile for speed (PyTorch 2.0+)
        """
        self.model_type = model_type
        self.model_size = model_size
        self.dtype = dtype

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🔧 Initializing {model_type}-{model_size} on {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)

        if dtype == torch.float16 and self.device.type == "cuda":
            self.model = self.model.half()

        # Compile for speed (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                print("✅ Model compiled with torch.compile")
            except:
                print("⚠️  torch.compile not available")

        self.model.eval()

        # Stats
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Model parameters: {num_params:,}")

    def _load_model(self) -> torch.nn.Module:
        """Load model based on type"""
        if self.model_type == "transformer":
            return create_byte_model(self.model_size)
        elif self.model_type == "mamba":
            return create_mamba_model(self.model_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def encode(self, text: str) -> List[int]:
        """Convert text to bytes"""
        return list(text.encode('utf-8'))

    def decode(self, byte_ids: List[int]) -> str:
        """Convert bytes to text"""
        try:
            return bytes(byte_ids).decode('utf-8', errors='ignore')
        except:
            return ''.join(chr(b) if 0 <= b < 256 else '?' for b in byte_ids)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> Dict[str, Any]:
        """
        Generate text from prompt

        Args:
            prompt: Input text
            config: Generation configuration

        Returns:
            Dictionary with generated text and metadata
        """
        if config is None:
            config = GenerationConfig()

        # Encode prompt
        prompt_bytes = self.encode(prompt)
        start_time = time.time()

        # Generate
        if config.stream:
            # Streaming generation
            return self._generate_stream(prompt_bytes, config)
        else:
            # Standard generation
            generated_bytes = self._generate_standard(prompt_bytes, config)

            # Decode
            output_text = self.decode(generated_bytes)

            # Calculate stats
            elapsed = time.time() - start_time
            new_bytes = len(generated_bytes) - len(prompt_bytes)
            tokens_per_sec = new_bytes / elapsed if elapsed > 0 else 0

            return {
                "text": output_text,
                "prompt": prompt,
                "bytes_generated": new_bytes,
                "total_bytes": len(generated_bytes),
                "time_seconds": elapsed,
                "bytes_per_second": tokens_per_sec,
                "device": str(self.device)
            }

    def _generate_standard(
        self,
        prompt_bytes: List[int],
        config: GenerationConfig
    ) -> List[int]:
        """Standard non-streaming generation"""

        # Use model's generate method
        if hasattr(self.model, 'generate'):
            # Check if model is Mamba (only accepts subset of params)
            model_name = self.model.__class__.__name__
            if 'Mamba' in model_name:
                return self.model.generate(
                    prompt_bytes=prompt_bytes,
                    max_new_bytes=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    device=str(self.device)
                )
            else:
                return self.model.generate(
                    prompt_bytes=prompt_bytes,
                    max_new_bytes=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    device=str(self.device)
                )
        else:
            # Fallback manual generation
            return self._generate_manual(prompt_bytes, config)

    def _generate_manual(
        self,
        prompt_bytes: List[int],
        config: GenerationConfig
    ) -> List[int]:
        """Manual generation loop (fallback)"""

        current = torch.tensor([prompt_bytes], dtype=torch.long, device=self.device)
        generated = prompt_bytes.copy()

        for _ in range(config.max_new_tokens):
            # Forward pass
            logits = self.model(current)
            next_logits = logits[0, -1, :] / config.temperature

            # Top-k
            if config.top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, config.top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, 1)
            byte_val = next_byte.item()

            # Append
            current = torch.cat([current, next_byte.unsqueeze(0)], dim=1)
            generated.append(byte_val)

            # Stop condition
            if byte_val in config.stop_tokens:
                break

        return generated

    @torch.inference_mode()
    def _generate_stream(
        self,
        prompt_bytes: List[int],
        config: GenerationConfig
    ) -> Iterator[Dict[str, Any]]:
        """Streaming generation (yields tokens as generated)"""

        current = torch.tensor([prompt_bytes], dtype=torch.long, device=self.device)
        generated = prompt_bytes.copy()
        start_time = time.time()

        for step in range(config.max_new_tokens):
            # Forward pass
            logits = self.model(current)
            next_logits = logits[0, -1, :] / config.temperature

            # Sampling (simplified for streaming)
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, 1)
            byte_val = next_byte.item()

            # Append
            current = torch.cat([current, next_byte.unsqueeze(0)], dim=1)
            generated.append(byte_val)

            # Decode new byte
            try:
                token_text = chr(byte_val) if 0 <= byte_val < 128 else ''
            except:
                token_text = ''

            # Yield update
            yield {
                "token": token_text,
                "byte_value": byte_val,
                "step": step,
                "total_bytes": len(generated),
                "elapsed": time.time() - start_time
            }

            # Stop condition
            if byte_val in config.stop_tokens:
                break

    def batch_generate(
        self,
        prompts: List[str],
        config: GenerationConfig = None
    ) -> List[Dict[str, Any]]:
        """
        Generate for multiple prompts in batch

        Args:
            prompts: List of input texts
            config: Generation configuration

        Returns:
            List of generation results
        """
        if config is None:
            config = GenerationConfig()

        results = []
        for prompt in prompts:
            result = self.generate(prompt, config)
            results.append(result)

        return results

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'model_size': self.model_size,
            'dtype': self.dtype
        }, path)
        print(f"💾 Saved model to {path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 Loaded model from {path}")


def main():
    """Demo the inference engine"""
    print("=" * 80)
    print("BYTE-LEVEL INFERENCE ENGINE DEMO")
    print("=" * 80)
    print()

    # Initialize engine
    engine = ByteInferenceEngine(
        model_type="transformer",
        model_size="small",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test prompts
    prompts = [
        "The future of AI is",
        "Quantum computing will",
        "Byte-level models are"
    ]

    # Generation config
    config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        stream=False
    )

    # Generate
    print("\n🚀 GENERATING TEXT\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Prompt: {prompt}")

        result = engine.generate(prompt, config)

        print(f"Generated: {result['text']}")
        print(f"Stats: {result['bytes_generated']} bytes in {result['time_seconds']:.2f}s "
              f"({result['bytes_per_second']:.1f} bytes/sec)")
        print()

    print("=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
