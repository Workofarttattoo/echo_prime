"""
Echo Lite - Embedded Cognitive Architecture for Raspberry Pi

Minimal version of Echo Prime optimized for:
- ARM64 processors (Raspberry Pi 5)
- CPU-only inference (no GPU)
- Low memory footprint (<500MB)
- Real-time reasoning
- Embedded deployment

Key differences from Echo Prime:
- No heavy transformers (use TinyLLM or GGUF quantized models)
- Simplified cognitive hierarchy (2 levels instead of 5)
- Lightweight embeddings (sentence-transformers mini)
- No visualization dashboard
- Core reasoning only
"""

import numpy as np
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import sys


@dataclass
class EchoLiteConfig:
    """Configuration for Echo Lite"""

    # Model settings
    model_type: str = "tiny"  # tiny, small
    max_memory_mb: int = 400
    cpu_threads: int = 4

    # Cognitive settings
    cognitive_levels: int = 2  # Reduced from 5
    context_window: int = 512  # Reduced from 2048

    # Inference settings
    temperature: float = 0.7
    max_tokens: int = 100

    # System
    enable_logging: bool = True
    log_file: str = "echo_lite.log"

    # Paths
    model_path: Optional[str] = None
    data_path: str = "./data"


class MinimalCognitiveArchitecture:
    """
    Lightweight 2-level cognitive architecture

    Level 1: Sensory Processing (input/output)
    Level 2: Executive Function (reasoning/decision)
    """

    def __init__(self, config: EchoLiteConfig):
        self.config = config

        # Cognitive state
        self.state = {
            'level_1': np.zeros(64, dtype=np.float32),  # Sensory
            'level_2': np.zeros(128, dtype=np.float32),  # Executive
        }

        # Short-term memory (limited)
        self.short_term_memory = []
        self.max_memory_items = 10

        # Prediction error tracking
        self.prediction_errors = []

        # Initialize weight matrix for executive reasoning (reused across calls)
        self.executive_weights = np.random.randn(64, 128).astype(np.float32) * 0.1

        print("🧠 Minimal Cognitive Architecture initialized")
        print(f"   Levels: {config.cognitive_levels}")
        print(f"   Memory footprint: ~{self._estimate_memory_mb():.1f}MB")

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage"""
        # State vectors
        state_mem = sum(v.nbytes for v in self.state.values()) / 1024 / 1024
        # Memory buffer
        mem_mem = len(self.short_term_memory) * 0.01  # ~10KB per item
        return state_mem + mem_mem + 50  # +50MB for Python overhead

    def process_input(self, text: str) -> np.ndarray:
        """
        Level 1: Process sensory input

        Simple character-level encoding for minimal overhead
        """
        # Convert to character codes (simple encoding)
        chars = [ord(c) for c in text[:64]]  # Limit length

        # Pad to fixed size
        chars = chars + [0] * (64 - len(chars))

        # Normalize
        sensory_state = np.array(chars, dtype=np.float32) / 255.0

        # Update level 1
        self.state['level_1'] = sensory_state

        return sensory_state

    def executive_reasoning(self, sensory_input: np.ndarray) -> Dict[str, Any]:
        """
        Level 2: Executive reasoning

        Simple feedforward processing
        """
        # Simple transformation (no heavy computation)
        # In real implementation, this would be a small neural network

        # Weighted sum with learned parameters
        executive_state = np.tanh(
            np.dot(sensory_input, self.executive_weights)
        )

        self.state['level_2'] = executive_state

        # Extract features
        attention = float(np.mean(np.abs(executive_state[:32])))
        confidence = float(np.mean(executive_state[32:64]))

        return {
            'attention': attention,
            'confidence': confidence,
            'state': executive_state
        }

    def cognitive_cycle(self, input_text: str) -> Dict[str, Any]:
        """
        Full cognitive cycle: Input → Process → Reason → Output
        """
        start_time = time.time()

        # Level 1: Sensory processing
        sensory = self.process_input(input_text)

        # Level 2: Executive reasoning
        reasoning = self.executive_reasoning(sensory)

        # Update short-term memory
        self.short_term_memory.append({
            'input': input_text,
            'reasoning': reasoning,
            'timestamp': time.time()
        })

        # Limit memory
        if len(self.short_term_memory) > self.max_memory_items:
            self.short_term_memory.pop(0)

        elapsed = time.time() - start_time

        return {
            'sensory': sensory.tolist(),
            'reasoning': reasoning,
            'memory_items': len(self.short_term_memory),
            'processing_time_ms': elapsed * 1000
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current cognitive state"""
        return {
            'level_1_activity': float(np.mean(np.abs(self.state['level_1']))),
            'level_2_activity': float(np.mean(np.abs(self.state['level_2']))),
            'memory_size': len(self.short_term_memory),
            'memory_mb': self._estimate_memory_mb()
        }


class EchoLite:
    """
    Main Echo Lite system

    Lightweight cognitive architecture for embedded systems
    """

    def __init__(self, config: Optional[EchoLiteConfig] = None):
        if config is None:
            config = EchoLiteConfig()

        self.config = config

        print("\n" + "="*60)
        print("🌟 ECHO LITE - Embedded Cognition")
        print("="*60)
        print(f"Platform: Raspberry Pi 5 / ARM64")
        print(f"Memory Target: <{config.max_memory_mb}MB")
        print(f"CPU Threads: {config.cpu_threads}")
        print("="*60 + "\n")

        # Initialize cognitive architecture
        self.cognitive = MinimalCognitiveArchitecture(config)

        # Initialize lightweight inference (optional)
        self.inference_available = False
        try:
            from .inference.tiny_inference import TinyInference
            self.inference = TinyInference(config)
            self.inference_available = True
            print("✅ Inference engine loaded")
        except ImportError:
            print("⚠️  Inference engine not available (optional)")

        # System state
        self.running = True
        self.cycle_count = 0

    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Main processing function

        Args:
            input_text: Input text to process

        Returns:
            Processing results with reasoning and response
        """
        self.cycle_count += 1

        # Cognitive processing
        cognitive_result = self.cognitive.cognitive_cycle(input_text)

        # Optional: Generate response with inference
        response = None
        if self.inference_available:
            try:
                response = self.inference.generate(
                    prompt=input_text,
                    max_tokens=self.config.max_tokens
                )
            except Exception as e:
                print(f"⚠️  Inference error: {e}")

        return {
            'input': input_text,
            'cognitive': cognitive_result,
            'response': response,
            'cycle': self.cycle_count,
            'system_state': self.cognitive.get_state()
        }

    def run_repl(self):
        """
        Interactive REPL for testing
        """
        print("\n🎮 Echo Lite REPL")
        print("Type 'quit' to exit, 'status' for system info\n")

        while self.running:
            try:
                user_input = input("→ ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("👋 Shutting down Echo Lite")
                    break

                if user_input.lower() == 'status':
                    state = self.cognitive.get_state()
                    print(f"\n📊 System Status:")
                    print(f"   Cycles: {self.cycle_count}")
                    print(f"   Memory: {state['memory_mb']:.1f}MB")
                    print(f"   L1 Activity: {state['level_1_activity']:.3f}")
                    print(f"   L2 Activity: {state['level_2_activity']:.3f}")
                    print()
                    continue

                # Process input
                result = self.process(user_input)

                # Display results
                print(f"\n⚡ Processed in {result['cognitive']['processing_time_ms']:.2f}ms")
                print(f"   Attention: {result['cognitive']['reasoning']['attention']:.3f}")
                print(f"   Confidence: {result['cognitive']['reasoning']['confidence']:.3f}")

                if result['response']:
                    print(f"\n💭 Response: {result['response']}")

                print()

            except KeyboardInterrupt:
                print("\n\n👋 Shutting down Echo Lite")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

    def benchmark(self, num_cycles: int = 100):
        """
        Run performance benchmark
        """
        print(f"\n⚡ Running benchmark ({num_cycles} cycles)...")

        test_inputs = [
            "Hello Echo",
            "Process this information",
            "What is consciousness?",
            "Analyze the pattern",
            "Execute reasoning cycle"
        ]

        start_time = time.time()
        processing_times = []

        for i in range(num_cycles):
            input_text = test_inputs[i % len(test_inputs)]
            result = self.process(input_text)
            processing_times.append(result['cognitive']['processing_time_ms'])

        total_time = time.time() - start_time

        print(f"\n📊 Benchmark Results:")
        print(f"   Total cycles: {num_cycles}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time/cycle: {np.mean(processing_times):.2f}ms")
        print(f"   Min time: {np.min(processing_times):.2f}ms")
        print(f"   Max time: {np.max(processing_times):.2f}ms")
        print(f"   Cycles/second: {num_cycles / total_time:.1f}")

        state = self.cognitive.get_state()
        print(f"   Memory usage: {state['memory_mb']:.1f}MB")
        print()


def main():
    """Main entry point"""

    # Create config
    config = EchoLiteConfig(
        model_type="tiny",
        max_memory_mb=400,
        cpu_threads=4,
        cognitive_levels=2
    )

    # Initialize Echo Lite
    echo = EchoLite(config)

    # Run benchmark
    echo.benchmark(num_cycles=100)

    # Start REPL
    echo.run_repl()


if __name__ == "__main__":
    main()
