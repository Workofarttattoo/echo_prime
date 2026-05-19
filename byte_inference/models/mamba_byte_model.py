"""
Mamba State Space Model for Byte-Level Processing
Linear time complexity O(n) instead of O(n²) for transformers!

Key advantages:
- Processes unlimited context length
- Constant memory per token
- Fast inference
- No tokenization needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MambaByteModel(nn.Module):
    """
    Mamba-based byte-level language model

    Uses selective state space models for linear-time sequence processing
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        max_seq_length: int = 1000000,  # Can handle very long sequences!
        use_cuda: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Byte embedding
        self.byte_embedding = nn.Embedding(256, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm_f = RMSNorm(d_model)

        # Output projection
        self.lm_head = nn.Linear(d_model, 256, bias=False)

        # Tie byte embedding and output weights (optional)
        # self.lm_head.weight = self.byte_embedding.weight

    def forward(
        self,
        byte_ids: torch.Tensor,
        cache: Optional[list] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass

        Args:
            byte_ids: [batch, seq_len] byte values (0-255)
            cache: Optional cache from previous forward pass
            return_cache: Whether to return cache for next iteration

        Returns:
            logits: [batch, seq_len, 256] next byte predictions
            cache: Optional cache for incremental decoding
        """
        # Embed bytes
        x = self.byte_embedding(byte_ids)

        # Initialize cache if needed
        if cache is None and return_cache:
            cache = [None] * self.n_layers

        # Process through Mamba layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_cache = layer(x, cache=layer_cache)
            if return_cache:
                new_cache.append(layer_cache)

        # Final norm
        x = self.norm_f(x)

        # Project to logits
        logits = self.lm_head(x)

        if return_cache:
            return logits, new_cache
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_bytes: list,
        max_new_bytes: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        device: str = "cuda"
    ) -> list:
        """
        Generate bytes with caching for fast inference

        Thanks to caching, each step is O(1) instead of O(n)!
        """
        self.eval()

        # Convert to tensor
        current = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

        cache = None
        generated = []

        for step in range(max_new_bytes):
            # Forward pass with cache
            if cache is None:
                # First pass: process full prompt
                logits, cache = self.forward(current, return_cache=True)
                next_logits = logits[0, -1, :]
            else:
                # Subsequent passes: only process last byte
                last_byte = current[:, -1:]
                logits, cache = self.forward(last_byte, cache=cache, return_cache=True)
                next_logits = logits[0, 0, :]

            # Temperature scaling
            next_logits = next_logits / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=0), dim=0)

            # Remove tokens above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=0)
            next_byte = torch.multinomial(probs, 1)

            # Append
            current = torch.cat([current, next_byte.unsqueeze(0)], dim=1)
            generated.append(next_byte.item())

            # Stop on null
            if next_byte.item() == 0:
                break

        return prompt_bytes + generated

    def get_num_params(self) -> int:
        """Count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective SSM
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto"
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter (state transition matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Norm
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with selective SSM

        Args:
            x: [batch, seq_len, d_model]
            cache: Optional cache from previous call

        Returns:
            output: [batch, seq_len, d_model]
            new_cache: Cache for next iteration
        """
        batch, seqlen, dim = x.shape

        # Residual
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Split input projection into two branches
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x, z = xz.chunk(2, dim=-1)  # Each [batch, seq_len, d_inner]

        # Convolution (with caching for incremental decoding)
        x = x.transpose(1, 2)  # [batch, d_inner, seq_len]

        if cache is not None and "conv_state" in cache:
            # Use cached conv state (pre-conv input from previous step)
            conv_state = cache["conv_state"]
            x_with_cache = torch.cat([conv_state, x], dim=2)
            # Extract causal outputs at correct positions (d_conv-1 through d_conv-1+seqlen-1)
            x_conv = self.conv1d(x_with_cache)[:, :, (self.d_conv-1):(self.d_conv-1+seqlen)]
            # Save pre-conv input tail for next iteration
            new_conv_state = x_with_cache[:, :, -(self.d_conv - 1):]
            x = x_conv
        else:
            # First pass: save input tail before convolution
            x_conv = self.conv1d(x)[:, :, :seqlen]
            new_conv_state = x[:, :, -(self.d_conv - 1):]
            x = x_conv

        x = x.transpose(1, 2)  # [batch, seq_len, d_inner]
        x = F.silu(x)

        # SSM
        x_dbl = self.x_proj(x)  # [batch, seq_len, dt_rank + 2 * d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))  # [batch, seq_len, d_inner]

        # Discretize and apply SSM
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        y = self._selective_scan(x, dt, A, B, C, self.D, cache)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        # Residual
        output = output + residual

        # Update cache with both conv_state and ssm_state
        new_cache = {"conv_state": new_conv_state}
        if cache is not None and "ssm_state" in cache:
            new_cache["ssm_state"] = cache["ssm_state"]

        return output, new_cache

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        cache: Optional[dict]
    ) -> torch.Tensor:
        """
        Selective SSM scan (simplified version)

        In practice, use optimized CUDA kernels for this
        """
        batch, seq_len, d_inner = x.shape

        # Discretize
        dt = dt.unsqueeze(-1)  # [batch, seq_len, d_inner, 1]
        dA = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # [batch, seq_len, d_inner, d_state]
        dB = dt * B.unsqueeze(2)  # [batch, seq_len, d_inner, d_state]

        # Initialize state
        if cache is not None and "ssm_state" in cache:
            h = cache["ssm_state"]
        else:
            h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []

        # Scan over sequence
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            y_t = y_t + D * x[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]

        # Cache final state
        if cache is not None:
            cache["ssm_state"] = h

        return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def create_mamba_model(
    model_size: str = "small"
) -> MambaByteModel:
    """
    Create Mamba model with predefined sizes

    Args:
        model_size: "tiny", "small", "medium", "large"

    Returns:
        MambaByteModel
    """

    configs = {
        "tiny": {"d_model": 256, "n_layers": 12, "d_state": 8},
        "small": {"d_model": 512, "n_layers": 24, "d_state": 16},
        "medium": {"d_model": 768, "n_layers": 32, "d_state": 16},
        "large": {"d_model": 1024, "n_layers": 48, "d_state": 16},
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")

    config = configs[model_size]
    model = MambaByteModel(**config)

    num_params = model.get_num_params()
    print(f"Created Mamba-{model_size} with {num_params:,} parameters")

    return model


if __name__ == "__main__":
    print("Mamba Byte Model Demo\n")

    # Create model
    model = create_mamba_model("small")

    # Test
    text = "Mamba processes bytes in linear time!"
    bytes_input = list(text.encode('utf-8'))

    print(f"Input: {text}")
    print(f"Bytes: {len(bytes_input)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"\nGenerating on {device}...")
    generated = model.generate(
        prompt_bytes=bytes_input,
        max_new_bytes=100,
        temperature=0.8,
        device=device
    )

    output_text = bytes(generated).decode('utf-8', errors='ignore')
    print(f"\nGenerated:\n{output_text}")
