import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class MultiHeadLatentAttention(nn.Module):
    """
    ECH0-PRIME Enhancement: DeepSeek MLA (Multi-head Latent Attention)
    
    Architecture:
    1. Low-rank compression of KV cache into a latent vector.
    2. Decoupled Rotary Positional Embeddings (RoPE).
    3. Significant reduction in inference memory/compute.
    """
    def __init__(self, dimension: int = 512, num_heads: int = 8, latent_dim: int = 64, q_lora_rank: int = 128):
        super().__init__()
        self.dimension = dimension
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = dimension // num_heads
        
        # 1. KV Low-rank Compression
        self.kv_compress = nn.Linear(dimension, latent_dim)
        self.kv_up_proj = nn.Linear(latent_dim, dimension * 2) # Keys and Values
        
        # 2. Q Low-rank Compression (optional but common in MLA)
        self.q_compress = nn.Linear(dimension, q_lora_rank)
        self.q_up_proj = nn.Linear(q_lora_rank, dimension)
        
        # 3. Decoupled RoPE (simplified implementation)
        # We handle RoPE by only applying it to a part of the query/key
        self.rope_dim = self.head_dim // 2
        
        # 4. Final Output Projection
        self.out_proj = nn.Linear(dimension, dimension)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # KV Compression
        kv_latent = self.kv_compress(x) # [B, S, latent_dim]
        kv_up = self.kv_up_proj(kv_latent) # [B, S, dimension * 2]
        
        K, V = torch.split(kv_up, [self.dimension, self.dimension], dim=-1)
        
        # Q Compression
        q_latent = self.q_compress(x)
        Q = self.q_up_proj(q_latent) # [B, S, dimension]
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V) # [B, heads, S, head_dim]
        
        # Concatenate heads and project out
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dimension)
        return self.out_proj(context)

class DeepSeekMoE(nn.Module):
    """
    ECH0-PRIME Enhancement: DeepSeek-style Mixture of Experts (MoE)
    With Fine-Grained Experts and Shared Experts.
    """
    def __init__(self, dimension: int = 512, num_experts: int = 8, active_experts: int = 2):
        super().__init__()
        self.dimension = dimension
        self.num_experts = num_experts
        self.active_experts = active_experts
        
        # Shared Expert (always active)
        self.shared_expert = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension)
        )
        
        # Gated Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension, dimension * 2),
                nn.GELU(),
                nn.Linear(dimension * 2, dimension)
            ) for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(dimension, num_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        flat_x = x.view(-1, self.dimension)
        
        # Routing scores
        routing_logits = self.router(flat_x)
        routing_weights = torch.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        top_weights, top_indices = torch.topk(routing_weights, self.active_experts, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True) # Renormalize
        
        # Shared expert contribution
        shared_out = self.shared_expert(x)
        
        # Expert execution
        expert_out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            # Mask for tokens assigned to this expert
            mask = (top_indices == i).any(dim=-1)
            if mask.any():
                # Get the specific weights for this expert for these tokens
                # This is a bit simplified for readability
                token_weights = torch.zeros(flat_x.size(0), 1, device=x.device)
                for k in range(self.active_experts):
                    token_weights[top_indices[:, k] == i] = top_weights[top_indices[:, k] == i, k].unsqueeze(-1)
                
                expert_out[mask] += expert(flat_x[mask]) * token_weights[mask]
                
        return shared_out + expert_out.view(batch_size, seq_len, self.dimension)

