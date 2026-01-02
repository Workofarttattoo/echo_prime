"""
Novel cognitive architectures inspired by neuroscience.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class DifferentiableNeuralComputer(nn.Module):
    """
    Differentiable Neural Computer (DNC) - external memory architecture.
    """
    def __init__(self, input_size: int, output_size: int, memory_size: int = 128, 
                 memory_dim: int = 20, num_read_heads: int = 1):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        
        # Memory matrix
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))
        
        # Controller (LSTM)
        self.controller = nn.LSTM(input_size, 256)
        
        # Read heads
        self.read_heads = nn.ModuleList([
            nn.Linear(256, memory_dim) for _ in range(num_read_heads)
        ])
        
        # Write head
        self.write_head = nn.Linear(256, memory_dim * 2)  # key + value
        
        # Output layer
        self.output_layer = nn.Linear(256 + num_read_heads * memory_dim, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DNC"""
        batch_size = x.size(0)
        
        # Controller forward
        controller_out, _ = self.controller(x)
        
        # Read from memory
        read_vectors = []
        for read_head in self.read_heads:
            read_key = read_head(controller_out)
            # Content-based addressing
            similarities = torch.matmul(self.memory, read_key.t())
            attention = F.softmax(similarities, dim=0)
            read_vector = torch.matmul(attention.t(), self.memory)
            read_vectors.append(read_vector)
        
        read_concat = torch.cat(read_vectors, dim=-1)
        
        # Write to memory
        write_output = self.write_head(controller_out)
        write_key = write_output[:, :self.memory_dim]
        write_value = write_output[:, self.memory_dim:]
        
        # Update memory (simplified)
        write_attention = F.softmax(torch.matmul(self.memory, write_key.t()), dim=0)
        self.memory = (1 - write_attention) * self.memory + write_attention.t() * write_value
        
        # Output
        combined = torch.cat([controller_out, read_concat], dim=-1)
        output = self.output_layer(combined)
        
        return output


class AttentionPattern(nn.Module):
    """
    Novel attention patterns inspired by neuroscience.
    """
    def __init__(self, dim: int, num_heads: int = 8, pattern_type: str = "gaussian"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pattern_type = pattern_type
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with novel attention pattern"""
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.dim)
        
        # Apply attention pattern
        if self.pattern_type == "gaussian":
            # Gaussian attention (localized)
            mask = self._gaussian_mask(seq_len, device=x.device)
            scores = scores * mask
        elif self.pattern_type == "hierarchical":
            # Hierarchical attention
            mask = self._hierarchical_mask(seq_len, device=x.device)
            scores = scores * mask
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        output = self.out_proj(output)
        
        return output
    
    def _gaussian_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create Gaussian attention mask"""
        positions = torch.arange(seq_len, device=device).float()
        center = seq_len / 2
        sigma = seq_len / 4
        
        mask = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask
    
    def _hierarchical_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create hierarchical attention mask"""
        mask = torch.ones(seq_len, seq_len, device=device)
        
        # Reduce attention for distant positions
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                mask[i, j] = 1.0 / (1.0 + distance)
        
        return mask.unsqueeze(0)


class SpikingNeuralNetwork(nn.Module):
    """
    Spiking Neural Network (SNN) for energy-efficient processing.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 threshold: float = 1.0, decay: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Membrane potentials
        self.register_buffer('membrane_potentials', torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spiking dynamics"""
        batch_size = x.size(0)
        
        # Reset membrane potentials for each batch
        spikes = []
        
        for t in range(x.size(1)):  # Time steps
            # Input current
            input_current = self.input_layer(x[:, t, :])
            
            # Update membrane potentials
            self.membrane_potentials = self.decay * self.membrane_potentials + input_current.mean(dim=0)
            
            # Generate spikes
            spike_mask = self.membrane_potentials > self.threshold
            spikes_t = spike_mask.float()
            
            # Reset spiked neurons
            self.membrane_potentials[spike_mask] = 0.0
            
            spikes.append(spikes_t)
        
        # Aggregate spikes
        spike_aggregate = torch.stack(spikes).mean(dim=0)
        
        # Output
        output = self.output_layer(spike_aggregate)
        
        return output


class HybridLearning:
    """
    Combines multiple learning paradigms.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.supervised_loss = nn.CrossEntropyLoss()
        self.unsupervised_loss = nn.MSELoss()
        self.reinforcement_optimizer = None
    
    def combined_loss(self, supervised_data: Tuple, unsupervised_data: torch.Tensor,
                     reward: Optional[float] = None) -> torch.Tensor:
        """
        Compute combined loss from multiple learning paradigms.
        """
        total_loss = torch.tensor(0.0)
        
        # Supervised learning component
        if supervised_data:
            x_sup, y_sup = supervised_data
            pred_sup = self.model(x_sup)
            loss_sup = self.supervised_loss(pred_sup, y_sup)
            total_loss += loss_sup
        
        # Unsupervised learning component
        if unsupervised_data is not None:
            # Autoencoder-like reconstruction
            encoded = self.model(unsupervised_data)
            # Simplified: use same model for encoding/decoding
            loss_unsup = self.unsupervised_loss(encoded, unsupervised_data)
            total_loss += 0.5 * loss_unsup
        
        # Reinforcement learning component
        if reward is not None:
            # Policy gradient (simplified)
            loss_rl = -torch.tensor(reward, requires_grad=True)
            total_loss += 0.1 * loss_rl
        
        return total_loss

