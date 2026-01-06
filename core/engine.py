import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from torch.distributions import Normal, kl_divergence


class NeuralCorticalLevel(nn.Module):
    """
    A cortical level with learnable neural network parameters.
    Implements predictive coding with proper generative functions.
    """
    def __init__(self, level_id: int, input_dim: int, output_dim: int, hidden_dim: int, name: str):
        super().__init__()
        self.level_id = level_id
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Encoder: bottom-up (error to expectation)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Decoder: top-down (expectation to prediction)
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Precision (uncertainty) network
        self.precision_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Softplus()  # Ensure positive precision
        )
        
        # Prior distribution parameters (learnable)
        self.prior_mu = nn.Parameter(torch.zeros(output_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(output_dim))
        
        # Current state
        self.expectation = None
        self.prediction = None
        self.error = None
        self.precision = None
        
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        """Bottom-up: encode input to expectation"""
        return self.encoder(input_data)
    
    def decode(self, expectation: torch.Tensor) -> torch.Tensor:
        """Top-down: decode expectation to prediction"""
        return self.decoder(expectation)
    
    def compute_precision(self, expectation: torch.Tensor) -> torch.Tensor:
        """Compute precision (inverse variance) from expectation"""
        return self.precision_net(expectation) + 1e-6  # Add small epsilon for stability
    
    def forward(self, input_data: torch.Tensor, top_down_prediction: Optional[torch.Tensor] = None):
        """
        Forward pass through the cortical level.
        
        Args:
            input_data: Bottom-up input (from lower level or sensory)
            top_down_prediction: Top-down prediction from higher level (optional)
        """
        # Encode input to expectation
        self.expectation = self.encode(input_data)
        
        # Generate top-down prediction if not provided
        if top_down_prediction is None:
            self.prediction = self.decode(self.expectation)
        else:
            self.prediction = top_down_prediction
        
        # Compute prediction error
        self.error = input_data - self.prediction
        
        # Compute precision
        self.precision = self.compute_precision(self.expectation)
        
        return self.expectation, self.error, self.precision
    
    def get_prior_distribution(self):
        """Get prior distribution for KL divergence calculation"""
        return Normal(self.prior_mu, torch.exp(0.5 * self.prior_logvar))
    
    def get_posterior_distribution(self):
        """Get posterior distribution (approximated as Normal)"""
        if self.expectation is None:
            return None
        # Approximate posterior as Normal with mean=expectation
        # In full implementation, this would use learned variance
        posterior_var = 1.0 / (self.precision.mean() + 1e-6)
        return Normal(self.expectation, torch.sqrt(posterior_var))


class HierarchicalGenerativeModel(nn.Module):
    """
    A hierarchy of NeuralCorticalLevel objects with proper message passing.
    Level 0: Sensory (V1/A1) - 1M dim
    Level 1: Perceptual (Object/Word recognition) - 100K dim
    Level 2: Associative (Relations/Grammar) - 10K dim
    Level 3: Prefrontal (Plans/Narratives) - 1K dim
    Level 4: Meta-cortex (Strategy/Self-model) - 100 dim
    """
    def __init__(self, use_cuda: bool = False, lightweight: bool = False):
        super().__init__()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # Define dimensions for each level. Lightweight mode keeps the shape but
        # shrinks sizes to avoid memory blow-ups during benchmarks/smoke tests.
        if lightweight:
            dims = [
                (4096, 1024, "Sensory"),      # Level 0: 4K -> 1K
                (1024, 256, "Perceptual"),    # Level 1: 1K -> 256
                (256, 128, "Associative"),    # Level 2: 256 -> 128
                (128, 64, "Prefrontal"),      # Level 3: 128 -> 64
                (64, 16, "Meta")              # Level 4: 64 -> 16
            ]
        else:
            dims = [
                (1000000, 100000, "Sensory"),      # Level 0: 1M -> 100K
                (100000, 10000, "Perceptual"),     # Level 1: 100K -> 10K
                (10000, 1000, "Associative"),      # Level 2: 10K -> 1K
                (1000, 100, "Prefrontal"),         # Level 3: 1K -> 100
                (100, 10, "Meta")                  # Level 4: 100 -> 10
            ]
        
        # Create neural cortical levels
        self.levels = nn.ModuleList()
        for i, (input_dim, output_dim, name) in enumerate(dims):
            hidden_dim = max(256, output_dim // 4)  # Adaptive hidden dimension
            level = NeuralCorticalLevel(
                level_id=i,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                name=name
            )
            self.levels.append(level)
        
        self.to(self.device)
        
    def step(self, sensory_input: Optional[torch.Tensor] = None):
        """
        A single cycle of predictive processing.
        1. Top-down: Level L predicts Level L-1.
        2. Bottom-up: Level L-1 sends error to Level L.
        3. Local: Update expectations to minimize error.
        """
        # Initialize sensory input if not provided
        first_level_dim = self.levels[0].input_dim
        if sensory_input is None:
            sensory_input = torch.zeros(1, first_level_dim, device=self.device)
        else:
            # Ensure it's on the right device and right size
            if isinstance(sensory_input, np.ndarray):
                sensory_input = torch.from_numpy(sensory_input).float()
            sensory_input = sensory_input.to(self.device).view(1, -1)
            
            # Pad or truncate to match expected size
            current_dim = sensory_input.shape[-1]
            if current_dim < first_level_dim:
                sensory_input = F.pad(sensory_input, (0, first_level_dim - current_dim))
            elif current_dim > first_level_dim:
                sensory_input = sensory_input[:, :first_level_dim]
        
        # Bottom-up pass: encode from sensory to meta
        current_input = sensory_input
        expectations = []
        
        for i, level in enumerate(self.levels):
            print(f"DEBUG: Level {i} {level.name} - Input shape: {current_input.shape}")
            expectation, error, precision = level(current_input)
            expectations.append(expectation)
            current_input = expectation  # Next level's input is this level's expectation
            print(f"DEBUG: Level {i} {level.name} - Output shape: {expectation.shape}")
        
        # Top-down pass: decode from Level L to Level L-1
        top_down_predictions = []
        for i in range(len(self.levels) - 1, -1, -1):
            level = self.levels[i]
            current_expectation = expectations[i]
            prediction = level.decode(current_expectation)
            top_down_predictions.insert(0, prediction)
        
        # Recompute with top-down predictions
        for i, level in enumerate(self.levels):
            if i == 0:
                level_input = sensory_input
            else:
                level_input = expectations[i-1]
            
            if i < len(top_down_predictions):
                top_down = top_down_predictions[i]
                # Resize top-down to match level input if needed
                if len(top_down) != len(level_input):
                    top_down = F.interpolate(
                        top_down.unsqueeze(0).unsqueeze(0),
                        size=len(level_input),
                        mode='linear',
                        align_corners=False
                    ).squeeze()
                level(level_input, top_down_prediction=top_down)
            else:
                level(level_input)
        
        return expectations
    
    def forward(self, sensory_input: Optional[torch.Tensor] = None):
        """Forward pass through the hierarchy"""
        return self.step(sensory_input)


class FreeEnergyEngine:
    """
    Optimization controller that minimizes Variational Free Energy across the hierarchy.
    F = Complexity (KL divergence) - Accuracy (log likelihood)
    """
    def __init__(self, model: HierarchicalGenerativeModel, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def calculate_free_energy(self, sensory_input: Optional[torch.Tensor] = None) -> float:
        """
        Calculates total free energy across all levels.
        F = -log p(x|z) + KL(q(z|x) || p(z))
        """
        first_level_dim = self.model.levels[0].input_dim
        if sensory_input is None:
            sensory_input = torch.zeros(1, first_level_dim, device=self.model.device)
        else:
            if isinstance(sensory_input, np.ndarray):
                sensory_input = torch.from_numpy(sensory_input).float()
            sensory_input = sensory_input.to(self.model.device).view(1, -1)
            
            current_dim = sensory_input.shape[-1]
            if current_dim < first_level_dim:
                sensory_input = F.pad(sensory_input, (0, first_level_dim - current_dim))
            elif current_dim > first_level_dim:
                sensory_input = sensory_input[:, :first_level_dim]
        
        # Forward pass
        self.model.step(sensory_input)
        
        total_f = 0.0
        
        for i, level in enumerate(self.model.levels):
            if level.error is not None and level.precision is not None:
                # Accuracy term: precision-weighted squared error (negative log likelihood)
                # -log p(x|z) ≈ precision * error^2
                accuracy_term = torch.sum(level.precision * (level.error ** 2))
                
                # Complexity term: KL divergence from prior
                prior_dist = level.get_prior_distribution()
                posterior_dist = level.get_posterior_distribution()
                
                if posterior_dist is not None:
                    kl_term = kl_divergence(posterior_dist, prior_dist).sum()
                else:
                    kl_term = torch.tensor(0.0, device=self.model.device)
                
                # Free energy = accuracy + complexity
                level_f = accuracy_term + kl_term
                total_f += level_f.item()
        
        return float(total_f)
    
    def optimize(self, sensory_input: Optional[torch.Tensor] = None, iterations: int = 5):
        """
        Optimize the model to minimize free energy.
        """
        self.model.train()
        
        for iteration in range(iterations):
            self.optimizer.zero_grad()
            
            # Calculate free energy
            fe = self.calculate_free_energy(sensory_input)
            
            # Backward pass (automatic differentiation)
            # We need to create a tensor from the scalar for backward
            fe_tensor = torch.tensor(fe, device=self.model.device, requires_grad=True)
            
            # Recompute to get gradients
            first_level_dim = self.model.levels[0].input_dim
            if sensory_input is None:
                current_input = torch.zeros(1, first_level_dim, device=self.model.device, requires_grad=True)
            else:
                if isinstance(sensory_input, np.ndarray):
                    current_input = torch.from_numpy(sensory_input).float()
                else:
                    current_input = sensory_input.clone()
                
                current_input = current_input.to(self.model.device).view(1, -1)
                if not current_input.requires_grad:
                    current_input = current_input.requires_grad_(True)
                
                current_dim = current_input.shape[-1]
                if current_dim < first_level_dim:
                    current_input = F.pad(current_input, (0, first_level_dim - current_dim))
                elif current_dim > first_level_dim:
                    current_input = current_input[:, :first_level_dim]
            
            # Forward pass
            self.model.step(current_input)
            
            # Compute loss
            loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
            for level in self.model.levels:
                if level.error is not None and level.precision is not None:
                    accuracy = torch.sum(level.precision * (level.error ** 2))
                    prior_dist = level.get_prior_distribution()
                    posterior_dist = level.get_posterior_distribution()
                    if posterior_dist is not None:
                        kl = kl_divergence(posterior_dist, prior_dist).sum()
                    else:
                        kl = torch.tensor(0.0, device=self.model.device)
                    loss = loss + accuracy + kl
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
        
        self.model.eval()
        return self.calculate_free_energy(sensory_input)


class GlobalWorkspace:
    """
    Simulates the 40Hz thalamocortical resonance for conscious access.
    Broadcasts high-precision information across the hierarchy.
    """
    def __init__(self, model: HierarchicalGenerativeModel):
        self.model = model
        self.frequency = 40.0  # Hz
        self.synchrony = 0.0
        self.broadcast_buffer = None
        
    def broadcast(self):
        """
        Synchronizes high-precision signals to the workspace.
        Implements competitive selection based on precision.
        """
        # Collect high-precision signals from all levels
        signals = []
        precisions = []
        
        for level in self.model.levels:
            if level.expectation is not None and level.precision is not None:
                # Weight by precision (high precision = high confidence)
                weighted_signal = level.expectation * level.precision.mean()
                signals.append(weighted_signal)
                precisions.append(level.precision.mean().item())
        
        if signals:
            # Competitive selection: signals with highest precision win
            max_precision_idx = np.argmax(precisions)
            self.broadcast_buffer = signals[max_precision_idx]
            
            # Calculate synchrony (how aligned the signals are)
            if len(signals) > 1:
                # Compute cosine similarity between top signals
                top_signals = sorted(zip(signals, precisions), key=lambda x: x[1], reverse=True)[:3]
                if len(top_signals) > 1:
                    similarities = []
                    for i in range(len(top_signals) - 1):
                        s1 = top_signals[i][0]
                        s2 = top_signals[i+1][0]
                        # Normalize and compute cosine similarity
                        s1_norm = s1 / (s1.norm() + 1e-9)
                        s2_norm = s2 / (s2.norm() + 1e-9)
                        sim = torch.dot(s1_norm, s2_norm).item()
                        similarities.append(sim)
                    self.synchrony = np.mean(similarities) if similarities else 0.0
                else:
                    self.synchrony = 1.0
            else:
                self.synchrony = 1.0
        else:
            self.broadcast_buffer = None
            self.synchrony = 0.0
        
        return self.broadcast_buffer, self.synchrony
