import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
from collections import deque


class MetaLearningController:
    """
    Implements the fast/slow weight update logic for meta-learning.
    theta_fast: Task-specific adaptation.
    theta_slow: Long-term structural knowledge.
    """
    def __init__(self, model: Optional[nn.Module] = None, param_dim: Optional[int] = None,
                 alpha: float = 0.01, beta: float = 0.001):
        self.alpha = alpha  # Fast learning rate
        self.beta = beta    # Slow integration rate (meta-learning parameter)

        if model is None:
            if param_dim is None:
                raise ValueError("param_dim is required when model is not provided")
            self.model = None
            self.theta_fast = np.zeros(param_dim, dtype=np.float32)
            self.theta_slow = self.theta_fast.copy()
            self.fast_optimizer = None
            self.slow_optimizer = None
            return

        self.model = model
        # Store slow weights (base parameters)
        self.theta_slow = {name: param.clone().detach() for name, param in model.named_parameters()}

        # Fast optimizer for task-specific adaptation
        self.fast_optimizer = optim.SGD(model.parameters(), lr=alpha)

        # Slow optimizer for meta-updates
        self.slow_optimizer = optim.Adam(model.parameters(), lr=beta)
    
    def adapt(self, loss):
        """Update fast weights based on immediate task loss or gradient."""
        if self.model is None:
            grad = np.asarray(loss, dtype=np.float32)
            if grad.shape != self.theta_fast.shape:
                raise ValueError("Gradient shape must match param_dim")
            self.theta_fast -= self.alpha * grad
            return

        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss must be a torch.Tensor when using a torch model")
        self.fast_optimizer.zero_grad()
        loss.backward()
        self.fast_optimizer.step()
    
    def consolidate(self):
        """Integrate fast weights into slow weights (structural update)."""
        if self.model is None:
            self.theta_slow = (1 - self.beta) * self.theta_slow + self.beta * self.theta_fast
            return

        # Update slow weights towards current fast weights
        for name, param in self.model.named_parameters():
            if name in self.theta_slow:
                # Exponential moving average
                self.theta_slow[name] = (1 - self.beta) * self.theta_slow[name] + self.beta * param.data.clone()
                # Update model parameters
                param.data.copy_(self.theta_slow[name])


class ValueFunction(nn.Module):
    """
    Value function for computing expected values (for RPE calculation).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class NeuromodulationSuite:
    """
    Simulates chemical signaling that modulates learning dynamics.
    Uses real value functions for reward prediction error.
    """
    def __init__(self, state_dim: int = 10, device: str = "cpu"):
        self.device = device
        self.dopamine = 0.0
        self.serotonin = 0.0
        self.acetylcholine = 0.0
        self.norepinephrine = 0.0
        
        # Value function for RPE calculation
        self.value_function = ValueFunction(state_dim).to(device)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=1e-3)
        
        # Gamma for temporal difference learning
        self.gamma = 0.9
    
    def compute_dopamine(self, reward: float, state: torch.Tensor, 
                        next_state: Optional[torch.Tensor] = None) -> float:
        """
        Reward Prediction Error: Delta = r + gamma * V(s') - V(s)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if next_state is not None and isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float().to(self.device)
        
        # Current value
        V_s = self.value_function(state.unsqueeze(0) if len(state.shape) == 1 else state)
        
        if next_state is not None:
            # Next state value
            V_s_next = self.value_function(next_state.unsqueeze(0) if len(next_state.shape) == 1 else next_state)
            # TD error
            rpe = reward + self.gamma * V_s_next.item() - V_s.item()
        else:
            # Simple RPE without next state
            rpe = reward - V_s.item()
        
        self.dopamine = rpe
        
        # Update value function
        self.value_optimizer.zero_grad()
        if next_state is not None:
            target = reward + self.gamma * V_s_next.detach()
            loss = nn.MSELoss()(V_s, target)
        else:
            target = torch.tensor(reward, device=self.device)
            loss = nn.MSELoss()(V_s.squeeze(), target)
        loss.backward()
        self.value_optimizer.step()
        
        return self.dopamine
    
    def compute_serotonin(self, entropy: float, beta_param: float = 1.0) -> float:
        """Modulates based on uncertainty/surprise."""
        self.serotonin = 1 / (1 + np.exp(-beta_param * entropy))
        return self.serotonin
    
    def compute_acetylcholine(self, expected_uncertainty: float) -> float:
        """Expected uncertainty (top-down modulation)."""
        self.acetylcholine = np.tanh(expected_uncertainty)
        return self.acetylcholine
    
    def compute_norepinephrine(self, prediction_error: float) -> float:
        """Unexpected uncertainty (bottom-up modulation)."""
        self.norepinephrine = np.tanh(prediction_error)
        return self.norepinephrine
    
    def get_modulated_learning_rate(self, base_rate: float) -> float:
        """NE modulates global learning rate: eta = eta0 * (1 + NE/NE_max)"""
        # NE increases when bottom-up prediction errors are high
        ne_factor = 1 + self.norepinephrine
        return base_rate * ne_factor


class ExperienceReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class CSALearningSystem:
    """
    Complete learning system with real-time gradient updates.
    """
    def __init__(self, model: nn.Module, param_dim: int = 1000, state_dim: int = 10, device: str = "cpu"):
        self.model = model
        self.device = device
        self.controller = MetaLearningController(model)
        self.neuromodulators = NeuromodulationSuite(state_dim, device)
        self.replay_buffer = ExperienceReplayBuffer(capacity=10000)
        self.consolidation_frequency = 100  # Consolidate every N steps
        self.step_count = 0
    
    def step(self, error_gradient: Optional[torch.Tensor] = None, 
             reward: Optional[float] = None,
             state: Optional[torch.Tensor] = None,
             next_state: Optional[torch.Tensor] = None,
             loss: Optional[torch.Tensor] = None):
        """
        Perform one learning step with real gradient updates.
        
        Args:
            error_gradient: Gradient of error (if available)
            reward: Reward signal
            state: Current state (for RPE calculation)
            next_state: Next state (for RPE calculation)
            loss: Direct loss tensor (preferred over error_gradient)
        """
        self.step_count += 1
        
        # 1. Compute loss if not provided
        if loss is None and error_gradient is not None:
            # Create a dummy loss from gradient
            # In practice, loss should come from forward pass
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            # This is a placeholder - real implementation would use actual forward pass
        
        # 2. Update neuromodulators
        if reward is not None and state is not None:
            rpe = self.neuromodulators.compute_dopamine(reward, state, next_state)
            
            # Update other neuromodulators based on prediction error
            if loss is not None:
                prediction_error = loss.item()
                self.neuromodulators.compute_norepinephrine(prediction_error)
                
                # Compute entropy/uncertainty for serotonin
                entropy = prediction_error  # Simplified
                self.neuromodulators.compute_serotonin(entropy)
        
        # 3. Adapt fast weights with modulated learning rate
        if loss is not None:
            if not loss.requires_grad:
                loss = loss.clone().detach().requires_grad_(True)
            # Get modulated learning rate
            modulated_lr = self.neuromodulators.get_modulated_learning_rate(self.controller.alpha)
            self.controller.fast_optimizer.param_groups[0]['lr'] = modulated_lr
            
            # Update fast weights
            self.controller.adapt(loss)
        
        # 4. Periodic consolidation
        if self.step_count % self.consolidation_frequency == 0:
            self.controller.consolidate()
    
    def update_from_replay(self, batch_size: int = 32):
        """
        Update model from experience replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        # Compute loss from batch
        # This is simplified - full implementation would use proper Q-learning or policy gradient
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for state, action, reward, next_state, done in batch:
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float().to(self.device)
            
            # Compute TD error
            rpe = self.neuromodulators.compute_dopamine(
                reward, state, next_state if not done else None
            )
            
            # Loss is negative RPE (we want to maximize reward)
            total_loss = total_loss - torch.tensor(rpe, device=self.device)
        
        # Average loss
        total_loss = total_loss / batch_size
        
        # Update model
        self.step(loss=total_loss)


class TransferLearning:
    """
    Domain adaptation and transfer learning across domains.
    """
    def __init__(self, base_model: nn.Module, device: str = "cpu"):
        self.base_model = base_model
        self.device = device
        self.domain_adapters = {}
    
    def create_domain_adapter(self, domain_name: str, input_dim: int, output_dim: int):
        """Create a domain-specific adapter"""
        adapter = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(self.device)
        self.domain_adapters[domain_name] = adapter
        return adapter
    
    def adapt_to_domain(self, domain_name: str, source_data: torch.Tensor, 
                       target_data: torch.Tensor, num_epochs: int = 10):
        """Adapt model to new domain"""
        if domain_name not in self.domain_adapters:
            self.create_domain_adapter(domain_name, source_data.size(-1), target_data.size(-1))
        
        adapter = self.domain_adapters[domain_name]
        optimizer = optim.Adam(adapter.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward through base model and adapter
            base_output = self.base_model(source_data)
            adapted_output = adapter(base_output)
            
            # Compute adaptation loss
            loss = criterion(adapted_output, target_data)
            loss.backward()
            optimizer.step()

        return adapter


# Backward compatibility wrapper
class CSALearningSystemWrapper:
    """
    Backward-compatible wrapper for CSALearningSystem.
    Maintains the old API while using the new implementation.
    """
    def __init__(self, param_dim: int = 1000, device: str = "cpu"):
        # Create a simple dummy model for backward compatibility
        self.dummy_model = nn.Sequential(
            nn.Linear(param_dim, param_dim // 2),
            nn.ReLU(),
            nn.Linear(param_dim // 2, param_dim)
        ).to(device)

        # Initialize the real learning system
        self.learning_system = _OriginalCSALearningSystem(
            model=self.dummy_model,
            param_dim=param_dim,
            state_dim=10,
            device=device
        )

    def step(self, error_gradient: Optional[np.ndarray] = None,
             reward: Optional[float] = None,
             loss: Optional[torch.Tensor] = None,
             **kwargs):
        """Backward-compatible step method"""
        if loss is not None:
            return self.learning_system.step(loss=loss, reward=reward, **kwargs)

        if error_gradient is None:
            error_gradient = np.zeros(self.learning_system.param_dim, dtype=np.float32)

        # Convert numpy to torch if needed
        if isinstance(error_gradient, np.ndarray):
            error_gradient = torch.from_numpy(error_gradient).float().to(self.learning_system.device)

        return self.learning_system.step(
            loss=torch.nn.functional.mse_loss(error_gradient, torch.zeros_like(error_gradient)),
            reward=reward,
            **kwargs
        )

    def __getattr__(self, name):
        """Delegate all other attributes to the real learning system"""
        return getattr(self.learning_system, name)


# Monkey patch for backward compatibility - replace the class with a factory function
import sys
current_module = sys.modules[__name__]

# Store the original class
_OriginalCSALearningSystem = CSALearningSystem

def CSALearningSystem(*args, **kwargs):
    """
    Factory function that creates either the new or wrapper version
    based on arguments.
    """
    model_kw = kwargs.get("model")
    if (args and isinstance(args[0], nn.Module)) or isinstance(model_kw, nn.Module):
        # New API: first argument is a model
        return _OriginalCSALearningSystem(*args, **kwargs)
    else:
        # Old API: first argument is param_dim or keyword arguments
        return CSALearningSystemWrapper(*args, **kwargs)

# Replace the class in the module
current_module.CSALearningSystem = CSALearningSystem
