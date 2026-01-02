"""
Real probabilistic reasoning with variational inference and MCMC.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Dict, List, Optional, Tuple, Callable
import warnings

try:
    import pyro
    import pyro.distributions as pyro_dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.mcmc import MCMC, NUTS
    from pyro.optim import Adam
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    warnings.warn("Pyro not available. Using simplified probabilistic reasoning.")


class VariationalInference:
    """
    Implements proper variational inference with reparameterization trick.
    """
    def __init__(self, latent_dim: int = 100, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device = device
        
        # Encoder: q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),  # Assuming 28x28 input
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log-variance
        ).to(device)
        
        # Decoder: p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-3
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode x to latent distribution parameters"""
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode z to x"""
        return self.decoder(z)
    
    def calculate_elbo(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                     x_recon: torch.Tensor) -> torch.Tensor:
        """
        Calculate Evidence Lower Bound.
        ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        """
        # Reconstruction loss (log likelihood)
        recon_loss = nn.functional.binary_cross_entropy(
            x_recon, x, reduction='sum'
        )
        
        # KL divergence: KL(q(z|x) || N(0, I))
        kl_div = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # ELBO = -recon_loss - kl_div (we minimize negative ELBO)
        elbo = -recon_loss - kl_div
        
        return elbo
    
    def infer(self, x: torch.Tensor, num_iterations: int = 100) -> Dict[str, torch.Tensor]:
        """
        Perform variational inference.
        
        Args:
            x: Input data [batch_size, input_dim]
            num_iterations: Number of optimization steps
        
        Returns:
            Dictionary with mu, logvar, z, x_recon
        """
        x = x.to(self.device)
        
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            
            # Encode
            mu, logvar = self.encode(x)
            
            # Reparameterize
            z = self.reparameterize(mu, logvar)
            
            # Decode
            x_recon = self.decode(z)
            
            # Calculate ELBO
            elbo = self.calculate_elbo(x, mu, logvar, x_recon)
            
            # Maximize ELBO = minimize negative ELBO
            loss = -elbo
            loss.backward()
            self.optimizer.step()
        
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "x_recon": x_recon,
            "elbo": elbo.item()
        }


class MCMCSampling:
    """
    Implements MCMC sampling for exact inference.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def sample_posterior(self, log_likelihood_fn: Callable, prior: dist.Distribution,
                        num_samples: int = 1000, num_warmup: int = 100) -> torch.Tensor:
        """
        Sample from posterior using MCMC.
        
        Args:
            log_likelihood_fn: Function that computes log p(x|z)
            prior: Prior distribution p(z)
            num_samples: Number of samples
            num_warmup: Number of warmup samples
        
        Returns:
            Samples from posterior
        """
        if PYRO_AVAILABLE:
            # Use Pyro's NUTS sampler
            def model():
                z = pyro.sample("z", prior)
                log_prob = log_likelihood_fn(z)
                pyro.factor("obs", log_prob)
                return z
            
            # NUTS kernel
            nuts_kernel = NUTS(model)
            
            # MCMC
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=num_warmup)
            mcmc.run()
            
            # Get samples
            samples = mcmc.get_samples()["z"]
            return samples
        else:
            # Simplified: Metropolis-Hastings
            samples = []
            current = prior.sample()
            current_log_prob = log_likelihood_fn(current) + prior.log_prob(current).sum()
            
            for i in range(num_samples + num_warmup):
                # Propose new sample
                proposal = current + torch.randn_like(current) * 0.1
                proposal_log_prob = log_likelihood_fn(proposal) + prior.log_prob(proposal).sum()
                
                # Accept/reject
                accept_prob = torch.exp(proposal_log_prob - current_log_prob)
                if torch.rand(1) < accept_prob:
                    current = proposal
                    current_log_prob = proposal_log_prob
                
                # Store sample (after warmup)
                if i >= num_warmup:
                    samples.append(current.clone())
            
            return torch.stack(samples)


class ProbabilisticReasoning:
    """
    Complete probabilistic reasoning system with VI and MCMC.
    """
    def __init__(self, latent_dim: int = 100, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device = device
        self.vi = VariationalInference(latent_dim, device)
        self.mcmc = MCMCSampling(device)
    
    def calculate_elbo(self, x: torch.Tensor, model_log_likelihood_fn: Optional[Callable] = None) -> float:
        """
        Computes the Evidence Lower Bound.
        """
        if model_log_likelihood_fn is None:
            # Use default VAE
            result = self.vi.infer(x, num_iterations=10)
            return result["elbo"]
        else:
            # Custom model
            mu, logvar = self.vi.encode(x)
            z = self.vi.reparameterize(mu, logvar)
            
            # Log likelihood from custom function
            log_px_z = model_log_likelihood_fn(x, z)
            
            # KL divergence
            kl_div = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            
            elbo = log_px_z - kl_div
            return elbo.item()
    
    def infer(self, x: torch.Tensor, iterations: int = 50, method: str = "vi") -> Dict:
        """
        Perform probabilistic inference.
        
        Args:
            x: Input data
            iterations: Number of iterations
            method: "vi" (variational) or "mcmc" (MCMC)
        
        Returns:
            Inference results
        """
        if method == "vi":
            return self.vi.infer(x, num_iterations=iterations)
        elif method == "mcmc":
            # Define prior
            prior = dist.Normal(torch.zeros(self.latent_dim, device=self.device),
                              torch.ones(self.latent_dim, device=self.device))
            
            # Define likelihood
            def log_likelihood(z):
                x_recon = self.vi.decode(z)
                return dist.Normal(x_recon, 0.1).log_prob(x).sum()
            
            samples = self.mcmc.sample_posterior(log_likelihood, prior, num_samples=iterations)
            
            return {
                "samples": samples,
                "mean": samples.mean(dim=0),
                "std": samples.std(dim=0)
            }
        else:
            raise ValueError(f"Unknown method: {method}")


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian neural network with weight uncertainty.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Weight distributions (mean and log-variance)
        self.w1_mu = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.w1_logvar = nn.Parameter(torch.randn(input_dim, hidden_dim) * -3)
        self.w2_mu = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        self.w2_logvar = nn.Parameter(torch.randn(hidden_dim, output_dim) * -3)
        
        self.b1_mu = nn.Parameter(torch.zeros(hidden_dim))
        self.b1_logvar = nn.Parameter(torch.ones(hidden_dim) * -3)
        self.b2_mu = nn.Parameter(torch.zeros(output_dim))
        self.b2_logvar = nn.Parameter(torch.ones(output_dim) * -3)
    
    def sample_weights(self):
        """Sample weights from their distributions"""
        w1 = self.w1_mu + torch.exp(0.5 * self.w1_logvar) * torch.randn_like(self.w1_mu)
        w2 = self.w2_mu + torch.exp(0.5 * self.w2_logvar) * torch.randn_like(self.w2_mu)
        b1 = self.b1_mu + torch.exp(0.5 * self.b1_logvar) * torch.randn_like(self.b1_mu)
        b2 = self.b2_mu + torch.exp(0.5 * self.b2_logvar) * torch.randn_like(self.b2_mu)
        return w1, w2, b1, b2
    
    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Forward pass with weight sampling"""
        outputs = []
        for _ in range(num_samples):
            w1, w2, b1, b2 = self.sample_weights()
            h = torch.relu(x @ w1 + b1)
            y = h @ w2 + b2
            outputs.append(y)
        
        # Average over samples
        return torch.stack(outputs).mean(dim=0)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from prior"""
        kl = 0.0
        
        # KL for each weight matrix
        for mu, logvar in [(self.w1_mu, self.w1_logvar), (self.w2_mu, self.w2_logvar),
                          (self.b1_mu, self.b1_logvar), (self.b2_mu, self.b2_logvar)]:
            kl += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl

