"""
Distributed training and processing infrastructure.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import List, Optional, Dict
import os


class DistributedTraining:
    """
    Distributed training across multiple GPUs/nodes.
    """
    def __init__(self, backend: str = "nccl", init_method: str = "env://"):
        self.backend = backend
        self.init_method = init_method
        self.world_size = None
        self.rank = None
        self.is_initialized = False
    
    def initialize(self, rank: int = None, world_size: int = None):
        """Initialize distributed training"""
        if dist.is_available():
            if rank is None:
                rank = int(os.environ.get("RANK", 0))
            if world_size is None:
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                rank=rank,
                world_size=world_size
            )
            
            self.rank = rank
            self.world_size = world_size
            self.is_initialized = True
        else:
            print("Warning: Distributed training not available")
    
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model with DDP"""
        if self.is_initialized and self.world_size > 1:
            model = model.to(device)
            model = DDP(model, device_ids=[device.index] if device.index is not None else None)
        return model
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_initialized:
            dist.destroy_process_group()


class ModelParallelism:
    """
    Split large models across multiple devices.
    """
    def __init__(self, devices: List[torch.device]):
        self.devices = devices
    
    def split_model(self, model: nn.Module) -> nn.Module:
        """Split model across devices"""
        if len(self.devices) == 1:
            return model.to(self.devices[0])
        
        # Simple strategy: split by layers
        layers = list(model.children())
        num_layers = len(layers)
        layers_per_device = num_layers // len(self.devices)
        
        parallel_layers = []
        for i, device in enumerate(self.devices):
            start_idx = i * layers_per_device
            end_idx = (i + 1) * layers_per_device if i < len(self.devices) - 1 else num_layers
            
            device_layers = nn.Sequential(*layers[start_idx:end_idx]).to(device)
            parallel_layers.append(device_layers)
        
        return ParallelModel(parallel_layers, self.devices)


class ParallelModel(nn.Module):
    """Model split across multiple devices"""
    def __init__(self, layers: List[nn.Module], devices: List[torch.device]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.devices = devices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass across devices"""
        for i, layer in enumerate(self.layers):
            x = x.to(self.devices[i])
            x = layer(x)
        return x


class DataParallelism:
    """
    Parallel data processing.
    """
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
    
    def create_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """Create DataLoader with parallel processing"""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )


class FaultTolerance:
    """
    Handle node failures gracefully.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer, epoch: int, loss: float):
        """Save checkpoint"""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only last N checkpoints
        self._cleanup_old_checkpoints(keep_last=5)
    
    def load_checkpoint(self, model: nn.Module, optimizer, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["loss"]
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints"""
        import glob
        checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt")))
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                os.remove(checkpoint)

