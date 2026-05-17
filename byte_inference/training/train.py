"""
Training Pipeline for Byte-Level Models

Train on your 4 million papers without tokenization!

Features:
- Multi-GPU distributed training
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Checkpointing
- Weights & Biases integration
- Resume from checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import sys

# Add models to path
sys.path.append(str(Path(__file__).parent.parent / "models"))
from byte_level_transformer import create_byte_model
from mamba_byte_model import create_mamba_model


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model
    model_type: str = "transformer"  # or "mamba"
    model_size: str = "small"

    # Data
    data_dir: str = "invention_data"
    max_seq_length: int = 2048
    batch_size: int = 8
    num_workers: int = 4

    # Training
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Optimization
    use_amp: bool = True  # Mixed precision
    dtype: str = "float16"  # or "bfloat16"

    # Checkpointing
    save_every_n_steps: int = 1000
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "byte-inference"

    # Distributed
    use_ddp: bool = False
    world_size: int = 1
    rank: int = 0


class PaperDataset(Dataset):
    """
    Dataset for training on scientific papers (byte-level)

    Loads papers from invention_data/ directory
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_length: int = 2048,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_length = max_seq_length
        self.split = split

        # Load all papers
        self.papers = self._load_papers()

        print(f"📚 Loaded {len(self.papers)} papers for {split}")

    def _load_papers(self):
        """Load papers from JSON files"""
        papers = []

        if not self.data_dir.exists():
            print(f"⚠️  Data directory {self.data_dir} not found")
            return papers

        # Load from all category directories
        for category_dir in self.data_dir.iterdir():
            if not category_dir.is_dir():
                continue

            papers_file = category_dir / "papers.json"
            if not papers_file.exists():
                continue

            try:
                with open(papers_file) as f:
                    category_papers = json.load(f)
                    papers.extend(category_papers)
            except Exception as e:
                print(f"⚠️  Error loading {papers_file}: {e}")

        return papers

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        """
        Get paper and convert to bytes

        Returns:
            input_bytes: [seq_len] tensor of byte values
            target_bytes: [seq_len] tensor (shifted by 1 for next-byte prediction)
        """
        paper = self.papers[idx]

        # Combine title and abstract
        text = f"{paper['title']}\n\n{paper['abstract']}"

        # Convert to bytes
        byte_values = list(text.encode('utf-8'))

        # Truncate or pad to max_seq_length
        if len(byte_values) > self.max_seq_length:
            byte_values = byte_values[:self.max_seq_length]
        else:
            # Pad with zeros
            byte_values = byte_values + [0] * (self.max_seq_length - len(byte_values))

        # Create input and target
        input_bytes = torch.tensor(byte_values[:-1], dtype=torch.long)
        target_bytes = torch.tensor(byte_values[1:], dtype=torch.long)

        return input_bytes, target_bytes


class Trainer:
    """
    Trainer for byte-level models
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Setup device
        if config.use_ddp:
            self.device = torch.device(f"cuda:{config.rank}")
            dist.init_process_group(backend='nccl')
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔧 Training on {self.device}")

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)

        # Wrap with DDP if needed
        if config.use_ddp:
            self.model = DDP(self.model, device_ids=[config.rank])

        # Create datasets
        self.train_dataset = PaperDataset(
            config.data_dir,
            max_seq_length=config.max_seq_length,
            split="train"
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

        # Wandb
        if config.use_wandb and config.rank == 0:
            try:
                import wandb
                wandb.init(project=config.wandb_project, config=vars(config))
            except:
                print("⚠️  Wandb not available")

    def _create_model(self):
        """Create model"""
        if self.config.model_type == "transformer":
            return create_byte_model(
                self.config.model_size,
                max_seq_length=self.config.max_seq_length
            )
        elif self.config.model_type == "mamba":
            return create_mamba_model(self.config.model_size)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting training for {self.config.num_epochs} epochs")
        print(f"   Total steps: {len(self.train_loader) * self.config.num_epochs:,}")
        print()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.train_epoch()

        print("\n✅ Training complete!")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_bytes, target_bytes) in enumerate(self.train_loader):
            input_bytes = input_bytes.to(self.device)
            target_bytes = target_bytes.to(self.device)

            # Forward pass
            if self.config.use_amp:
                with autocast():
                    logits = self.model(input_bytes)
                    loss = self._compute_loss(logits, target_bytes)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                logits = self.model(input_bytes)
                loss = self._compute_loss(logits, target_bytes)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item()
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]

                    print(
                        f"Epoch {self.epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

                    if self.config.use_wandb and self.config.rank == 0:
                        try:
                            import wandb
                            wandb.log({
                                "loss": avg_loss,
                                "learning_rate": lr,
                                "epoch": self.epoch,
                                "step": self.global_step
                            })
                        except:
                            pass

                # Checkpointing
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint()

    def _compute_loss(self, logits, targets):
        """Compute cross-entropy loss"""
        # logits: [batch, seq_len, 256]
        # targets: [batch, seq_len]
        batch_size, seq_len, vocab_size = logits.shape

        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=0)
        return loss

    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"

        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': vars(self.config)
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        print(f"📂 Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"✅ Resumed from step {self.global_step}")


def main():
    """Main training entry point"""

    # Configuration
    config = TrainingConfig(
        model_type="transformer",
        model_size="small",
        data_dir="invention_data",
        batch_size=4,
        num_epochs=3,
        learning_rate=5e-5,
        use_amp=True,
        log_every_n_steps=10,
        save_every_n_steps=500
    )

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
