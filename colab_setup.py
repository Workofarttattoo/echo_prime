#!/usr/bin/env python3
"""
ECH0-PRIME Google Colab GPU Setup
Run this in Google Colab for Phase 2 GPU acceleration.
"""

# Enable GPU runtime first: Runtime > Change runtime type > GPU

!git clone https://github.com/your-repo/echo-prime.git
%cd echo-prime

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate
!pip install pillow librosa

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

# Run GPU-accelerated AGI
from phase2_gpu_integration import demo_gpu_accelerated_agi
demo_gpu_accelerated_agi()

print("\n🎉 Colab GPU setup complete!")
print("ECH0-PRIME Phase 2 running on GPU acceleration")
