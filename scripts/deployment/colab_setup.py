#!/usr/bin/env python3
"""
ECH0-PRIME Google Colab GPU Setup
Run this in Google Colab for Phase 2 GPU acceleration.

Usage (in a Colab cell):
    # First cell — clone & install:
    #   !git clone https://github.com/Workofarttattoo/echo_prime.git
    #   %cd echo_prime
    #   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    #   !pip install transformers accelerate pillow librosa
    #
    # Second cell — run this script:
    #   %run scripts/deployment/colab_setup.py
"""

import torch


def main():
    print("GPU Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU Memory: {mem_gb:.1f} GB")

    # Run GPU-accelerated AGI demo
    try:
        from scripts.capabilities.phase2_gpu_integration import demo_gpu_accelerated_agi
        demo_gpu_accelerated_agi()
    except ImportError:
        print("⚠️  phase2_gpu_integration module not found — skipping demo.")

    print("\n🎉 Colab GPU setup complete!")
    print("ECH0-PRIME Phase 2 running on GPU acceleration")


if __name__ == "__main__":
    main()
