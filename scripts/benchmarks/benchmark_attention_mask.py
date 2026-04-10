import torch
import time
from research.novel_architectures import AttentionPattern

def original_hierarchical_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create hierarchical attention mask (Original)"""
    mask = torch.ones(seq_len, seq_len, device=device)

    # Reduce attention for distant positions
    for i in range(seq_len):
        for j in range(seq_len):
            distance = abs(i - j)
            mask[i, j] = 1.0 / (1.0 + distance)

    return mask.unsqueeze(0)

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pattern = AttentionPattern(dim=64, num_heads=8, pattern_type="hierarchical").to(device)

    seq_lengths = [32, 128, 512, 1024]

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")

        # Warmup and correctness check
        orig_out = original_hierarchical_mask(seq_len, device)
        opt_out = pattern._hierarchical_mask(seq_len, device)

        if not torch.allclose(orig_out, opt_out):
            print("WARNING: Outputs do not match!")
        else:
            print("Correctness check passed.")

        # Benchmark Original
        start = time.time()
        # Run multiple iterations for better measurement
        iters = 10
        for _ in range(iters):
            _ = original_hierarchical_mask(seq_len, device)
        orig_time = (time.time() - start) / iters
        print(f"Original Time:  {orig_time:.6f} seconds")

        # Benchmark Optimized
        start = time.time()
        for _ in range(iters):
            _ = pattern._hierarchical_mask(seq_len, device)
        opt_time = (time.time() - start) / iters
        print(f"Optimized Time: {opt_time:.6f} seconds")

        if opt_time > 0:
            speedup = orig_time / opt_time
            print(f"Speedup:        {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
