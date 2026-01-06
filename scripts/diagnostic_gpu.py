import torch
import sys
import os
import time

# Add project root
sys.path.insert(0, os.getcwd())

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

print("\n1. Checking MPS...")
if torch.backends.mps.is_available():
    print("✅ MPS is available")
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"✅ Tensor created on MPS: {x}")
else:
    print("❌ MPS not available")

print("\n2. Initializing ECH0-PRIME (Verbose)...")
try:
    from main_orchestrator import EchoPrimeAGI
    start = time.time()
    agi = EchoPrimeAGI(enable_voice=False, device="auto")
    print(f"✅ ECH0-PRIME initialized in {time.time() - start:.2f}s")
    
    print("\n3. Testing Cognitive Cycle...")
    import numpy as np
    dummy_input = np.zeros(1000)
    res = agi.cognitive_cycle(dummy_input, "System diagnostic test")
    print(f"✅ Cycle complete. Result type: {type(res)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
