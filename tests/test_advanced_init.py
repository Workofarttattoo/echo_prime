import sys
import os
import torch
import numpy as np
from main_orchestrator import EchoPrimeAGI

def test_init():
    print("Testing ECH0-PRIME Advanced Initialization...")
    # Force CPU and try to be as lightweight as possible while keeping advanced components
    try:
        agi = EchoPrimeAGI(enable_voice=False, device="cpu", lightweight=False)
        print("✓ Successfully initialized with advanced components")
        
        if agi.prompt_masterworks:
            print("✓ Prompt Masterworks available")
        else:
            print("❌ Prompt Masterworks NOT available")
            
        if agi.performance_profiler:
            print("✓ Performance Profiler available")
        else:
            print("❌ Performance Profiler NOT available")
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

if __name__ == "__main__":
    test_init()

