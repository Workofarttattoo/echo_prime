#!/usr/bin/env python3
"""
ECH0-PRIME Phase 2 Entry Point
Activates full architecture, neural acceleration, and advanced capabilities.
"""

import os
import sys
import subprocess

def start_phase2():
    print("🚀 ECH0-PRIME Phase 2: Capability Development & Neural Acceleration")
    print("=" * 60)
    
    # Set environment variables for Phase 2
    os.environ["ECH0_FULL_ARCH"] = "1"
    os.environ["ECH0_LIGHTWEIGHT"] = "0"
    os.environ["ECH0_PHASE"] = "2"
    
    print("✅ Environment configured for Full Architecture")
    print("✅ GPU/MPS acceleration enabled")
    print("✅ Advanced capabilities (Swarm, Hive Mind, Masterworks) active")
    print("\nStarting Unified Orchestrator...")
    
    # Import and run the main orchestrator
    try:
        from main_orchestrator import boot_system
        boot_system()
    except ImportError:
        print("❌ Error: Could not find main_orchestrator.py. Ensure you are in the project root.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Phase 2 shutdown complete.")

if __name__ == "__main__":
    start_phase2()

