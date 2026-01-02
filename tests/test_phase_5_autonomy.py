import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from main_orchestrator import EchoPrimeAGI

def test_phase_5_autonomy():
    print("--- Verifying ECH0-PRIME Phase 5: Goal-Directed Autonomy ---")
    
    agi = EchoPrimeAGI()
    
    # Mission: Create a directory, list it, and delete it (Resetting state)
    mission = (
        "Perform a system clean-up: "
        "1. Create a directory named 'temp_audit'. "
        "2. List the directory content. "
        "3. Remove the directory 'temp_audit'. "
        "Mission status must be MISSION_STATUS: ACHIEVED once done."
    )
    
    # Run the mission
    agi.execute_mission(mission, max_cycles=6)
    
    print("\n[Final System State Verification]")
    if not os.path.exists('/Users/noone/.gemini/antigravity/scratch/echo_prime/temp_audit'):
        print("SUCCESS: Target directory 'temp_audit' does not exist (Goal Achievement).")
    else:
        print("WARNING: Target directory 'temp_audit' still exists.")

if __name__ == "__main__":
    test_phase_5_autonomy()
