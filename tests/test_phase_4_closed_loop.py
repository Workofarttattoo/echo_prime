import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from main_orchestrator import EchoPrimeAGI

def test_phase_4_closed_loop():
    print("--- Verifying ECH0-PRIME Phase 4: Closed-Loop Action & Memory ---")
    
    agi = EchoPrimeAGI()
    
    # Mock visual sensory input
    mock_input = np.random.randn(1000000)
    
    # Mock a situation where the LLM proposes an action
    # We'll modify the reasoning result temporarily to simulate an action-oriented thought
    print("\n[Executing Cognitive Cycle with Action Intent]")
    
    # We use a prompt that nudges the LLM toward an action
    outcome = agi.cognitive_cycle(mock_input, "Observe the environment and create a directory named 'agi_log'. ACTION: {\"tool\": \"mkdir\", \"args\": [\"agi_log\"]}")
    
    print("\n[Cycle Outcome]")
    print(f"Status: {outcome['status']}")
    print(f"Free Energy: {outcome['free_energy']:.4f}")
    print(f"Surprise: {outcome['surprise']}")
    print(f"Memory Count: {len(agi.memory.episodic.storage)}")
    
    print("\n[Actuation Result]")
    for res in outcome['actions']:
        print(f"Actuator Feedback: {res}")
        
    # Verify file system change
    if os.path.exists('/Users/noone/.gemini/antigravity/scratch/echo_prime/agi_log'):
        print("\nSUCCESS: AGI successfully performed a system action (mkdir agi_log).")
    else:
        print("\nFAILURE: Directory 'agi_log' was not created.")

if __name__ == "__main__":
    test_phase_4_closed_loop()
