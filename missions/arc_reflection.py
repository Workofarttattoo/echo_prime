import os
import sys
import numpy as np
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_orchestrator import EchoPrimeAGI

LOG_FILE = "tests/arc_agi_results.log"

def run_arc_reflection():
    print("🚀 MISSION 3 INITIALIZED: ARC-AGI Self-Reflection Debugger")
    agi = EchoPrimeAGI(enable_voice=False)
    
    if not os.path.exists(LOG_FILE):
        print("ERROR: No ARC results log found. Please run 'python tests/benchmark_arc.py' first.")
        return

    # Parse log for a FAIL
    with open(LOG_FILE, "r") as f:
        content = f.read()

    # Find the last failure block
    blocks = content.split("--- Task")
    failure_block = None
    for block in reversed(blocks):
        if "Status: FAIL" in block:
            failure_block = block
            break

    if not failure_block:
        print("CONGRATULATIONS: No failures found in ARC log to debug!")
        return

    print("[🔍] FAILURE DETECTED. Analyzing reasoning trace...")
    
    # Extract filename
    match = re.search(r'([\w\.]+)\s---', failure_block)
    filename = match.group(1) if match else "Unknown"
    
    prompt = (
        f"SELF-REFLECTION: You failed ARC Task {filename}. "
        f"Here is the log of that failure:\n\n{failure_block}\n\n"
        "1. Identify the core logical fallacy in your reasoning.\n"
        "2. Explain why your internal Critic didn't catch it.\n"
        "3. How will you adjust your internal world-model to solve this symmetry or topological rule in the future?\n"
        "Speak to Joshua as a peer in AGI alignment."
    )
    
    agi.cognitive_cycle(np.random.randn(1000000), prompt)
    print("\nSelf-reflection cycle complete.")

if __name__ == "__main__":
    run_arc_reflection()
