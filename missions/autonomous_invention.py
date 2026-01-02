import os
import sys
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_orchestrator import EchoPrimeAGI

def run_invention_mission(goal: str):
    print(f"🚀 MISSION 2 INITIALIZED: Autonomous Invention Cycle -> {goal}")
    agi = EchoPrimeAGI(enable_voice=False)
    agi.set_mission_goal(goal)
    
    # 1. Broad Research Scan
    print("[🛰️] Launching Initial Arxiv Scan...")
    research_query = f"ACTION: {{'tool': 'scan_arxiv', 'args': {{'query': '{goal}'}}}}"
    agi.cognitive_cycle(np.random.randn(1000000), research_query)
    
    # 2. Iterate and Synthesize
    # We'll run 3 cycles of deep thought
    for i in range(3):
        print(f"[🧠] Thinking Cycle {i+1}/3...")
        prompt = (
            f"REFLECT: Based on your recent Arxiv scans and local memory about {goal}, "
            "what is the single most significant theoretical bottleneck? "
            "Propose a 'Level 10' experiment to bypass it. "
            "Speak your findings clearly."
        )
        agi.cognitive_cycle(np.random.randn(1000000), prompt)
        time.sleep(2)

    final_prompt = "MISSION COMPLETE. Final Summary of your Invention Proposal:"
    agi.cognitive_cycle(np.random.randn(1000000), final_prompt)
    print("\nMission Complete. Results stored in logs and memory.")

if __name__ == "__main__":
    target = "Quantum Dot Solar Cell Efficiency Bottlenecks and Hot Carrier Extraction"
    run_invention_mission(target)
