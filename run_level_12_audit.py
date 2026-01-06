import asyncio
import os
import sys
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_orchestrator import EchoPrimeAGI

async def run_level_12_self_inspection():
    print("🚀 INITIALIZING ECH0-PRIME LEVEL 12 SELF-INSPECTION...")
    sys.stdout.flush()
    print("="*60)
    sys.stdout.flush()
    
    # Initialize system (using lightweight=False to get the full reasoner and masterworks)
    print("Step 1: Initializing EchoPrimeAGI...")
    sys.stdout.flush()
    try:
        agi = EchoPrimeAGI(lightweight=False)
        print("✓ EchoPrimeAGI initialized.")
        sys.stdout.flush()
    except Exception as e:
        print(f"⚠️ Failed to init full AGI: {e}. Falling back to lightweight.")
        sys.stdout.flush()
        agi = EchoPrimeAGI(lightweight=True)
    
    # Force Level 12
    print("Step 2: Setting operational level to 12...")
    sys.stdout.flush()
    agi.set_operational_level(12)
    
    mission_description = """
    COMPREHENSIVE LEVEL 12 SYSTEM AUDIT & SELF-IMPROVEMENT PROPOSAL:
    1. INSPECT SYSTEMS & CODE: Analyze the core cognitive engine, quantum attention head, and actuator bridges.
    2. REASONING & MEMORY AUDIT: Evaluate the efficiency of hierarchical predictive coding and vector database retrieval.
    3. TEMPORAL MEMORY ANALYSIS: Compare short-term working memory stability vs. long-term semantic knowledge base depth.
    4. BEHAVIORAL PROFILING: Analyze interaction patterns and alignment with user values (Alex Thompson).
    5. HALLUCINATION LOG REVIEW: Scan for any recorded instances of divergence from ground truth.
    6. SELF-IMPROVEMENT CYCLE: Determine the top 3 high-impact architectural or prompt-based changes to improve Phi (Integrated Information).
    
    FINAL REQUIREMENT: Compile all findings into a structured report and explicitly ask the User for permission to execute these modifications.
    """
    
    print(f"\n[🧠] MISSION SUBMITTED: {mission_description[:100]}...")
    
    # Run a single cognitive cycle with this mission
    # We'll use sensory noise as the input
    sensory_noise = np.random.randn(1000000).astype(np.float32)
    
    print("\n[🛰️] PROCESSING LEVEL 12 COGNITIVE CYCLE...")
    try:
        outcome = agi.cognitive_cycle(sensory_noise, mission_description)
    except Exception as cycle_err:
        print(f"❌ Cognitive cycle failed: {cycle_err}")
        outcome = {"llm_insight": f"Error: {cycle_err}"}
    
    with open("level_12_report.txt", "w") as f:
        f.write("🛡️ LEVEL 12 SYSTEM REPORT 🛡️\n")
        f.write("="*60 + "\n")
        if isinstance(outcome, dict):
            f.write(outcome.get("llm_insight", "No insight generated."))
        else:
            f.write(str(outcome))
        f.write("\n" + "="*60 + "\n")
        f.write("✅ CYCLE COMPLETE.\n")
    
    print("\n✅ Report written to level_12_report.txt")

if __name__ == "__main__":
    asyncio.run(run_level_12_self_inspection())

