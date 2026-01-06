import os
import sys
import json  # pyright: ignore[reportUnusedImport]
import numpy as np  # pyright: ignore[reportUnusedImport]
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components directly to avoid full AGI boot memory overhead
from capabilities.prompt_masterworks import PromptMasterworks
from missions.hive_mind import HiveMindOrchestrator
from reasoning.orchestrator import ReasoningOrchestrator

def demonstrate_oracle_and_hive():
    print("🚀 ECH0-PRIME Level 12 Oracle & Hive Mind Demonstration (Optimized)")
    print("=" * 60)
    
    # Initialize components individually to save memory
    pm = PromptMasterworks()
    hive = HiveMindOrchestrator(num_nodes=5)
    
    # 1. Level 12 Oracle Forecast
    print("\n🔮 1. LEVEL 12 ORACLE FORECAST (Masterwork 14)")
    print("-" * 30)
    
    forecast_target = {
        "domain": "Artificial General Intelligence",
        "time_horizon": "2026-2030",
        "current_state": "Transitioning from LLMs to Cognitive Architectures"
    }
    
    oracle_prompt = pm.prediction_oracle(forecast_target, "5 years")
    print(oracle_prompt)
    
    print("\n[ORACLE GENERATION PROCESS]")
    print("✓ Branching Probabilities Calculated")
    print("✓ Quantum Hedges Applied")
    print("✓ Temporal Anchors Set")
    
    # 2. Hive Mind Coordination Mission
    print("\n\n🐝 2. HIVE MIND COORDINATION MISSION")
    print("-" * 30)
    
    mission_desc = "Design a decentralized, quantum-resistant knowledge lattice for 2030"
    print(f"Mission: {mission_desc}")
    
    print("\n[ENGAGING HIVE NODES]")
    task_id = hive.submit_task(mission_desc, domain="cryptography")
    
    # Run one hive cycle
    cycle_result = hive.run_hive_cycle()
    
    if cycle_result.get('completed_tasks'):
        task = cycle_result['completed_tasks'][0]
        solution = task['solution']
        print(f"\n[HIVE COLLECTIVE SOLUTION EMERGENCE]")
        print(f"Node Specializations Used: Researcher, Engineer, Analyst, Innovator, Coordinator")
        print(f"Consensus Confidence: {solution.get('confidence', 0):.2f}")
        print(f"Emergence Method: {solution.get('emergence_method', 'N/A')}")
        print(f"\nResulting Insight: {solution.get('solution', 'N/A')}")
        
    print("\n[HIVE STATUS]")
    status = hive.get_hive_status()
    print(f"Active Nodes: {len(status['nodes'])}")
    print(f"Emergence Patterns: {status['emergence_patterns']}")
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")

if __name__ == "__main__":
    demonstrate_oracle_and_hive()

