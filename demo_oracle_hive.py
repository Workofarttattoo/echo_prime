import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_orchestrator import EchoPrimeAGI

def demonstrate_oracle_and_hive():
    print("🚀 ECH0-PRIME Level 12 Oracle & Hive Mind Demonstration")
    print("=" * 60)
    
    # Initialize AGI in lightweight mode for the demo
    agi = EchoPrimeAGI(lightweight=False)
    
    # 1. Level 12 Oracle Forecast
    print("\n🔮 1. LEVEL 12 ORACLE FORECAST")
    print("-" * 30)
    
    forecast_target = {
        "domain": "Artificial General Intelligence",
        "time_horizon": "2026-2030",
        "current_state": "Transitioning from LLMs to Cognitive Architectures"
    }
    
    # Use Masterwork 14: Prediction Oracle
    oracle_prompt = agi.prediction_oracle(forecast_target, "5 years")
    print(f"Oracle Prompt Generated (Masterwork 14)")
    
    # Simulate high-level reasoning for the forecast
    print("\n[ORACLE INSIGHTS]")
    forecast_result = agi.reasoner.reason_about_scenario(
        {"goal": "Generate a Level 12 Oracle Forecast for AGI evolution"},
        {"state": forecast_target, "protocol": "Prediction Oracle"}
    )
    print(forecast_result.get("llm_insight", "Thinking..."))
    
    # 2. Hive Mind Coordination Mission
    print("\n\n🐝 2. HIVE MIND COORDINATION MISSION")
    print("-" * 30)
    
    mission = "Design a decentralized, quantum-resistant knowledge lattice for 2030"
    print(f"Mission: {mission}")
    
    # engage hive mind
    print("\n[ENGAGING HIVE NODES]")
    task_id = agi.submit_hive_task(mission, specialization_hint="cryptography_research")
    print(f"✓ {task_id}")
    
    # Run a hive cycle to show collective intelligence
    hive_result = agi.run_hive_cycle()
    
    if hive_result.get('completed_tasks'):
        task = hive_result['completed_tasks'][0]
        solution = task['solution']
        print(f"\n[HIVE COLLECTIVE SOLUTION]")
        print(f"Consensus Confidence: {solution.get('confidence', 0):.2f}")
        print(f"Emergence Method: {solution.get('emergence_method', 'N/A')}")
        print(f"Solution Summary: {solution.get('solution', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE")

if __name__ == "__main__":
    demonstrate_oracle_and_hive()

