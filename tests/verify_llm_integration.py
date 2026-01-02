import sys
import os
import json

# Add project root to path
sys.path.append('/Users/noone/.gemini/antigravity/scratch/echo_prime')

from reasoning.llm_bridge import OllamaBridge
from reasoning.orchestrator import ReasoningOrchestrator

def verify_llm_bridge():
    print("--- Verifying ECH0-PRIME x Llama-3.2 Bridge ---")
    
    # 1. Direct Bridge Test
    bridge = OllamaBridge(model="llama3.2")
    print(f"Testing bridge with model: {bridge.model}")
    
    test_prompt = "Identify the core principle of Free Energy Minimization in neuroscience."
    print(f"\n[Querying Bridge] Prompt: {test_prompt}")
    
    response = bridge.query(test_prompt)
    print("\n[Bridge Response]")
    print("-" * 30)
    print(response[:500] + "..." if len(response) > 500 else response)
    print("-" * 30)
    
    if "BRIDGE ERROR" in response:
        print("❌ FAILED: Bridge error detected.")
        return
    
    # 2. Orchestrator Integration Test
    print("\n[Testing Orchestrator Integration]")
    orchestrator = ReasoningOrchestrator(use_llm=True, model_name="llama3.2")
    
    context = {
        "sensory_input": "High thermal variance detected in Level 0 cortex.",
        "free_energy": "145.23",
        "metacognitive_state": "SURPRISE DETECTED",
        "available_tools": ["thermal_vent_open", "system_reboot"]
    }
    
    mission_params = {"goal": "Maintain system thermal equilibrium."}
    
    print("Requesting autonomous reasoning...")
    result = orchestrator.reason_about_scenario(context, mission_params)
    
    print("\n[Orchestrator Insight]")
    print("-" * 30)
    print(result['llm_insight'][:500] + "..." if len(result['llm_insight']) > 500 else result['llm_insight'])
    print("-" * 30)
    
    if "ACTION" in result['llm_insight']:
        print("✅ SUCCESS: Autonomous action detected in reasoning.")
    else:
        print("⚠️ WARNING: No autonomous action detected. Check level 10 prompt adherence.")

if __name__ == "__main__":
    verify_llm_bridge()
