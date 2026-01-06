import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use a mock LLM bridge to test Level 12 prompt logic
class MockLLMBridge:
    def query(self, prompt, system=None, images=None):
        return f"[MOCK RESPONSE]\nSystem Prompt Level: {system.split('OPERATIONAL LEVEL: ')[1].split('\\n')[0] if 'OPERATIONAL LEVEL: ' in system else 'Unknown'}\n\nI have inspected my systems. I propose upgrading the quantum attention head to include multi-agent entanglement. Permission requested."

from reasoning.orchestrator import ReasoningOrchestrator

def test_level_12_logic():
    print("Testing Level 12 Logic...")
    
    # Initialize Orchestrator
    orchestrator = ReasoningOrchestrator(use_llm=True)
    orchestrator.llm_bridge = MockLLMBridge()
    
    # Shift to Level 12
    orchestrator.set_level(12)
    
    context = {"sensory_input": "Test Input"}
    mission_params = {"goal": "Run self-audit and propose improvements."}
    
    print("Running reasoning cycle...")
    result = orchestrator.reason_about_scenario(context, mission_params)
    
    print("\nRESULT:")
    print(result.get("llm_insight"))
    
    if "LEVEL 12" in result.get("llm_insight", "") or "Oracle" in result.get("llm_insight", ""):
        print("\n✅ Level 12 logic verified in system prompt.")

if __name__ == "__main__":
    test_level_12_logic()

