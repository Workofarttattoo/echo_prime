
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.registry import ToolRegistry
from core.actuator import ActuatorBridge
from reasoning.orchestrator import ReasoningOrchestrator

# 1. Register a dummy tool
@ToolRegistry.register()
def calculate_sum(a: int, b: int) -> int:
    """Calculates the sum of two integers."""
    return a + b

def test_tool_schema():
    print("\n--- Testing Tool Schema ---")
    schemas = ToolRegistry.get_schemas()
    print(json.dumps(schemas, indent=2))
    
    # Verify structure
    assert schemas[-1]["type"] == "function"
    assert schemas[-1]["function"]["name"] == "calculate_sum"
    print("✅ Schema verification passed.")

def test_actuator_registry_integration():
    print("\n--- Testing Actuator-Registry Integration ---")
    actuator = ActuatorBridge("/tmp")
    
    intent = {"tool": "calculate_sum", "args": {"a": 10, "b": 20}}
    result = actuator.execute_intent(intent)
    print(f"Execution Result: {result}")
    
    assert "30" in result
    print("✅ Actuator execution passed.")

def test_reasoner_history():
    print("\n--- Testing Reasoner History Injection ---")
    reasoner = ReasoningOrchestrator(use_llm=False)
    
    action = {"tool": "calculate_sum", "args": {"a": 5, "b": 5}}
    result = "10"
    
    reasoner.record_action_result(action, result)
    
    last_two = reasoner.mission_history[-2:]
    print("Mission History Tail:")
    print(last_two)
    
    assert "ACTION:" in last_two[0]
    assert "OBSERVATION: 10" in last_two[1]
    print("✅ Reasoner history injection passed.")

if __name__ == "__main__":
    test_tool_schema()
    test_actuator_registry_integration()
    test_reasoner_history()
