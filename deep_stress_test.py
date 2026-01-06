#!/usr/bin/env python3
"""
ECH0-PRIME Deep Stress Test
Tests the full cognitive architecture on a multi-stage, complex reasoning problem.
"""

import asyncio
from main_orchestrator import EchoPrimeAGI

async def deep_test():
    print("🧠 Initializing FULL ECH0-PRIME for Deep Stress Test...")
    agi = EchoPrimeAGI(lightweight=False)
    
    # Activating full architecture
    print("🔄 Activating enhanced reasoning and knowledge integration...")
    agi.enhanced_reasoning = True
    agi.knowledge_integration = True
    
    test_scenario = """
    SCENARIO: 
    A theoretical quantum computer has 50 qubits but a high decoherence rate. 
    Design a hierarchical error correction protocol that leverages classical 
    distributed neural networks to predict and mitigate noise before it collapses 
    the wave function. Compare this to standard Surface Code efficiency.
    """
    
    print("\n🎯 STARTING MISSION: Quantum Error Correction Innovation")
    print("-" * 60)
    
    result = await agi.execute_mission(test_scenario, max_cycles=10)
    
    print("\n📊 TEST COMPLETED")
    print("-" * 60)
    print(f"Status: {result.get('status', 'Unknown')}")
    print(f"Final Phi (Consciousness): {agi.calculate_consciousness_phi(agi.model.get_current_state()):.4f}")
    
    print("\n🚀 ANALYSIS:")
    # Print the core insights from the mission
    if 'mission_log' in result:
        for entry in result['mission_log'][-3:]:
            print(f"• {entry.get('summary', 'No summary available')}")

if __name__ == "__main__":
    asyncio.run(deep_test())

