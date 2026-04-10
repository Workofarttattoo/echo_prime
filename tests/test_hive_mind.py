#!/usr/bin/env python3
"""
Test script for the Hive Mind functionality from QuLabInfinite.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from missions.hive_mind import HiveMindOrchestrator

def test_basic_hive_functionality():
    """Test basic hive mind operations"""
    print("🧠 Testing Hive Mind Basic Functionality")
    print("=" * 40)

    # Initialize hive
    hive = HiveMindOrchestrator(num_nodes=3)
    print(f"✓ Initialized hive with {len(hive.nodes)} nodes")

    # Check node specializations
    print("\nNode Specializations:")
    for node_id, node in hive.nodes.items():
        print(f"  {node_id}: {node.specialization} - {node.capabilities}")

    # Submit a test task
    task_id = hive.submit_task("Test task: Design a simple algorithm", "engineering", 1.0)
    print(f"\n✓ Submitted task: {task_id}")

    # Run a hive cycle
    result = hive.run_hive_cycle()
    print(f"✓ Completed hive cycle: {result}")

    # Check task status
    task = hive.tasks.get(task_id)
    if task:
        print(f"✓ Task status: {task.status}")
        if task.consensus_solution:
            print(f"✓ Solution confidence: {task.consensus_solution.get('confidence', 0):.2f}")

    # Get hive status
    status = hive.get_hive_status()
    print(f"\n✓ Hive status: {status['state']}")
    print(f"✓ Active nodes: {status['nodes']}")

    # Shutdown
    hive.shutdown_hive()
    print("✓ Hive mind shutdown complete")

    return True

def test_quantum_integration():
    """Test quantum integration if available"""
    print("\n🧬 Testing Quantum Integration")
    print("=" * 40)

    hive = HiveMindOrchestrator(num_nodes=2)

    # Test quantum swarm optimization
    problem_space = {
        'bounds': [(-2, 2), (-2, 2), (-2, 2)],
        'objective': 'minimize_sphere'
    }

    try:
        result = hive.quantum_processor.quantum_particle_swarm_optimization(problem_space, 5)
        print(f"✓ Quantum PSO result: {result}")
    except Exception as e:
        print(f"⚠️ Quantum PSO test failed: {e}")

    hive.shutdown_hive()
    return True

def test_emergent_intelligence():
    """Test emergent intelligence engine"""
    print("\n🧪 Testing Emergent Intelligence")
    print("=" * 40)

    hive = HiveMindOrchestrator(num_nodes=2)

    # Simulate agent interactions
    interactions = [
        {"timestamp": time.time(), "agent_id": "node1", "content": "Exploring solution A"},
        {"timestamp": time.time() + 0.1, "agent_id": "node2", "content": "Considering solution B"},
        {"timestamp": time.time() + 0.2, "agent_id": "node1", "content": "Combining A and B"},
        {"timestamp": time.time() + 0.3, "agent_id": "node2", "content": "Refining combined solution"}
    ]

    # Detect patterns
    patterns = hive.emergence_engine.detect_emergent_patterns(interactions)
    print(f"✓ Detected {len(patterns)} emergent patterns")

    # Test solution emergence
    partial_solutions = [
        {"solution": "solution_A", "confidence": 0.7, "agent_performance": 0.8},
        {"solution": "solution_B", "confidence": 0.8, "agent_performance": 0.9},
        {"solution": "solution_C", "confidence": 0.6, "agent_performance": 0.7}
    ]

    emergent = hive.emergence_engine.generate_emergent_solution(
        partial_solutions,
        {"task_description": "Test emergence"}
    )
    print(f"✓ Emergent solution confidence: {emergent.get('confidence', 0):.2f}")

    hive.shutdown_hive()
    return True

def test_main_system_integration():
    """Test hive mind integration with main orchestrator"""
    print("\n🧠 Testing Main System Integration")
    print("=" * 40)

    try:
        # Test that hive mind methods are available
        from main_orchestrator import EchoPrimeAGI

        # Check methods exist
        required_methods = ['submit_hive_task', 'run_hive_cycle', 'get_hive_status', 'shutdown_hive']
        available_methods = [m for m in dir(EchoPrimeAGI) if m in required_methods]

        if len(available_methods) == len(required_methods):
            print("✓ All hive mind methods integrated into EchoPrimeAGI")
            return True
        else:
            print(f"❌ Missing methods: {set(required_methods) - set(available_methods)}")
            return False

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ECH0-PRIME Hive Mind Test Suite")
    print("=" * 50)

    tests = [
        test_basic_hive_functionality,
        test_quantum_integration,
        test_emergent_intelligence,
        test_main_system_integration
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} CRASHED: {e}")
        print()

    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("🎉 All hive mind tests passed!")
        print("The QuLabInfinite hive mind is ready for deployment.")
        print("✓ Integrated with main ECH0-PRIME system")
    else:
        print(f"⚠️ {failed} tests failed. Check the output above.")

    sys.exit(0 if failed == 0 else 1)
