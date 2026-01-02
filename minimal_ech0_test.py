#!/usr/bin/env python3
"""
Minimal ECH0-PRIME test - just core components
"""

import sys
import os

def test_minimal():
    """Test minimal ECH0-PRIME components"""
    print("🧪 MINIMAL ECH0-PRIME COMPONENT TEST")
    print("=" * 50)

    try:
        print("Testing core engine import...")
        from core.engine import HierarchicalGenerativeModel, FreeEnergyEngine, GlobalWorkspace
        print("✅ Core engine imports OK")

        print("Testing model initialization...")
        model = HierarchicalGenerativeModel(use_cuda=False)
        print("✅ Model initialized")

        print("Testing free energy engine...")
        fe_engine = FreeEnergyEngine(model)
        print("✅ Free energy engine OK")

        print("Testing global workspace...")
        workspace = GlobalWorkspace(model)
        print("✅ Global workspace OK")

        print("Testing memory manager...")
        from memory.manager import MemoryManager
        memory = MemoryManager()
        print("✅ Memory manager OK")

        print("Testing learning system...")
        from learning.meta import CSALearningSystem
        learning = CSALearningSystem(param_dim=100, device="cpu")
        print("✅ Learning system OK")

        print("\n🎉 ALL CORE COMPONENTS WORKING!")
        print("The issue is in the full orchestrator integration.")

        return True

    except Exception as e:
        print(f"❌ COMPONENT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_orchestrator():
    """Test a very simple orchestrator"""
    print("\n🔧 TESTING SIMPLE ORCHESTRATOR")
    print("=" * 40)

    try:
        from core.engine import HierarchicalGenerativeModel

        class SimpleOrchestrator:
            def __init__(self):
                self.model = HierarchicalGenerativeModel(use_cuda=False)
                print("✅ Simple orchestrator created")

            def solve_simple(self, problem):
                # Just return a mock answer for testing
                if "2+2" in problem:
                    return "4"
                elif "capital of France" in problem.lower():
                    return "Paris"
                else:
                    return "I don't know"

        orch = SimpleOrchestrator()

        # Test simple problems
        test1 = orch.solve_simple("What is 2+2?")
        print(f"2+2 = {test1}")

        test2 = orch.solve_simple("What is the capital of France?")
        print(f"Capital of France = {test2}")

        print("✅ Simple orchestrator works!")

        return True

    except Exception as e:
        print(f"❌ Simple orchestrator failed: {e}")
        return False

if __name__ == "__main__":
    components_ok = test_minimal()
    simple_ok = test_simple_orchestrator()

    print("\n" + "=" * 50)
    print("📊 FINAL DIAGNOSIS:")

    if components_ok and simple_ok:
        print("✅ ECH0-PRIME components are working")
        print("❌ Issue is in full orchestrator initialization")
        print("🔧 Solution: Simplify the main_orchestrator.py or fix initialization loop")
    elif components_ok:
        print("✅ Core components work, simple orchestrator failed")
        print("🔧 Check orchestrator logic")
    else:
        print("❌ Core components have issues")
        print("🔧 Debug individual component imports")


