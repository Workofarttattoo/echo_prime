#!/usr/bin/env python3
"""
Quick test to verify ECH0-PRIME can solve basic problems
"""

import numpy as np
from main_orchestrator import EchoPrimeAGI

def test_basic_math():
    """Test basic mathematical problem solving"""
    print("🧮 Testing ECH0-PRIME basic math capabilities...")

    try:
        ech0 = EchoPrimeAGI(lightweight=True)

        # Simple math problem
        problem = "What is 2 + 2?"
        print(f"Problem: {problem}")

        result = ech0.solve_mathematical_problem(problem)
        print(f"Answer: {result}")

        # Check if answer contains "4"
        if "4" in str(result):
            print("✅ Basic math works!")
            return True
        else:
            print("❌ Basic math failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_creative_problem():
    """Test creative problem solving"""
    print("\n🎨 Testing ECH0-PRIME creative problem solving...")

    try:
        ech0 = EchoPrimeAGI(lightweight=True)

        problem = {
            "question": "If you have 3 apples and give away 1, how many do you have left?",
            "choices": ["1", "2", "3", "4"],
            "domain": "basic_reasoning"
        }

        solutions = ech0.solve_creatively(problem)
        print(f"Solutions: {solutions}")

        if solutions and len(solutions) > 0:
            answer = solutions[0].get("answer", "")
            print(f"Answer: {answer}")
            if "2" in str(answer):
                print("✅ Creative problem solving works!")
                return True
            else:
                print("❌ Creative problem solving failed")
                return False
        else:
            print("❌ No solutions returned")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cognitive_cycle():
    """Test cognitive cycle"""
    print("\n🧠 Testing ECH0-PRIME cognitive cycle...")

    try:
        ech0 = EchoPrimeAGI(lightweight=True)

        question = "What is the capital of France?"
        input_data = np.array([ord(c) for c in question[:100]])

        result = ech0.cognitive_cycle(input_data, question)
        print(f"Question: {question}")
        print(f"Result: {result}")

        if result and "paris" in str(result).lower():
            print("✅ Cognitive cycle works!")
            return True
        else:
            print("❌ Cognitive cycle failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ECH0-PRIME BASIC CAPABILITY TEST")
    print("=" * 50)

    math_ok = test_basic_math()
    creative_ok = test_creative_problem()
    cognitive_ok = test_cognitive_cycle()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"  Math: {'✅' if math_ok else '❌'}")
    print(f"  Creative: {'✅' if creative_ok else '❌'}")
    print(f"  Cognitive: {'✅' if cognitive_ok else '❌'}")

    if math_ok or creative_ok or cognitive_ok:
        print("\n🎉 ECH0-PRIME has some working capabilities!")
        print("Ready to run full benchmarks.")
    else:
        print("\n❌ ECH0-PRIME is not working. Needs debugging.")


