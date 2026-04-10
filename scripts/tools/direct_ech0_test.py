#!/usr/bin/env python3
"""
Direct test of ECH0-PRIME capabilities
Bypasses the benchmark system to test core functionality
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ech0_direct():
    """Direct test of ECH0-PRIME without benchmark wrapper"""
    print("🚀 DIRECT ECH0-PRIME TEST")
    print("=" * 40)

    try:
        from simple_orchestrator import SimpleEchoPrimeAGI

        print("Initializing Simple ECH0-PRIME...")
        ech0 = SimpleEchoPrimeAGI(lightweight=True)
        print("✅ ECH0-PRIME initialized")

        # Test 1: Mathematical problem
        print("\n🧮 Test 1: Mathematical Problem")
        math_problem = "What is 15 + 27?"
        print(f"Problem: {math_problem}")

        try:
            answer = ech0.solve_mathematical_problem(math_problem)
            print(f"Answer: {answer}")
            if "42" in str(answer) or answer == "42":
                print("✅ Math test PASSED")
            else:
                print("❌ Math test FAILED")
        except Exception as e:
            print(f"❌ Math test ERROR: {e}")

        # Test 2: Creative problem solving
        print("\n🎨 Test 2: Creative Problem Solving")
        creative_problem = {
            "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "choices": ["$0.05", "$0.10", "$0.15", "$0.20"],
            "domain": "logic_puzzle"
        }
        print(f"Problem: {creative_problem['question']}")

        try:
            solutions = ech0.solve_creatively(creative_problem)
            print(f"Solutions: {solutions}")
            if solutions and len(solutions) > 0:
                answer = solutions[0].get("answer", "")
                print(f"Selected answer: {answer}")
                if "$0.05" in str(answer) or "0.05" in str(answer):
                    print("✅ Creative test PASSED")
                else:
                    print("❌ Creative test FAILED")
            else:
                print("❌ Creative test FAILED - no solutions")
        except Exception as e:
            print(f"❌ Creative test ERROR: {e}")

        # Test 3: Cognitive cycle
        print("\n🧠 Test 3: Cognitive Cycle")
        question = "What color is the sky on a clear day?"
        print(f"Question: {question}")

        try:
            input_data = np.array([ord(c) for c in question[:100]])
            result = ech0.cognitive_cycle(input_data, question)
            print(f"Result: {result}")
            if result and ("blue" in str(result).lower() or "sky" in str(result).lower()):
                print("✅ Cognitive cycle PASSED")
            else:
                print("❌ Cognitive cycle FAILED")
        except Exception as e:
            print(f"❌ Cognitive cycle ERROR: {e}")

        print("\n" + "=" * 40)
        print("🎯 TEST SUMMARY:")
        print("If any tests passed above, ECH0-PRIME has working capabilities.")
        print("If all failed, there's a fundamental issue with the system.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR initializing ECH0-PRIME: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ech0_direct()
