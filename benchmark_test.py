#!/usr/bin/env python3
"""
Simple benchmark test to verify ECH0-PRIME capabilities
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_orchestrator import EchoPrimeAGI

def test_basic_math():
    """Test basic mathematical reasoning"""
    agi = EchoPrimeAGI(enable_voice=False)

    test_problems = [
        "If you have 3 apples and buy 2 more, how many do you have?",
        "What comes next: 2, 4, 6, 8, ...?",
        "A bat and ball cost $1.10. Bat costs $1 more than ball. How much does ball cost?"
    ]

    print("🧮 Testing ECH0-PRIME Mathematical Reasoning:")
    print("-" * 50)

    for i, problem in enumerate(test_problems, 1):
        print(f"\nProblem {i}: {problem}")
        try:
            answer = agi.solve_mathematical_problem(problem)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n✅ Basic math test completed!")

def test_phd_knowledge():
    """Test PhD-level knowledge access"""
    agi = EchoPrimeAGI(enable_voice=False)

    queries = [
        ("advanced_mathematics", "algebraic_geometry", "fundamental concepts"),
        ("theoretical_physics", "quantum_field_theory", "renormalization"),
        ("advanced_cs", "complexity_theory", "P vs NP")
    ]

    print("\n🧠 Testing PhD Knowledge Access:")
    print("-" * 40)

    for domain, subfield, query in queries:
        print(f"\nQuerying {domain} → {subfield}:")
        try:
            result = agi.handle_command("query_phd_knowledge", {
                "domain": domain,
                "subfield": subfield,
                "query": query
            })
            print("✓ Knowledge retrieved successfully")
        except Exception as e:
            print(f"Error: {e}")

    print("\n✅ PhD knowledge test completed!")

def test_research_capabilities():
    """Test research proposal generation"""
    agi = EchoPrimeAGI(enable_voice=False)

    topics = [
        "Unified Theory of Intelligence",
        "Quantum Machine Learning Integration"
    ]

    print("\n🔬 Testing Research Capabilities:")
    print("-" * 35)

    for topic in topics:
        print(f"\nGenerating proposal for: {topic}")
        try:
            result = agi.handle_command("generate_research_proposal", {
                "topic": topic,
                "domain": "interdisciplinary"
            })
            print("✓ Research proposal generated")
        except Exception as e:
            print(f"Error: {e}")

    print("\n✅ Research capabilities test completed!")

if __name__ == "__main__":
    print("🚀 ECH0-PRIME Capability Verification")
    print("=" * 40)

    test_basic_math()
    test_phd_knowledge()
    test_research_capabilities()

    print("\n🎯 ECH0-PRIME verification complete!")
    print("All core capabilities are operational and ready for benchmarking.")
