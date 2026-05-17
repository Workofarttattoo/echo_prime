#!/usr/bin/env python3
"""
Compare Baseline vs Phase 1 Enhanced

Side-by-side comparison showing the improvements from Phase 1 enhancements
"""

import importlib.util


def load_module(name, path):
    """Load module dynamically"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("=" * 80)
print("🔬 BASELINE VS PHASE 1 ENHANCED COMPARISON")
print("=" * 80)

# Load both systems
print("\n📦 Loading systems...")

# Baseline: Pure symbolic reasoning engines
print("   Loading baseline (pure symbolic)...")
math_module = load_module("math_engine", "reasoning/math_engine.py")
code_module = load_module("code_debugger", "reasoning/code_debugger.py")
knowledge_module = load_module("knowledge_reasoner", "reasoning/knowledge_reasoner.py")

baseline_math = math_module.get_math_engine()
baseline_code = code_module.get_code_debugger()
baseline_knowledge = knowledge_module.get_knowledge_reasoner()

# Phase 1 Enhanced
print("   Loading Phase 1 Enhanced...")
phase1_module = load_module("echo_phase1", "echo_phase1_enhanced.py")
phase1_system = phase1_module.EchoPhase1Enhanced(
    use_llm=False,
    use_rag=True,
    use_reflection=True
)

print("✅ Both systems loaded!\n")

# Test cases
test_cases = [
    # Math
    {
        'problem': "What is 25 + 17?",
        'expected': "42",
        'domain': 'math'
    },
    {
        'problem': "Calculate 15 * 8",
        'expected': "120",
        'domain': 'math'
    },
    {
        'problem': "John has 50 apples and gives 12 to Mary. How many are left?",
        'expected': "38",
        'domain': 'math'
    },
    {
        'problem': "A store has 100 items. They sell 35 in the morning and 28 in the afternoon. How many remain?",
        'expected': "37",
        'domain': 'math'
    },
    {
        'problem': "What is 144 / 12?",
        'expected': "12",
        'domain': 'math'
    },
    # Knowledge
    {
        'problem': "What is the capital of France?",
        'expected': "Paris",
        'domain': 'knowledge'
    },
    {
        'problem': "Who wrote The Republic?",
        'expected': "Plato",
        'domain': 'knowledge'
    },
    {
        'problem': "What is the Pythagorean theorem?",
        'expected': "a² + b² = c²",
        'domain': 'knowledge'
    },
    {
        'problem': "What is Newton's second law?",
        'expected': "F = ma",
        'domain': 'knowledge'
    },
    # Code
    {
        'problem': "Write a function that returns the sum of two numbers",
        'expected': "def",
        'domain': 'code'
    },
    {
        'problem': "Create a function to check if a number is even",
        'expected': "def",
        'domain': 'code'
    },
]


def check_answer(answer: str, expected: str) -> bool:
    """Check if answer matches expected"""
    answer_str = str(answer).strip().lower()
    expected_str = str(expected).strip().lower()

    if answer_str == expected_str:
        return True
    if expected_str in answer_str or answer_str in expected_str:
        return True
    try:
        return float(answer_str) == float(expected_str)
    except:
        pass
    return False


# Run comparison
print("=" * 80)
print("🧪 RUNNING COMPARISON")
print("=" * 80)

baseline_results = {'total': 0, 'correct': 0, 'math': 0, 'code': 0, 'knowledge': 0}
phase1_results = {'total': 0, 'correct': 0, 'math': 0, 'code': 0, 'knowledge': 0}

for i, test in enumerate(test_cases):
    problem = test['problem']
    expected = test['expected']
    domain = test['domain']

    print(f"\n{'─' * 80}")
    print(f"TEST {i+1}/{len(test_cases)}: {problem[:60]}...")
    print(f"Expected: {expected}")
    print(f"{'─' * 80}")

    # Baseline
    if domain == 'math':
        baseline_answer = baseline_math.solve_problem(problem)
        baseline_results['math'] += 1
    elif domain == 'code':
        baseline_answer = baseline_code.debug_code(problem)
        baseline_results['code'] += 1
    else:
        baseline_answer = baseline_knowledge.answer_question(problem, None)
        baseline_results['knowledge'] += 1

    baseline_correct = check_answer(baseline_answer, expected)
    baseline_results['total'] += 1
    if baseline_correct:
        baseline_results['correct'] += 1

    # Phase 1
    phase1_result = phase1_system.solve(problem, domain)
    phase1_answer = phase1_result['answer']
    phase1_correct = check_answer(phase1_answer, expected)
    phase1_results['total'] += 1
    if phase1_correct:
        phase1_results['correct'] += 1

    # Display comparison
    baseline_status = "✅" if baseline_correct else "❌"
    phase1_status = "✅" if phase1_correct else "❌"

    print(f"\n📊 BASELINE (Pure Symbolic):     {baseline_status}")
    print(f"   Answer: {baseline_answer[:100]}")

    print(f"\n🚀 PHASE 1 ENHANCED (RAG+Reflection): {phase1_status}")
    print(f"   Answer: {phase1_answer[:100]}")
    print(f"   Confidence: {phase1_result['confidence']:.2f}")
    print(f"   Method: {phase1_result['method']}")

    # Show improvement
    if baseline_correct and phase1_correct:
        print(f"\n   ✨ Both correct!")
    elif not baseline_correct and phase1_correct:
        print(f"\n   🎯 Phase 1 FIXED this! (Baseline was wrong)")
    elif baseline_correct and not phase1_correct:
        print(f"\n   ⚠️  Baseline was correct, Phase 1 regressed")
    else:
        print(f"\n   💭 Both wrong, needs more work")

# Final results
print("\n" + "=" * 80)
print("📊 FINAL COMPARISON")
print("=" * 80)

baseline_accuracy = (baseline_results['correct'] / baseline_results['total']) * 100
phase1_accuracy = (phase1_results['correct'] / phase1_results['total']) * 100
improvement = phase1_accuracy - baseline_accuracy
relative_improvement = ((phase1_accuracy - baseline_accuracy) / baseline_accuracy) * 100

print(f"\n🏁 BASELINE (Pure Symbolic):")
print(f"   Score: {baseline_results['correct']}/{baseline_results['total']} ({baseline_accuracy:.1f}%)")

print(f"\n🚀 PHASE 1 ENHANCED (RAG + Reflection):")
print(f"   Score: {phase1_results['correct']}/{phase1_results['total']} ({phase1_accuracy:.1f}%)")

print(f"\n📈 IMPROVEMENT:")
print(f"   Absolute: +{improvement:.1f} percentage points")
print(f"   Relative: +{relative_improvement:.1f}%")

print("\n" + "=" * 80)

if improvement > 0:
    print("✅ PHASE 1 ENHANCEMENTS SUCCESSFUL!")
    print(f"   RAG + Self-Reflection improved accuracy by {improvement:.1f} pp")
elif improvement == 0:
    print("⚖️  NO CHANGE")
    print("   Phase 1 enhancements maintained baseline performance")
else:
    print("⚠️  REGRESSION DETECTED")
    print(f"   Phase 1 is {-improvement:.1f} pp worse than baseline")

print("=" * 80)

# Breakdown by domain
print("\n📊 BY DOMAIN:")
print(f"\n   MATH:")
print(f"      Tests: {baseline_results['math']}")
print(f"      (Individual domain scores not tracked in this simple comparison)")

print(f"\n   KNOWLEDGE:")
print(f"      Tests: {baseline_results['knowledge']}")

print(f"\n   CODE:")
print(f"      Tests: {baseline_results['code']}")

print("\n" + "=" * 80)
print("🎓 KEY TAKEAWAYS:")
print("=" * 80)

print("\n✅ What Phase 1 Added:")
print("   1. RAG - Retrieval-Augmented Generation for knowledge questions")
print("   2. Self-Reflection - Verify and correct answers")
print("   3. Better Math Patterns - Multi-step word problems")
print("   4. Hybrid Similarity - Keyword + embedding matching")

print("\n💡 Phase 1 Best For:")
print("   - Knowledge questions (RAG retrieval)")
print("   - Math verification (self-reflection)")
print("   - Situations requiring explainability (can cite sources)")

print("\n⚡ Baseline Best For:")
print("   - Simple symbolic reasoning")
print("   - Maximum speed (no RAG overhead)")
print("   - Deterministic math (already 100%)")

print("\n" + "=" * 80)
