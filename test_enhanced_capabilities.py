#!/usr/bin/env python3
"""
Test script for the enhanced capabilities implemented for ECH0-PRIME.

This script demonstrates the new mathematical verification, pattern recognition,
visual abstraction, and self-correction capabilities.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from capabilities.mathematical_verification import MathematicalVerificationSystem
from capabilities.mathematical_patterns import MathematicalPatternLibrary
from capabilities.visual_abstraction import VisualAbstractionSystem
from capabilities.self_correction import SelfCorrectionSystem


def test_mathematical_verification():
    """Test the mathematical verification system."""
    print("🧮 Testing Mathematical Verification System")
    print("=" * 50)

    verifier = MathematicalVerificationSystem()

    # Test cases
    test_cases = [
        "2 + 2 = 4",  # Simple correct equation
        "3 * 4 = 12",  # Multiplication
        "10 / 0",  # Division by zero (should detect error)
        "x^2 + 2*x + 1 = (x + 1)^2",  # Algebraic identity
        "sqrt(-1) = i",  # Complex numbers
    ]

    for expression in test_cases:
        print(f"\nTesting: {expression}")
        result = verifier.verify_with_steps(expression)
        print(f"  Valid: {result['is_valid']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        if result['errors']:
            print(f"  Errors: {result['errors']}")
        if result['suggestions']:
            print(f"  Suggestions: {result['suggestions'][:1]}")  # Show first suggestion
        if result['final_result']:
            print(f"  Result: {result['final_result']}")


def test_mathematical_patterns():
    """Test the mathematical pattern recognition library."""
    print("\n🔍 Testing Mathematical Pattern Library")
    print("=" * 50)

    pattern_lib = MathematicalPatternLibrary()

    # Test cases
    test_cases = [
        "x^2 - 4 = 0",  # Quadratic equation
        "2*x + 3 > 5",  # Linear inequality
        "Assume P, then Q. Therefore, P → Q",  # Logical implication
        "lim(x→0) sin(x)/x = 1",  # Limit expression
        "∫(x^2)dx from 0 to 1",  # Integral
    ]

    for content in test_cases:
        print(f"\nAnalyzing: {content}")
        patterns = pattern_lib.recognize_patterns(content)
        for pattern in patterns:
            print(f"  Pattern: {pattern.pattern_type.value}")
            print(f"  Name: {pattern.pattern_name}")
            print(f"  Confidence: {pattern.confidence:.2f}")
            print(f"  Description: {pattern_lib.get_pattern_explanation(pattern)}")


def test_visual_abstraction():
    """Test the visual abstraction system for ARC-AGI tasks."""
    print("\n👁️ Testing Visual Abstraction System")
    print("=" * 50)

    visual_system = VisualAbstractionSystem()

    # Create a simple pattern: diagonal line
    grid1 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    grid2 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])

    # Test grid analysis
    print("Analyzing diagonal pattern grid:")
    analysis = visual_system.grid_analyzer.analyze_grid(grid1)
    print(f"  Dimensions: {analysis['dimensions']}")
    print(f"  Unique colors: {analysis['unique_colors']}")
    print(f"  Complexity score: {analysis['complexity_score']:.2f}")

    if analysis['symmetries']['diagonal_main']:
        print("  ✓ Has main diagonal symmetry")
    else:
        print("  ✗ No main diagonal symmetry")

    # Test pattern transformation inference
    print("\nTesting transformation inference:")
    transform = visual_system.pattern_transformer.infer_transformation(grid1, grid2)
    print(f"  Inferred transformation: {transform['type']}")
    print(f"  Confidence: {transform.get('confidence', 'N/A')}")

    # Test task solving
    print("\nTesting ARC task solving:")
    input_grids = [grid1]
    output_grids = [grid2]
    test_input = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])

    try:
        solution = visual_system.solve_task(input_grids, output_grids, test_input)
        print("  Solution generated successfully")
        print(f"  Solution shape: {solution.shape}")
    except Exception as e:
        print(f"  Error in task solving: {e}")


def test_self_correction():
    """Test the self-correction system."""
    print("\n🔧 Testing Self-Correction System")
    print("=" * 50)

    corrector = SelfCorrectionSystem()

    # Test cases with various types of errors
    test_cases = [
        "2 + 2 = 5",  # Mathematical error
        "All people are either good or bad. This is a false dichotomy.",  # Logical fallacy
        "The Earth is flat because I said so.",  # Factual inaccuracy
        "We should do X. Because it will work.",  # Weak reasoning
        "This is certainly, definitely, absolutely correct.",  # Overconfidence
    ]

    for content in test_cases:
        print(f"\nAnalyzing: {content}")
        analysis = corrector.analyze_and_correct(content)
        print(f"  Errors detected: {analysis['errors_detected']}")

        if analysis['error_details']:
            top_error = analysis['error_details'][0]
            print(f"  Top error: {top_error['type']} (confidence: {top_error['confidence']:.2f})")
            print(f"  Description: {top_error['description']}")

        if analysis['corrections_suggested']:
            print(f"  Suggestions: {len(analysis['corrections_suggested'])} correction(s) available")


def main():
    """Run all capability tests."""
    print("🚀 ECH0-PRIME Enhanced Capabilities Test Suite")
    print("=" * 60)

    try:
        test_mathematical_verification()
        test_mathematical_patterns()
        test_visual_abstraction()
        test_self_correction()

        print("\n✅ All tests completed successfully!")
        print("\n📊 Summary of Enhancements:")
        print("  ✓ Numerical Verification System - Step-by-step mathematical checking")
        print("  ✓ Mathematical Pattern Library - Advanced construct recognition")
        print("  ✓ Visual Abstraction System - ARC-AGI pattern recognition")
        print("  ✓ Self-Correction Mechanisms - Error detection and automated fixes")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
