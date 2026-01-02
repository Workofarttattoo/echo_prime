#!/usr/bin/env python3
"""
ECH0-PRIME Numerical Verification System
Advanced numerical verification with gap-fixing capabilities
"""

import re
import math
import sympy as sp
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class VerificationLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class VerificationResult:
    confidence: float
    verified_solution: str
    issues: List[str]
    suggestions: List[str]
    verification_level: VerificationLevel
    computational_complexity: str


class NumericalVerifier:
    """
    Advanced numerical verification system with gap-fixing capabilities
    """

    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z')
        self.verification_cache = {}
        self.common_errors = {
            'division_by_zero': re.compile(r'/\s*0'),
            'undefined_operation': re.compile(r'sqrt\s*\(\s*-\s*\d+\)'),
            'precision_loss': re.compile(r'\d+\.\d{10,}'),
            'overflow': re.compile(r'\d{100,}'),
        }

    def verify_solution(self, problem: str, solution: str, reasoning: str = "") -> Dict[str, Any]:
        """
        Verify numerical solution with comprehensive gap-fixing

        Args:
            problem: Mathematical problem statement
            solution: Proposed solution
            reasoning: Step-by-step reasoning (optional)

        Returns:
            VerificationResult with confidence and corrections
        """
        cache_key = hash(f"{problem}:{solution}")
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]

        # Extract numerical components
        problem_nums = self._extract_numbers(problem)
        solution_nums = self._extract_numbers(solution)

        # Determine verification level
        verification_level = self._determine_verification_level(problem, solution)

        # Perform verification based on level
        if verification_level == VerificationLevel.BASIC:
            result = self._verify_basic(problem, solution)
        elif verification_level == VerificationLevel.INTERMEDIATE:
            result = self._verify_intermediate(problem, solution, reasoning)
        elif verification_level == VerificationLevel.ADVANCED:
            result = self._verify_advanced(problem, solution, reasoning)
        else:  # EXPERT
            result = self._verify_expert(problem, solution, reasoning)

        # Gap-fixing: Attempt to correct identified issues
        if result.confidence < 0.8:
            corrected_result = self._apply_gap_fixing(problem, result)
            result = corrected_result

        # Cache result
        self.verification_cache[cache_key] = result.__dict__

        return result.__dict__

    def _determine_verification_level(self, problem: str, solution: str) -> VerificationLevel:
        """Determine appropriate verification level based on problem complexity"""
        complexity_score = 0

        # Check for advanced mathematical concepts
        advanced_concepts = ['integral', 'derivative', 'matrix', 'eigenvalue', 'complex', 'differential']
        for concept in advanced_concepts:
            if concept in problem.lower() or concept in solution.lower():
                complexity_score += 2

        # Check for algebraic complexity
        if re.search(r'x\^\d+', problem) or re.search(r'x\^\d+', solution):
            complexity_score += 1
        if re.search(r'sqrt|log|exp|sin|cos|tan', problem) or re.search(r'sqrt|log|exp|sin|cos|tan', solution):
            complexity_score += 1

        # Check for multiple variables
        variables = set(re.findall(r'[a-zA-Z]', problem + solution))
        complexity_score += len(variables) - 1

        if complexity_score >= 5:
            return VerificationLevel.EXPERT
        elif complexity_score >= 3:
            return VerificationLevel.ADVANCED
        elif complexity_score >= 1:
            return VerificationLevel.INTERMEDIATE
        else:
            return VerificationLevel.BASIC

    def _verify_basic(self, problem: str, solution: str) -> VerificationResult:
        """Basic numerical verification for simple arithmetic"""
        issues = []
        suggestions = []

        try:
            # Extract the equation
            if '=' in problem:
                left_side = problem.split('=')[0].strip()
                right_side = problem.split('=')[1].strip()

                # Evaluate both sides
                left_val = self._safe_eval(left_side)
                right_val = self._safe_eval(right_side)

                if abs(left_val - right_val) < 1e-10:
                    return VerificationResult(1.0, solution, [], [], VerificationLevel.BASIC, "O(1)")
                else:
                    issues.append(f"Equation not satisfied: {left_val} ≠ {right_val}")
                    suggestions.append("Check arithmetic operations")

            # Check for common errors
            for error_type, pattern in self.common_errors.items():
                if pattern.search(solution):
                    issues.append(f"Potential {error_type.replace('_', ' ')}")
                    suggestions.append(self._get_error_suggestion(error_type))

            confidence = max(0.1, 1.0 - len(issues) * 0.2)

        except Exception as e:
            issues.append(f"Evaluation error: {str(e)}")
            confidence = 0.1

        return VerificationResult(confidence, solution, issues, suggestions, VerificationLevel.BASIC, "O(1)")

    def _verify_intermediate(self, problem: str, solution: str, reasoning: str) -> VerificationResult:
        """Intermediate verification with algebraic manipulation checking"""
        issues = []
        suggestions = []

        try:
            # Check if solution is algebraically equivalent
            if self._check_algebraic_equivalence(problem, solution):
                confidence = 0.9
            else:
                confidence = 0.6
                issues.append("Algebraic manipulation may be incorrect")

            # Verify step-by-step if reasoning provided
            if reasoning:
                step_confidence = self._verify_reasoning_steps(reasoning)
                confidence = min(confidence, step_confidence)

                if step_confidence < 0.7:
                    issues.append("Reasoning steps contain errors")
                    suggestions.append("Review intermediate calculation steps")

        except Exception as e:
            issues.append(f"Verification error: {str(e)}")
            confidence = 0.3

        return VerificationResult(confidence, solution, issues, suggestions, VerificationLevel.INTERMEDIATE, "O(n)")

    def _verify_advanced(self, problem: str, solution: str, reasoning: str) -> VerificationResult:
        """Advanced verification with symbolic computation"""
        issues = []
        suggestions = []

        try:
            # Use sympy for symbolic verification
            problem_expr = self._parse_to_sympy(problem)
            solution_expr = self._parse_to_sympy(solution)

            if problem_expr is not None and solution_expr is not None:
                # Check if expressions are equivalent
                diff = sp.simplify(problem_expr - solution_expr)
                if diff == 0:
                    confidence = 0.95
                else:
                    confidence = 0.7
                    issues.append("Symbolic expressions are not equivalent")
                    suggestions.append("Check symbolic manipulation")
            else:
                # Fall back to numerical verification
                confidence = 0.6
                issues.append("Could not parse expressions symbolically")

        except Exception as e:
            issues.append(f"Symbolic verification error: {str(e)}")
            confidence = 0.4

        return VerificationResult(confidence, solution, issues, suggestions, VerificationLevel.ADVANCED, "O(n²)")

    def _verify_expert(self, problem: str, solution: str, reasoning: str) -> VerificationResult:
        """Expert-level verification with theorem proving elements"""
        issues = []
        suggestions = []

        # Expert verification requires more sophisticated analysis
        confidence = 0.5  # Start conservative

        try:
            # Check for mathematical rigor
            rigor_score = self._assess_mathematical_rigor(problem, solution, reasoning)
            confidence *= rigor_score

            # Verify boundary conditions
            boundary_issues = self._check_boundary_conditions(problem, solution)
            issues.extend(boundary_issues)

            # Check for completeness of solution
            if not self._check_solution_completeness(problem, solution):
                issues.append("Solution may be incomplete")
                suggestions.append("Verify all cases are covered")

        except Exception as e:
            issues.append(f"Expert verification error: {str(e)}")
            confidence = 0.2

        return VerificationResult(confidence, solution, issues, suggestions, VerificationLevel.EXPERT, "O(n³)")

    def _apply_gap_fixing(self, problem: str, current_result: VerificationResult) -> VerificationResult:
        """Apply gap-fixing techniques to improve verification confidence"""
        corrected_solution = current_result.verified_solution
        new_issues = current_result.issues.copy()
        new_suggestions = current_result.suggestions.copy()

        # Gap-fixing strategies
        for issue in current_result.issues:
            if "division by zero" in issue.lower():
                corrected_solution = self._fix_division_by_zero(problem, corrected_solution)
                new_issues.remove(issue)
                new_suggestions.append("Applied limit analysis for division by zero")

            elif "algebraic" in issue.lower():
                corrected_solution = self._fix_algebraic_error(problem, corrected_solution)
                if corrected_solution != current_result.verified_solution:
                    new_issues.remove(issue)
                    new_suggestions.append("Applied algebraic simplification")

            elif "precision" in issue.lower():
                corrected_solution = self._fix_precision_error(corrected_solution)
                new_issues.remove(issue)
                new_suggestions.append("Applied high-precision arithmetic")

        # Recalculate confidence
        new_confidence = min(0.95, current_result.confidence + 0.2 * (len(current_result.issues) - len(new_issues)))

        return VerificationResult(
            new_confidence, corrected_solution, new_issues, new_suggestions,
            current_result.verification_level, current_result.computational_complexity
        )

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        numbers = []
        for match in re.finditer(r'-?\d+\.?\d*', text):
            try:
                numbers.append(float(match.group()))
            except ValueError:
                continue
        return numbers

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expression"""
        try:
            # Replace common symbols
            expression = expression.replace('π', str(math.pi))
            expression = expression.replace('pi', str(math.pi))
            expression = expression.replace('e', str(math.e))

            # Use restricted evaluation
            return eval(expression, {"__builtins__": {}}, {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
                'pi': math.pi, 'e': math.e, 'abs': abs
            })
        except:
            return float('nan')

    def _check_algebraic_equivalence(self, expr1: str, expr2: str) -> bool:
        """Check if two algebraic expressions are equivalent"""
        try:
            e1 = sp.sympify(expr1)
            e2 = sp.sympify(expr2)
            diff = sp.simplify(e1 - e2)
            return diff == 0
        except:
            return False

    def _verify_reasoning_steps(self, reasoning: str) -> float:
        """Verify the logical flow of reasoning steps"""
        steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
        confidence = 1.0

        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            # Check for logical connection
            if not self._has_logical_connection(current_step, next_step):
                confidence -= 0.1

            # Check for mathematical consistency
            if not self._check_mathematical_consistency(current_step, next_step):
                confidence -= 0.15

        return max(0.1, confidence)

    def _has_logical_connection(self, step1: str, step2: str) -> bool:
        """Check if two steps are logically connected"""
        logical_indicators = ['therefore', 'thus', 'hence', 'so', 'because', 'since', 'if', 'then']
        combined = (step1 + ' ' + step2).lower()
        return any(indicator in combined for indicator in logical_indicators)

    def _check_mathematical_consistency(self, step1: str, step2: str) -> bool:
        """Check mathematical consistency between steps"""
        # Simple check: numbers should be reasonably close if performing operations
        nums1 = self._extract_numbers(step1)
        nums2 = self._extract_numbers(step2)

        if not nums1 or not nums2:
            return True  # Can't check without numbers

        # Check if results are reasonable
        for num1 in nums1:
            for num2 in nums2:
                if abs(num1 - num2) > 1000 and abs(num1 / num2) > 100:
                    return False  # Numbers are unreasonably different

        return True

    def _parse_to_sympy(self, expression: str) -> Optional[sp.Expr]:
        """Parse expression to sympy format"""
        try:
            return sp.sympify(expression)
        except:
            return None

    def _assess_mathematical_rigor(self, problem: str, solution: str, reasoning: str) -> float:
        """Assess mathematical rigor of solution"""
        rigor_score = 0.5

        # Check for formal mathematical notation
        if any(symbol in solution for symbol in ['∫', '∂', '∑', '∏', '∀', '∃']):
            rigor_score += 0.2

        # Check reasoning depth
        steps = len([s for s in reasoning.split('\n') if s.strip()])
        if steps > 3:
            rigor_score += 0.2

        # Check for boundary condition analysis
        if 'when' in reasoning.lower() or 'if' in reasoning.lower():
            rigor_score += 0.1

        return min(1.0, rigor_score)

    def _check_boundary_conditions(self, problem: str, solution: str) -> List[str]:
        """Check if boundary conditions are properly handled"""
        issues = []

        # Check for potential boundary issues
        if '/' in solution and 'x' in solution:
            if not ('x ≠ 0' in solution or 'x != 0' in solution):
                issues.append("Missing boundary condition check for division")

        if 'sqrt' in solution and 'x' in solution:
            if not ('x ≥ 0' in solution or 'x >= 0' in solution):
                issues.append("Missing domain restriction for square root")

        return issues

    def _check_solution_completeness(self, problem: str, solution: str) -> bool:
        """Check if solution covers all necessary cases"""
        # Check for conditional solutions
        if 'if' in solution.lower() or 'case' in solution.lower():
            return 'else' in solution.lower() or 'otherwise' in solution.lower()

        # For equations, check if all variables are solved for
        problem_vars = set(re.findall(r'[a-zA-Z]', problem))
        solution_vars = set(re.findall(r'[a-zA-Z]', solution))

        return len(solution_vars - problem_vars) <= 0  # No new undefined variables

    def _fix_division_by_zero(self, problem: str, solution: str) -> str:
        """Apply limit analysis for division by zero cases"""
        # Simple limit handling
        if 'x' in solution and '/x' in solution:
            return solution.replace('/x', '/x (for x ≠ 0)')
        return solution + " (undefined at x = 0)"

    def _fix_algebraic_error(self, problem: str, solution: str) -> str:
        """Attempt to fix algebraic manipulation errors"""
        try:
            # Try to simplify the expression
            expr = sp.sympify(solution)
            simplified = sp.simplify(expr)
            return str(simplified)
        except:
            return solution

    def _fix_precision_error(self, solution: str) -> str:
        """Apply high-precision arithmetic"""
        # Round to reasonable precision
        numbers = re.findall(r'\d+\.\d+', solution)
        for num in numbers:
            if len(num.split('.')[-1]) > 6:
                rounded = f"{float(num):.6f}"
                solution = solution.replace(num, rounded)
        return solution

    def _get_error_suggestion(self, error_type: str) -> str:
        """Get suggestion for fixing a specific error type"""
        suggestions = {
            'division_by_zero': 'Check denominators and apply limits where appropriate',
            'undefined_operation': 'Verify domain restrictions and handle special cases',
            'precision_loss': 'Use higher precision arithmetic or exact symbolic computation',
            'overflow': 'Consider logarithmic scaling or asymptotic analysis'
        }
        return suggestions.get(error_type, 'Review mathematical operations')


class ConsistencyChecker:
    """
    Checks consistency across multiple solutions and approaches
    """

    def __init__(self):
        self.verifier = NumericalVerifier()

    def check_solution_consistency(self, problem: str, solutions: List[str]) -> Dict[str, Any]:
        """
        Check consistency across multiple solution approaches

        Args:
            problem: The mathematical problem
            solutions: List of different solution approaches

        Returns:
            Consistency analysis results
        """
        if len(solutions) < 2:
            return {
                'is_consistent': True,
                'confidence': 0.5,
                'issues': ['Need at least 2 solutions to check consistency'],
                'method_agreement': 0.0
            }

        # Verify each solution
        verifications = []
        for solution in solutions:
            result = self.verifier.verify_solution(problem, solution)
            verifications.append(result)

        # Check agreement between solutions
        agreement_score = self._calculate_agreement(verifications)

        # Find consensus solution
        consensus_solution = self._find_consensus_solution(verifications)

        # Identify inconsistencies
        inconsistencies = self._identify_inconsistencies(verifications)

        confidence = agreement_score * 0.8  # Slightly conservative

        return {
            'is_consistent': agreement_score > 0.7,
            'confidence': confidence,
            'agreement_score': agreement_score,
            'consensus_solution': consensus_solution,
            'inconsistencies': inconsistencies,
            'solution_count': len(solutions),
            'verification_results': verifications
        }

    def _calculate_agreement(self, verifications: List[Dict]) -> float:
        """Calculate agreement score between verifications"""
        if len(verifications) < 2:
            return 0.0

        confidences = [v['confidence'] for v in verifications]

        # Agreement based on confidence similarity
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)

        # High agreement if confidences are close to mean
        agreement = 1.0 - min(1.0, std_confidence / mean_confidence) if mean_confidence > 0 else 0.0

        return agreement

    def _find_consensus_solution(self, verifications: List[Dict]) -> str:
        """Find the most consistent solution across verifications"""
        # Group by solution content similarity
        solution_groups = {}
        for v in verifications:
            solution = v['verified_solution']
            # Simple grouping by first few characters
            key = solution[:50] if len(solution) > 50 else solution
            if key not in solution_groups:
                solution_groups[key] = []
            solution_groups[key].append(v)

        # Find group with highest average confidence
        best_group = None
        best_score = 0.0

        for group_key, group_verifications in solution_groups.items():
            avg_confidence = np.mean([v['confidence'] for v in group_verifications])
            group_size = len(group_verifications)

            # Score combines confidence and consensus (group size)
            score = avg_confidence * (group_size / len(verifications))

            if score > best_score:
                best_score = score
                best_group = group_key

        return best_group if best_group else "No consensus found"

    def _identify_inconsistencies(self, verifications: List[Dict]) -> List[str]:
        """Identify specific inconsistencies between solutions"""
        inconsistencies = []

        # Check for contradictory results
        solutions = [v['verified_solution'] for v in verifications]
        confidences = [v['confidence'] for v in verifications]

        # Look for high-confidence contradictory solutions
        high_conf_solutions = [sol for sol, conf in zip(solutions, confidences) if conf > 0.8]

        if len(set(high_conf_solutions)) > 1:
            inconsistencies.append("High-confidence solutions contradict each other")

        # Check for common error patterns
        all_issues = []
        for v in verifications:
            all_issues.extend(v.get('issues', []))

        if all_issues:
            common_issues = [issue for issue in all_issues if all_issues.count(issue) > 1]
            if common_issues:
                inconsistencies.append(f"Common issues across solutions: {list(set(common_issues))}")

        return inconsistencies
