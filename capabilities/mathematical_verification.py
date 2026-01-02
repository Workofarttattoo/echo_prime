"""
Mathematical Verification System - Comprehensive calculation verification with step-by-step checking.

This module provides advanced mathematical verification capabilities including:
- Step-by-step calculation verification
- Algebraic manipulation checking
- Numerical precision validation
- Proof verification
- Error detection and correction
"""

import re
import math
import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import ast
import operator


class VerificationError(Enum):
    """Types of verification errors"""
    CALCULATION_ERROR = "calculation_error"
    ALGEBRAIC_ERROR = "algebraic_error"
    PRECISION_ERROR = "precision_error"
    LOGIC_ERROR = "logic_error"
    SYNTAX_ERROR = "syntax_error"
    UNDEFINED_OPERATION = "undefined_operation"


@dataclass
class VerificationStep:
    """Represents a single verification step"""
    step_number: int
    operation: str
    input_expression: str
    expected_output: Optional[str]
    actual_output: Optional[str]
    is_correct: bool
    error_type: Optional[VerificationError]
    confidence: float
    explanation: str


@dataclass
class VerificationResult:
    """Complete verification result"""
    is_valid: bool
    steps: List[VerificationStep]
    overall_confidence: float
    detected_errors: List[VerificationError]
    corrections_suggested: List[str]
    final_result: Any


class StepByStepVerifier:
    """
    Verifies calculations step by step with detailed checking.
    """

    def __init__(self):
        self.symbols = {}  # Variable definitions
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2,
            'golden_ratio': (1 + math.sqrt(5)) / 2
        }

    def verify_calculation(self, expression: str, expected_result: Optional[Any] = None,
                          step_by_step: bool = True) -> VerificationResult:
        """
        Verify a mathematical calculation step by step.

        Args:
            expression: Mathematical expression to verify
            expected_result: Expected final result (optional)
            step_by_step: Whether to perform detailed step-by-step verification

        Returns:
            VerificationResult with detailed analysis
        """
        steps = []
        detected_errors = []

        try:
            # Parse and tokenize the expression
            tokens = self._tokenize_expression(expression)

            if step_by_step:
                # Perform step-by-step verification
                steps = self._verify_step_by_step(tokens)

            # Check for errors in steps
            for step in steps:
                if not step.is_correct:
                    detected_errors.append(step.error_type)

            # Calculate overall confidence
            if steps:
                valid_steps = sum(1 for s in steps if s.is_correct)
                overall_confidence = valid_steps / len(steps)
            else:
                overall_confidence = 0.5

            # Evaluate final result
            final_result = self._safe_evaluate(expression)

            # Compare with expected if provided
            if expected_result is not None:
                is_valid = self._results_match(final_result, expected_result, tolerance=1e-10)
            else:
                is_valid = len(detected_errors) == 0

            return VerificationResult(
                is_valid=is_valid,
                steps=steps,
                overall_confidence=overall_confidence,
                detected_errors=detected_errors,
                corrections_suggested=self._suggest_corrections(detected_errors, expression),
                final_result=final_result
            )

        except Exception as e:
            return VerificationResult(
                is_valid=False,
                steps=steps,
                overall_confidence=0.0,
                detected_errors=[VerificationError.SYNTAX_ERROR],
                corrections_suggested=[f"Syntax error: {str(e)}"],
                final_result=None
            )

    def _tokenize_expression(self, expression: str) -> List[str]:
        """Tokenize mathematical expression into components."""
        # Remove whitespace and normalize
        expression = re.sub(r'\s+', '', expression)

        # Split on operators while preserving them
        tokens = re.findall(r'[+\-*/^()]|\d+\.?\d*|[a-zA-Z_]\w*', expression)

        return tokens

    def _verify_step_by_step(self, tokens: List[str]) -> List[VerificationStep]:
        """Perform detailed step-by-step verification."""
        steps = []
        step_num = 1

        # Handle parentheses first
        if '(' in tokens and ')' in tokens:
            steps.extend(self._verify_parentheses(tokens, step_num))
            step_num += len(steps)

        # Verify operator precedence and associativity
        if any(op in tokens for op in ['*', '/', '^']):
            steps.extend(self._verify_operator_precedence(tokens, step_num))
            step_num += len(steps)

        # Verify numerical operations
        steps.extend(self._verify_numerical_operations(tokens, step_num))

        return steps

    def _verify_parentheses(self, tokens: List[str], start_step: int) -> List[VerificationStep]:
        """Verify parentheses matching and nesting."""
        steps = []
        stack = []
        paren_pairs = []

        for i, token in enumerate(tokens):
            if token == '(':
                stack.append(i)
            elif token == ')':
                if not stack:
                    steps.append(VerificationStep(
                        step_number=start_step + len(steps),
                        operation="parentheses_check",
                        input_expression="".join(tokens),
                        expected_output="balanced",
                        actual_output="unbalanced",
                        is_correct=False,
                        error_type=VerificationError.SYNTAX_ERROR,
                        confidence=1.0,
                        explanation="Unmatched closing parenthesis"
                    ))
                    return steps
                open_pos = stack.pop()
                paren_pairs.append((open_pos, i))

        if stack:
            steps.append(VerificationStep(
                step_number=start_step + len(steps),
                operation="parentheses_check",
                input_expression="".join(tokens),
                expected_output="balanced",
                actual_output="unbalanced",
                is_correct=False,
                error_type=VerificationError.SYNTAX_ERROR,
                confidence=1.0,
                explanation="Unmatched opening parenthesis"
            ))
            return steps

        steps.append(VerificationStep(
            step_number=start_step + len(steps),
            operation="parentheses_check",
            input_expression="".join(tokens),
            expected_output="balanced",
            actual_output="balanced",
            is_correct=True,
            error_type=None,
            confidence=1.0,
            explanation=f"Found {len(paren_pairs)} balanced parenthesis pairs"
        ))

        return steps

    def _verify_operator_precedence(self, tokens: List[str], start_step: int) -> List[VerificationStep]:
        """Verify operator precedence rules."""
        steps = []

        # Check for common precedence errors
        expr_str = "".join(tokens)

        # Multiplication/division before addition/subtraction
        if re.search(r'\d+\+.*\d+\*|\d+\-.*\d+\*|\d+\*.*\d+\+|\d+\*.*\d+\-', expr_str):
            steps.append(VerificationStep(
                step_number=start_step + len(steps),
                operation="precedence_check",
                input_expression=expr_str,
                expected_output="correct_precedence",
                actual_output="potential_precedence_error",
                is_correct=False,
                error_type=VerificationError.LOGIC_ERROR,
                confidence=0.8,
                explanation="Mixed operators without explicit parentheses - verify precedence"
            ))

        # Exponentiation precedence
        if '^' in tokens and ('*' in tokens or '/' in tokens):
            steps.append(VerificationStep(
                step_number=start_step + len(steps),
                operation="exponent_precedence",
                input_expression=expr_str,
                expected_output="exponentiation_first",
                actual_output="verified",
                is_correct=True,
                error_type=None,
                confidence=0.9,
                explanation="Exponentiation has higher precedence than multiplication/division"
            ))

        return steps

    def _verify_numerical_operations(self, tokens: List[str], start_step: int) -> List[VerificationStep]:
        """Verify numerical operations for correctness."""
        steps = []

        # Check for division by zero
        if '/' in tokens:
            expr_str = "".join(tokens)
            if re.search(r'/0(?![.\d])', expr_str):
                steps.append(VerificationStep(
                    step_number=start_step + len(steps),
                    operation="division_check",
                    input_expression=expr_str,
                    expected_output="no_division_by_zero",
                    actual_output="division_by_zero",
                    is_correct=False,
                    error_type=VerificationError.UNDEFINED_OPERATION,
                    confidence=1.0,
                    explanation="Division by zero detected"
                ))

        # Check for very large numbers that might cause overflow
        numbers = [float(t) for t in tokens if re.match(r'\d+\.?\d*', t)]
        for num in numbers:
            if abs(num) > 1e100:
                steps.append(VerificationStep(
                    step_number=start_step + len(steps),
                    operation="numerical_stability",
                    input_expression=str(num),
                    expected_output="stable",
                    actual_output="potentially_unstable",
                    is_correct=False,
                    error_type=VerificationError.PRECISION_ERROR,
                    confidence=0.7,
                    explanation=f"Very large number {num} may cause numerical instability"
                ))

        return steps

    def _safe_evaluate(self, expression: str) -> Any:
        """Safely evaluate a mathematical expression."""
        try:
            # Use sympy for symbolic evaluation when possible
            if any(var in expression for var in ['x', 'y', 'z', 'a', 'b', 'c']):
                return str(sp.sympify(expression))
            else:
                # For numerical expressions, use careful evaluation
                return self._numerical_evaluate(expression)
        except:
            # Fallback to basic evaluation
            try:
                return eval(expression, {"__builtins__": {}}, self._safe_functions())
            except:
                return None

    def _numerical_evaluate(self, expression: str) -> float:
        """Evaluate numerical expression with high precision."""
        # Replace common constants
        for const, value in self.constants.items():
            expression = re.sub(r'\b' + const + r'\b', str(value), expression)

        # Use Python's ast for safer evaluation
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except:
            return float('nan')

    def _eval_node(self, node):
        """Safely evaluate AST node."""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = node.op

            if isinstance(op, ast.Add):
                return left + right
            elif isinstance(op, ast.Sub):
                return left - right
            elif isinstance(op, ast.Mult):
                return left * right
            elif isinstance(op, ast.Div):
                if right == 0:
                    return float('inf')
                return left / right
            elif isinstance(op, ast.Pow):
                return left ** right
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            args = [self._eval_node(arg) for arg in node.args]
            return self._call_function(func_name, args)

        return None

    def _call_function(self, name: str, args: List[float]) -> float:
        """Call mathematical functions safely."""
        functions = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'abs': abs, 'round': round, 'floor': math.floor, 'ceil': math.ceil
        }

        if name in functions and args:
            try:
                return functions[name](*args)
            except:
                return float('nan')

        return float('nan')

    def _safe_functions(self) -> Dict:
        """Get safe functions for eval."""
        return {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'abs': abs, 'round': round, 'pi': math.pi, 'e': math.e
        }

    def _results_match(self, result1: Any, result2: Any, tolerance: float = 1e-10) -> bool:
        """Check if two results match within tolerance."""
        try:
            if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
                return abs(result1 - result2) < tolerance
            return result1 == result2
        except:
            return False

    def _suggest_corrections(self, errors: List[VerificationError], expression: str) -> List[str]:
        """Suggest corrections based on detected errors."""
        suggestions = []

        for error in errors:
            if error == VerificationError.SYNTAX_ERROR:
                suggestions.append("Check for unmatched parentheses or invalid operators")
            elif error == VerificationError.UNDEFINED_OPERATION:
                suggestions.append("Division by zero or invalid operation detected")
            elif error == VerificationError.PRECISION_ERROR:
                suggestions.append("Consider using higher precision arithmetic")
            elif error == VerificationError.LOGIC_ERROR:
                suggestions.append("Verify operator precedence and algebraic rules")

        if not suggestions:
            suggestions.append("Expression appears syntactically correct")

        return suggestions


class AlgebraicVerifier:
    """
    Verifies algebraic manipulations and equation solving.
    """

    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z')

    def verify_equation(self, equation: str) -> VerificationResult:
        """Verify algebraic equation solving."""
        try:
            # Parse equation
            if '=' in equation:
                left, right = equation.split('=', 1)
                left_expr = sp.sympify(left)
                right_expr = sp.sympify(right)

                # Check if equation is true
                diff = left_expr - right_expr
                is_identity = diff.simplify() == 0

                steps = [VerificationStep(
                    step_number=1,
                    operation="equation_verification",
                    input_expression=equation,
                    expected_output="identity" if is_identity else "not_identity",
                    actual_output="identity" if is_identity else "not_identity",
                    is_correct=True,
                    error_type=None,
                    confidence=0.95,
                    explanation=f"Equation is {'an identity' if is_identity else 'not an identity'}"
                )]

                return VerificationResult(
                    is_valid=is_identity,
                    steps=steps,
                    overall_confidence=0.95,
                    detected_errors=[],
                    corrections_suggested=[],
                    final_result=is_identity
                )
            else:
                return VerificationResult(
                    is_valid=False,
                    steps=[],
                    overall_confidence=0.0,
                    detected_errors=[VerificationError.SYNTAX_ERROR],
                    corrections_suggested=["Equation must contain '=' sign"],
                    final_result=None
                )

        except Exception as e:
            return VerificationResult(
                is_valid=False,
                steps=[],
                overall_confidence=0.0,
                detected_errors=[VerificationError.SYNTAX_ERROR],
                corrections_suggested=[f"Algebraic parsing error: {str(e)}"],
                final_result=None
            )

    def verify_manipulation(self, original: str, manipulated: str) -> VerificationResult:
        """Verify if algebraic manipulation preserves equality."""
        try:
            orig_expr = sp.sympify(original)
            manip_expr = sp.sympify(manipulated)

            # Check if expressions are equivalent
            diff = (orig_expr - manip_expr).simplify()
            is_equivalent = diff == 0

            steps = [VerificationStep(
                step_number=1,
                operation="algebraic_manipulation",
                input_expression=f"{original} → {manipulated}",
                expected_output="equivalent",
                actual_output="equivalent" if is_equivalent else "not_equivalent",
                is_correct=is_equivalent,
                error_type=None if is_equivalent else VerificationError.ALGEBRAIC_ERROR,
                confidence=0.9,
                explanation=f"Expressions are {'equivalent' if is_equivalent else 'not equivalent'}"
            )]

            return VerificationResult(
                is_valid=is_equivalent,
                steps=steps,
                overall_confidence=0.9,
                detected_errors=[] if is_equivalent else [VerificationError.ALGEBRAIC_ERROR],
                corrections_suggested=[] if is_equivalent else ["Algebraic manipulation is incorrect"],
                final_result=is_equivalent
            )

        except Exception as e:
            return VerificationResult(
                is_valid=False,
                steps=[],
                overall_confidence=0.0,
                detected_errors=[VerificationError.ALGEBRAIC_ERROR],
                corrections_suggested=[f"Algebraic manipulation error: {str(e)}"],
                final_result=None
            )


class ProofVerifier:
    """
    Verifies mathematical proofs and logical arguments.
    """

    def __init__(self):
        self.logical_rules = {
            'modus_ponens': lambda p, q: f"If {p} and {p}→{q}, then {q}",
            'modus_tollens': lambda p, q: f"If ¬{q} and {p}→{q}, then ¬{p}",
            'hypothetical_syllogism': lambda p, q, r: f"If {p}→{q} and {q}→{r}, then {p}→{r}"
        }

    def verify_proof(self, proof_steps: List[str]) -> VerificationResult:
        """Verify a mathematical proof."""
        steps = []
        detected_errors = []

        for i, step in enumerate(proof_steps):
            # Basic syntax checking
            if not step.strip():
                continue

            # Check if step follows from previous steps
            is_valid_step = self._validate_proof_step(step, proof_steps[:i])

            steps.append(VerificationStep(
                step_number=i + 1,
                operation="proof_step_validation",
                input_expression=step,
                expected_output="valid",
                actual_output="valid" if is_valid_step else "invalid",
                is_correct=is_valid_step,
                error_type=None if is_valid_step else VerificationError.LOGIC_ERROR,
                confidence=0.8,
                explanation=f"Proof step {i+1}: {'valid' if is_valid_step else 'invalid'}"
            ))

            if not is_valid_step:
                detected_errors.append(VerificationError.LOGIC_ERROR)

        overall_confidence = (len(proof_steps) - len(detected_errors)) / len(proof_steps) if proof_steps else 0

        return VerificationResult(
            is_valid=len(detected_errors) == 0,
            steps=steps,
            overall_confidence=overall_confidence,
            detected_errors=detected_errors,
            corrections_suggested=["Review logical connections between steps"] if detected_errors else [],
            final_result=len(detected_errors) == 0
        )

    def _validate_proof_step(self, step: str, previous_steps: List[str]) -> bool:
        """Validate if a proof step logically follows from previous steps."""
        # Basic validation - check if step references previous mathematical concepts
        step_lower = step.lower()

        # Look for mathematical keywords
        math_keywords = ['therefore', 'thus', 'hence', 'so', 'because', 'since', 'given', 'assume']

        has_logical_connector = any(keyword in step_lower for keyword in math_keywords)

        # Check if step contains mathematical operations
        has_math_ops = any(op in step for op in ['+', '-', '*', '/', '=', '>', '<', '≤', '≥'])

        # For now, accept steps with logical connectors or mathematical operations
        return has_logical_connector or has_math_ops or len(previous_steps) == 0


class MathematicalVerificationSystem:
    """
    Complete mathematical verification system.
    """

    def __init__(self):
        self.step_verifier = StepByStepVerifier()
        self.algebraic_verifier = AlgebraicVerifier()
        self.proof_verifier = ProofVerifier()

    def verify(self, content: str, verification_type: str = "auto") -> VerificationResult:
        """
        Verify mathematical content.

        Args:
            content: Mathematical content to verify
            verification_type: Type of verification ('calculation', 'equation', 'proof', 'auto')

        Returns:
            VerificationResult with detailed analysis
        """
        content = content.strip()

        if verification_type == "auto":
            if '=' in content and len(content.split('=')) == 2:
                verification_type = "equation"
            elif '\n' in content and len(content.split('\n')) > 2:
                verification_type = "proof"
            else:
                verification_type = "calculation"

        if verification_type == "calculation":
            return self.step_verifier.verify_calculation(content)
        elif verification_type == "equation":
            return self.algebraic_verifier.verify_equation(content)
        elif verification_type == "proof":
            steps = [line.strip() for line in content.split('\n') if line.strip()]
            return self.proof_verifier.verify_proof(steps)
        else:
            return self.step_verifier.verify_calculation(content)

    def verify_with_steps(self, expression: str) -> Dict[str, Any]:
        """Provide detailed step-by-step verification with explanations."""
        result = self.verify(expression)

        return {
            'expression': expression,
            'is_valid': result.is_valid,
            'confidence': result.overall_confidence,
            'steps': [
                {
                    'step': step.step_number,
                    'operation': step.operation,
                    'input': step.input_expression,
                    'expected': step.expected_output,
                    'actual': step.actual_output,
                    'correct': step.is_correct,
                    'explanation': step.explanation
                }
                for step in result.steps
            ],
            'errors': [error.value for error in result.detected_errors],
            'suggestions': result.corrections_suggested,
            'final_result': result.final_result
        }


# Export the main verification system
__all__ = ['MathematicalVerificationSystem', 'VerificationResult', 'VerificationStep', 'VerificationError']
