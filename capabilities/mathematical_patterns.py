"""
Mathematical Pattern Library - Recognition patterns for advanced mathematical constructs.

This module provides comprehensive pattern recognition for:
- Inequalities and their properties
- Complex equations and systems
- Mathematical proofs and theorems
- Geometric patterns and transformations
- Calculus patterns (limits, derivatives, integrals)
- Algebraic structures and groups
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Pattern
from dataclasses import dataclass
from enum import Enum
import sympy as sp


class PatternType(Enum):
    """Types of mathematical patterns"""
    INEQUALITY = "inequality"
    EQUATION_SYSTEM = "equation_system"
    PROOF_STRUCTURE = "proof_structure"
    GEOMETRIC_TRANSFORMATION = "geometric_transformation"
    CALCULUS_PATTERN = "calculus_pattern"
    ALGEBRAIC_STRUCTURE = "algebraic_structure"
    NUMBER_THEORY = "number_theory"
    LOGIC_PATTERN = "logic_pattern"


@dataclass
class PatternMatch:
    """Represents a pattern match result"""
    pattern_type: PatternType
    pattern_name: str
    matched_text: str
    confidence: float
    properties: Dict[str, Any]
    transformations: List[str]
    related_theorems: List[str]


@dataclass
class InequalityPattern:
    """Pattern for inequality recognition"""
    inequality_type: str
    left_expr: str
    right_expr: str
    direction: str  # '<', '>', '<=', '>='
    variables: Set[str]
    properties: Dict[str, Any]


class InequalityRecognizer:
    """
    Recognizes and analyzes mathematical inequalities.
    """

    def __init__(self):
        self.inequality_patterns = {
            'linear': re.compile(r'([a-zA-Z][a-zA-Z0-9]*)\s*([<>]=?)\s*([+-]?\d*\.?\d+[a-zA-Z]?|[a-zA-Z][a-zA-Z0-9]*)'),
            'quadratic': re.compile(r'[a-zA-Z]\^2\s*[<>]=?\s*.*'),
            'fractional': re.compile(r'\d+/\d+\s*[<>]=?\s*\d+/\d+'),
            'absolute_value': re.compile(r'\|[^|]+\|\s*[<>]=?\s*.*'),
            'exponential': re.compile(r'\d+\^\([a-zA-Z][a-zA-Z0-9]*\)\s*[<>]=?\s*.*'),
        }

    def recognize_inequality(self, expression: str) -> Optional[InequalityPattern]:
        """Recognize inequality patterns in expressions."""
        expression = expression.strip()

        # Check for inequality symbols
        inequality_ops = ['<=', '>=', '<', '>']
        found_op = None
        for op in inequality_ops:
            if op in expression:
                found_op = op
                break

        if not found_op:
            return None

        # Split on the inequality
        parts = expression.split(found_op, 1)
        if len(parts) != 2:
            return None

        left_expr, right_expr = parts[0].strip(), parts[1].strip()

        # Extract variables
        variables = set(re.findall(r'[a-zA-Z][a-zA-Z0-9]*', expression))

        # Determine inequality type
        inequality_type = self._classify_inequality(left_expr, right_expr, found_op)

        # Analyze properties
        properties = self._analyze_inequality_properties(left_expr, right_expr, found_op)

        return InequalityPattern(
            inequality_type=inequality_type,
            left_expr=left_expr,
            right_expr=right_expr,
            direction=found_op,
            variables=variables,
            properties=properties
        )

    def _classify_inequality(self, left: str, right: str, op: str) -> str:
        """Classify the type of inequality."""
        if '^2' in left or '^2' in right:
            return 'quadratic'
        elif '|' in left or '|' in right:
            return 'absolute_value'
        elif '/' in left or '/' in right:
            return 'fractional'
        elif '^' in left or '^' in right:
            return 'exponential'
        elif re.match(r'[a-zA-Z]\s*[<>]=?\s*\d+', left + op + right):
            return 'linear'
        else:
            return 'general'

    def _analyze_inequality_properties(self, left: str, right: str, op: str) -> Dict[str, Any]:
        """Analyze properties of the inequality."""
        properties = {}

        # Check if inequality can be solved
        try:
            # Simple variable isolation
            variables = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', left)
            if len(variables) == 1 and variables[0] not in right:
                properties['solvable'] = True
                properties['solution_direction'] = self._get_solution_direction(left, right, op, variables[0])
            else:
                properties['solvable'] = False
        except:
            properties['solvable'] = False

        # Check for boundary conditions
        if '=' in op:
            properties['includes_boundary'] = True
        else:
            properties['includes_boundary'] = False

        return properties

    def _get_solution_direction(self, left: str, right: str, op: str, var: str) -> str:
        """Get the solution direction for simple inequalities."""
        # For ax + b > c, solution depends on sign of a
        # This is a simplified analysis
        if var in left:
            return f"Solve for {var} considering the inequality direction"

        return "Complex inequality requiring algebraic manipulation"


class EquationSystemRecognizer:
    """
    Recognizes and analyzes systems of equations.
    """

    def __init__(self):
        self.system_patterns = {
            'linear_system': re.compile(r'.*=\s*.*(?:\n.*=\s*.*)+'),
            'nonlinear_system': re.compile(r'.*\^2.*=.*|.*sqrt.*=.*'),
            'differential_system': re.compile(r'd\w+/d\w+.*=.*'),
        }

    def analyze_system(self, equations: List[str]) -> Dict[str, Any]:
        """Analyze a system of equations."""
        if not equations:
            return {'type': 'empty', 'properties': {}}

        # Determine system type
        system_type = self._classify_system(equations)

        # Extract variables
        all_vars = set()
        for eq in equations:
            vars_in_eq = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', eq)
            all_vars.update(vars_in_eq)

        # Check if system is consistent
        consistency = self._check_consistency(equations)

        # Determine if solvable
        solvable = self._check_solvability(equations, all_vars)

        return {
            'type': system_type,
            'num_equations': len(equations),
            'variables': list(all_vars),
            'is_consistent': consistency['is_consistent'],
            'is_solvable': solvable,
            'properties': consistency.get('properties', {}),
            'solution_method': self._suggest_solution_method(system_type, len(equations), len(all_vars))
        }

    def _classify_system(self, equations: List[str]) -> str:
        """Classify the type of equation system."""
        combined = ' '.join(equations)

        if re.search(r'd\w+/d\w+', combined):
            return 'differential'
        elif any('^2' in eq or 'sqrt' in eq for eq in equations):
            return 'nonlinear'
        elif all('=' in eq and not any(char in eq for char in ['^', 'sqrt', 'sin', 'cos']) for eq in equations):
            return 'linear'
        else:
            return 'mixed'

    def _check_consistency(self, equations: List[str]) -> Dict[str, Any]:
        """Check if the system is mathematically consistent."""
        # Simplified consistency check
        try:
            # Count variables and equations
            num_eq = len(equations)

            # For linear systems, check if number of equations makes sense
            if num_eq == 0:
                return {'is_consistent': True, 'properties': {'trivial': True}}
            elif num_eq == 1:
                return {'is_consistent': True, 'properties': {'single_equation': True}}
            else:
                # For multiple equations, assume consistent unless obviously contradictory
                return {'is_consistent': True, 'properties': {'multiple_equations': True}}

        except:
            return {'is_consistent': False, 'properties': {'error': 'analysis_failed'}}

    def _check_solvability(self, equations: List[str], variables: Set[str]) -> bool:
        """Check if the system is solvable."""
        num_eq = len(equations)
        num_var = len(variables)

        # Basic solvability check
        if num_eq == num_var:
            return True  # Square system
        elif num_eq > num_var:
            return True  # Overdetermined (may have solution)
        else:
            return False  # Underdetermined (infinite solutions or none)

    def _suggest_solution_method(self, system_type: str, num_eq: int, num_var: int) -> str:
        """Suggest appropriate solution method."""
        if system_type == 'linear':
            if num_eq == num_var == 2:
                return 'substitution or elimination'
            elif num_eq == num_var == 3:
                return 'Gaussian elimination or Cramer\'s rule'
            else:
                return 'matrix methods'
        elif system_type == 'nonlinear':
            return 'numerical methods or substitution'
        elif system_type == 'differential':
            return 'analytical methods or numerical integration'
        else:
            return 'case-by-case analysis'


class ProofStructureRecognizer:
    """
    Recognizes mathematical proof structures and patterns.
    """

    def __init__(self):
        self.proof_patterns = {
            'direct_proof': ['assume', 'show', 'therefore'],
            'contradiction': ['assume', 'contradiction', 'therefore'],
            'induction': ['base_case', 'inductive_hypothesis', 'inductive_step'],
            'contrapositive': ['assume_not', 'show_not', 'therefore'],
        }

    def analyze_proof_structure(self, proof_text: str) -> Dict[str, Any]:
        """Analyze the structure of a mathematical proof."""
        lines = [line.strip() for line in proof_text.split('\n') if line.strip()]

        # Identify proof type
        proof_type = self._identify_proof_type(lines)

        # Extract logical flow
        logical_flow = self._extract_logical_flow(lines)

        # Check for common proof errors
        errors = self._check_proof_errors(lines)

        # Validate logical consistency
        consistency_score = self._check_logical_consistency(lines)

        return {
            'proof_type': proof_type,
            'logical_flow': logical_flow,
            'num_steps': len(lines),
            'errors': errors,
            'consistency_score': consistency_score,
            'is_complete': self._check_proof_completeness(lines, proof_type)
        }

    def _identify_proof_type(self, lines: List[str]) -> str:
        """Identify the type of proof."""
        text = ' '.join(lines).lower()

        if 'contradiction' in text and 'assume' in text:
            return 'proof_by_contradiction'
        elif 'inductive' in text or 'base case' in text:
            return 'mathematical_induction'
        elif 'assume not' in text or 'contrapositive' in text:
            return 'proof_by_contrapositive'
        elif any(word in text for word in ['assume', 'suppose', 'let']):
            return 'direct_proof'
        else:
            return 'unknown'

    def _extract_logical_flow(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract the logical flow of the proof."""
        flow = []

        for i, line in enumerate(lines):
            step_type = self._classify_proof_step(line)
            flow.append({
                'step_number': i + 1,
                'content': line,
                'type': step_type,
                'logical_connectors': self._find_logical_connectors(line)
            })

        return flow

    def _classify_proof_step(self, line: str) -> str:
        """Classify a proof step."""
        line_lower = line.lower()

        if any(word in line_lower for word in ['assume', 'suppose', 'let', 'given']):
            return 'assumption'
        elif any(word in line_lower for word in ['therefore', 'thus', 'hence', 'so']):
            return 'conclusion'
        elif any(word in line_lower for word in ['because', 'since', 'by']):
            return 'justification'
        elif '=' in line or any(op in line for op in ['+', '-', '*', '/']):
            return 'computation'
        else:
            return 'statement'

    def _find_logical_connectors(self, line: str) -> List[str]:
        """Find logical connectors in a proof step."""
        connectors = []
        line_lower = line.lower()

        logical_words = [
            'and', 'or', 'not', 'if', 'then', 'because', 'therefore',
            'thus', 'hence', 'so', 'since', 'by', 'given', 'assume'
        ]

        for word in logical_words:
            if word in line_lower:
                connectors.append(word)

        return connectors

    def _check_proof_errors(self, lines: List[str]) -> List[str]:
        """Check for common proof errors."""
        errors = []

        # Check for circular reasoning
        if len(lines) > 3:
            first_half = ' '.join(lines[:len(lines)//2])
            second_half = ' '.join(lines[len(lines)//2:])
            if any(word in first_half and word in second_half
                   for word in ['assume', 'suppose', 'let']):
                errors.append('Potential circular reasoning')

        # Check for unjustified leaps
        large_gaps = []
        for i in range(1, len(lines)):
            if self._is_large_logical_gap(lines[i-1], lines[i]):
                large_gaps.append(i)

        if large_gaps:
            errors.append(f'Large logical gaps at steps: {large_gaps}')

        return errors

    def _is_large_logical_gap(self, line1: str, line2: str) -> bool:
        """Check if there's a large logical gap between steps."""
        # Simplified check - look for missing intermediate steps
        connectors1 = self._find_logical_connectors(line1)
        connectors2 = self._find_logical_connectors(line2)

        # If neither has strong logical connectors, might be a gap
        strong_connectors = ['therefore', 'thus', 'hence', 'so']
        has_strong1 = any(conn in connectors1 for conn in strong_connectors)
        has_strong2 = any(conn in connectors2 for conn in strong_connectors)

        return not (has_strong1 or has_strong2)

    def _check_logical_consistency(self, lines: List[str]) -> float:
        """Check logical consistency of the proof."""
        # Simplified consistency scoring
        consistency_indicators = 0
        total_checks = 0

        for line in lines:
            if any(word in line.lower() for word in ['therefore', 'thus', 'hence']):
                total_checks += 1
                # Check if there's a preceding justification
                if any(word in line.lower() for word in ['because', 'since', 'by']):
                    consistency_indicators += 1

        return consistency_indicators / total_checks if total_checks > 0 else 0.5

    def _check_proof_completeness(self, lines: List[str], proof_type: str) -> bool:
        """Check if the proof is complete."""
        if not lines:
            return False

        # Check for conclusion
        last_line = lines[-1].lower()
        has_conclusion = any(word in last_line for word in ['therefore', 'thus', 'hence', 'q.e.d', 'proven'])

        # Check for required elements based on proof type
        if proof_type == 'proof_by_contradiction':
            has_contradiction = any('contradiction' in line.lower() for line in lines)
            return has_conclusion and has_contradiction
        elif proof_type == 'mathematical_induction':
            has_base = any('base' in line.lower() for line in lines)
            has_inductive = any('inductive' in line.lower() for line in lines)
            return has_conclusion and has_base and has_inductive
        else:
            return has_conclusion


class CalculusPatternRecognizer:
    """
    Recognizes calculus patterns (limits, derivatives, integrals).
    """

    def __init__(self):
        self.limit_pattern = re.compile(r'\\?lim(?:_{([^}]+)})?\s*([^=]+)=(.+)')
        self.derivative_pattern = re.compile(r'(?:d/dx|∂/∂x|\\frac{d}{dx})\\?\(([^)]+)\\?\)')
        self.integral_pattern = re.compile(r'\\?int(?:_{([^}]+)})?(?:\\?\^\\?{([^}]+)})?\\?\(([^)]+)\\?\)')

    def recognize_calculus_pattern(self, expression: str) -> Dict[str, Any]:
        """Recognize calculus patterns in expressions."""
        expression = expression.strip()

        if self.limit_pattern.search(expression):
            return self._analyze_limit(expression)
        elif self.derivative_pattern.search(expression):
            return self._analyze_derivative(expression)
        elif self.integral_pattern.search(expression):
            return self._analyze_integral(expression)
        else:
            return {'type': 'unknown', 'properties': {}}

    def _analyze_limit(self, expression: str) -> Dict[str, Any]:
        """Analyze a limit expression."""
        match = self.limit_pattern.search(expression)

        if match:
            variable_part = match.group(1) or 'x'
            approaching = match.group(2).strip()
            function = match.group(3).strip()

            # Extract variable and approach value
            if '→' in approaching:
                var, value = approaching.split('→', 1)
                var = var.strip()
                value = value.strip()
            else:
                var = variable_part
                value = approaching

            # Check for indeterminate forms
            indeterminate = self._check_indeterminate_form(function, var, value)

            return {
                'type': 'limit',
                'variable': var,
                'approach_value': value,
                'function': function,
                'indeterminate_form': indeterminate,
                'solution_method': self._suggest_limit_method(indeterminate)
            }

        return {'type': 'limit', 'properties': {'parsing_error': True}}

    def _analyze_derivative(self, expression: str) -> Dict[str, Any]:
        """Analyze a derivative expression."""
        match = self.derivative_pattern.search(expression)

        if match:
            function = match.group(1)

            # Classify derivative type
            derivative_type = 'ordinary'
            if '∂' in expression:
                derivative_type = 'partial'

            # Analyze function complexity
            complexity = self._analyze_function_complexity(function)

            return {
                'type': 'derivative',
                'derivative_type': derivative_type,
                'function': function,
                'complexity': complexity,
                'solution_method': self._suggest_derivative_method(complexity)
            }

        return {'type': 'derivative', 'properties': {'parsing_error': True}}

    def _analyze_integral(self, expression: str) -> Dict[str, Any]:
        """Analyze an integral expression."""
        match = self.integral_pattern.search(expression)

        if match:
            lower_limit = match.group(1)
            upper_limit = match.group(2)
            integrand = match.group(3)

            # Determine integral type
            integral_type = 'definite' if lower_limit and upper_limit else 'indefinite'

            # Analyze integrand complexity
            complexity = self._analyze_function_complexity(integrand)

            return {
                'type': 'integral',
                'integral_type': integral_type,
                'integrand': integrand,
                'lower_limit': lower_limit,
                'upper_limit': upper_limit,
                'complexity': complexity,
                'solution_method': self._suggest_integral_method(complexity, integral_type)
            }

        return {'type': 'integral', 'properties': {'parsing_error': True}}

    def _check_indeterminate_form(self, function: str, var: str, value: str) -> Optional[str]:
        """Check if limit is an indeterminate form."""
        # Common indeterminate forms: 0/0, ∞/∞, 0*∞, ∞-∞, 0^0, 1^∞, ∞^0

        if value in ['∞', 'infinity']:
            if '/' in function:
                return '∞/∞'
        elif value == '0':
            if '/' in function:
                return '0/0'

        return None

    def _analyze_function_complexity(self, function: str) -> str:
        """Analyze the complexity of a mathematical function."""
        complexity_score = 0

        if '^' in function or '**' in function:
            complexity_score += 2
        if any(trig in function.lower() for trig in ['sin', 'cos', 'tan', 'log', 'ln', 'exp']):
            complexity_score += 1
        if '/' in function:
            complexity_score += 1
        if 'sqrt' in function or '^0.5' in function:
            complexity_score += 1

        if complexity_score >= 3:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _suggest_limit_method(self, indeterminate_form: Optional[str]) -> str:
        """Suggest method for evaluating limit."""
        if indeterminate_form == '0/0':
            return "L'Hôpital's rule or algebraic simplification"
        elif indeterminate_form == '∞/∞':
            return "L'Hôpital's rule or divide by highest power"
        else:
            return "Direct substitution or algebraic manipulation"

    def _suggest_derivative_method(self, complexity: str) -> str:
        """Suggest method for computing derivative."""
        if complexity == 'low':
            return 'power rule, product rule, quotient rule'
        elif complexity == 'medium':
            return 'chain rule, trigonometric derivatives, logarithmic derivatives'
        else:
            return 'advanced rules, implicit differentiation, or numerical methods'

    def _suggest_integral_method(self, complexity: str, integral_type: str) -> str:
        """Suggest method for computing integral."""
        if complexity == 'low':
            return 'power rule, basic trigonometric integrals'
        elif complexity == 'medium':
            return 'u-substitution, integration by parts'
        else:
            return 'advanced techniques, numerical integration, or lookup tables'


class MathematicalPatternLibrary:
    """
    Complete mathematical pattern recognition library.
    """

    def __init__(self):
        self.inequality_recognizer = InequalityRecognizer()
        self.equation_system_recognizer = EquationSystemRecognizer()
        self.proof_recognizer = ProofStructureRecognizer()
        self.calculus_recognizer = CalculusPatternRecognizer()

    def recognize_patterns(self, content: str) -> List[PatternMatch]:
        """
        Recognize all applicable mathematical patterns in content.

        Args:
            content: Mathematical content to analyze

        Returns:
            List of pattern matches with confidence scores
        """
        matches = []

        # Split content into lines for analysis
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Check for inequalities
        for line in lines:
            inequality = self.inequality_recognizer.recognize_inequality(line)
            if inequality:
                matches.append(PatternMatch(
                    pattern_type=PatternType.INEQUALITY,
                    pattern_name=inequality.inequality_type,
                    matched_text=line,
                    confidence=0.85,
                    properties={
                        'left_expr': inequality.left_expr,
                        'right_expr': inequality.right_expr,
                        'direction': inequality.direction,
                        'variables': list(inequality.variables),
                        'solvable': inequality.properties.get('solvable', False)
                    },
                    transformations=self._get_inequality_transformations(inequality),
                    related_theorems=['transitive_property', 'triangle_inequality']
                ))

        # Check for equation systems
        if len(lines) > 1:
            system_analysis = self.equation_system_recognizer.analyze_system(lines)
            if system_analysis['type'] != 'empty':
                matches.append(PatternMatch(
                    pattern_type=PatternType.EQUATION_SYSTEM,
                    pattern_name=system_analysis['type'] + '_system',
                    matched_text='\n'.join(lines),
                    confidence=0.8,
                    properties=system_analysis,
                    transformations=[],
                    related_theorems=['cramer_rule', 'gaussian_elimination']
                ))

        # Check for proof structures
        if len(lines) > 3:
            proof_analysis = self.proof_recognizer.analyze_proof_structure(content)
            if proof_analysis['proof_type'] != 'unknown':
                matches.append(PatternMatch(
                    pattern_type=PatternType.PROOF_STRUCTURE,
                    pattern_name=proof_analysis['proof_type'],
                    matched_text=content,
                    confidence=0.75,
                    properties=proof_analysis,
                    transformations=[],
                    related_theorems=['logical_equivalence', 'deduction_theorem']
                ))

        # Check for calculus patterns
        calculus_analysis = self.calculus_recognizer.recognize_calculus_pattern(content)
        if calculus_analysis['type'] != 'unknown':
            matches.append(PatternMatch(
                pattern_type=PatternType.CALCULUS_PATTERN,
                pattern_name=calculus_analysis['type'],
                matched_text=content,
                confidence=0.9,
                properties=calculus_analysis,
                transformations=[],
                related_theorems=['fundamental_theorem_calculus', 'chain_rule']
            ))

        return matches

    def _get_inequality_transformations(self, inequality: InequalityPattern) -> List[str]:
        """Get possible transformations for an inequality."""
        transformations = []

        if inequality.inequality_type == 'linear':
            transformations.extend([
                f"Add {inequality.right_expr} to both sides",
                f"Subtract {inequality.left_expr} from both sides",
                f"Multiply both sides by -1 (reverse inequality)"
            ])

        if inequality.inequality_type == 'quadratic':
            transformations.extend([
                "Move all terms to one side",
                "Factor the quadratic expression",
                "Use quadratic formula if applicable"
            ])

        return transformations

    def get_pattern_explanation(self, pattern_match: PatternMatch) -> str:
        """Get a human-readable explanation of a pattern match."""
        if pattern_match.pattern_type == PatternType.INEQUALITY:
            props = pattern_match.properties
            return f"Inequality: {props['left_expr']} {props['direction']} {props['right_expr']}"

        elif pattern_match.pattern_type == PatternType.EQUATION_SYSTEM:
            props = pattern_match.properties
            return f"{props['type'].title()} system with {props['num_equations']} equations in {len(props['variables'])} variables"

        elif pattern_match.pattern_type == PatternType.PROOF_STRUCTURE:
            props = pattern_match.properties
            return f"{props['proof_type'].replace('_', ' ').title()} with {props['consistency_score']:.1%} logical consistency"

        elif pattern_match.pattern_type == PatternType.CALCULUS_PATTERN:
            props = pattern_match.properties
            if props['type'] == 'limit':
                return f"Limit of {props['function']} as {props['variable']} approaches {props['approach_value']}"
            elif props['type'] == 'derivative':
                return f"Derivative of {props['function']}"
            elif props['type'] == 'integral':
                return f"{'Definite' if props.get('integral_type') == 'definite' else 'Indefinite'} integral of {props['integrand']}"

        return f"Pattern: {pattern_match.pattern_name}"


# Export the main pattern library
__all__ = ['MathematicalPatternLibrary', 'PatternMatch', 'PatternType']
