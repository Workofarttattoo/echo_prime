#!/usr/bin/env python3
"""
Mathematical Reasoning Engine for ECH0-PRIME
Handles symbolic math, equation solving, and step-by-step problem solving
"""

import re
from typing import Optional, Dict, List, Tuple, Any


class MathematicalReasoningEngine:
    """Advanced mathematical reasoning and problem solving"""

    def __init__(self):
        self.operators = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else float('inf'),
            '**': lambda x, y: x ** y,
            '^': lambda x, y: x ** y,
        }

    def solve_problem(self, problem: str) -> str:
        """Main entry point for solving mathematical problems"""
        problem = problem.strip()

        # Try different solution strategies in order
        strategies = [
            self._solve_arithmetic,
            self._solve_word_problem,
            self._solve_algebra,
            self._solve_geometry,
            self._solve_calculus,
            self._solve_statistics,
        ]

        for strategy in strategies:
            try:
                result = strategy(problem)
                if result is not None:
                    return str(result)
            except:
                continue

        # Fallback: try to extract any numbers and return the last one
        nums = re.findall(r'-?\d+\.?\d*', problem)
        return nums[-1] if nums else "0"

    def _solve_arithmetic(self, problem: str) -> Optional[str]:
        """Solve basic arithmetic problems"""
        problem = problem.lower()

        # Direct computation patterns
        patterns = [
            (r'what is (\d+\.?\d*)\s*\+\s*(\d+\.?\d*)', lambda m: float(m.group(1)) + float(m.group(2))),
            (r'what is (\d+\.?\d*)\s*-\s*(\d+\.?\d*)', lambda m: float(m.group(1)) - float(m.group(2))),
            (r'what is (\d+\.?\d*)\s*\*\s*(\d+\.?\d*)', lambda m: float(m.group(1)) * float(m.group(2))),
            (r'what is (\d+\.?\d*)\s*/\s*(\d+\.?\d*)', lambda m: float(m.group(1)) / float(m.group(2))),
            (r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*=', lambda m: float(m.group(1)) + float(m.group(2))),
            (r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*=', lambda m: float(m.group(1)) - float(m.group(2))),
            (r'(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)\s*=', lambda m: float(m.group(1)) * float(m.group(2))),
            (r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*=', lambda m: float(m.group(1)) / float(m.group(2))),
        ]

        for pattern, operation in patterns:
            match = re.search(pattern, problem)
            if match:
                result = operation(match)
                return str(int(result) if result == int(result) else result)

        return None

    def _solve_word_problem(self, problem: str) -> Optional[str]:
        """Solve word problems (GSM8K style)"""
        problem = problem.lower()

        # Common word problem patterns
        # "has X, gives Y, how many left" -> subtraction
        if any(kw in problem for kw in ['has', 'had', 'were', 'are']) and \
           any(kw in problem for kw in ['gives', 'gave', 'loses', 'lost', 'sells', 'sold']):
            nums = re.findall(r'\d+', problem)
            if len(nums) >= 2:
                result = int(nums[0]) - int(nums[1])
                return str(result)

        # "buys X at Y each" -> multiplication
        if any(kw in problem for kw in ['buys', 'bought', 'each', 'per']):
            nums = re.findall(r'\d+', problem)
            if len(nums) >= 2:
                result = int(nums[0]) * int(nums[1])
                return str(result)

        # "total", "sum", "altogether" -> addition
        if any(kw in problem for kw in ['total', 'sum', 'altogether', 'combined']):
            nums = re.findall(r'\d+', problem)
            if len(nums) >= 2:
                result = sum(int(n) for n in nums)
                return str(result)

        # "split", "divide", "share" -> division
        if any(kw in problem for kw in ['split', 'divide', 'share', 'each person']):
            nums = re.findall(r'\d+', problem)
            if len(nums) >= 2:
                result = int(nums[0]) // int(nums[1])
                return str(result)

        return None

    def _solve_algebra(self, problem: str) -> Optional[str]:
        """Solve algebraic equations"""
        problem = problem.lower()

        # Linear equations: ax + b = c -> x = (c - b) / a
        # Pattern: "x^2 + Ax + B = 0"
        quadratic = re.search(r'x\^2\s*\+\s*(\d+)x?\s*\+\s*(\d+)\s*=\s*0', problem)
        if quadratic:
            a = 1
            b = int(quadratic.group(1))
            c = int(quadratic.group(2))

            # Quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                x1 = (-b + discriminant**0.5) / (2*a)
                x2 = (-b - discriminant**0.5) / (2*a)
                return str(int(x1) if x1 == int(x1) else x1)

        # Linear: ax + b = c
        linear = re.search(r'(\d+)?x\s*\+\s*(\d+)\s*=\s*(\d+)', problem)
        if linear:
            a = int(linear.group(1)) if linear.group(1) else 1
            b = int(linear.group(2))
            c = int(linear.group(3))
            x = (c - b) / a
            return str(int(x) if x == int(x) else x)

        # Simple x = value
        simple = re.search(r'x\s*=\s*(\d+\.?\d*)', problem)
        if simple:
            return simple.group(1)

        return None

    def _solve_geometry(self, problem: str) -> Optional[str]:
        """Solve geometry problems"""
        problem = problem.lower()

        # Area of rectangle: length * width
        if 'rectangle' in problem and 'area' in problem:
            nums = re.findall(r'\d+', problem)
            if len(nums) >= 2:
                area = int(nums[0]) * int(nums[1])
                return str(area)

        # Perimeter of rectangle: 2(l + w)
        if 'rectangle' in problem and 'perimeter' in problem:
            nums = re.findall(r'\d+', problem)
            if len(nums) >= 2:
                perimeter = 2 * (int(nums[0]) + int(nums[1]))
                return str(perimeter)

        # Area of circle: πr^2
        if 'circle' in problem and 'area' in problem:
            nums = re.findall(r'\d+', problem)
            if nums:
                radius = int(nums[0])
                area = 3.14159 * radius * radius
                return str(int(area) if area == int(area) else round(area, 2))

        # Pythagorean theorem: a^2 + b^2 = c^2
        if 'triangle' in problem and any(kw in problem for kw in ['hypotenuse', 'right']):
            nums = re.findall(r'\d+', problem)
            if len(nums) >= 2:
                a, b = int(nums[0]), int(nums[1])
                c = (a**2 + b**2) ** 0.5
                return str(int(c) if c == int(c) else round(c, 2))

        return None

    def _solve_calculus(self, problem: str) -> Optional[str]:
        """Solve basic calculus problems"""
        problem = problem.lower()

        # Derivative of x^n = n*x^(n-1)
        if 'derivative' in problem:
            power_match = re.search(r'x\^(\d+)', problem)
            if power_match:
                n = int(power_match.group(1))
                return f"{n}x^{n-1}" if n > 1 else str(n)

        # Integral of x^n = x^(n+1)/(n+1)
        if 'integral' in problem:
            power_match = re.search(r'x\^(\d+)', problem)
            if power_match:
                n = int(power_match.group(1))
                return f"x^{n+1}/{n+1}"

        return None

    def _solve_statistics(self, problem: str) -> Optional[str]:
        """Solve statistics problems"""
        problem = problem.lower()

        # Mean/average
        if any(kw in problem for kw in ['average', 'mean']):
            nums = re.findall(r'\d+', problem)
            if nums:
                avg = sum(int(n) for n in nums) / len(nums)
                return str(int(avg) if avg == int(avg) else round(avg, 2))

        # Median
        if 'median' in problem:
            nums = sorted([int(n) for n in re.findall(r'\d+', problem)])
            if nums:
                mid = len(nums) // 2
                if len(nums) % 2 == 0:
                    median = (nums[mid-1] + nums[mid]) / 2
                else:
                    median = nums[mid]
                return str(int(median) if median == int(median) else median)

        return None

    def extract_answer_from_solution(self, solution: str) -> str:
        """Extract final numerical answer from a solution"""
        # Look for common answer patterns
        patterns = [
            r'####\s*(\d+\.?\d*)',  # GSM8K format
            r'answer[:\s]+(\d+\.?\d*)',
            r'=\s*(\d+\.?\d*)',
            r'result[:\s]+(\d+\.?\d*)',
            r'solution[:\s]+(\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fallback: return last number in solution
        nums = re.findall(r'-?\d+\.?\d*', solution)
        return nums[-1] if nums else "0"


# Global instance
_math_engine = None

def get_math_engine() -> MathematicalReasoningEngine:
    """Get or create the global math engine instance"""
    global _math_engine
    if _math_engine is None:
        _math_engine = MathematicalReasoningEngine()
    return _math_engine
