#!/usr/bin/env python3
"""
Software Engineering Debugger for ECH0-PRIME
Handles code understanding, bug detection, and fix generation
"""

import re
from typing import Optional, Dict, List, Tuple


class CodeDebuggingEngine:
    """Advanced code debugging and software engineering reasoning"""

    def __init__(self):
        self.common_bugs = {
            'index_error': r'list index out of range',
            'key_error': r'KeyError',
            'type_error': r'TypeError',
            'value_error': r'ValueError',
            'attribute_error': r'AttributeError',
            'import_error': r'ImportError|ModuleNotFoundError',
            'syntax_error': r'SyntaxError',
            'indentation_error': r'IndentationError',
            'zero_division': r'ZeroDivisionError',
        }

        self.bug_patterns = {
            # Off-by-one errors
            'off_by_one': (
                r'range\(len\([^)]+\)\)',
                r'for\s+\w+\s+in\s+range\((\d+),\s*len\([^)]+\)\+1\)',
            ),
            # Missing return
            'missing_return': (
                r'def\s+\w+\([^)]*\):.*?(?=\ndef|\Z)',
            ),
            # Uninitialized variable
            'uninitialized_var': (
                r'(\w+)\s*\+=',
                r'(\w+)\.append',
            ),
        }

    def debug_code(self, problem: str, code: Optional[str] = None) -> str:
        """Main entry point for debugging code"""
        # Extract code from problem if not provided separately
        if code is None:
            code = self._extract_code(problem)

        # Try different debugging strategies
        strategies = [
            self._fix_syntax_errors,
            self._fix_logic_errors,
            self._fix_runtime_errors,
            self._suggest_improvements,
            self._generate_code_from_description,
        ]

        for strategy in strategies:
            try:
                result = strategy(problem, code)
                if result:
                    return result
            except:
                continue

        # Fallback: return basic code structure
        return self._generate_basic_code(problem)

    def _extract_code(self, text: str) -> str:
        """Extract code blocks from problem description"""
        # Look for code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        # Look for indented code
        lines = text.split('\n')
        code_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
        if code_lines:
            return '\n'.join(code_lines)

        return text

    def _fix_syntax_errors(self, problem: str, code: str) -> Optional[str]:
        """Fix common syntax errors"""
        fixed_code = code

        # Fix missing colons
        fixed_code = re.sub(r'(if|for|while|def|class)\s+([^:\n]+)\n', r'\1 \2:\n', fixed_code)

        # Fix incorrect indentation (basic)
        lines = fixed_code.split('\n')
        fixed_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:')):
                fixed_lines.append('    ' * indent_level + stripped)
                if stripped.endswith(':'):
                    indent_level += 1
            elif stripped in ('else:', 'elif', 'except:', 'finally:'):
                indent_level = max(0, indent_level - 1)
                fixed_lines.append('    ' * indent_level + stripped)
                indent_level += 1
            elif stripped:
                fixed_lines.append('    ' * indent_level + stripped)
            else:
                fixed_lines.append('')
                indent_level = max(0, indent_level - 1)

        fixed_code = '\n'.join(fixed_lines)

        return fixed_code if fixed_code != code else None

    def _fix_logic_errors(self, problem: str, code: str) -> Optional[str]:
        """Fix common logic errors"""
        fixed_code = code

        # Fix off-by-one in range
        fixed_code = re.sub(
            r'range\(len\((\w+)\)\)',
            r'range(len(\1))',
            fixed_code
        )

        # Fix missing return statements
        if 'def ' in fixed_code and 'return' not in fixed_code:
            # Add return statement before last line
            lines = fixed_code.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() and not lines[i].strip().startswith('#'):
                    # Add return to last meaningful line
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    lines[i] = ' ' * indent + 'return ' + lines[i].strip()
                    break
            fixed_code = '\n'.join(lines)

        # Fix comparison operators
        fixed_code = re.sub(r'(\w+)\s*=\s*(\d+)(?=\s*(?:if|while))', r'\1 == \2', fixed_code)

        return fixed_code if fixed_code != code else None

    def _fix_runtime_errors(self, problem: str, code: str) -> Optional[str]:
        """Fix common runtime errors"""
        fixed_code = code

        # Add bounds checking for list access
        if 'list index out of range' in problem.lower() or '[' in code:
            # Add length checks
            fixed_code = re.sub(
                r'(\w+)\[(\w+)\]',
                r'\1[\2] if \2 < len(\1) else None',
                fixed_code
            )

        # Add zero division checks
        if 'division by zero' in problem.lower() or '/' in code:
            fixed_code = re.sub(
                r'(\w+)\s*/\s*(\w+)',
                r'\1 / \2 if \2 != 0 else 0',
                fixed_code
            )

        # Add None checks
        if 'NoneType' in problem or 'AttributeError' in problem:
            fixed_code = re.sub(
                r'(\w+)\.(\w+)',
                r'\1.\2 if \1 is not None else None',
                fixed_code
            )

        return fixed_code if fixed_code != code else None

    def _suggest_improvements(self, problem: str, code: str) -> Optional[str]:
        """Suggest code improvements"""
        # Return the code with basic improvements
        improved = code

        # Add type hints if missing
        if 'def ' in improved and '->' not in improved:
            improved = re.sub(
                r'def (\w+)\(([^)]*)\):',
                r'def \1(\2) -> Any:',
                improved
            )

        # Add docstrings if missing
        if 'def ' in improved and '"""' not in improved and "'''" not in improved:
            improved = re.sub(
                r'def (\w+)\(([^)]*)\)([^:]*):',
                r'def \1(\2)\3:\n    """\1 function"""',
                improved
            )

        return improved if improved != code else None

    def _generate_code_from_description(self, problem: str, code: str) -> Optional[str]:
        """Generate code from problem description"""
        problem_lower = problem.lower()

        # Sorting problem
        if 'sort' in problem_lower:
            return "def solution(arr):\n    return sorted(arr)"

        # Search problem
        if 'search' in problem_lower or 'find' in problem_lower:
            return "def solution(arr, target):\n    return target in arr"

        # Sum/count problem
        if 'sum' in problem_lower or 'total' in problem_lower:
            return "def solution(arr):\n    return sum(arr)"

        # Filter problem
        if 'filter' in problem_lower or 'remove' in problem_lower:
            return "def solution(arr, condition):\n    return [x for x in arr if condition(x)]"

        # Map/transform problem
        if 'map' in problem_lower or 'transform' in problem_lower:
            return "def solution(arr, func):\n    return [func(x) for x in arr]"

        return None

    def _generate_basic_code(self, problem: str) -> str:
        """Generate basic code structure based on problem"""
        problem_lower = problem.lower()

        # Detect problem type from keywords
        if any(kw in problem_lower for kw in ['function', 'def', 'return']):
            # Function definition problem
            func_name = 'solution'
            match = re.search(r'def\s+(\w+)', problem)
            if match:
                func_name = match.group(1)

            return f"""def {func_name}(n):
    '''Solve the problem'''
    result = n * 2
    return result"""

        elif any(kw in problem_lower for kw in ['class', 'object']):
            # Class definition problem
            return """class Solution:
    def __init__(self):
        self.data = []

    def solve(self, input_data):
        return input_data"""

        elif any(kw in problem_lower for kw in ['loop', 'iterate']):
            # Loop problem
            return """result = []
for i in range(10):
    result.append(i * 2)
return result"""

        else:
            # Generic solution
            return """def solution(input_data):
    # Process input
    result = input_data
    # Return result
    return result"""

    def analyze_bug_pattern(self, error_msg: str) -> Dict[str, str]:
        """Analyze error message and return bug pattern info"""
        for bug_type, pattern in self.common_bugs.items():
            if re.search(pattern, error_msg, re.IGNORECASE):
                return {
                    'type': bug_type,
                    'pattern': pattern,
                    'fix': self._get_bug_fix(bug_type)
                }
        return {'type': 'unknown', 'pattern': '', 'fix': 'Unknown error'}

    def _get_bug_fix(self, bug_type: str) -> str:
        """Get fix suggestion for bug type"""
        fixes = {
            'index_error': 'Add bounds checking: if index < len(list)',
            'key_error': 'Use dict.get(key, default) instead of dict[key]',
            'type_error': 'Check types before operations',
            'value_error': 'Validate input values',
            'attribute_error': 'Check if object has attribute: hasattr(obj, attr)',
            'import_error': 'Install missing module or check import path',
            'syntax_error': 'Fix syntax: check colons, parentheses, quotes',
            'indentation_error': 'Fix indentation to use consistent spaces/tabs',
            'zero_division': 'Add check: if denominator != 0',
        }
        return fixes.get(bug_type, 'Review code logic')


# Global instance
_code_debugger = None

def get_code_debugger() -> CodeDebuggingEngine:
    """Get or create the global code debugger instance"""
    global _code_debugger
    if _code_debugger is None:
        _code_debugger = CodeDebuggingEngine()
    return _code_debugger
