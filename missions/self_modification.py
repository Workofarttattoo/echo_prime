"""
Autonomous self-modification and code improvement system.
True AGI self-improvement with code analysis, optimization, and safe deployment.
"""
import ast
import os
import subprocess
import time
import json
import inspect
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Tuple, Any, Callable
import hashlib
import shutil
import sys
from pathlib import Path
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
import psutil
import memory_profiler


class CodeAnalyzer:
    """
    Analyzes code for performance, complexity, and improvement opportunities.
    """
    def __init__(self):
        self.analysis_cache = {}

    def analyze_code(self, code: str, filename: str = None) -> Dict[str, Any]:
        """
        Comprehensive code analysis including:
        - Complexity metrics
        - Performance profiling
        - Code quality metrics
        - Improvement suggestions
        """
        try:
            tree = ast.parse(code)

            analysis = {
                'complexity': self._analyze_complexity(tree),
                'performance': self._analyze_performance(code),
                'quality': self._analyze_quality(tree),
                'suggestions': self._generate_improvements(tree, code),
                'timestamp': time.time(),
                'code_hash': hashlib.md5(code.encode()).hexdigest()
            }

            if filename:
                self.analysis_cache[filename] = analysis

            return analysis

        except SyntaxError as e:
            return {'error': f'Syntax error: {e}', 'line': e.lineno}

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.functions = []
                self.classes = []

            def visit_FunctionDef(self, node):
                self.functions.append({
                    'name': node.name,
                    'args': len(node.args.args),
                    'lines': node.end_lineno - node.lineno if node.end_lineno else 0
                })
                self.complexity += 1
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                self.classes.append(node.name)
                self.complexity += 2  # Classes are more complex
                self.generic_visit(node)

            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        return {
            'cyclomatic_complexity': visitor.complexity,
            'functions': visitor.functions,
            'classes': visitor.classes,
            'total_lines': len(tree.body) if hasattr(tree, 'body') else 0
        }

    def _analyze_performance(self, code: str) -> Dict[str, Any]:
        """Profile code performance"""
        try:
            # Create temporary file for profiling
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Profile the code execution
            pr = cProfile.Profile()
            pr.enable()

            # Execute in isolated environment
            exec_globals = {'__name__': '__main__'}
            exec(code, exec_globals)

            pr.disable()

            # Get profiling stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions

            return {
                'profile_stats': s.getvalue(),
                'execution_time': ps.total_tt,
                'function_calls': ps.total_calls
            }

        except Exception as e:
            return {'error': f'Performance analysis failed: {e}'}
        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file)

    def _analyze_quality(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        class QualityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
                self.metrics = {
                    'docstrings': 0,
                    'type_hints': 0,
                    'constants': 0,
                    'magic_numbers': 0
                }

            def visit_FunctionDef(self, node):
                # Check for docstring
                if not ast.get_docstring(node):
                    self.issues.append(f"Function '{node.name}' missing docstring")

                # Check for type hints
                has_type_hints = any(arg.annotation for arg in node.args.args)
                if has_type_hints:
                    self.metrics['type_hints'] += 1

                self.metrics['docstrings'] += 1 if ast.get_docstring(node) else 0

            def visit_Num(self, node):
                # Detect magic numbers (simple heuristic)
                if isinstance(node.n, (int, float)) and node.n not in [0, 1, -1]:
                    self.issues.append(f"Magic number: {node.n}")

        visitor = QualityVisitor()
        visitor.visit(tree)

        return {
            'issues': visitor.issues,
            'metrics': visitor.metrics,
            'quality_score': self._calculate_quality_score(visitor.metrics, visitor.issues)
        }

    def _calculate_quality_score(self, metrics: Dict, issues: List) -> float:
        """Calculate overall code quality score"""
        base_score = 100
        penalty_per_issue = 5

        # Bonus for good practices
        base_score += metrics.get('docstrings', 0) * 2
        base_score += metrics.get('type_hints', 0) * 3

        # Penalties for issues
        base_score -= len(issues) * penalty_per_issue

        return max(0, min(100, base_score))

    def _generate_improvements(self, tree: ast.AST, code: str) -> List[Dict]:
        """Generate specific improvement suggestions"""
        suggestions = []

        # Simple improvement heuristics
        if 'print(' in code and 'import logging' not in code:
            suggestions.append({
                'type': 'logging',
                'description': 'Replace print statements with proper logging',
                'priority': 'medium'
            })

        if 'except:' in code and 'Exception' not in code:
            suggestions.append({
                'type': 'exception_handling',
                'description': 'Use specific exception types instead of bare except',
                'priority': 'high'
            })

        if len(code.split('\n')) > 100:
            suggestions.append({
                'type': 'refactoring',
                'description': 'Consider breaking large functions into smaller ones',
                'priority': 'medium'
            })

        return suggestions


class PerformanceProfiler:
    """
    Advanced performance profiling and optimization.
    """
    def __init__(self):
        self.baseline_metrics = {}

    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a function's performance"""
        # Memory profiling
        mem_usage_before = memory_profiler.memory_usage()[0]

        # Time profiling
        start_time = time.time()
        start_cpu = psutil.cpu_percent()

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            error = str(e)

        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        mem_usage_after = memory_profiler.memory_usage()[0]

        metrics = {
            'execution_time': end_time - start_time,
            'cpu_usage': (start_cpu + end_cpu) / 2,
            'memory_usage': mem_usage_after - mem_usage_before,
            'memory_peak': mem_usage_after,
            'success': success
        }

        if not success:
            metrics['error'] = error

        return metrics

    def compare_performance(self, baseline: Dict, current: Dict) -> Dict[str, Any]:
        """Compare current performance against baseline"""
        comparison = {}

        for metric in ['execution_time', 'cpu_usage', 'memory_usage']:
            if metric in baseline and metric in current:
                improvement = baseline[metric] - current[metric]
                percent_change = (improvement / baseline[metric]) * 100 if baseline[metric] != 0 else 0
                comparison[metric] = {
                    'improvement': improvement,
                    'percent_change': percent_change,
                    'better': improvement > 0
                }

        return comparison


class AutonomousImprover:
    """
    Autonomous code improvement and optimization system.
    """
    def __init__(self, llm_bridge=None):
        self.llm_bridge = llm_bridge
        self.analyzer = CodeAnalyzer()
        self.profiler = PerformanceProfiler()
        self.improvement_history = []
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)

    def analyze_and_improve(self, code: str, target_file: str = None,
                          performance_target: Dict = None) -> Dict[str, Any]:
        """
        Complete autonomous improvement cycle:
        1. Analyze current code
        2. Profile performance
        3. Generate improvements
        4. Test and validate
        5. Deploy if better
        """
        print(f"🔍 Analyzing code for improvement...")

        # Step 1: Code analysis
        analysis = self.analyzer.analyze_code(code, target_file)

        # Step 2: Performance profiling (if executable)
        performance = self._profile_code_performance(code)

        # Step 3: Generate improvement suggestions
        improvements = self._generate_improvement_plan(analysis, performance, performance_target)

        # Step 4: Apply improvements
        improved_code = self._apply_improvements(code, improvements)

        # Step 5: Validate improvements
        validation = self._validate_improvements(code, improved_code, performance_target)

        result = {
            'original_analysis': analysis,
            'original_performance': performance,
            'improvements_applied': improvements,
            'improved_code': improved_code,
            'validation': validation,
            'success': validation.get('overall_improvement', False),
            'timestamp': time.time()
        }

        self.improvement_history.append(result)
        return result

    def _profile_code_performance(self, code: str) -> Dict:
        """Profile code performance safely"""
        try:
            # Try to extract main function or create test execution
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if functions:
                # Profile the first function
                func_name = functions[0].name
                test_code = f"""
import sys
sys.path.insert(0, '.')
{code}
result = {func_name}()
"""
                return self.profiler.profile_function(exec, test_code)
            else:
                # Execute whole script
                return self.profiler.profile_function(exec, code)

        except Exception as e:
            return {'error': f'Profiling failed: {e}'}

    def _generate_improvement_plan(self, analysis: Dict, performance: Dict,
                                 target: Dict = None) -> List[Dict]:
        """Generate comprehensive improvement plan"""
        improvements = []

        # Code quality improvements
        if 'quality' in analysis:
            quality_issues = analysis['quality'].get('issues', [])
            for issue in quality_issues[:3]:  # Limit to top 3 issues
                improvements.append({
                    'type': 'quality',
                    'description': f"Fix: {issue}",
                    'priority': 'medium',
                    'estimated_impact': 'maintainability'
                })

        # Performance improvements
        if 'execution_time' in performance and performance['execution_time'] > 1.0:
            improvements.append({
                'type': 'performance',
                'description': 'Optimize execution time',
                'priority': 'high',
                'estimated_impact': 'speed'
            })

        # Complexity improvements
        if 'complexity' in analysis:
            complexity = analysis['complexity'].get('cyclomatic_complexity', 0)
            if complexity > 10:
                improvements.append({
                    'type': 'complexity',
                    'description': 'Reduce cyclomatic complexity by refactoring',
                    'priority': 'medium',
                    'estimated_impact': 'maintainability'
                })

        # LLM-based improvements if available
        if self.llm_bridge:
            llm_suggestions = self._get_llm_improvements(analysis, performance)
            improvements.extend(llm_suggestions)

        return improvements

    def _get_llm_improvements(self, analysis: Dict, performance: Dict) -> List[Dict]:
        """Get improvement suggestions from LLM"""
        try:
            prompt = f"""
Analyze this code analysis and suggest specific improvements:

Analysis: {json.dumps(analysis, indent=2)}
Performance: {json.dumps(performance, indent=2)}

Provide 2-3 specific, actionable improvement suggestions.
Format as JSON list with keys: type, description, priority, estimated_impact
"""

            response = self.llm_bridge.generate(prompt)

            # Try to parse JSON response
            try:
                suggestions = json.loads(response)
                return suggestions if isinstance(suggestions, list) else []
            except:
                # Fallback: extract suggestions from text
                return []

        except Exception as e:
            print(f"LLM improvement generation failed: {e}")
            return []

    def _apply_improvements(self, code: str, improvements: List[Dict]) -> str:
        """Apply selected improvements to code"""
        improved_code = code

        for improvement in improvements:
            if improvement['type'] == 'quality':
                # Apply quality improvements
                improved_code = self._apply_quality_improvement(improved_code, improvement)
            elif improvement['type'] == 'performance':
                # Apply performance improvements
                improved_code = self._apply_performance_improvement(improved_code, improvement)

        return improved_code

    def _apply_quality_improvement(self, code: str, improvement: Dict) -> str:
        """Apply specific quality improvement"""
        # Simple heuristics for common improvements
        if 'docstring' in improvement.get('description', '').lower():
            # Add basic docstrings
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and i + 1 < len(lines):
                    if not lines[i + 1].strip().startswith('"""'):
                        func_name = line.split('def ')[1].split('(')[0]
                        lines.insert(i + 1, f'    """{func_name} function."""')
                        break
            return '\n'.join(lines)

        return code

    def _apply_performance_improvement(self, code: str, improvement: Dict) -> str:
        """Apply performance optimizations"""
        # Basic performance improvements
        improved_code = code

        # Replace common inefficient patterns
        if 'for i in range(len(' in improved_code:
            improved_code = improved_code.replace(
                'for i in range(len(list)):',
                'for i, item in enumerate(list):'
            )

        return improved_code

    def _validate_improvements(self, original_code: str, improved_code: str,
                              performance_target: Dict = None) -> Dict[str, Any]:
        """Validate that improvements actually help"""
        if original_code == improved_code:
            return {'overall_improvement': False, 'reason': 'No changes made'}

        # Profile both versions
        original_perf = self._profile_code_performance(original_code)
        improved_perf = self._profile_code_performance(improved_code)

        # Compare performance
        if 'error' not in original_perf and 'error' not in improved_perf:
            comparison = self.profiler.compare_performance(original_perf, improved_perf)

            # Check if improvements meet targets
            meets_target = True
            if performance_target:
                for metric, target_value in performance_target.items():
                    if metric in comparison and not comparison[metric]['better']:
                        meets_target = False
                        break

            return {
                'overall_improvement': meets_target,
                'performance_comparison': comparison,
                'original_metrics': original_perf,
                'improved_metrics': improved_perf
            }

        return {'overall_improvement': False, 'reason': 'Performance profiling failed'}

    def safe_deploy_improvement(self, improved_code: str, target_file: str) -> Dict[str, Any]:
        """Safely deploy code improvements with rollback capability"""
        backup_path = None

        try:
            # Create backup
            if os.path.exists(target_file):
                backup_path = self.backup_dir / f"{Path(target_file).name}.backup.{int(time.time())}"
                shutil.copy2(target_file, backup_path)

            # Validate improved code syntax
            ast.parse(improved_code)

            # Write improved code
            with open(target_file, 'w') as f:
                f.write(improved_code)

            # Test that the module can still be imported
            if target_file.endswith('.py'):
                module_name = Path(target_file).stem
                try:
                    # Remove from cache if already imported
                    if module_name in sys.modules:
                        del sys.modules[module_name]

                    __import__(module_name)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)

                    # Rollback on failure
                    if backup_path:
                        shutil.copy2(backup_path, target_file)

            else:
                success = True
                error = None

            return {
                'deployed': success,
                'backup_created': backup_path is not None,
                'backup_path': str(backup_path) if backup_path else None,
                'error': error
            }

        except Exception as e:
            # Rollback on any error
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, target_file)

            return {
                'deployed': False,
                'error': str(e),
                'backup_restored': True
            }


class SelfModificationSystem:
    """
    Complete autonomous self-modification system for AGI improvement.
    """
    def __init__(self, llm_bridge=None):
        self.llm_bridge = llm_bridge
        self.improver = AutonomousImprover(llm_bridge)
        self.modification_history = []
        self.self_improvement_goals = []

    def propose_improvement(self, current_code: str, performance_metrics: Dict,
                          improvement_description: str) -> Dict[str, Any]:
        """
        Propose autonomous improvement to the system itself.
        """
        print(f"🤖 Analyzing self-improvement opportunity: {improvement_description}")

        # Analyze current implementation
        analysis = self.improver.analyzer.analyze_code(current_code)

        # Generate improvement plan
        improvement_plan = {
            'description': improvement_description,
            'current_analysis': analysis,
            'performance_baseline': performance_metrics,
            'proposed_changes': [],
            'risk_assessment': self._assess_improvement_risk(current_code, improvement_description),
            'timestamp': time.time()
        }

        # Use LLM to generate specific improvement suggestions
        if self.llm_bridge:
            prompt = f"""
Analyze this code and suggest specific improvements for: {improvement_description}

Current code:
```python
{current_code}
```

Performance metrics: {json.dumps(performance_metrics, indent=2)}
Code analysis: {json.dumps(analysis, indent=2)}

Provide detailed improvement suggestions with code examples.
"""

            llm_response = self.llm_bridge.generate(prompt)
            improvement_plan['llm_suggestions'] = llm_response

        self.modification_history.append(improvement_plan)
        return improvement_plan

    def apply_improvement(self, file_path: str, new_code: str, description: str) -> Dict[str, Any]:
        """
        Apply and validate a self-improvement.
        """
        print(f"🔧 Applying self-improvement: {description}")

        # Analyze the improvement
        result = self.improver.analyze_and_improve(new_code, file_path)

        if result['success']:
            # Deploy the improvement
            deployment = self.improver.safe_deploy_improvement(new_code, file_path)

            if deployment['deployed']:
                print(f"✅ Self-improvement deployed successfully: {description}")
                return {
                    'success': True,
                    'description': description,
                    'validation': result['validation'],
                    'deployment': deployment
                }
            else:
                print(f"❌ Self-improvement deployment failed: {deployment.get('error')}")
                return {
                    'success': False,
                    'error': deployment.get('error'),
                    'rolled_back': deployment.get('backup_restored', False)
                }
        else:
            print(f"❌ Self-improvement validation failed")
            return {
                'success': False,
                'reason': 'Validation failed',
                'details': result['validation']
            }

    def _assess_improvement_risk(self, code: str, description: str) -> Dict[str, Any]:
        """Assess the risk level of a proposed improvement"""
        risk_factors = []

        # High-risk indicators
        if 'database' in description.lower() or 'file' in description.lower():
            risk_factors.append('data_integrity')
        if 'network' in description.lower() or 'socket' in description.lower():
            risk_factors.append('connectivity')
        if 'security' in description.lower() or 'auth' in description.lower():
            risk_factors.append('security')

        # Code complexity risk
        tree = ast.parse(code)
        complexity = len(list(ast.walk(tree)))

        risk_level = 'low'
        if len(risk_factors) > 0:
            risk_level = 'high'
        elif complexity > 50:
            risk_level = 'medium'

        return {
            'level': risk_level,
            'factors': risk_factors,
            'complexity_score': complexity,
            'requires_testing': risk_level in ['medium', 'high']
        }

    def get_self_improvement_status(self) -> Dict[str, Any]:
        """Get current status of self-improvement efforts"""
        successful_improvements = len([m for m in self.modification_history
                                     if m.get('validation', {}).get('overall_improvement', False)])

        return {
            'total_proposed': len(self.modification_history),
            'successful_improvements': successful_improvements,
            'success_rate': successful_improvements / len(self.modification_history) if self.modification_history else 0,
            'active_goals': len(self.self_improvement_goals),
            'last_improvement': self.modification_history[-1] if self.modification_history else None
        }


# Legacy compatibility
class CodeGenerator:
    """
    Legacy code generation class for backward compatibility.
    Use SelfModificationSystem for true autonomous improvement.
    """
    def __init__(self, llm_bridge=None):
        self.llm_bridge = llm_bridge
        self.generated_code_history = []
    
    def generate_code(self, description: str, context: Optional[str] = None) -> str:
        """
        Generate Python code from natural language description.
        
        Args:
            description: What the code should do
            context: Existing code context
        
        Returns:
            Generated Python code
        """
        prompt = f"""Generate Python code that: {description}

Requirements:
- Use proper error handling
- Include type hints where appropriate
- Add docstrings
- Follow PEP 8 style guide
- Do not use dangerous functions (eval, exec, subprocess, etc.)

"""
        
        if context:
            prompt += f"\nContext:\n```python\n{context}\n```\n"
        
        prompt += "\nReturn only the Python code, no explanations."
        
        if self.llm_bridge:
            try:
                response = self.llm_bridge.generate(prompt)
                # Extract code from response (might be in markdown code blocks)
                code = self._extract_code(response)
                self.generated_code_history.append({
                    "description": description,
                    "code": code,
                    "timestamp": time.time()
                })
                return code
            except Exception as e:
                print(f"Error generating code: {e}")
                return ""
        else:
            # Fallback: return template
            return f"""# Generated code for: {description}
def generated_function():
    \"\"\"{description}\"\"\"
    pass
"""
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code blocks
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        else:
            return text.strip()


class CodeValidator:
    """
    Validates generated code for safety and correctness.
    """
    def __init__(self):
        self.dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "subprocess",
            "os.system",
            "rm -rf",
            "delete *",
            "shutil.rmtree",
            "open(",
            "__builtins__",
            "globals()",
            "locals()"
        ]
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def validate_safety(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code safety.
        
        Returns:
            (is_safe, list_of_violations)
        """
        violations = []
        code_lower = code.lower()
        
        for pattern in self.dangerous_patterns:
            if pattern in code_lower:
                violations.append(f"Dangerous pattern detected: {pattern}")
        
        # Check for file operations outside safe directories
        if "open(" in code and "safe_dir" not in code:
            violations.append("File operations may not be in safe directory")
        
        return len(violations) == 0, violations
    
    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate that imports are safe.
        
        Returns:
            (is_safe, list_of_unsafe_imports)
        """
        tree = ast.parse(code)
        unsafe_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["subprocess", "os", "sys", "shutil"]:
                        unsafe_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module in ["subprocess", "os", "sys", "shutil"]:
                    unsafe_imports.append(node.module)
        
        return len(unsafe_imports) == 0, unsafe_imports
    
    def validate(self, code: str) -> Dict[str, any]:
        """
        Complete validation of code.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "syntax_valid": False,
            "safety_valid": False,
            "imports_valid": False,
            "errors": [],
            "warnings": []
        }
        
        # Syntax validation
        syntax_ok, syntax_error = self.validate_syntax(code)
        results["syntax_valid"] = syntax_ok
        if not syntax_ok:
            results["errors"].append(f"Syntax error: {syntax_error}")
        
        # Safety validation
        safety_ok, violations = self.validate_safety(code)
        results["safety_valid"] = safety_ok
        if not safety_ok:
            results["errors"].extend(violations)
        
        # Import validation
        imports_ok, unsafe_imports = self.validate_imports(code)
        results["imports_valid"] = imports_ok
        if not imports_ok:
            results["warnings"].append(f"Unsafe imports: {unsafe_imports}")
        
        results["valid"] = syntax_ok and safety_ok
        
        return results


class SafeSandbox:
    """
    Safe execution sandbox for generated code.
    """
    def __init__(self, safe_dir: str = "sandbox"):
        self.safe_dir = safe_dir
        if not os.path.exists(safe_dir):
            os.makedirs(safe_dir)
        
        # Create isolated environment
        self.isolated_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "all": all,
                "any": any,
            }
        }
    
    def execute(self, code: str, timeout: int = 5) -> Tuple[bool, any, Optional[str]]:
        """
        Execute code in safe sandbox.
        
        Returns:
            (success, result, error_message)
        """
        try:
            # Compile code
            compiled = compile(code, "<string>", "exec")
            
            # Execute in isolated environment
            exec(compiled, self.isolated_globals.copy(), {})
            
            return True, None, None
        except Exception as e:
            return False, None, str(e)
    
    def test_function(self, code: str, function_name: str, test_inputs: List[any]) -> Dict:
        """
        Test a generated function.
        
        Returns:
            Test results
        """
        results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Execute code to define function
            exec(code, self.isolated_globals.copy(), {})
            
            # Get function
            if function_name in self.isolated_globals:
                func = self.isolated_globals[function_name]
                
                # Test with inputs
                for test_input in test_inputs:
                    try:
                        result = func(test_input)
                        results["passed"] += 1
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append(str(e))
            else:
                results["errors"].append(f"Function {function_name} not found")
        except Exception as e:
            results["errors"].append(f"Execution error: {e}")
        
        return results


class VersionControl:
    """
    Simple version control for code modifications.
    """
    def __init__(self, repo_dir: str = "code_history"):
        self.repo_dir = repo_dir
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)
        
        self.history_file = os.path.join(repo_dir, "history.json")
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load version history"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save version history"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def commit(self, file_path: str, code: str, description: str) -> str:
        """
        Commit code change.
        
        Returns:
            Commit hash
        """
        # Compute hash
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:8]
        
        # Save code
        commit_dir = os.path.join(self.repo_dir, code_hash)
        os.makedirs(commit_dir, exist_ok=True)
        
        commit_file = os.path.join(commit_dir, os.path.basename(file_path))
        with open(commit_file, 'w') as f:
            f.write(code)
        
        # Record in history
        self.history.append({
            "hash": code_hash,
            "file": file_path,
            "description": description,
            "timestamp": time.time()
        })
        self._save_history()
        
        return code_hash
    
    def revert(self, commit_hash: str) -> bool:
        """
        Revert to a previous commit.
        
        Returns:
            Success status
        """
        # Find commit in history
        commit = None
        for entry in self.history:
            if entry["hash"] == commit_hash:
                commit = entry
                break
        
        if not commit:
            return False
        
        # Restore file
        commit_dir = os.path.join(self.repo_dir, commit_hash)
        commit_file = os.path.join(commit_dir, os.path.basename(commit["file"]))
        
        if os.path.exists(commit_file):
            shutil.copy(commit_file, commit["file"])
            return True
        
        return False


class SelfModificationSystem:
    """
    Complete self-modification system with code generation, validation, and version control.
    """
    def __init__(self, llm_bridge=None, safe_dir: str = "sandbox"):
        self.code_generator = CodeGenerator(llm_bridge)
        self.validator = CodeValidator()
        self.sandbox = SafeSandbox(safe_dir)
        self.version_control = VersionControl()
    
    def propose_improvement(self, current_code: str, performance_metrics: Dict,
                           description: str) -> Dict:
        """
        Propose code improvement.
        
        Returns:
            Improvement proposal with validation results
        """
        # Generate improvement description
        improvement_desc = f"""
        {description}
        
        Current performance metrics: {performance_metrics}
        """
        
        # Generate code
        new_code = self.code_generator.generate_code(improvement_desc, current_code)
        
        # Validate
        validation = self.validator.validate(new_code)
        
        # Test in sandbox
        test_results = None
        if validation["valid"]:
            test_results = self.sandbox.execute(new_code)
        
        return {
            "code": new_code,
            "validation": validation,
            "test_results": test_results,
            "proposed": validation["valid"] and (test_results[0] if test_results else False)
        }
    
    def apply_improvement(self, file_path: str, new_code: str, description: str) -> Dict:
        """
        Apply code improvement with safety checks.
        
        Returns:
            Application results
        """
        # Validate
        validation = self.validator.validate(new_code)
        
        if not validation["valid"]:
            return {
                "success": False,
                "error": "Validation failed",
                "validation": validation
            }
        
        # Backup current code
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                old_code = f.read()
            
            # Commit old version
            self.version_control.commit(file_path, old_code, f"Backup before: {description}")
        
        # Test new code
        test_results = self.sandbox.execute(new_code)
        if not test_results[0]:
            return {
                "success": False,
                "error": f"Test failed: {test_results[2]}",
                "test_results": test_results
            }
        
        # Apply improvement
        try:
            with open(file_path, 'w') as f:
                f.write(new_code)
            
            # Commit new version
            commit_hash = self.version_control.commit(file_path, new_code, description)
            
            return {
                "success": True,
                "commit_hash": commit_hash,
                "file": file_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def rollback(self, commit_hash: str) -> bool:
        """Rollback to previous version"""
        return self.version_control.revert(commit_hash)

