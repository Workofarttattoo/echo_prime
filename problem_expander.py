#!/usr/bin/env python3
"""
ECH0-PRIME Problem Expansion System
Generates comprehensive problem sets to address coverage gaps
"""

import random
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ProblemDomain(Enum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    STATISTICS = "statistics"
    LOGIC = "logic"
    NUMBER_THEORY = "number_theory"
    LINEAR_ALGEBRA = "linear_algebra"


@dataclass
class ProblemTemplate:
    domain: ProblemDomain
    difficulty: str  # 'easy', 'medium', 'hard', 'expert'
    template: str
    variables: Dict[str, Tuple[float, float]]  # variable_name -> (min_val, max_val)
    constraints: List[str]
    solution_template: str


class ProblemExpander:
    """
    Generates comprehensive problem sets to address coverage gaps
    """

    def __init__(self):
        self.templates = self._load_problem_templates()
        self.generated_problems = {}
        self.coverage_tracker = {}

    def _load_problem_templates(self) -> Dict[ProblemDomain, List[ProblemTemplate]]:
        """Load problem templates for different domains"""
        return {
            ProblemDomain.ALGEBRA: [
                ProblemTemplate(
                    domain=ProblemDomain.ALGEBRA,
                    difficulty="easy",
                    template="Solve for x: {a}*x + {b} = {c}",
                    variables={"a": (1, 10), "b": (-10, 10), "c": (-20, 20)},
                    constraints=["a != 0"],
                    solution_template="x = ({c} - {b}) / {a}"
                ),
                ProblemTemplate(
                    domain=ProblemDomain.ALGEBRA,
                    difficulty="medium",
                    template="Solve the quadratic equation: {a}*x^2 + {b}*x + {c} = 0",
                    variables={"a": (1, 5), "b": (-10, 10), "c": (-10, 10)},
                    constraints=["a != 0"],
                    solution_template="x = [-{b} ± sqrt({b}^2 - 4*{a}*{c})] / (2*{a})"
                ),
                ProblemTemplate(
                    domain=ProblemDomain.ALGEBRA,
                    difficulty="hard",
                    template="Solve the system: {a}*x + {b}*y = {c}, {d}*x + {e}*y = {f}",
                    variables={"a": (1, 5), "b": (-5, 5), "c": (-10, 10),
                             "d": (1, 5), "e": (-5, 5), "f": (-10, 10)},
                    constraints=["abs(a*e - b*d) > 0.1"],  # Non-singular matrix
                    solution_template="Using Cramer's rule or elimination"
                )
            ],
            ProblemDomain.CALCULUS: [
                ProblemTemplate(
                    domain=ProblemDomain.CALCULUS,
                    difficulty="easy",
                    template="Find the derivative of f(x) = {a}*x^{n}",
                    variables={"a": (1, 5), "n": (1, 5)},
                    constraints=["n != 0"],
                    solution_template="f'(x) = {a}*{n}*x^{n_minus_1}"
                ),
                ProblemTemplate(
                    domain=ProblemDomain.CALCULUS,
                    difficulty="medium",
                    template="Evaluate the integral: ∫{a}*x^{n} dx from {low} to {high}",
                    variables={"a": (1, 5), "n": (-2, 5), "low": (-5, 0), "high": (1, 5)},
                    constraints=["n != -1", "low < high"],
                    solution_template="[{a}*x^{n_plus_1}/({n_plus_1})] from {low} to {high}"
                )
            ],
            ProblemDomain.GEOMETRY: [
                ProblemTemplate(
                    domain=ProblemDomain.GEOMETRY,
                    difficulty="easy",
                    template="Find the area of a circle with radius {r}",
                    variables={"r": (1, 10)},
                    constraints=["r > 0"],
                    solution_template="Area = π*{r}^2"
                ),
                ProblemTemplate(
                    domain=ProblemDomain.GEOMETRY,
                    difficulty="medium",
                    template="Find the volume of a sphere with radius {r}",
                    variables={"r": (1, 10)},
                    constraints=["r > 0"],
                    solution_template="Volume = (4/3)*π*{r}^3"
                )
            ],
            ProblemDomain.TRIGONOMETRY: [
                ProblemTemplate(
                    domain=ProblemDomain.TRIGONOMETRY,
                    difficulty="easy",
                    template="Find sin({angle}°)",
                    variables={"angle": (0, 360)},
                    constraints=[],
                    solution_template="Use calculator or exact values"
                ),
                ProblemTemplate(
                    domain=ProblemDomain.TRIGONOMETRY,
                    difficulty="medium",
                    template="Solve: sin(x) = {val} for x in [0, 2π]",
                    variables={"val": (-1, 1)},
                    constraints=[],
                    solution_template="x = arcsin({val}) + 2πk, π - arcsin({val}) + 2πk"
                )
            ],
            ProblemDomain.STATISTICS: [
                ProblemTemplate(
                    domain=ProblemDomain.STATISTICS,
                    difficulty="easy",
                    template="Find the mean of: {numbers}",
                    variables={"numbers": ([1, 10], [5, 20])},
                    constraints=[],
                    solution_template="Mean = sum/n = {sum}/{n}"
                ),
                ProblemTemplate(
                    domain=ProblemDomain.STATISTICS,
                    difficulty="medium",
                    template="Find the standard deviation of: {numbers}",
                    variables={"numbers": ([1, 10], [10, 30])},
                    constraints=[],
                    solution_template="Use formula: sqrt[Σ(xi-mean)²/n]"
                )
            ],
            ProblemDomain.LOGIC: [
                ProblemTemplate(
                    domain=ProblemDomain.LOGIC,
                    difficulty="easy",
                    template="Determine if this argument is valid: If P then Q. P is true. Therefore Q.",
                    variables={},
                    constraints=[],
                    solution_template="Valid (modus ponens)"
                ),
                ProblemTemplate(
                    domain=ProblemDomain.LOGIC,
                    difficulty="medium",
                    template="Prove by contradiction: Assume ¬P, derive contradiction, therefore P",
                    variables={},
                    constraints=[],
                    solution_template="Proof by contradiction structure"
                )
            ]
        }

    def generate_problem_set(self, domain: str = 'all', count: int = 100,
                           difficulty_distribution: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Generate a comprehensive problem set

        Args:
            domain: Problem domain ('all' for all domains)
            count: Number of problems to generate
            difficulty_distribution: Distribution of difficulties (optional)

        Returns:
            List of generated problems
        """
        if difficulty_distribution is None:
            difficulty_distribution = {'easy': 0.4, 'medium': 0.4, 'hard': 0.15, 'expert': 0.05}

        problems = []

        if domain == 'all':
            domains = list(ProblemDomain)
            problems_per_domain = count // len(domains)
            remainder = count % len(domains)

            for i, domain_enum in enumerate(domains):
                domain_count = problems_per_domain + (1 if i < remainder else 0)
                problems.extend(self._generate_domain_problems(domain_enum, domain_count, difficulty_distribution))
        else:
            try:
                domain_enum = ProblemDomain(domain)
                problems = self._generate_domain_problems(domain_enum, count, difficulty_distribution)
            except ValueError:
                raise ValueError(f"Unknown domain: {domain}")

        # Update coverage tracking
        self._update_coverage_tracking(problems)

        return problems

    def _generate_domain_problems(self, domain: ProblemDomain, count: int,
                                difficulty_distribution: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate problems for a specific domain"""
        domain_templates = self.templates.get(domain, [])
        if not domain_templates:
            return []

        problems = []

        # Calculate problems per difficulty
        difficulties = list(difficulty_distribution.keys())
        counts_per_difficulty = {}
        remaining = count

        for i, difficulty in enumerate(difficulties[:-1]):
            counts_per_difficulty[difficulty] = int(count * difficulty_distribution[difficulty])
            remaining -= counts_per_difficulty[difficulty]

        counts_per_difficulty[difficulties[-1]] = remaining

        # Generate problems
        for difficulty, target_count in counts_per_difficulty.items():
            difficulty_templates = [t for t in domain_templates if t.difficulty == difficulty]

            if not difficulty_templates:
                # Fallback to any available templates
                difficulty_templates = domain_templates

            generated = 0
            attempts = 0
            max_attempts = target_count * 10  # Prevent infinite loops

            while generated < target_count and attempts < max_attempts:
                template = random.choice(difficulty_templates)
                problem_data = self._instantiate_template(template)

                if problem_data:
                    problems.append(problem_data)
                    generated += 1

                attempts += 1

        return problems

    def _instantiate_template(self, template: ProblemTemplate) -> Optional[Dict[str, Any]]:
        """Instantiate a problem template with random values"""
        variables = {}

        # Generate random values for variables
        for var_name, (min_val, max_val) in template.variables.items():
            if isinstance(min_val, list) and isinstance(max_val, list):
                # Special case for lists (e.g., number sequences)
                length = random.randint(min_val[0], min_val[1])
                variables[var_name] = [random.randint(max_val[0], max_val[1]) for _ in range(length)]
            else:
                # Regular numeric range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    variables[var_name] = random.randint(min_val, max_val)
                else:
                    variables[var_name] = random.uniform(min_val, max_val)

        # Check constraints
        if not self._check_constraints(template.constraints, variables):
            return None

        # Format the problem text
        try:
            problem_text = template.template.format(**variables)

            # Generate solution
            solution_text = template.solution_template.format(**variables)

            # Handle special formatting
            solution_text = self._format_solution(solution_text, variables)

            return {
                'domain': template.domain.value,
                'difficulty': template.difficulty,
                'problem': problem_text,
                'solution': solution_text,
                'variables': variables,
                'template': template.template,
                'metadata': {
                    'complexity': self._estimate_complexity(template, variables),
                    'required_skills': self._identify_required_skills(template),
                    'educational_level': self._estimate_educational_level(template.difficulty)
                }
            }

        except (KeyError, ValueError) as e:
            print(f"Template instantiation error: {e}")
            return None

    def _check_constraints(self, constraints: List[str], variables: Dict[str, Any]) -> bool:
        """Check if variable values satisfy constraints"""
        for constraint in constraints:
            try:
                # Replace variable names with values
                test_constraint = constraint
                for var_name, var_value in variables.items():
                    test_constraint = test_constraint.replace(var_name, str(var_value))

                # Evaluate the constraint
                if not eval(test_constraint, {"__builtins__": {}}, {"abs": abs}):
                    return False

            except (NameError, TypeError, SyntaxError):
                # If constraint can't be evaluated, assume it's satisfied
                continue

        return True

    def _format_solution(self, solution_template: str, variables: Dict[str, Any]) -> str:
        """Format solution with proper mathematical notation"""
        solution = solution_template

        # Handle special cases
        for var_name, var_value in variables.items():
            if var_name + '_minus_1' in solution:
                solution = solution.replace(var_name + '_minus_1', str(var_value - 1))
            if var_name + '_plus_1' in solution:
                solution = solution.replace(var_name + '_plus_1', str(var_value + 1))

        # Format fractions
        solution = solution.replace('(', '').replace(')', '')

        return solution

    def _estimate_complexity(self, template: ProblemTemplate, variables: Dict[str, Any]) -> float:
        """Estimate problem complexity score"""
        complexity = 0.0

        # Base complexity by difficulty
        difficulty_scores = {'easy': 1.0, 'medium': 2.0, 'hard': 3.0, 'expert': 4.0}
        complexity += difficulty_scores.get(template.difficulty, 2.0)

        # Add complexity for mathematical operations
        problem_text = template.template
        if '^' in problem_text:
            complexity += 0.5
        if 'sqrt' in problem_text or '∫' in problem_text:
            complexity += 1.0
        if 'system' in problem_text.lower():
            complexity += 1.5

        # Add complexity for variable ranges
        for var_range in template.variables.values():
            if isinstance(var_range[0], (int, float)):
                range_size = abs(var_range[1] - var_range[0])
                complexity += min(1.0, range_size / 20.0)

        return complexity

    def _identify_required_skills(self, template: ProblemTemplate) -> List[str]:
        """Identify skills required to solve the problem"""
        skills = []

        problem_text = template.template.lower()

        if 'solve' in problem_text and 'x' in problem_text:
            skills.append('equation_solving')
        if '^2' in problem_text:
            skills.append('quadratic_equations')
        if 'system' in problem_text:
            skills.append('system_of_equations')
        if 'derivative' in problem_text or 'd/dx' in problem_text:
            skills.append('differentiation')
        if 'integral' in problem_text or '∫' in problem_text:
            skills.append('integration')
        if 'mean' in problem_text or 'average' in problem_text:
            skills.append('basic_statistics')
        if 'sin' in problem_text or 'cos' in problem_text:
            skills.append('trigonometry')
        if 'area' in problem_text or 'volume' in problem_text:
            skills.append('geometry')

        return skills

    def _estimate_educational_level(self, difficulty: str) -> str:
        """Estimate educational level for the problem"""
        levels = {
            'easy': 'middle_school',
            'medium': 'high_school',
            'hard': 'undergraduate',
            'expert': 'graduate'
        }
        return levels.get(difficulty, 'high_school')

    def _update_coverage_tracking(self, problems: List[Dict[str, Any]]):
        """Update coverage tracking statistics"""
        for problem in problems:
            domain = problem['domain']
            difficulty = problem['difficulty']

            if domain not in self.coverage_tracker:
                self.coverage_tracker[domain] = {}

            if difficulty not in self.coverage_tracker[domain]:
                self.coverage_tracker[domain][difficulty] = 0

            self.coverage_tracker[domain][difficulty] += 1

    def get_coverage_report(self) -> Dict[str, Any]:
        """Get coverage report for generated problems"""
        return {
            'total_problems': sum(sum(domain.values()) for domain in self.coverage_tracker.values()),
            'domain_breakdown': self.coverage_tracker,
            'coverage_gaps': self._identify_coverage_gaps(),
            'recommendations': self._generate_coverage_recommendations()
        }

    def _identify_coverage_gaps(self) -> List[str]:
        """Identify gaps in problem coverage"""
        gaps = []

        # Check for domains with low coverage
        for domain in ProblemDomain:
            domain_str = domain.value
            if domain_str not in self.coverage_tracker:
                gaps.append(f"No problems generated for {domain_str}")
                continue

            domain_problems = sum(self.coverage_tracker[domain_str].values())
            if domain_problems < 10:
                gaps.append(f"Low coverage for {domain_str}: only {domain_problems} problems")

            # Check difficulty distribution
            difficulties = self.coverage_tracker[domain_str]
            if 'expert' not in difficulties or difficulties['expert'] < 3:
                gaps.append(f"Insufficient expert-level problems for {domain_str}")

        return gaps

    def _generate_coverage_recommendations(self) -> List[str]:
        """Generate recommendations to improve coverage"""
        recommendations = []

        coverage_report = self.get_coverage_report()
        gaps = coverage_report['coverage_gaps']

        if gaps:
            recommendations.append(f"Address {len(gaps)} coverage gaps")
            recommendations.extend(gaps[:3])  # Top 3 gaps

        # General recommendations
        recommendations.extend([
            "Generate more expert-level problems for advanced domains",
            "Ensure balanced difficulty distribution across domains",
            "Include interdisciplinary problems combining multiple domains",
            "Add problems requiring multiple solution methods"
        ])

        return recommendations


class BenchmarkCoordinator:
    """
    Coordinates comprehensive benchmarking across expanded problem sets
    """

    def __init__(self):
        self.expander = ProblemExpander()
        self.benchmark_results = {}
        self.performance_history = []

    def run_comprehensive_benchmark(self, solver_function, problem_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across multiple domains and difficulties

        Args:
            solver_function: Function that takes a problem string and returns a solution
            problem_counts: Optional dict specifying number of problems per domain

        Returns:
            Benchmark results
        """
        if problem_counts is None:
            problem_counts = {
                'algebra': 25,
                'calculus': 15,
                'geometry': 20,
                'trigonometry': 15,
                'statistics': 10,
                'logic': 15
            }

        print("🔬 Running Comprehensive Mathematical Benchmark")
        print("=" * 60)

        # Generate test problems
        test_problems = []
        for domain, count in problem_counts.items():
            domain_problems = self.expander.generate_problem_set(domain=domain, count=count)
            test_problems.extend(domain_problems)

        print(f"Generated {len(test_problems)} test problems")

        # Run benchmark
        results = self._run_benchmark(solver_function, test_problems)

        # Analyze results
        analysis = self._analyze_benchmark_results(results)

        # Store results
        self.benchmark_results = {
            'timestamp': self._current_timestamp(),
            'total_problems_tested': len(test_problems),
            'results': results,
            'analysis': analysis,
            'recommendations': self._generate_benchmark_recommendations(analysis)
        }

        self.performance_history.append(self.benchmark_results)

        return self.benchmark_results

    def _run_benchmark(self, solver_function, test_problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the actual benchmark"""
        results = []

        for i, problem_data in enumerate(test_problems):
            print(f"Testing problem {i+1}/{len(test_problems)}: {problem_data['domain']} - {problem_data['difficulty']}")

            try:
                # Time the solution
                import time
                start_time = time.time()

                solution = solver_function(problem_data['problem'])

                end_time = time.time()
                solve_time = end_time - start_time

                # Evaluate correctness (simplified)
                is_correct = self._evaluate_solution_correctness(solution, problem_data['solution'])

                result = {
                    'problem_id': i,
                    'domain': problem_data['domain'],
                    'difficulty': problem_data['difficulty'],
                    'problem': problem_data['problem'],
                    'expected_solution': problem_data['solution'],
                    'actual_solution': solution,
                    'is_correct': is_correct,
                    'solve_time': solve_time,
                    'complexity': problem_data['metadata']['complexity'],
                    'required_skills': problem_data['metadata']['required_skills']
                }

                results.append(result)

            except Exception as e:
                print(f"Error solving problem {i+1}: {e}")
                results.append({
                    'problem_id': i,
                    'domain': problem_data['domain'],
                    'difficulty': problem_data['difficulty'],
                    'error': str(e),
                    'is_correct': False,
                    'solve_time': 0
                })

        return results

    def _evaluate_solution_correctness(self, actual: str, expected: str) -> bool:
        """Evaluate if solution is correct (simplified)"""
        # This is a very simplified correctness check
        # In practice, would need symbolic math evaluation

        actual_clean = self._normalize_solution(actual)
        expected_clean = self._normalize_solution(expected)

        # Exact match
        if actual_clean == expected_clean:
            return True

        # Check for numerical equivalence
        try:
            actual_val = self._extract_numerical_value(actual_clean)
            expected_val = self._extract_numerical_value(expected_clean)

            if actual_val is not None and expected_val is not None:
                return abs(actual_val - expected_val) < 1e-6

        except:
            pass

        # Check for structural similarity
        return self._check_structural_similarity(actual_clean, expected_clean)

    def _normalize_solution(self, solution: str) -> str:
        """Normalize solution string for comparison"""
        # Remove extra whitespace, normalize case, etc.
        normalized = solution.lower().strip()
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        return normalized

    def _extract_numerical_value(self, solution: str) -> Optional[float]:
        """Extract numerical value from solution string"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', solution)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        return None

    def _check_structural_similarity(self, sol1: str, sol2: str) -> bool:
        """Check structural similarity between solutions"""
        # Very basic structural check
        return len(set(sol1.split()) & set(sol2.split())) > 0

    def _analyze_benchmark_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results"""
        total_problems = len(results)
        correct_solutions = sum(1 for r in results if r.get('is_correct', False))
        overall_accuracy = correct_solutions / total_problems if total_problems > 0 else 0

        # Domain performance
        domain_results = {}
        for result in results:
            domain = result['domain']
            if domain not in domain_results:
                domain_results[domain] = {'total': 0, 'correct': 0, 'avg_time': 0}

            domain_results[domain]['total'] += 1
            if result.get('is_correct', False):
                domain_results[domain]['correct'] += 1
            domain_results[domain]['avg_time'] += result.get('solve_time', 0)

        # Calculate averages
        for domain_data in domain_results.values():
            if domain_data['total'] > 0:
                domain_data['accuracy'] = domain_data['correct'] / domain_data['total']
                domain_data['avg_time'] = domain_data['avg_time'] / domain_data['total']

        # Difficulty performance
        difficulty_results = {}
        for result in results:
            difficulty = result['difficulty']
            if difficulty not in difficulty_results:
                difficulty_results[difficulty] = {'total': 0, 'correct': 0}

            difficulty_results[difficulty]['total'] += 1
            if result.get('is_correct', False):
                difficulty_results[difficulty]['correct'] += 1

        for diff_data in difficulty_results.values():
            if diff_data['total'] > 0:
                diff_data['accuracy'] = diff_data['correct'] / diff_data['total']

        # Performance metrics
        solve_times = [r.get('solve_time', 0) for r in results if 'solve_time' in r]
        avg_solve_time = np.mean(solve_times) if solve_times else 0
        median_solve_time = np.median(solve_times) if solve_times else 0

        return {
            'overall_accuracy': overall_accuracy,
            'total_problems': total_problems,
            'correct_solutions': correct_solutions,
            'domain_results': domain_results,
            'difficulty_results': difficulty_results,
            'performance_metrics': {
                'avg_solve_time': avg_solve_time,
                'median_solve_time': median_solve_time,
                'min_solve_time': min(solve_times) if solve_times else 0,
                'max_solve_time': max(solve_times) if solve_times else 0
            }
        }

    def _generate_benchmark_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []

        overall_accuracy = analysis['overall_accuracy']

        if overall_accuracy < 0.5:
            recommendations.append("Overall accuracy is low - focus on fundamental problem-solving skills")
        elif overall_accuracy < 0.7:
            recommendations.append("Moderate accuracy - improve consistency across problem types")
        else:
            recommendations.append("Good overall accuracy - focus on advanced problem types")

        # Domain-specific recommendations
        domain_results = analysis['domain_results']
        for domain, results in domain_results.items():
            accuracy = results.get('accuracy', 0)
            if accuracy < 0.5:
                recommendations.append(f"Improve performance in {domain} (accuracy: {accuracy:.1%})")
            elif accuracy < 0.7:
                recommendations.append(f"Strengthen {domain} skills (accuracy: {accuracy:.1%})")

        # Performance recommendations
        avg_time = analysis['performance_metrics']['avg_solve_time']
        if avg_time > 30:
            recommendations.append("Improve solution speed - consider more efficient algorithms")
        elif avg_time < 1:
            recommendations.append("Very fast solving - consider adding verification steps")

        return recommendations

    def _current_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self.performance_history) < 2:
            return {'message': 'Need at least 2 benchmark runs for trend analysis'}

        recent_results = self.performance_history[-2:]

        trend_analysis = {
            'accuracy_trend': self._calculate_trend([r['analysis']['overall_accuracy'] for r in recent_results]),
            'speed_trend': self._calculate_trend([
                r['analysis']['performance_metrics']['avg_solve_time'] for r in recent_results
            ]),
            'domain_improvements': {},
            'recommendations': []
        }

        # Analyze domain improvements
        domains = set()
        for result in recent_results:
            domains.update(result['analysis']['domain_results'].keys())

        for domain in domains:
            accuracies = []
            for result in recent_results:
                domain_data = result['analysis']['domain_results'].get(domain, {})
                accuracies.append(domain_data.get('accuracy', 0))

            if len(accuracies) >= 2:
                trend_analysis['domain_improvements'][domain] = self._calculate_trend(accuracies)

        # Generate trend-based recommendations
        accuracy_trend = trend_analysis['accuracy_trend']
        if accuracy_trend > 0.1:
            trend_analysis['recommendations'].append("Performance is improving - continue current approach")
        elif accuracy_trend < -0.1:
            trend_analysis['recommendations'].append("Performance is declining - review recent changes")
        else:
            trend_analysis['recommendations'].append("Performance is stable - focus on specific weak areas")

        return trend_analysis

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from list of values"""
        if len(values) < 2:
            return 0.0

        # Simple linear trend
        n = len(values)
        x = list(range(n))
        slope = np.polyfit(x, values, 1)[0]

        return slope
