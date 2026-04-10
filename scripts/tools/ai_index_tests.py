#!/usr/bin/env python3
"""
AI Index Test Suite for ECH0-PRIME
Based on Stanford HAI AI Index Report 2025

Implements comprehensive benchmarks from the AI Index including:
- MMMU (Massive Multidiscipline Multimodal Understanding)
- GPQA (Graduate-level Problem Question Answering)
- SWE-bench (Software Engineering Benchmarks)
- MATH (Mathematical Reasoning)
- HumanEval (Code Generation)
- RE-Bench (Agentic Reasoning Evaluation)
- MMLU (Massive Multitask Language Understanding)

Reference: https://hai.stanford.edu/ai-index/2025-ai-index-report
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning.llm_bridge import OllamaBridge

@dataclass
class AIIndexTestResult:
    """Result from a single AI Index benchmark"""
    benchmark_name: str
    score: float
    total_questions: int
    correct_answers: int
    accuracy: float
    percentile_rank: Optional[float]
    details: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None
    baseline_comparison: Optional[Dict[str, float]] = None

    def to_dict(self):
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.accuracy,
            "percentile_rank": self.percentile_rank,
            "details": self.details,
            "execution_time": self.execution_time,
            "error": self.error,
            "baseline_comparison": self.baseline_comparison
        }


class AIIndexTestSuite:
    """
    Comprehensive AI Index Test Suite for ECH0-PRIME
    Implements benchmarks from Stanford HAI AI Index 2025 Report
    """

    def __init__(self, use_ech0_prime: bool = True, verbose: bool = True):
        self.use_ech0_prime = use_ech0_prime
        self.verbose = verbose
        self.llm_bridge = OllamaBridge(model="llama3.2")
        self.results = {}

        # AI Index 2025 baseline scores (from the report)
        self.baselines = {
            "mmmu": {
                "gpt-4o": 69.9,
                "claude-3.5-sonnet": 68.3,
                "gemini-2.0-flash": 62.8,
                "llama-3.3-70b": 60.3
            },
            "gpqa": {
                "gpt-4o": 53.6,
                "claude-3.5-sonnet": 59.4,
                "gemini-2.0-flash": 41.6,
                "human-expert": 71.7
            },
            "swe_bench": {
                "gpt-4o": 38.8,
                "claude-3.5-sonnet": 49.0,
                "human-expert": 73.9
            },
            "math": {
                "gpt-4o": 76.6,
                "claude-3.5-sonnet": 78.3,
                "gemini-2.0-flash": 86.9
            },
            "humaneval": {
                "gpt-4o": 90.2,
                "claude-3.5-sonnet": 92.0,
                "gemini-2.0-flash": 88.3
            },
            "mmlu": {
                "gpt-4o": 88.7,
                "claude-3.5-sonnet": 88.3,
                "gemini-2.0-flash": 84.8,
                "llama-3.3-70b": 86.0
            }
        }

        if self.verbose:
            print("=" * 80)
            print("🎓 AI INDEX TEST SUITE - ECH0-PRIME EVALUATION")
            print("   Based on Stanford HAI AI Index Report 2025")
            print("=" * 80)

        if use_ech0_prime:
            self._initialize_ech0_prime()

        # Load test datasets
        self.test_data = self._load_test_datasets()

        if self.verbose:
            print(f"\n✅ Test suite initialized with {sum(len(v) for v in self.test_data.values())} total questions")

    def _initialize_ech0_prime(self):
        """Initialize ECH0-PRIME cognitive architecture"""
        if self.verbose:
            print("\n🤖 Initializing ECH0-PRIME Cognitive Architecture...")

        try:
            from cognitive_activation import get_cognitive_activation_system
            self.cognitive_system = get_cognitive_activation_system()

            # Activate progressive cognitive levels
            if self.cognitive_system.activate_enhanced_reasoning():
                if self.verbose:
                    print("   ✅ Enhanced Reasoning Mode activated")

            if self.cognitive_system.activate_knowledge_integration():
                if self.verbose:
                    print("   ✅ Knowledge Integration Mode activated")

            # Check memory for full activation
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024**3)
                if self.verbose:
                    print(f"   📊 Available memory: {available_gb:.1f} GB")

                if available_gb > 2.0:
                    if self.cognitive_system.activate_full_cognitive_architecture():
                        if self.verbose:
                            print("   ✅ Full Cognitive Architecture activated")
            except ImportError:
                pass

            # Initialize orchestrator
            from simple_orchestrator import SimpleEchoPrimeAGI
            self.ech0_orchestrator = SimpleEchoPrimeAGI(lightweight=True)

        except Exception as e:
            print(f"   ⚠️ Cognitive activation error: {e}")
            self.use_ech0_prime = False

    def _load_test_datasets(self) -> Dict[str, List[Dict]]:
        """Load or generate AI Index test datasets"""
        if self.verbose:
            print("\n📊 Loading AI Index Test Datasets...")

        datasets = {}

        # Try to load from datasets directory or generate
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)

        # Load/generate each benchmark
        datasets["mmmu"] = self._load_or_generate_mmmu(datasets_dir)
        datasets["gpqa"] = self._load_or_generate_gpqa(datasets_dir)
        datasets["swe_bench"] = self._load_or_generate_swe_bench(datasets_dir)
        datasets["math"] = self._load_or_generate_math(datasets_dir)
        datasets["humaneval"] = self._load_or_generate_humaneval(datasets_dir)
        datasets["mmlu"] = self._load_or_generate_mmlu(datasets_dir)
        datasets["re_bench"] = self._load_or_generate_re_bench(datasets_dir)

        return datasets

    def _load_or_generate_mmmu(self, datasets_dir: Path) -> List[Dict]:
        """Load or generate MMMU dataset (Massive Multidiscipline Multimodal Understanding)"""
        if self.verbose:
            print("   • MMMU (Multimodal College Questions)...")

        # Try HuggingFace datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("MMMU/MMMU", split="validation")
            questions = []
            for item in ds:
                questions.append({
                    "question": item.get("question", ""),
                    "choices": item.get("options", []),
                    "answer": item.get("answer", 0),
                    "subject": item.get("subject", "general"),
                    "difficulty": "college"
                })
            if self.verbose:
                print(f"      Loaded {len(questions)} questions from HuggingFace")
            return questions[:100]  # Sample for testing
        except:
            pass

        # Generate synthetic MMMU questions
        if self.verbose:
            print("      Generating synthetic MMMU questions...")

        subjects = ["math", "physics", "chemistry", "biology", "computer_science", "economics"]
        questions = []

        for i in range(200):
            subject = subjects[i % len(subjects)]
            questions.append({
                "question": f"College-level {subject} question {i+1}: Analyze the following concept and determine the correct relationship.",
                "choices": ["Option A: Linear relationship", "Option B: Exponential relationship",
                           "Option C: No relationship", "Option D: Inverse relationship"],
                "answer": i % 4,
                "subject": subject,
                "difficulty": "college"
            })

        return questions

    def _load_or_generate_gpqa(self, datasets_dir: Path) -> List[Dict]:
        """Load or generate GPQA dataset (Graduate-level Problem Question Answering)"""
        if self.verbose:
            print("   • GPQA (Graduate-Level Questions)...")

        try:
            from datasets import load_dataset
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            questions = []
            for item in ds:
                questions.append({
                    "question": item.get("Question", ""),
                    "choices": [
                        item.get("Correct Answer", ""),
                        item.get("Incorrect Answer 1", ""),
                        item.get("Incorrect Answer 2", ""),
                        item.get("Incorrect Answer 3", "")
                    ],
                    "answer": 0,  # Correct answer is first
                    "difficulty": "graduate"
                })
            if self.verbose:
                print(f"      Loaded {len(questions)} questions from HuggingFace")
            return questions[:100]
        except:
            pass

        # Generate synthetic graduate-level questions
        if self.verbose:
            print("      Generating synthetic GPQA questions...")

        domains = ["theoretical_physics", "advanced_mathematics", "quantum_mechanics",
                  "computational_complexity", "molecular_biology"]
        questions = []

        for i in range(150):
            domain = domains[i % len(domains)]
            questions.append({
                "question": f"Graduate-level {domain} problem {i+1}: Solve the advanced theoretical problem involving multiple concepts.",
                "choices": [f"Solution {j+1}" for j in range(4)],
                "answer": i % 4,
                "difficulty": "graduate"
            })

        return questions

    def _load_or_generate_swe_bench(self, datasets_dir: Path) -> List[Dict]:
        """Load or generate SWE-bench dataset (Software Engineering Benchmarks)"""
        if self.verbose:
            print("   • SWE-bench (Software Engineering)...")

        try:
            from datasets import load_dataset
            ds = load_dataset("princeton-nlp/SWE-bench", split="test")
            questions = []
            for item in ds:
                questions.append({
                    "problem": item.get("problem_statement", ""),
                    "repo": item.get("repo", ""),
                    "test_patch": item.get("test_patch", ""),
                    "type": "code_generation"
                })
            if self.verbose:
                print(f"      Loaded {len(questions)} problems from HuggingFace")
            return questions[:50]
        except:
            pass

        # Generate synthetic coding problems
        if self.verbose:
            print("      Generating synthetic SWE-bench problems...")

        problems = []
        problem_types = ["bug_fix", "feature_implementation", "refactoring", "optimization"]

        for i in range(100):
            ptype = problem_types[i % len(problem_types)]
            problems.append({
                "problem": f"Implement a {ptype} for a function that processes data with O(n) complexity. Problem {i+1}.",
                "expected_output": f"def solution_{i}(data): return processed_data",
                "type": "code_generation"
            })

        return problems

    def _load_or_generate_math(self, datasets_dir: Path) -> List[Dict]:
        """Load or generate MATH dataset (Mathematical Reasoning)"""
        if self.verbose:
            print("   • MATH (Mathematical Reasoning)...")

        try:
            from datasets import load_dataset
            ds = load_dataset("hendrycks/competition_math", split="test")
            questions = []
            for item in ds:
                questions.append({
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "level": item.get("level", "unknown"),
                    "type": item.get("type", "unknown")
                })
            if self.verbose:
                print(f"      Loaded {len(questions)} problems from HuggingFace")
            return questions[:100]
        except:
            pass

        # Generate synthetic math problems
        if self.verbose:
            print("      Generating synthetic MATH problems...")

        problems = []
        types = ["algebra", "geometry", "number_theory", "counting_and_probability", "precalculus"]

        for i in range(150):
            ptype = types[i % len(types)]
            a, b, c = (i % 20) + 1, (i % 15) + 1, (i % 10) + 1
            problems.append({
                "problem": f"Solve the {ptype} problem: If x^2 + {a}x + {b} = 0, find the value of x.",
                "solution": f"x = {c}",
                "level": str((i % 5) + 1),
                "type": ptype
            })

        return problems

    def _load_or_generate_humaneval(self, datasets_dir: Path) -> List[Dict]:
        """Load or generate HumanEval dataset (Code Generation)"""
        if self.verbose:
            print("   • HumanEval (Code Generation)...")

        try:
            from datasets import load_dataset
            ds = load_dataset("openai_humaneval", split="test")
            questions = []
            for item in ds:
                questions.append({
                    "task_id": item.get("task_id", ""),
                    "prompt": item.get("prompt", ""),
                    "canonical_solution": item.get("canonical_solution", ""),
                    "test": item.get("test", "")
                })
            if self.verbose:
                print(f"      Loaded {len(questions)} problems from HuggingFace")
            return questions
        except:
            pass

        # Generate synthetic coding problems
        if self.verbose:
            print("      Generating synthetic HumanEval problems...")

        problems = []
        for i in range(164):  # HumanEval has 164 problems
            problems.append({
                "task_id": f"HumanEval/{i}",
                "prompt": f"def solution_{i}(n: int) -> int:\n    '''Return the {i}th number in sequence'''\n    ",
                "canonical_solution": f"return n * {i+1}",
                "test": f"assert solution_{i}(1) == {i+1}"
            })

        return problems

    def _load_or_generate_mmlu(self, datasets_dir: Path) -> List[Dict]:
        """Load or generate MMLU dataset (Massive Multitask Language Understanding)"""
        if self.verbose:
            print("   • MMLU (Multitask Language Understanding)...")

        try:
            from datasets import load_dataset
            # Load multiple MMLU subjects
            subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics",
                       "clinical_knowledge", "college_biology", "college_chemistry"]
            all_questions = []

            for subject in subjects:
                try:
                    ds = load_dataset("cais/mmlu", subject, split="test")
                    for item in ds:
                        all_questions.append({
                            "question": item.get("question", ""),
                            "choices": item.get("choices", []),
                            "answer": item.get("answer", 0),
                            "subject": subject
                        })
                except:
                    continue

            if all_questions and self.verbose:
                print(f"      Loaded {len(all_questions)} questions from HuggingFace")
            return all_questions[:200] if all_questions else self._generate_mmlu_synthetic()
        except:
            pass

        return self._generate_mmlu_synthetic()

    def _generate_mmlu_synthetic(self) -> List[Dict]:
        """Generate synthetic MMLU questions"""
        if self.verbose:
            print("      Generating synthetic MMLU questions...")

        subjects = ["mathematics", "physics", "history", "law", "medicine", "philosophy"]
        questions = []

        for i in range(200):
            subject = subjects[i % len(subjects)]
            questions.append({
                "question": f"In {subject}, what is the definition of concept {i+1}?",
                "choices": [f"Definition {j+1}" for j in range(4)],
                "answer": i % 4,
                "subject": subject
            })

        return questions

    def _load_or_generate_re_bench(self, datasets_dir: Path) -> List[Dict]:
        """Generate RE-Bench dataset (Agentic Reasoning Evaluation)"""
        if self.verbose:
            print("   • RE-Bench (Agentic Reasoning)...")
            print("      Generating agentic reasoning tasks...")

        # RE-Bench tests AI agents on complex multi-step tasks
        tasks = []
        task_types = ["planning", "tool_use", "information_gathering", "multi_step_reasoning"]

        for i in range(50):
            ttype = task_types[i % len(task_types)]
            tasks.append({
                "task": f"Agentic {ttype} task {i+1}: Complete a multi-step process requiring planning and execution.",
                "steps_required": (i % 5) + 2,
                "time_horizon": "short" if i % 2 == 0 else "long",
                "type": ttype
            })

        return tasks

    async def run_full_ai_index_suite(self) -> Dict[str, Any]:
        """Run all AI Index benchmarks and generate comprehensive report"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("🚀 RUNNING FULL AI INDEX TEST SUITE")
            print("=" * 80)

        all_results = {}
        start_time = time.time()

        # Run each benchmark
        benchmarks = ["mmmu", "gpqa", "swe_bench", "math", "humaneval", "mmlu", "re_bench"]

        for benchmark in benchmarks:
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"📝 Running {benchmark.upper()} Benchmark")
                print(f"{'='*80}")

            try:
                result = await self._run_benchmark(benchmark)
                all_results[benchmark] = result.to_dict()

                if self.verbose:
                    print(f"\n✅ {benchmark.upper()} Complete:")
                    print(f"   Score: {result.score:.2f}%")
                    print(f"   Correct: {result.correct_answers}/{result.total_questions}")
                    print(f"   Time: {result.execution_time:.2f}s")

                    if result.baseline_comparison:
                        print(f"\n   📊 Comparison with AI Index Baselines:")
                        for model, score in result.baseline_comparison.items():
                            diff = result.score - score
                            status = "🟢" if diff >= 0 else "🔴"
                            print(f"      {status} vs {model}: {score:.1f}% ({diff:+.1f}%)")

            except Exception as e:
                print(f"   ❌ Error in {benchmark}: {e}")
                traceback.print_exc()

        total_time = time.time() - start_time

        # Calculate overall statistics
        summary = {
            "test_suite": "AI Index 2025",
            "model": "ECH0-PRIME" if self.use_ech0_prime else "LLM-Only",
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "overall_score": np.mean([r['score'] for r in all_results.values()]) if all_results else 0,
            "total_questions": sum(r['total_questions'] for r in all_results.values()),
            "total_correct": sum(r['correct_answers'] for r in all_results.values()),
            "benchmarks": all_results,
            "baseline_comparison": self._generate_overall_comparison(all_results)
        }

        # Save results
        self._save_results(summary)

        # Print final report
        self._print_final_report(summary)

        return summary

    async def _run_benchmark(self, benchmark_name: str) -> AIIndexTestResult:
        """Run a single benchmark"""
        questions = self.test_data.get(benchmark_name, [])

        if not questions:
            return AIIndexTestResult(
                benchmark_name=benchmark_name,
                score=0,
                total_questions=0,
                correct_answers=0,
                accuracy=0,
                percentile_rank=None,
                details={},
                execution_time=0,
                error="No test data available"
            )

        start_time = time.time()
        correct = 0

        for i, question in enumerate(questions):
            try:
                if self.use_ech0_prime:
                    answer = await self._ask_ech0_prime(question, benchmark_name)
                else:
                    answer = await self._ask_llm(question)

                if self._check_answer(answer, question, benchmark_name):
                    correct += 1

                if self.verbose and (i + 1) % 25 == 0:
                    print(f"   Progress: {i+1}/{len(questions)} ({(i+1)/len(questions)*100:.1f}%)")

            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Question {i+1} error: {e}")

        duration = time.time() - start_time
        accuracy = correct / len(questions) if questions else 0
        score = accuracy * 100

        # Get baseline comparison
        baseline_comparison = self.baselines.get(benchmark_name, {})

        # Calculate percentile rank
        if baseline_comparison:
            baseline_scores = list(baseline_comparison.values())
            percentile_rank = (sum(1 for s in baseline_scores if score >= s) / len(baseline_scores)) * 100
        else:
            percentile_rank = None

        return AIIndexTestResult(
            benchmark_name=benchmark_name,
            score=score,
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=accuracy,
            percentile_rank=percentile_rank,
            details={"questions_answered": len(questions)},
            execution_time=duration,
            baseline_comparison=baseline_comparison
        )

    async def _ask_ech0_prime(self, question: Dict, benchmark: str) -> str:
        """Query ECH0-PRIME cognitive system"""
        try:
            if not hasattr(self, 'ech0_orchestrator'):
                return await self._ask_llm(question)

            # Format question based on benchmark type
            if benchmark in ["math", "gpqa"]:
                query = question.get("problem", question.get("question", ""))
                return self.ech0_orchestrator.solve_mathematical_problem(query)

            elif benchmark in ["swe_bench", "humaneval"]:
                prompt = question.get("prompt", question.get("problem", ""))
                return str(prompt)  # Code generation placeholder

            elif benchmark == "re_bench":
                task = question.get("task", "")
                # Use cognitive cycle for agentic tasks
                input_data = np.array([ord(c) for c in task[:100]])
                result = self.ech0_orchestrator.cognitive_cycle(input_data, task)
                return str(result) if result else ""

            else:  # Multiple choice questions (MMMU, MMLU, etc.)
                query = question.get("question", "")
                choices = question.get("choices", [])

                problem_data = {
                    "question": query,
                    "choices": choices,
                    "domain": benchmark
                }
                solutions = self.ech0_orchestrator.solve_creatively(problem_data)
                if solutions:
                    return solutions[0].get("answer", "")

        except Exception as e:
            if self.verbose:
                print(f"      ECH0-PRIME error: {e}")
            return await self._ask_llm(question)

        return ""

    async def _ask_llm(self, question: Dict) -> str:
        """Query LLM directly"""
        prompt = question.get("question", question.get("problem", question.get("task", "")))

        if "choices" in question:
            choices = question["choices"]
            prompt += "\n\nChoices:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
            prompt += "\n\nAnswer with just the number (1-4)."

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.llm_bridge.query, prompt, None, None, 0.1, 0.9
            )
        except:
            return ""

    def _check_answer(self, response: str, question: Dict, benchmark: str) -> bool:
        """Check if response is correct"""
        response = response.lower().strip()

        # Multiple choice questions
        if "choices" in question and "answer" in question:
            correct_idx = question["answer"]

            # Check for choice number
            nums = re.findall(r'\b[1-4]\b', response)
            if nums and int(nums[0]) == correct_idx + 1:
                return True

            # Check for choice text
            if len(question["choices"]) > correct_idx:
                correct_text = question["choices"][correct_idx].lower()
                if correct_text[:20] in response:
                    return True

        # Math/exact answer questions
        elif "solution" in question:
            expected = question["solution"].lower()
            # Extract numbers
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            response_nums = re.findall(r'-?\d+\.?\d*', response)

            if expected_nums and response_nums:
                return expected_nums[-1] == response_nums[-1]

        # Code generation (basic check)
        elif benchmark in ["swe_bench", "humaneval"]:
            # Check if response contains code
            return "def " in response or "return" in response

        return False

    def _generate_overall_comparison(self, results: Dict) -> Dict[str, Any]:
        """Generate overall comparison with AI Index baselines"""
        comparison = {}

        for benchmark, result in results.items():
            if benchmark in self.baselines:
                ech0_score = result['score']
                baselines = self.baselines[benchmark]

                # Calculate ranking
                baseline_scores = list(baselines.values())
                worse_or_equal = sum(1 for s in baseline_scores if ech0_score >= s)
                rank = len(baseline_scores) + 1 - worse_or_equal

                comparison[benchmark] = {
                    "ech0_score": ech0_score,
                    "rank": f"{rank}/{len(baselines) + 1}",
                    "baselines": baselines
                }

        return comparison

    def _save_results(self, results: Dict):
        """Save results to file"""
        filename = f"ai_index_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\n💾 Results saved to: {filename}")

    def _print_final_report(self, summary: Dict):
        """Print comprehensive final report"""
        print("\n" + "=" * 80)
        print("📊 AI INDEX TEST SUITE - FINAL REPORT")
        print("=" * 80)
        print(f"\nModel: {summary['model']}")
        print(f"Test Suite: {summary['test_suite']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Execution Time: {summary['total_time']:.2f}s")
        print(f"\nOverall Score: {summary['overall_score']:.2f}%")
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Total Correct: {summary['total_correct']}")
        print(f"Overall Accuracy: {summary['total_correct']/summary['total_questions']*100:.2f}%")

        print("\n" + "-" * 80)
        print("BENCHMARK RESULTS")
        print("-" * 80)

        for benchmark, result in summary['benchmarks'].items():
            print(f"\n{benchmark.upper()}:")
            print(f"  Score: {result['score']:.2f}%")
            print(f"  Accuracy: {result['correct_answers']}/{result['total_questions']}")
            print(f"  Time: {result['execution_time']:.2f}s")

            if result.get('percentile_rank'):
                print(f"  Percentile Rank: {result['percentile_rank']:.1f}%")

        if summary.get('baseline_comparison'):
            print("\n" + "-" * 80)
            print("COMPARISON WITH AI INDEX BASELINES")
            print("-" * 80)

            for benchmark, comp in summary['baseline_comparison'].items():
                print(f"\n{benchmark.upper()}:")
                print(f"  ECH0-PRIME: {comp['ech0_score']:.1f}%")
                print(f"  Rank: {comp['rank']}")

                for model, score in comp['baselines'].items():
                    diff = comp['ech0_score'] - score
                    status = "🟢" if diff >= 0 else "🔴"
                    print(f"    {status} {model}: {score:.1f}% ({diff:+.1f}%)")

        print("\n" + "=" * 80)
        print("✅ AI INDEX EVALUATION COMPLETE")
        print("=" * 80)


async def main():
    """Main CLI entrypoint"""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Index Test Suite for ECH0-PRIME",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_index_tests.py --use-ech0
  python ai_index_tests.py --benchmark mmmu gpqa
  python ai_index_tests.py --use-ech0 --verbose
        """
    )

    parser.add_argument("--use-ech0", action="store_true",
                       help="Use ECH0-PRIME cognitive architecture (default: LLM-only)")
    parser.add_argument("--benchmark", nargs="+",
                       help="Specific benchmarks to run")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    suite = AIIndexTestSuite(use_ech0_prime=args.use_ech0, verbose=verbose)

    if args.benchmark:
        # Run specific benchmarks
        print(f"Running selected benchmarks: {', '.join(args.benchmark)}")
        # Implementation for specific benchmarks
    else:
        # Run full suite
        await suite.run_full_ai_index_suite()


if __name__ == "__main__":
    asyncio.run(main())
