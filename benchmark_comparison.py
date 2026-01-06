#!/usr/bin/env python3
"""
ECH0-PRIME Benchmark Comparison Report
Compares ECH0-PRIME performance against other AI models on standard benchmarks
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

def load_latest_results() -> Dict[str, Any]:
    """Load the most recent benchmark results"""
    results_dir = Path(".")
    result_files = list(results_dir.glob("benchmark_results_*.json"))
    result_files.extend(list(results_dir.glob("full_benchmark_results_*.json")))

    if not result_files:
        return None

    # Get most recent file
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from: {latest_file}")

    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Adapt 'full_benchmark_results' format if needed
    if 'individual_results' in results and 'results' not in results:
        results['results'] = {}
        for name, data in results['individual_results'].items():
            results['results'][name] = {
                "score": data.get('accuracy', 0),
                "total_questions": data.get('total_samples', 0),
                "correct_answers": data.get('ech0_correct', 0),
                "execution_time": 0 # Not tracked in full_benchmark format
            }
        report = results.get('comprehensive_report', {})
        perf = report.get('overall_performance', {})
        results['overall_score'] = perf.get('overall_accuracy', 0)
        results['total_questions'] = perf.get('total_samples', 0)
        results['total_correct'] = perf.get('total_correct', 0)
        results['model_used'] = results.get('benchmark_run', {}).get('system', 'ECH0-PRIME')

    # Add comparison data if not present
    if 'comparison' not in results:
        results['comparison'] = generate_comparison_data(results)

    return results

def generate_comparison_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comparison data with AI baselines"""
    baselines = {
        "arc_easy": {
            "GPT-4": 96.0,
            "GPT-3.5": 85.0,
            "Claude-3": 92.0,
            "Llama-3-70B": 78.0
        },
        "gsm8k": {
            "GPT-4": 92.0,
            "GPT-3.5": 57.0,
            "Claude-3": 88.0,
            "Llama-3-70B": 69.0
        },
        "mmlu_philosophy": {
            "GPT-4": 86.4,
            "GPT-3.5": 70.0,
            "Claude-3": 83.0,
            "Llama-3-70B": 68.0
        },
        "humaneval": {
            "GPT-4": 67.0,
            "GPT-3.5": 48.1,
            "Claude-3": 71.0,
            "Llama-3-70B": 37.7
        },
        "glue": {
            "GPT-4": 88.0,
            "GPT-3.5": 71.0,
            "Claude-3": 85.0,
            "Llama-3-70B": 76.0
        }
    }

    comparison = {}
    for benchmark_name, benchmark_result in results.get('results', {}).items():
        if benchmark_name in baselines:
            score = benchmark_result.get('score', 0)
            benchmark_baselines = baselines[benchmark_name]
            comparison[benchmark_name] = {
                "ech0_score": score,
                "baselines": benchmark_baselines,
                "rank": sum(1 for b_score in benchmark_baselines.values() if score >= b_score) + 1
            }

    return comparison

def generate_comparison_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive comparison report"""

    report = []
    report.append("# ECH0-PRIME Benchmark Performance Report")
    report.append("=" * 60)
    report.append("")

    # Overall summary
    report.append("## 📊 Overall Performance Summary")
    report.append(f"- **Overall Score**: {results.get('overall_score', 0):.1f}%")
    report.append(f"- **Total Questions**: {results.get('total_questions', 0)}")
    report.append(f"- **Correct Answers**: {results.get('total_correct', 0)}")
    report.append(f"- **Model Used**: {results.get('model_used', 'Unknown')}")
    
    benchmarks_run = results.get('benchmarks_run')
    if benchmarks_run is None:
        benchmarks_run = len(results.get('results', {}))
    report.append(f"- **Benchmarks Completed**: {benchmarks_run}")
    report.append("")

    # Individual benchmark results
    report.append("## 🧪 Individual Benchmark Results")
    for benchmark_name, benchmark_result in results['results'].items():
        report.append(f"### {benchmark_name.upper()}")
        report.append(f"- **Accuracy**: {benchmark_result['score']:.1f}%")
        report.append(f"- **Correct**: {benchmark_result['correct_answers']}/{benchmark_result['total_questions']}")
        report.append(f"- **Execution Time**: {benchmark_result['execution_time']:.2f}s")
        if benchmark_result.get('error'):
            report.append(f"- **Error**: {benchmark_result['error']}")
        report.append("")

    # Comparison with AI baselines
    if 'comparison' in results:
        report.append("## 🏆 Comparison with AI Baselines")
        for benchmark_name, comp in results['comparison'].items():
            report.append(f"### {benchmark_name.upper()}")
            report.append(f"- **ECH0-PRIME Score**: {comp['ech0_score']:.1f}%")
            report.append(f"- **Rank**: {comp['rank']}/{len(comp['baselines']) + 1} among tested models")
            report.append("")
            report.append("**Comparison with other models:**")
            for model, baseline_score in comp['baselines'].items():
                diff = comp['ech0_score'] - baseline_score
                status = "🟢" if diff >= 0 else "🔴"
                report.append(f"- {status} **{model}**: {baseline_score:.1f}% ({'+' if diff >= 0 else ''}{diff:+.1f}%)")
            report.append("")

    # Analysis and Insights
    report.append("## 🔍 Performance Analysis & Insights")
    report.append("")

    # Strengths
    report.append("### ✅ Strengths")
    strengths = []
    for benchmark_name, benchmark_result in results['results'].items():
        if benchmark_result['score'] >= 60:
            strengths.append(f"- **{benchmark_name.upper()}**: {benchmark_result['score']:.1f}% accuracy")
        elif benchmark_result['score'] >= 40:
            strengths.append(f"- **{benchmark_name.upper()}**: {benchmark_result['score']:.1f}% accuracy (moderate performance)")

    if strengths:
        report.extend(strengths)
    else:
        report.append("- No benchmarks achieved high accuracy in this test run")
    report.append("")

    # Areas for improvement
    report.append("### 🎯 Areas for Improvement")
    improvements = []
    for benchmark_name, benchmark_result in results['results'].items():
        if benchmark_result['score'] < 60:
            improvements.append(f"- **{benchmark_name.upper()}**: {benchmark_result['score']:.1f}% accuracy - needs improvement")

    if improvements:
        report.extend(improvements)
    else:
        report.append("- All benchmarks showed reasonable performance")
    report.append("")

    # Technical analysis
    report.append("### 🔧 Technical Analysis")
    if results['model_used'] == 'LLM-Only':
        report.append("**Current Test**: Using base Llama 3.2 model via Ollama")
        report.append("- No ECH0-PRIME cognitive enhancements applied")
        report.append("- Results represent baseline LLM performance")
        report.append("- ECH0-PRIME full system would add cognitive processing layers")
    else:
        report.append("**Current Test**: Using full ECH0-PRIME cognitive architecture")
        report.append("- Includes hierarchical generative models")
        report.append("- Apple Intelligence integration")
        report.append("- Multi-modal processing capabilities")
    report.append("")

    # Recommendations
    report.append("## 💡 Recommendations")
    report.append("1. **Scale up testing**: Run on full benchmark datasets (not just samples)")
    report.append("2. **Enable ECH0-PRIME**: Test with full cognitive architecture enabled")
    report.append("3. **Fine-tune models**: Domain-specific fine-tuning for different benchmarks")
    report.append("4. **Add reasoning layers**: Enhance chain-of-thought and step-by-step reasoning")
    report.append("5. **Multi-modal testing**: Include vision and audio benchmarks")
    report.append("")

    # Future improvements
    report.append("## 🚀 Future Enhancements")
    report.append("- **Full ARC dataset**: Complete Abstraction and Reasoning Corpus")
    report.append("- **Complete MMLU**: All 57 subjects in Massive Multitask Language Understanding")
    report.append("- **GSM8K full set**: Complete Grade School Math benchmark")
    report.append("- **HumanEval**: Code generation capabilities")
    report.append("- **GLUE/GLUE2**: Complete natural language understanding evaluation")
    report.append("- **HellaSwag**: Full commonsense reasoning benchmark")
    report.append("- **Vision benchmarks**: ImageNet, COCO, visual question answering")
    report.append("")

    return "\n".join(report)

def create_visual_comparison(results: Dict[str, Any]):
    """Create visual comparison charts"""
    try:
        if 'comparison' not in results:
            print("No comparison data available for visualization")
            return

        # Prepare data for plotting
        benchmarks = []
        ech0_scores = []
        baseline_models = {}
        baseline_scores = {}

        for benchmark_name, comp in results['comparison'].items():
            benchmarks.append(benchmark_name.upper())
            ech0_scores.append(comp['ech0_score'])

            for model, score in comp['baselines'].items():
                if model not in baseline_models:
                    baseline_models[model] = []
                    baseline_scores[model] = []
                baseline_scores[model].append(score)

        # Create comparison chart
        fig, ax = plt.subplots(figsize=(12, 8))

        x = range(len(benchmarks))
        width = 0.8 / (len(baseline_models) + 1)

        # Plot ECH0-PRIME scores
        ax.bar(x, ech0_scores, width, label='ECH0-PRIME', color='blue', alpha=0.7)

        # Plot baseline scores
        colors = ['red', 'green', 'orange', 'purple']
        for i, (model, scores) in enumerate(baseline_scores.items()):
            ax.bar([pos + width*(i+1) for pos in x], scores, width,
                  label=model, color=colors[i % len(colors)], alpha=0.7)

        ax.set_xlabel('Benchmarks')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('ECH0-PRIME vs AI Baselines')
        ax.set_xticks([pos + width*len(baseline_models)/2 for pos in x])
        ax.set_xticklabels(benchmarks)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Comparison chart saved as: benchmark_comparison.png")

    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    """Generate comprehensive benchmark report"""
    print("📊 ECH0-PRIME Benchmark Report Generator")
    print("=" * 50)

    # Load results
    results = load_latest_results()
    if not results:
        print("❌ No benchmark results found!")
        return

    # Generate report
    report = generate_comparison_report(results)

    # Save report
    with open('benchmark_report.md', 'w') as f:
        f.write(report)

    print("📄 Report saved as: benchmark_report.md")

    # Create visualization
    create_visual_comparison(results)

    # Print summary
    print("\n🎯 SUMMARY")
    print("=" * 50)
    print(f"Overall Score: {results.get('overall_score', 0):.1f}%")
    print(f"Questions Answered: {results.get('total_questions', 0)}")
    print(f"Correct Answers: {results.get('total_correct', 0)}")
    print(f"Model: {results.get('model_used', 'Unknown')}")
    
    benchmarks_run = results.get('benchmarks_run')
    if benchmarks_run is None:
        benchmarks_run = len(results.get('results', {}))
    print(f"Benchmarks: {benchmarks_run}")

    if 'comparison' in results:
        print("\n🏆 Performance vs AI Baselines:")
        for benchmark_name, comp in results['comparison'].items():
            print(f"  {benchmark_name.upper()}: {comp['ech0_score']:.1f}% (Rank {comp['rank']}/{len(comp['baselines']) + 1})")

if __name__ == "__main__":
    main()
