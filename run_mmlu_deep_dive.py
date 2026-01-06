#!/usr/bin/env python3
"""
ECH0-PRIME MMLU Deep Dive Runner
Focuses on comprehensive testing across diverse MMLU subjects.
"""

import os
import sys
import asyncio
import json
import time
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_benchmark_suite import AIBenchmarkSuite

async def run_mmlu_deep_dive():
    print("🔬 ECH0-PRIME: MMLU DEEP DIVE INITIATED")
    print("=" * 60)
    
    # Representative subjects across domains
    subjects = [
        "mmlu_philosophy",
        "mmlu_college_physics",
        "mmlu_machine_learning",
        "mmlu_abstract_algebra",
        "mmlu_professional_law",
        "mmlu_world_religions",
        "mmlu_medical_genetics",
        "mmlu_us_foreign_policy",
        "mmlu_high_school_psychology",
        "mmlu_econometrics"
    ]
    
    # Initialize benchmark suite
    # We pass the subjects list to avoid loading all 57 MMLU subjects
    suite = AIBenchmarkSuite(use_ech0_prime=True, use_full_datasets=True, mmlu_subjects=subjects)
    
    # Set sample limit to 20 per subject for the deep dive to balance depth vs stability
    sample_limit = 20
    for s in subjects:
        if s in suite.benchmark_data:
            suite.benchmark_data[s] = suite.benchmark_data[s][:sample_limit]
    
    # Filter benchmark data to only included subjects
    available_subjects = [s for s in subjects if s in suite.benchmark_data]
    
    if not available_subjects:
        print("⚠️ No MMLU subjects found in loaded data. Attempting to load from HF specifically.")
        # This will be handled by the suite if we pass them to run_benchmark_suite
        # but let's make sure they are at least attempted.
        available_subjects = subjects

    print(f"📊 Targeted Subjects: {len(available_subjects)}")
    for s in available_subjects:
        count = len(suite.benchmark_data.get(s, []))
        print(f"   • {s}: {count} questions")

    # Run the benchmarks
    start_time = time.time()
    results = await suite.run_benchmark_suite(benchmarks=available_subjects)
    duration = time.time() - start_time

    # Generate a specialized MMLU report
    report_file = f"mmlu_deep_dive_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ MMLU DEEP DIVE COMPLETE")
    print(f"⏱️ Total Duration: {duration/60:.1f} minutes")
    print(f"📄 Full report saved to: {report_file}")
    
    # Display summary
    print("\n🏆 MMLU SUBJECT BREAKDOWN:")
    print("-" * 40)
    for name, res in results['results'].items():
        print(f"{name:25}: {res['score']:.1f}% ({res['correct_answers']}/{res['total_questions']})")
    
    print("-" * 40)
    print(f"{'OVERALL MMLU AVG':25}: {results['overall_score']:.1f}%")

if __name__ == "__main__":
    asyncio.run(run_mmlu_deep_dive())

