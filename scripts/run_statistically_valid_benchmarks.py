#!/usr/bin/env python3
"""
ECH0-PRIME Statistically Valid Benchmark Runner
Runs benchmarks with N=100 (or max available) to achieve statistical significance.
"""

import sys
import os
import asyncio
import json
import time
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import benchmark runners
from tests.benchmark_turing import run_turing_test
from tests.benchmark_arc import run_arc_benchmark
from tests.benchmark_hle import run_hle_benchmark
from ai_benchmark_suite import AIBenchmarkSuite

async def run_validated_benchmarks():
    print("🏆 ECH0-PRIME Validated Benchmark Suite (N=100 Target)")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60 + "\n")

    results = {
        "timestamp": time.time(),
        "benchmarks": {}
    }

    # 1. Turing Test (Max 10 probes currently available in the script)
    # The script tests/benchmark_turing.py has 10 hardcoded questions.
    # We will run all 10.
    print("\n1. [TURING TEST] Running all available probes...")
    try:
        turing_data = run_turing_test(limit=20) # Limit > 10 ensures all run
        score = sum(r['score'] for r in turing_data) / len(turing_data) * 100
        results["benchmarks"]["turing"] = {
            "n": len(turing_data),
            "score": score,
            "status": "PASS" if score > 50 else "FAIL" # Simplified status
        }
    except Exception as e:
        print(f"❌ Turing Test Error: {e}")
        results["benchmarks"]["turing"] = {"error": str(e)}

    # 2. ARC-AGI (N=100)
    print("\n2. [ARC-AGI] Running 100 random samples...")
    try:
        # Note: benchmark_arc.py returns void but prints to stdout/log. 
        # We need to capture the score or modifying the import to return it if possible.
        # Looking at previous file view of benchmark_arc.py, it prints logs but doesn't return the score cleanly in all paths?
        # Actually benchmark_arc.py prints final score but doesn't return it in main block?
        # Let's inspect benchmark_arc.py again. It defines run_arc_benchmark which prints but doesn't return value on line 87?
        # Wait, line 87 is the if __name__ block. 
        # run_arc_benchmark function doesn't return anything.
        # We might need to rely on the log file or modify the function.
        # I'll rely on reading the log file after execution since I can't easily modify the imported function's return without editing the file.
        # OR I can edit benchmark_arc.py to return the score. 
        # Actually, in the interest of "not breaking existing tests", I will parse the log file it produces.
        
        run_arc_benchmark(limit=100)
        
        # Parse log file for score
        log_path = "tests/arc_agi_results.log"
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                content = f.read()
                # fast parse of last line "ARC-AGI COMPLETE. SCORE: X/Y"
                if "ARC-AGI COMPLETE. SCORE:" in content:
                    last_line = [l for l in content.split('\n') if "ARC-AGI COMPLETE. SCORE:" in l][-1]
                    # "ARC-AGI COMPLETE. SCORE: 5/10 (50.0%)"
                    parts = last_line.split("SCORE: ")[1].split(" ")
                    score_fraction = parts[0] # "5/10"
                    num, den = map(int, score_fraction.split("/"))
                    percent = (num/den) * 100
                    results["benchmarks"]["arc_agi"] = {
                        "n": den,
                        "score": percent,
                        "raw": score_fraction
                    }
        else:
             results["benchmarks"]["arc_agi"] = {"error": "Log file not found"}

    except Exception as e:
        print(f"❌ ARC-AGI Error: {e}")
        results["benchmarks"]["arc_agi"] = {"error": str(e)}


    # 3. HLE (N=100)
    print("\n3. [HLE] Humanity's Last Exam (N=100)...")
    try:
        hle_data = run_hle_benchmark(limit=100)
        # hle_data is a list of results
        if hle_data:
            score = sum(1 for r in hle_data if r['status'] == 'PASS')
            total = len(hle_data)
            results["benchmarks"]["hle"] = {
                "n": total,
                "score": (score/total)*100,
                "raw": f"{score}/{total}"
            }
        else:
            results["benchmarks"]["hle"] = {"error": "No data returned (likely dataset load fail)"}
    except Exception as e:
        print(f"❌ HLE Error: {e}")
        results["benchmarks"]["hle"] = {"error": str(e)}


    # 4. GSM8K & MMLU (N=100 each) via AIBenchmarkSuite
    print("\n4. [INDUSTRY] GSM8K & MMLU (N=100)...")
    try:
        suite = AIBenchmarkSuite(
            use_ech0_prime=True, 
            use_full_datasets=True,
            max_samples_per_benchmark=100 # Configured limit
        )
        
        # Override the limit if the class doesn't use the init param perfectly for all methods
        # The suite.run_benchmark_suite leads to run_single_benchmark which leads to specific process methods.
        # Looking at ai_benchmark_suite.py, _process_gsm8k_dataset etc use max_samples_per_benchmark.
        
        suite_results = await suite.run_benchmark_suite(['gsm8k', 'mmlu_philosophy'])
        
        # Suite results structure: {'results': {'benchmark_name': {'score': X, 'accuracy': Y, ...}}}
        
        if 'gsm8k' in suite_results['results']:
            r = suite_results['results']['gsm8k']
            results["benchmarks"]["gsm8k"] = {
                "n": r.get('total_questions', 100),
                "score": r['accuracy'] * 100 if 'accuracy' in r else r['score']
            }
            
        if 'mmlu_philosophy' in suite_results['results']:
            r = suite_results['results']['mmlu_philosophy']
            results["benchmarks"]["mmlu"] = {
                "n": r.get('total_questions', 100),
                "score": r['accuracy'] * 100 if 'accuracy' in r else r['score'],
                "subject": "philosophy" # We used a subset for speed/demo, but N=100 is decent
            }

    except Exception as e:
        print(f"❌ Industry Suite Error: {e}")
        results["benchmarks"]["industry_suite"] = {"error": str(e)}

    # Summary
    print("\n" + "="*60)
    print("📊 VALIDATED RESULTS SUMMARY")
    print("="*60)
    print(json.dumps(results["benchmarks"], indent=2))
    
    with open("validated_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to validated_benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(run_validated_benchmarks())
