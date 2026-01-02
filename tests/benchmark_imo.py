import sys
import os
import csv
import random
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main_orchestrator import EchoPrimeAGI
from training.intelligent_grader import IntelligentGrader
from benchmark_tracker import update_benchmark_stats

CSV_PATH = os.path.join(os.path.dirname(__file__), "imobench/answerbench.csv")

def run_imo_benchmark(limit=None):
    print("Initializing ECH0-PRIME for Full IMO Benchmarking...")
    agi = EchoPrimeAGI()
    agi.voice_enabled = False
    grader = IntelligentGrader()
    
    # Load questions
    questions = []
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}")
        return

    # Use all if limit is None
    if limit is None:
        subset = questions
    else:
        subset = random.sample(questions, min(limit, len(questions)))
    
    print(f"\nSTARTING IMO BENCHMARK (N={len(subset)})...")
    print("="*60)
    
    score = 0
    results = []
    
    # Ensure artifacts/logs directory exists
    log_file = "tests/imobench/benchmark_results.log"
    with open(log_file, "w") as lf:
        lf.write(f"IMO Benchmark Run - {len(subset)} questions\n\n")

    for i, test in enumerate(subset):
        q_id = test.get('Problem ID')
        problem = test.get('Problem')
        expected = test.get('Short Answer')
        category = test.get('Category')
        
        print(f"\rProcessing {i+1}/{len(subset)}: {q_id}", end="", flush=True)
        
        # Inject query
        intent = f"Solve Math Problem: {problem}"
        outcome = agi.cognitive_cycle(np.random.randn(1000000), intent)
        response = outcome.get("llm_insight", "No response")
        
        # Intelligent grading
        score_val, justification = grader.grade(problem, expected, response)
        passed = score_val >= 0.8
        status = "PASS" if passed else "FAIL"
        score += score_val
        update_benchmark_stats("IMO-Math", len(subset), int(score), i + 1)
        
        results.append({
            "id": q_id,
            "category": category,
            "status": status,
            "expected": expected,
            "response": response[:200]
        })
        
        with open(log_file, "a") as lf:
            lf.write(f"--- {q_id} ---\n")
            lf.write(f"Problem: {problem}\n")
            lf.write(f"Expected: {expected}\n")
            lf.write(f"ECH0: {response}\n")
            lf.write(f"Score: {score_val} | Justification: {justification}\n")
            lf.write(f"Status: {status}\n\n")
            
    print("\n" + "="*60)
    print(f"COMPLETE. SCORE: {score}/{len(subset)} ({(score/len(subset))*100:.1f}%)")
    print(f"Full logs saved to: {log_file}")
    
    return results

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Run entire suite
    run_imo_benchmark(limit=None)
