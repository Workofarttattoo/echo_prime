import os
import json
import numpy as np
import sys
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simple_orchestrator import SimpleEchoPrimeAGI
from training.intelligent_grader import IntelligentGrader
try:
    from benchmark_tracker import update_benchmark_stats
except ImportError:
    try:
        from tests.benchmark_tracker import update_benchmark_stats
    except ImportError:
        def update_benchmark_stats(*args, **kwargs): pass

DATA_DIR = "tests/arc_agi/data/training"
LOG_FILE = "tests/arc_agi_results.log"

def grid_to_str(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def run_arc_benchmark(limit=10):
    print(f"Initializing ECH0-PRIME for ARC-AGI Benchmark (limit={limit})...")
    agi = SimpleEchoPrimeAGI(lightweight=True)
    grader = IntelligentGrader()
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    subset = random.sample(files, min(limit, len(files)))
    
    score = 0
    with open(LOG_FILE, "w") as lf:
        lf.write(f"ARC-AGI Benchmark Pilot - {len(subset)} tasks\n\n")

    for i, filename in enumerate(subset):
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            task = json.load(f)
        
        print(f"\rProcessing ARC Task {i+1}/{len(subset)}: {filename}", end="", flush=True)
        
        # Build prompt from training examples
        prompt = "TASK: Find the rule that transforms the input grid into the output grid. Here are examples:\n\n"
        for ex in task["train"]:
            prompt += f"INPUT:\n{grid_to_str(ex['input'])}\nOUTPUT:\n{grid_to_str(ex['output'])}\n\n"
        
        test_input = task["test"][0]["input"]
        expected_output = task["test"][0]["output"]
        
        prompt += f"Now, apply the rule to this test input and provide ONLY the output grid:\nTEST INPUT:\n{grid_to_str(test_input)}\n\nFINAL OUTPUT GRID:"
        
        outcome = agi.cognitive_cycle(np.random.randn(1000000), prompt)
        response = outcome.get("llm_insight", "")
        
        # Grading
        expected_str = grid_to_str(expected_output)
        score_val, justification = grader.grade(prompt, expected_str, response)
        
        if score_val >= 0.9:
            score += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        update_benchmark_stats("ARC-AGI", len(subset), score, i + 1)
            
        with open(LOG_FILE, "a") as lf:
            lf.write(f"--- Task {filename} ---\n")
            lf.write(f"Expected:\n{expected_str}\n")
            lf.write(f"ECH0 Response:\n{response}\n")
            lf.write(f"Grader Score: {score_val} | Jusitification: {justification}\n")
            lf.write(f"Status: {status}\n\n")

    print("\n" + "="*60)
    print(f"ARC-AGI COMPLETE. SCORE: {score}/{len(subset)} ({(score/len(subset))*100:.1f}%)")
    print(f"Logs saved to: {LOG_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    
    run_arc_benchmark(limit=args.limit)
