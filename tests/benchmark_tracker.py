import json
import os

BENCHMARK_STATE = "dashboard/v2/public/data/benchmarks.json"

def update_benchmark_stats(suite_name, total, passed, current_idx):
    """Updates a global JSON file for the dashboard to read."""
    state = {}
    if os.path.exists(BENCHMARK_STATE):
        try:
            with open(BENCHMARK_STATE, 'r') as f:
                state = json.load(f)
        except:
            pass
    
    state[suite_name] = {
        "total": total,
        "passed": passed,
        "current": current_idx,
        "progress": f"{(current_idx / total * 100):.1f}%" if total > 0 else "0%"
    }
    
    with open(BENCHMARK_STATE, 'w') as f:
        json.dump(state, f, indent=2)
