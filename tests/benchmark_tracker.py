import json
import os

# Use absolute path to avoid FileNotFoundError in different execution contexts
BENCHMARK_STATE = "/Users/noone/echo_prime/dashboard/v2/public/data/benchmarks.json"
BENCHMARK_STATE_FALLBACK = "dashboard/v2/public/data/benchmarks.json"

def update_benchmark_stats(suite_name, total, passed, current_idx):
    """Updates a global JSON file for the dashboard to read."""
    state = {}
    
    # Try absolute path first
    target_path = BENCHMARK_STATE
    target_dir = os.path.dirname(target_path)
    
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
    except Exception:
        # Fallback to relative path if absolute path directory creation fails
        target_path = BENCHMARK_STATE_FALLBACK
        target_dir = os.path.dirname(target_path)
        if target_dir and not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
    if os.path.exists(target_path):
        try:
            with open(target_path, 'r') as f:
                state = json.load(f)
        except:
            pass
    
    state[suite_name] = {
        "total": total,
        "passed": passed,
        "current": current_idx,
        "progress": f"{(current_idx / total * 100):.1f}%" if total > 0 else "0%"
    }
    
    try:
        with open(target_path, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write benchmark stats to {target_path}: {e}")
