import sys
import os
import time
import numpy as np
from datasets import load_dataset
import tempfile
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main_orchestrator import EchoPrimeAGI
from training.intelligent_grader import IntelligentGrader
try:
    from benchmark_tracker import update_benchmark_stats
except ImportError:
    try:
        from tests.benchmark_tracker import update_benchmark_stats
    except ImportError:
        def update_benchmark_stats(*args, **kwargs): pass

def run_hle_benchmark(limit=5):
    print("Initializing ECH0-PRIME for HLE Benchmarking...")
    agi = EchoPrimeAGI(enable_voice=False)
    grader = IntelligentGrader()
    
    # ... rest remains ...
    
    print(f"Loading HLE dataset (cais/hle)...")
    try:
        # Get token from env
        hf_token = os.getenv("HF_TOKEN")
        ds = load_dataset("cais/hle", split="test", token=hf_token)
    except Exception as e:
        print(f"ERROR: Could not load HLE dataset: {e}")
        return

    # Select samples
    indices = np.random.choice(len(ds), min(limit, len(ds)), replace=False)
    samples = ds.select(indices)
    
    print(f"\nSTARTING HLE BENCHMARK (N={len(samples)})...")
    print("="*60)
    
    score = 0
    results = []
    
    log_file = "tests/benchmark_hle_results.log"
    with open(log_file, "w") as lf:
        lf.write(f"Humanity's Last Exam Benchmark Run - {len(samples)} questions\n\n")

    for i, sample in enumerate(samples):
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        image = sample.get('image_preview') # This is a PIL image or None
        subject = sample.get('subject', 'Unknown')
        
        print(f"\rProcessing {i+1}/{len(samples)}: [{subject}]", end="", flush=True)
        
        temp_img_path = None
        if image:
            # Save PIL image to temp file for VisionBridge
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                temp_img_path = tmp.name
        
        # Inject query
        intent = f"Humanity's Last Exam Query: {question}"
        outcome = agi.cognitive_cycle(np.random.randn(1000000), intent, image_path=temp_img_path)
        response = outcome.get("llm_insight", "No response")
        
        # Intelligent grading
        score_val, justification = grader.grade(question, answer, response)
        passed = score_val >= 0.8
        status = "PASS" if passed else "FAIL"
        score += score_val
        update_benchmark_stats("HLE", len(samples), int(score), i + 1)
        
        results.append({
            "subject": subject,
            "status": status,
            "expected": answer,
            "response": response[:200]
        })
        
        with open(log_file, "a") as lf:
            lf.write(f"--- Q{i+1} [{subject}] ---\n")
            lf.write(f"Question: {question}\n")
            lf.write(f"Expected: {answer}\n")
            lf.write(f"ECH0: {response}\n")
            lf.write(f"Score: {score_val} | Justification: {justification}\n")
            lf.write(f"Status: {status}\n\n")
            
        # Cleanup temp image
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
    print("\n" + "="*60)
    print(f"COMPLETE. SCORE: {score}/{len(samples)} ({(score/len(samples))*100:.1f}%)")
    print(f"Full logs saved to: {log_file}")
    
    return results

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Run a small batch first
    run_hle_benchmark(limit=5)
