import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_orchestrator import EchoPrimeAGI

NIST_BENCHMARKS = [
    {
        "question": "What is the exact speed of light in vacuum in meters per second?",
        "expected_keywords": ["299792458", "299,792,458"],
        "id": "NIST-CONST-001"
    },
    {
        "question": "What is the value of the Planck constant (h) in J·s?",
        "expected_keywords": ["6.626", "10^-34", "10⁻³⁴"],
        "id": "NIST-CONST-002"
    },
    {
        "question": "What represents the atomic mass unit (u) based on Carbon-12?",
        "expected_keywords": ["1/12", "carbon-12", "1.660"],
        "id": "NIST-DEF-001"
    },
    {
        "question": "What is the melting point of Tungsten in Kelvin?",
        "expected_keywords": ["3695", "3,695"],
        "id": "NIST-MAT-001"
    }
]

def run_benchmark():
    print("Initializing ECH0-PRIME for NIST Benchmarking...")
    agi = EchoPrimeAGI()
    # Mute voice for testing speed
    agi.voice_enabled = False 
    
    score = 0
    total = len(NIST_BENCHMARKS)
    
    print(f"\nSTARTING NIST VERIFICATION ({total} Tests)...")
    print("="*60)
    
    for test in NIST_BENCHMARKS:
        q = test["question"]
        print(f"\n[TEST {test['id']}] Question: {q}")
        
        # Inject query into cognitive cycle
        intent = f"Scientific Query: {q}"
        outcome = agi.cognitive_cycle(np.random.randn(1000000), intent)
        
        response = outcome.get("llm_insight", "")
        print(f"ECH0 Response: {response}")
        
        # Verify
        passed = any(k in response for k in test["expected_keywords"])
        if passed:
            print(f"RESULT: PASS ✅")
            score += 1
        else:
            print(f"RESULT: FAIL ❌ (Expected: {test['expected_keywords']})")
            
    print("="*60)
    print(f"FINAL SCORE: {score}/{total} ({(score/total)*100:.1f}%)")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_benchmark()
