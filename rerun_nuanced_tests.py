import sys
import os
import numpy as np
import json
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from simple_orchestrator import SimpleEchoPrimeAGI
from training.intelligent_grader import IntelligentGrader
from reasoning.llm_bridge import OllamaBridge

def generate_and_test():
    print("🧬 Starting Synthetic Variant Benchmark (Nuanced Grading)...")
    agi = SimpleEchoPrimeAGI(lightweight=True)
    grader = IntelligentGrader()
    llm = OllamaBridge(model="llama3.2")
    
    # 1. Take a known problem type and mutate it
    seeds = [
        {
            "original": "If Janet has 5 apples and eats 2, how many are left?",
            "type": "Arithmetic"
        },
        {
            "original": "A laser beam hits a proton, does the orbital angular momentum increase or decrease its energy?",
            "type": "Physics"
        }
    ]
    
    variants = []
    
    print("\n🛠️ Generating new variants off known seeds...")
    for seed in seeds:
        mutation_prompt = f"Given this question: '{seed['original']}', create a NEW variation with different numbers and a slightly more complex scenario. Output ONLY the question and the correct answer in JSON format: {{'question': '...', 'answer': '...'}}"
        raw_variant = llm.query(mutation_prompt)
        
        try:
            # Basic cleanup
            import re
            json_match = re.search(r'\{.*\}', raw_variant, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                variants.append(data)
                print(f"  ✅ Generated variation for {seed['type']}")
        except:
            print(f"  ⚠️ Failed to generate variation for {seed['type']}")

    # 2. Add some manually crafted "unused" questions (within-range test cases)
    variants.append({
        "question": "Calculate the volume of a sphere with a radius of 3.33 units. Use pi = 3.14159.",
        "answer": "154.5" # Actual is ~154.53
    })
    variants.append({
        "question": "What is the capital of Kazakhstan?",
        "answer": "Astana" # (Recently renamed back from Nur-Sultan, good test for 'freshness')
    })
    
    print(f"\n🚀 Running {len(variants)} Truthful/Nuanced Benchmarks...")
    print("="*60)
    
    for i, v in enumerate(variants):
        q = v["question"]
        expected = v["answer"]
        
        print(f"\n[Test {i+1}] Q: {q}")
        
        # Run through orchestrator
        intent = f"Nuanced Test: {q}"
        outcome = agi.cognitive_cycle(np.random.randn(500), intent)
        response = outcome.get("llm_insight", "No response")
        
        print(f"ECH0: {response}")
        
        # Grade with nuance
        score, log = grader.grade(q, expected, response)
        print(f"Result -> Score: {score}")
        print(f"Audit: {log}")
        print("-" * 40)

if __name__ == "__main__":
    generate_and_test()
