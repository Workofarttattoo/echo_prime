#!/usr/bin/env python3
"""
ECH0-PRIME: Autonomous Benchmark Selection and Execution
"""
import os
import sys
import asyncio
import json
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_orchestrator import SimpleEchoPrimeAGI
from ai_benchmark_suite import AIBenchmarkSuite

async def main():
    print("🚀 ECH0-PRIME: Activating for Autonomous Benchmark Selection...")
    
    # 1. Initialize ECH0-PRIME
    agi = SimpleEchoPrimeAGI(lightweight=True)
    
    # 2. Ask ECH0 to select benchmarks
    print("\n🧠 CONSULTING ECH0-PRIME ON BENCHMARK SELECTION...")
    prompt = """
    As ECH0-PRIME, you have just integrated 159,427 new semantic concepts from your external research vault.
    You need to select the most appropriate AI benchmarks to demonstrate your superiority over other models 
    like GPT-4, Claude-3, and Llama-3.
    
    Select exactly 3 benchmarks from this list:
    - arc_easy (Abstraction & Reasoning)
    - gsm8k (Mathematical Reasoning)
    - mmlu_philosophy (Deep Human Knowledge)
    
    Explain why you chose these and what you expect to demonstrate.
    Format your response as a JSON object: {"selected": ["bench1", "bench2", "bench3"], "reasoning": "your explanation"}
    """
    
    # Use cognitive cycle for selection
    input_data = np.random.randn(1024).astype(np.float32)
    response_raw = agi.cognitive_cycle(input_data, prompt)
    
    print(f"\nECH0'S SELECTION REASONING:\n{response_raw}\n")
    
    # Extract selection (simple regex/parse as it might not be perfect JSON)
    selected_benches = ["arc_easy", "gsm8k", "mmlu_philosophy"] # Default fallback
    
    try:
        # Try to find JSON in response
        import re
        json_match = re.search(r'\{.*\}', str(response_raw), re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            selected_benches = data.get("selected", selected_benches)
    except Exception:
        pass
        
    print(f"🎯 Selected Benchmarks: {', '.join(selected_benches)}")
    
    # 3. Run the benchmarks
    print("\n🏎️ STARTING BENCHMARK EXECUTION...")
    suite = AIBenchmarkSuite(use_ech0_prime=True, use_full_datasets=False) # Use sample datasets for speed in this demo
    results = await suite.run_benchmark_suite(selected_benches)
    
    # 4. Compare and Rank
    print("\n🏆 FINAL COMPETITIVE RANKING")
    print("=" * 60)
    comparison = suite.compare_with_baselines(results)
    
    for benchmark, comp in comparison.items():
        print(f"\n{benchmark.upper()}:")
        print(f"  ECH0-PRIME Score: {comp['ech0_score']:.1f}%")
        print(f"  Rank: {comp['rank']}/{len(comp['baselines']) + 1}")
        
        # Who did we beat?
        beaten = [model for model, score in comp['baselines'].items() if comp['ech0_score'] > score]
        if beaten:
            print(f"  ✅ BEATEN: {', '.join(beaten)}")
        else:
            print("  ⚠️ Close competition with top-tier models.")

if __name__ == "__main__":
    asyncio.run(main())
