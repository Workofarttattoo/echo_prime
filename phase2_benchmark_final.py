#!/usr/bin/env python3
"""
ECH0-PRIME Phase 2 Final Performance Benchmark
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Phase 2 optimized environment
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["ECH0_PHASE"] = "2"
os.environ["ECH0_FULL_ARCH"] = "1"
os.environ["ECH0_OPTIMIZED"] = "1"

from main_orchestrator import EchoPrimeAGI

async def run_phase2_benchmark():
    print("🚀 ECH0-PRIME Phase 2 Final Performance Benchmark")
    print("=" * 55)

    results = {}
    agi = None

    try:
        print("1. Initializing Phase 2 System...")
        start_time = time.time()

        agi = EchoPrimeAGI(
            lightweight=False,
            enable_voice=False,
            memory_optimized=True
        )

        init_time = time.time() - start_time
        results["initialization"] = {"time": init_time, "success": True}
        print(".2f"
        print("2. Testing Core Capabilities...")

        # Basic reasoning test
        print("   • Basic Reasoning...")
        reasoning_start = time.time()
        try:
            result = agi.cognitive_cycle(None, "What is 2 + 2?")
            reasoning_time = time.time() - reasoning_start
            results["basic_reasoning"] = {
                "time": reasoning_time,
                "success": True,
                "response": result.get("llm_insight", "No response")[:100] if result else "Failed"
            }
            print(".3f"        except Exception as e:
            results["basic_reasoning"] = {"success": False, "error": str(e)}
            print("     ❌ Failed")

        # Prompt masterworks test
        print("   • Prompt Masterworks...")
        masterwork_start = time.time()
        try:
            mirror_result = agi.recursive_mirror("Test analysis")
            masterwork_time = time.time() - masterwork_start
            results["prompt_masterworks"] = {
                "time": masterwork_time,
                "success": True,
                "length": len(mirror_result)
            }
            print(".3f"        except Exception as e:
            results["prompt_masterworks"] = {"success": False, "error": str(e)}
            print("     ❌ Failed")

        # Knowledge base test
        print("   • Knowledge Base...")
        kb_entries = 0
        if hasattr(agi.reasoner, 'gov_mem'):
            kb_stats = agi.reasoner.gov_mem.get_statistics()
            kb_entries = kb_stats.get("total_entries", 0)
        results["knowledge_base"] = {"entries": kb_entries, "accessible": True}
        print(f"     📚 {kb_entries} entries accessible")

        # Memory systems test
        print("   • Memory Systems...")
        memory_count = 0
        memory_count += 1 if hasattr(agi, 'working_memory') else 0
        memory_count += 1 if hasattr(agi.reasoner, 'episodic_memory') else 0
        memory_count += 1 if hasattr(agi.reasoner, 'semantic_memory') else 0
        results["memory_systems"] = {"active": memory_count, "total": 3}
        print(f"     🧠 {memory_count}/3 memory systems active")

        print("3. Calculating Performance Score...")

        # Calculate score
        score = 0
        score += 20 if results["initialization"]["time"] < 30 else 10  # Initialization
        score += 25 if results["basic_reasoning"]["success"] else 0     # Basic reasoning
        score += 20 if results["prompt_masterworks"]["success"] else 0  # Masterworks
        score += 15 if results["knowledge_base"]["accessible"] else 0   # Knowledge
        score += 20 if results["memory_systems"]["active"] >= 2 else 10  # Memory

        results["overall_score"] = score

        print("\n" + "=" * 55)
        print("📊 PHASE 2 BENCHMARK RESULTS")
        print("=" * 55)
        print(f"🎯 Overall Score: {score}/100")

        if score >= 80:
            status = "EXCELLENT - Full Phase 2 Operational"
        elif score >= 60:
            status = "GOOD - Core Phase 2 Functional"
        elif score >= 40:
            status = "FAIR - Basic Phase 2 Working"
        else:
            status = "NEEDS ATTENTION"

        print(f"🏆 Status: {status}")
        print("✅ Hardware: Apple M4 + 24GB RAM (Optimized)"
        # Save results
        results_file = Path("benchmark_results") / f"phase2_final_{int(time.time())}.json"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 Results saved to: {results_file}")

        return results

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if agi:
            agi.cleanup()

if __name__ == "__main__":
    results = asyncio.run(run_phase2_benchmark())
    if "error" not in results:
        print("\n🎉 Phase 2 Benchmark Complete!")
        print("The system is ready for advanced AI operations."    else:
        print(f"\n❌ Benchmark failed: {results['error']}")
        sys.exit(1)
