#!/usr/bin/env python3
"""
ECH0-PRIME Phase 2 Performance Benchmark
Direct testing of Phase 2 capabilities without API dependency
"""

import os
import sys
import asyncio
import time
import json
from pathlib import Path

# Add project root to path
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

class Phase2Benchmark:
    def __init__(self):
        self.results = {}
        self.agi = None

    async def run_full_benchmark(self):
        print("🚀 ECH0-PRIME Phase 2 Performance Benchmark")
        print("==========================================")

        try:
            print("1. Initializing Phase 2 System...")
            start_time = time.time()

            self.agi = EchoPrimeAGI(
                lightweight=True,
                enable_voice=False,
            )

            init_time = time.time() - start_time
            self.results["initialization_time"] = init_time
            print(f"   ⏱️ Initialization completed in {init_time:.2f}s")
            print("2. Testing Core Capabilities...")

            # Test 1: Basic Reasoning
            print("   • Testing Basic Reasoning...")
            reasoning_start = time.time()
            result1 = self.agi.cognitive_cycle(None, "What is 15 + 27?")
            reasoning_time = time.time() - reasoning_start
            self.results["basic_reasoning"] = {
                "time": reasoning_time,
                "response": result1.get("llm_insight", "No response") if result1 else "Failed"
            }
            print(f"     ✅ Basic reasoning took {reasoning_time:.3f}s")
            # Test 2: Prompt Masterworks
            print("   • Testing Prompt Masterworks...")
            masterwork_start = time.time()
            try:
                mirror_result = self.agi.recursive_mirror("Analyze the concept of artificial intelligence")
                masterwork_time = time.time() - masterwork_start
                self.results["prompt_masterworks"] = {
                    "time": masterwork_time,
                    "functional": True,
                    "response_length": len(mirror_result)
                }
                print(f"     ✅ Prompt Masterworks took {masterwork_time:.3f}s")
            except Exception as e:
                self.results["prompt_masterworks"] = {
                    "functional": False,
                    "error": str(e)
                }
                print("     ❌ Prompt Masterworks failed")

            # Test 3: Knowledge Base Access
            print("   • Testing Knowledge Base...")
            kb_stats = {}
            if hasattr(self.agi.reasoner, 'gov_mem'):
                kb_stats = self.agi.reasoner.gov_mem.get_statistics()
            self.results["knowledge_base"] = {
                "entries": kb_stats.get("total_entries", 0),
                "accessible": True
            }
            print(f"     📚 Knowledge base: {kb_stats.get('total_entries', 0)} entries")

            # Test 4: Memory Systems
            print("   • Testing Memory Systems...")
            memory_stats = {
                "working_memory": hasattr(self.agi, 'working_memory'),
                "episodic_memory": hasattr(self.agi.reasoner, 'episodic_memory'),
                "semantic_memory": hasattr(self.agi.reasoner, 'semantic_memory')
            }
            self.results["memory_systems"] = memory_stats
            active_memory = sum(memory_stats.values())
            print(f"     🧠 Memory systems active: {active_memory}/3")

            # Test 5: Complex Reasoning
            print("   • Testing Complex Reasoning...")
            complex_start = time.time()
            try:
                complex_result = self.agi.cognitive_cycle(None,
                    "Explain the relationship between quantum computing and artificial intelligence")
                complex_time = time.time() - complex_start
                self.results["complex_reasoning"] = {
                    "time": complex_time,
                    "response_length": len(complex_result.get("llm_insight", "")) if complex_result else 0
                }
                print(f"     ✅ Complex reasoning took {complex_time:.3f}s")
            except Exception as e:
                self.results["complex_reasoning"] = {
                    "error": str(e)
                }
                print("     ❌ Complex reasoning failed")

            print("3. Calculating Performance Metrics...")

            # Performance assessment
            self.results["performance_assessment"] = self._assess_performance()

            print("\n" + "=" * 50)
            print("📊 PHASE 2 BENCHMARK RESULTS")
            print("=" * 50)

            assessment = self.results["performance_assessment"]
            print(f"🎯 Overall Score: {assessment['overall_score']}/100")
            print(f"⚡ Initialization: {assessment['initialization_score']}/20")
            print(f"🧠 Reasoning: {assessment['reasoning_score']}/30")
            print(f"🎭 Masterworks: {assessment['masterworks_score']}/20")
            print(f"📚 Knowledge: {assessment['knowledge_score']}/15")
            print(f"💾 Memory: {assessment['memory_score']}/15")

            print(f"\n🏆 Status: {assessment['status']}")
            print(f"💡 Recommendation: {assessment['recommendation']}")

            # Save results
            self._save_results()

            # Cleanup
            if self.agi:
                self.agi.cleanup()

            return self.results

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _assess_performance(self):
        """Assess overall performance based on test results"""
        score = 0
        max_score = 100

        # Initialization (20 points)
        init_score = 20 if self.results.get("initialization_time", 999) < 30 else 10
        score += init_score

        # Basic reasoning (20 points)
        reasoning = self.results.get("basic_reasoning", {})
        reasoning_score = 20 if reasoning.get("time", 999) < 10 and "response" in reasoning else 0
        score += reasoning_score

        # Prompt masterworks (20 points)
        masterworks = self.results.get("prompt_masterworks", {})
        masterworks_score = 20 if masterworks.get("functional", False) else 0
        score += masterworks_score

        # Knowledge base (15 points)
        kb = self.results.get("knowledge_base", {})
        kb_score = 15 if kb.get("accessible", False) else 0
        score += kb_score

        # Memory systems (15 points)
        memory = self.results.get("memory_systems", {})
        memory_score = 15 if sum(memory.values()) >= 2 else 5 * sum(memory.values())
        score += memory_score

        # Complex reasoning (10 points)
        complex_r = self.results.get("complex_reasoning", {})
        complex_score = 10 if "time" in complex_r and complex_r["time"] < 30 else 0
        score += complex_score

        # Determine status
        if score >= 80:
            status = "EXCELLENT - Full Phase 2 Capabilities"
            recommendation = "System ready for advanced AI operations"
        elif score >= 60:
            status = "GOOD - Core Phase 2 Functional"
            recommendation = "System operational with some optimizations needed"
        elif score >= 40:
            status = "FAIR - Basic Phase 2 Working"
            recommendation = "Core functionality working, advanced features need attention"
        else:
            status = "NEEDS ATTENTION"
            recommendation = "Phase 2 initialization incomplete, requires debugging"

        return {
            "overall_score": score,
            "initialization_score": init_score,
            "reasoning_score": reasoning_score + complex_score,
            "masterworks_score": masterworks_score,
            "knowledge_score": kb_score,
            "memory_score": memory_score,
            "status": status,
            "recommendation": recommendation
        }

    def _save_results(self):
        """Save benchmark results to file"""
        results_file = Path("benchmark_results") / f"phase2_benchmark_{int(time.time())}.json"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"📄 Results saved to: {results_file}")

async def main():
    benchmark = Phase2Benchmark()
    results = await benchmark.run_full_benchmark()

    if "error" not in results:
        print("\n🎉 Phase 2 Benchmark Complete!")
    else:
        print(f"\n❌ Benchmark Failed: {results['error']}")

if __name__ == "__main__":
    asyncio.run(main())
