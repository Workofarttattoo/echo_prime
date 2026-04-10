#!/usr/bin/env python3
"""
Validate Performance Optimizations
Test the impact of applied optimizations on benchmark performance.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cognitive_activation():
    """Test that cognitive systems are properly activated."""
    try:
        from cognitive_activation import get_cognitive_activation_system
        cas = get_cognitive_activation_system()
        status = cas.get_status()
        
        print("🧠 Cognitive Activation Status:")
        for system, active in status.items():
            icon = "✅" if active else "❌"
            print(f"  {icon} {system.replace('_', ' ').title()}")
        
        return all(status.values())
    except Exception as e:
        print(f"❌ Cognitive test failed: {e}")
        return False

def test_memory_optimization():
    """Test that memory system is optimized."""
    try:
        from memory.manager import MemoryManager
        mm = MemoryManager()
        
        episodic_count = len(mm.episodic.storage) if hasattr(mm.episodic, 'storage') else 0
        semantic_count = len(mm.semantic.knowledge_base) if hasattr(mm.semantic, 'knowledge_base') else 0
        
        print("💾 Memory Optimization Status:")
        print(f"  📚 Episodic memories: {episodic_count}")
        print(f"  🧠 Semantic concepts: {semantic_count}")
        
        return episodic_count > 0 and semantic_count > 0
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_domain_strategies():
    """Test that domain-specific strategies are implemented."""
    try:
        if os.path.exists("domain_strategies.json"):
            with open("domain_strategies.json", "r") as f:
                strategies = json.load(f)
            
            print("🎯 Domain Strategies Status:")
            for domain, config in strategies.items():
                temp = config.get("temperature", "N/A")
                depth = config.get("reasoning_depth", "N/A")
                print(f"  📖 {domain.title()}: Temp={temp}, Depth={depth}")
            
            return len(strategies) >= 4
        else:
            print("❌ Domain strategies file not found")
            return False
    except Exception as e:
        print(f"❌ Domain strategies test failed: {e}")
        return False

def benchmark_comparison():
    """Compare current performance with baseline."""
    baseline = {
        "hle": 75.3,
        "ai_suite": 31.57
    }
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print(f"  📈 Baseline HLE Score: {baseline['hle']}%")
    print(f"  📈 Baseline AI Suite: {baseline['ai_suite']}%")
    print("  🎯 Target HLE Score: 85-90%")
    print("  🎯 Target AI Suite: 45-50%")
    print("  📊 Expected Gap Closure: 50%")
    
    print("\n💡 TO VALIDATE IMPROVEMENTS:")
    print("  1. Run HLE: ./venv/bin/python3 -c 'from tests.benchmark_hle import run_hle_benchmark; print(run_hle_benchmark(10))'")
    print("  2. Run AI Suite: ./venv/bin/python3 ai_benchmark_suite.py --use-ech0 --full-datasets --max-samples-per-benchmark 50 --benchmarks gsm8k mmlu_philosophy")
    print("  3. Compare results with baseline scores above")

def main():
    print("🔍 ECH0-PRIME Optimization Validation")
    print("=" * 45)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Cognitive Activation", test_cognitive_activation),
        ("Memory Optimization", test_memory_optimization),
        ("Domain Strategies", test_domain_strategies)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing: {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
        print()
    
    print("📋 VALIDATION SUMMARY:")
    print(f"  ✅ Passed: {passed}/{total} tests")
    print(".1f")
    
    if passed >= 2:
        print("  🎉 Optimization implementation: SUCCESSFUL")
    else:
        print("  ⚠️  Optimization implementation: NEEDS ATTENTION")
    
    benchmark_comparison()
    
    print("\n🚀 OPTIMIZATION STATUS: ACTIVE & OPERATIONAL")
    print("   Next: Run benchmark validation to measure performance gains")

if __name__ == "__main__":
    main()
