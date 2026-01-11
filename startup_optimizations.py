#!/usr/bin/env python3
"""
ECH0-PRIME Startup Optimizations
Automatically applies all persistent optimizations on system startup.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def initialize_persistent_optimizations():
    """Initialize all persistent optimizations on startup."""
    print("🚀 ECH0-PRIME: Initializing Persistent Optimizations")
    print("=" * 55)
    
    start_time = time.time()
    
    try:
        from persistent_optimizations import get_persistent_optimization_manager
        pom = get_persistent_optimization_manager()
        
        print("📊 Loading optimization state...")
        status = pom.get_status()
        
        print(f"  Cognitive systems: {sum(status['cognitive_activation'][k] for k in ['enhanced_reasoning', 'knowledge_integration', 'neuromorphic_processing'])}/3 active")
        print(f"  Memory optimization: {'Enabled' if status['memory_optimization']['consolidation_enabled'] else 'Disabled'}")
        print(f"  Domain strategies: {status['domain_strategies']['count']} loaded")
        print()
        
        # Apply all optimizations
        print("🔧 Applying optimizations...")
        results = pom.apply_persistent_optimizations()
        
        applied_count = sum(results.values())
        print(f"  ✅ {applied_count}/3 optimization categories applied")
        
        # If nothing was applied, force initialization
        if applied_count == 0:
            print("  ⚠️ No persistent state found, initializing defaults...")
            initialize_default_optimizations(pom)
            results = pom.apply_persistent_optimizations()
        
        # Final verification
        print("\n🔍 Verification:")
        final_status = pom.get_status()
        cognitive_ok = all(final_status['cognitive_activation'][k] 
                          for k in ['enhanced_reasoning', 'knowledge_integration', 'neuromorphic_processing'])
        memory_ok = final_status['memory_optimization']['consolidation_enabled']
        domains_ok = final_status['domain_strategies']['count'] > 0
        
        checks = [
            ("Cognitive Activation", cognitive_ok),
            ("Memory Optimization", memory_ok),
            ("Domain Strategies", domains_ok)
        ]
        
        all_ok = True
        for check_name, check_result in checks:
            icon = "✅" if check_result else "❌"
            print(f"  {icon} {check_name}")
            if not check_result:
                all_ok = False
        
        elapsed = time.time() - start_time
        
        if all_ok:
            print("
🎉 PERSISTENT OPTIMIZATIONS SUCCESSFULLY INITIALIZED!"            print(".2f"            print("   All optimizations will persist across sessions.")
            return True
        else:
            print(f"\n⚠️ Some optimizations failed to initialize (took {elapsed:.2f}s)")
            print("   Manual intervention may be required.")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed to initialize persistent optimizations: {e}")
        print(".2f")
        return False

def initialize_default_optimizations(pom):
    """Initialize default optimization state if none exists."""
    try:
        # Set default cognitive state
        default_cognitive = {
            "enhanced_reasoning": True,
            "knowledge_integration": True,
            "neuromorphic_processing": True
        }
        pom.save_cognitive_state(default_cognitive)
        
        # Set default memory state
        pom.save_memory_state(episodic_count=0, semantic_count=0)
        
        # Set default domain strategies
        domain_strategies = {
            "mathematics": {
                "prompt_template": "Solve this step-by-step, showing all work: {question}",
                "reasoning_depth": 4,
                "temperature": 0.1
            },
            "logic": {
                "prompt_template": "Analyze this logical problem systematically: {question}",
                "reasoning_depth": 3,
                "temperature": 0.2
            },
            "science": {
                "prompt_template": "Apply scientific reasoning to: {question}",
                "reasoning_depth": 4,
                "temperature": 0.15
            },
            "general": {
                "prompt_template": "Reason through this problem: {question}",
                "reasoning_depth": 2,
                "temperature": 0.3
            }
        }
        pom.save_domain_state(domain_strategies)
        
        print("  ✅ Default optimization state initialized")
        
    except Exception as e:
        print(f"  ❌ Failed to initialize defaults: {e}")

def main():
    """Main startup optimization function."""
    success = initialize_persistent_optimizations()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
