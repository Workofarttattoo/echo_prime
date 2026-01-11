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
        
        cognitive_count = sum(status['cognitive_activation'][k] 
                             for k in ['enhanced_reasoning', 'knowledge_integration', 'neuromorphic_processing'])
        print(f"  Cognitive systems: {cognitive_count}/3 active")
        print(f"  Memory optimization: {'Enabled' if status['memory_optimization']['consolidation_enabled'] else 'Disabled'}")
        print(f"  Domain strategies: {status['domain_strategies']['count']} loaded")
        print()
        
        # Apply all optimizations
        print("🔧 Applying optimizations...")
        results = pom.apply_persistent_optimizations()
        
        applied_count = sum(results.values())
        print(f"  ✅ {applied_count}/3 optimization categories applied")
        
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
            print("\n🎉 PERSISTENT OPTIMIZATIONS SUCCESSFULLY INITIALIZED!")
            print(".2f")
            print("   All optimizations will persist across sessions.")
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

def main():
    """Main startup optimization function."""
    success = initialize_persistent_optimizations()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
