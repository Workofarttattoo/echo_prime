#!/usr/bin/env python3
"""
Apply Persistent Optimizations
Ensures all optimizations persist across sessions and are automatically restored.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🔄 Applying Persistent Optimizations")
    print("=" * 40)
    
    from persistent_optimizations import get_persistent_optimization_manager
    pom = get_persistent_optimization_manager()
    
    print("📊 Current State Before Application:")
    status = pom.get_status()
    print(f"  Cognitive: {status['cognitive_activation']['enhanced_reasoning']}")
    print(f"  Memory: {status['memory_optimization']['consolidation_enabled']}")
    print(f"  Domains: {status['domain_strategies']['count']}")
    print()
    
    # Apply all persistent optimizations
    print("🚀 Applying Optimizations...")
    results = pom.apply_persistent_optimizations()
    
    print("\n📋 Application Results:")
    for opt, applied in results.items():
        icon = "✅" if applied else "❌"
        print(f"  {icon} {opt.replace('_', ' ').title()}")
    print()
    
    # Force enable optimizations if not already active
    force_enable_optimizations(pom)
    
    # Create checkpoint
    checkpoint_name = pom.create_checkpoint("post_application")
    
    # Final validation
    print("🔍 Final Validation:")
    final_status = pom.get_status()
    cognitive_active = all(final_status['cognitive_activation'][k] 
                          for k in ['enhanced_reasoning', 'knowledge_integration', 'neuromorphic_processing'])
    memory_active = final_status['memory_optimization']['consolidation_enabled']
    domains_active = final_status['domain_strategies']['count'] > 0
    
    print(f"  ✅ Cognitive Systems: {'Active' if cognitive_active else 'Inactive'}")
    print(f"  ✅ Memory Optimization: {'Active' if memory_active else 'Inactive'}")
    print(f"  ✅ Domain Strategies: {'Active' if domains_active else 'Inactive'}")
    print()
    
    if cognitive_active and memory_active and domains_active:
        print("🎉 ALL PERSISTENT OPTIMIZATIONS SUCCESSFULLY APPLIED!")
        print("   Optimizations will now persist across sessions.")
        print(f"   Checkpoint saved: {checkpoint_name}")
        return True
    else:
        print("⚠️ Some optimizations may need manual intervention.")
        return False

def force_enable_optimizations(pom):
    """Force enable all optimizations if not already active."""
    print("🔧 Forcing Optimization Activation...")
    
    # Force cognitive activation
    try:
        from cognitive_activation import get_cognitive_activation_system
        cas = get_cognitive_activation_system()
        
        # Activate all systems
        cas.activate_enhanced_reasoning()
        cas.activate_knowledge_integration()
        cas.activate_neuromorphic_processing()
        
        # Get current state
        status = cas.get_status()
        
        # Save to persistent manager
        pom.save_cognitive_state(status)
        
        print("  ✅ Cognitive activation forced and saved")
        
    except Exception as e:
        print(f"  ❌ Cognitive activation failed: {e}")
    
    # Force memory optimization
    try:
        from memory.manager import MemoryManager
        mm = MemoryManager()
        
        # Apply optimizations
        mm.consolidate_now()
        mm.compress_memory(ratio=0.2)
        
        # Get memory stats
        episodic_count = len(mm.episodic.storage) if hasattr(mm.episodic, 'storage') else 0
        semantic_count = len(mm.semantic.knowledge_base) if hasattr(mm.semantic, 'knowledge_base') else 0
        
        # Save to persistent manager
        pom.save_memory_state(episodic_count, semantic_count)
        
        # Save memory state to disk
        mm.episodic.save(mm.episodic_path)
        mm.semantic.save(mm.semantic_path)
        
        print(f"  ✅ Memory optimization forced and saved ({episodic_count} episodic, {semantic_count} semantic)")
        
    except Exception as e:
        print(f"  ❌ Memory optimization failed: {e}")
    
    # Ensure domain strategies exist
    if not pom.domain_state:
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
        print("  ✅ Domain strategies created and saved")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
