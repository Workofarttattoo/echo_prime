#!/usr/bin/env python3
"""
Ensure Optimizations Are Active
Quick script to verify and activate all persistent optimizations.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🔍 Checking ECH0-PRIME Optimization Status")
    print("=" * 45)
    
    try:
        # Test cognitive activation
        from cognitive_activation import get_cognitive_activation_system
        cas = get_cognitive_activation_system()
        cognitive_status = cas.get_status()
        cognitive_active = all(cognitive_status.values())
        
        # Test memory system
        from memory.manager import MemoryManager
        mm = MemoryManager()
        episodic_count = len(mm.episodic.storage) if hasattr(mm.episodic, 'storage') else 0
        semantic_count = len(mm.semantic.knowledge_base) if hasattr(mm.semantic, 'knowledge_base') else 0
        
        # Test domain strategies
        from persistent_optimizations import get_persistent_optimization_manager
        pom = get_persistent_optimization_manager()
        domains_loaded = len(pom.domain_state) > 0
        
        print("📊 System Status:")
        print(f"  🧠 Cognitive: {'✅ ACTIVE' if cognitive_active else '❌ INACTIVE'}")
        print(f"  💾 Memory: {'✅ ACTIVE' if episodic_count + semantic_count > 0 else '⚠️  EMPTY'} ({episodic_count} episodic, {semantic_count} semantic)")
        print(f"  🎯 Domains: {'✅ LOADED' if domains_loaded else '❌ MISSING'} ({len(pom.domain_state)} strategies)")
        
        all_good = cognitive_active and domains_loaded
        
        if all_good:
            print("\n🎉 ALL OPTIMIZATIONS ARE ACTIVE AND PERSISTENT!")
            return True
        else:
            print("\n⚠️ Some optimizations need attention. Run:")
            print("  ./venv/bin/python3 apply_persistent_optimizations.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking optimizations: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
