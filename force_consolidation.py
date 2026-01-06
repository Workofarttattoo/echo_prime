import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.manager import MemoryManager

def force_consolidation():
    print("🧠 Initializing Memory Manager...")
    # Use the venv and same directory
    mm = MemoryManager(data_dir="memory_data")
    
    print(f"📊 Current state:")
    print(f"  Episodic entries: {len(mm.episodic.storage)}")
    print(f"  Semantic concepts: {len(mm.semantic.knowledge_base)}")
    
    print("🔄 Forcing consolidation cycle...")
    mm.consolidate_now()
    
    print("💾 Saving updated state...")
    mm.episodic.save(mm.episodic_path)
    mm.semantic.save(mm.semantic_path)
    
    print("✅ Consolidation complete and saved.")

if __name__ == "__main__":
    force_consolidation()

