import os
import sys
import time
import shutil
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.manager import MemoryManager
from wisdom_processor import WisdomProcessor

def run_integration():
    print("🚀 ECH0-PRIME FULL WISDOM INTEGRATION (LEVEL 12)")
    print("="*60)
    
    # 1. INGESTION PHASE
    src = "/Volumes/3NCRYPT3D_V4ULT"
    dst_pdf = "research_drop/pdfs"
    dst_json = "research_drop/json"
    
    os.makedirs(dst_pdf, exist_ok=True)
    os.makedirs(dst_json, exist_ok=True)
    
    print(f"📥 Scanning vault: {src}...")
    
    ingested = 0
    # Walk through the entire vault
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.lower().endswith((".pdf", ".json")):
                try:
                    target_dir = dst_pdf if file.lower().endswith(".pdf") else dst_json
                    shutil.copy2(os.path.join(root, file), os.path.join(target_dir, file))
                    ingested += 1
                    if ingested % 100 == 0:
                        print(f"  ...ingested {ingested} files")
                except Exception as e:
                    pass
    
    print(f"✅ Ingestion complete. {ingested} files moved to research_drop.")
    
    # 2. PROCESSING PHASE
    print("\n🧠 Beginning cognitive integration...")
    processor = WisdomProcessor()
    wisdom_files = processor.get_wisdom_files()
    
    # Process JSONs
    for file_path in wisdom_files['jsons']:
        processor.process_json_wisdom(file_path)
        processor.processed_count += 1
        
    # Process PDFs
    for file_path in wisdom_files['pdfs']:
        processor.process_pdf_wisdom(file_path)
        processor.processed_count += 1
        
    # 3. CONSOLIDATION PHASE
    print("\n💾 Committing to persistent knowledge lattice...")
    processor.consolidate_memories()
    
    print("\n🎉 INTEGRATION SUCCESSFUL.")
    print(f"  Episodic Memories: {len(processor.memory.episodic.storage)}")
    print(f"  Semantic Concepts: {len(processor.memory.semantic.knowledge_base)}")

if __name__ == "__main__":
    run_integration()

