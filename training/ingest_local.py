import os
import argparse
from ech0_governance.persistent_memory import PersistentMemory
from memory.manager import MemoryManager

def ingest_directory(directory: str):
    """
    Scans directory for .txt, .md, .pdf files and stores them in ECH0 memory.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    # Initialize Memory
    mem_manager = MemoryManager()
    governance_mem = PersistentMemory(mem_manager)
    
    print(f"Starting ingestion from: {directory}")
    
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.txt', '.md', '.markdown')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    # Simple chunking (e.g. by paragraphs)
                    chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
                    
                    for chunk in chunks:
                        # Store in memory
                        governance_mem.store(f"Source: {file} | Content: {chunk}")
                        count += 1
                        
                    print(f"Processed {file}: {len(chunks)} chunks.")
                    
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
                    
    # Save State
    mem_manager.episodic.save(mem_manager.episodic_path)
    print(f"Ingestion Complete. Stored {count} chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest local documents into ECH0 memory.")
    parser.add_argument("--dir", required=True, help="Directory to scan")
    args = parser.parse_args()
    
    ingest_directory(args.dir)
