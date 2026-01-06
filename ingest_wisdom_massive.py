import asyncio
import os
import sys
import json
import time
from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime
import fitz  # PyMuPDF
from learning.compressed_knowledge_base import CompressedKnowledgeBase, KnowledgeNode
from learning.data_compressor import CompressedChunk, DataCompressor

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

class WisdomMassiveIngestor:
    """
    Streams data from the research_drop directory into the massive knowledge base.
    """
    def __init__(self, research_drop_path: str = "research_drop"):
        self.path = research_drop_path
        self.pdf_path = os.path.join(research_drop_path, "pdfs")
        self.json_path = os.path.join(research_drop_path, "json")

    async def stream_wisdom(self, max_files: int = 1000) -> AsyncGenerator[Dict[str, Any], None]:
        count = 0
        
        # 1. Stream JSONs
        if os.path.exists(self.json_path):
            print(f"Scanning JSONs in {self.json_path}...")
            for file in os.listdir(self.json_path):
                if count >= max_files: break
                if file.lower().endswith(".json"):
                    file_path = os.path.join(self.json_path, file)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            # Convert JSON to a searchable text string
                            content = json.dumps(data, indent=2)
                            yield {
                                "content": content,
                                "modality": "text",
                                "domain": "wisdom_json",
                                "metadata": {"source": file, "type": "research_data"}
                            }
                            count += 1
                    except Exception as e:
                        print(f"Error reading JSON {file}: {e}")

        # 2. Stream PDFs
        if os.path.exists(self.pdf_path):
            print(f"Scanning PDFs in {self.pdf_path}...")
            for file in os.listdir(self.pdf_path):
                if count >= max_files: break
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(self.pdf_path, file)
                    try:
                        doc = fitz.open(file_path)
                        full_text = ""
                        # Only take first 10 pages to avoid massive nodes if needed, 
                        # or process fully. For "massive pipeline" let's do fully but 
                        # chunk them in the loop.
                        for page in doc:
                            full_text += page.get_text()
                        doc.close()
                        
                        if full_text.strip():
                            yield {
                                "content": full_text,
                                "modality": "text",
                                "domain": "wisdom_pdf",
                                "metadata": {"source": file, "type": "research_paper"}
                            }
                            count += 1
                    except Exception as e:
                        print(f"Error reading PDF {file}: {e}")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest research_drop wisdom into ECH0-PRIME massive KB")
    parser.add_argument("--max", type=int, default=500, help="Max files to process in this run")
    args = parser.parse_args()

    print(f"🚀 ECH0-PRIME: Massive Wisdom Ingestion Sequence Initiated...")
    print(f"Target: {args.max} files from research_drop/")

    # Initialize KB
    kb_path = "./massive_kb"
    kb = CompressedKnowledgeBase(kb_path)
    await kb.load_async()
    
    ingestor = WisdomMassiveIngestor()
    data_stream = ingestor.stream_wisdom(max_files=args.max)
    
    start_time = time.time()
    processed_count = 0
    node_count = 0
    
    # Process the stream
    async for item in data_stream:
        try:
            content = item["content"]
            metadata = item["metadata"]
            domain = item["domain"]
            
            # Split large content into chunks of ~4000 characters (approx 1000 tokens)
            # consistent with the massive pipeline approach
            chunk_size = 4000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip(): continue
                
                # Add chunk_id to metadata
                chunk_meta = metadata.copy()
                chunk_meta["chunk_id"] = i
                
                # Create CompressedChunk (Fast mode for massive data)
                original_tokens = len(chunk_text.split())
                chunk = CompressedChunk(
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    compression_ratio=1.0,
                    quality_score=0.9,
                    timestamp=datetime.now(),
                    modality="text",
                    domain=domain,
                    compressed_content=chunk_text,
                    metadata=chunk_meta
                )
                
                await kb.add_compressed_chunk(chunk)
                node_count += 1
            
            processed_count += 1
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  [🧠] Processed {processed_count} files | Generated {node_count} nodes | {processed_count/elapsed:.2f} files/sec")
                
        except Exception as e:
            print(f"Error processing item: {e}")

    # Save results
    print("\nSaving massive knowledge base indices...")
    await kb.save_async()
    
    stats = kb.get_statistics()
    print("\n" + "="*50)
    print("✅ Massive Wisdom Ingestion Complete")
    print(f"Files Processed: {processed_count}")
    print(f"Total Nodes: {stats['total_nodes']}")
    print(f"Knowledge Domains: {list(stats['domains'].keys())}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())

