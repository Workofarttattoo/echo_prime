import asyncio
import os
import json
import time
from typing import Dict, List, Any, AsyncGenerator
from learning.compressed_knowledge_base import CompressedKnowledgeBase, MassiveDataIngestor, KnowledgeNode
from learning.data_compressor import StreamingDataProcessor, CompressedChunk
from reasoning.tools.massive_data_ingestor import MassiveDataStreamer
from datetime import datetime

class GooAQIngestor(MassiveDataStreamer):
    """
    Specialized ingestor for the GooAQ dataset.
    """
    
    async def stream_gooaq(self, split: str = "train", max_samples: int = 100000) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream data from GooAQ (sentence-transformers/gooaq).
        """
        try:
            from datasets import load_dataset
            
            print(f"Loading GooAQ dataset (sentence-transformers/gooaq) (split: {split})...")
            # We use streaming=True to avoid downloading the entire dataset at once
            dataset = load_dataset("sentence-transformers/gooaq", split=split, streaming=True)
            
            count = 0
            for example in dataset:
                if max_samples and count >= max_samples:
                    break
                
                question = example.get("question", "")
                answer = example.get("answer", "")
                
                if question and answer:
                    content = f"Question: {question}\nAnswer: {answer}"
                    yield {
                        "content": content,
                        "modality": "text",
                        "domain": "web_qa",
                        "metadata": {
                            "source": "huggingface",
                            "dataset": "sentence-transformers/gooaq",
                            "split": split,
                            "example_id": count
                        }
                    }
                    count += 1
                        
        except Exception as e:
            print(f"GooAQ streaming error: {e}")

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest GooAQ dataset into ECH0-PRIME")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to ingest")
    parser.add_argument("--fast", action="store_true", help="Skip LLM compression (much faster)")
    args = parser.parse_args()

    print(f"🚀 ECH0-PRIME: Ingesting GooAQ Dataset ({'FAST' if args.fast else 'INTELLIGENT'} mode)...")
    
    # Initialize components
    kb_path = "./massive_kb"
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
        
    kb = CompressedKnowledgeBase(kb_path)
    await kb.load_async()
    
    ingestor = MassiveDataIngestor(kb)
    gooaq_streamer = GooAQIngestor()
    
    max_samples = args.samples
    print(f"Target: {max_samples} samples from sentence-transformers/gooaq")
    
    data_stream = gooaq_streamer.stream_gooaq(max_samples=max_samples)
    
    start_time = time.time()
    processed_count = 0
    
    print("\nStarting ingestion loop...")
    
    async for item in data_stream:
        try:
            content = item.get("content", "")
            if not content.strip():
                continue

            if args.fast:
                # Fast mode: bypass LLM compression and create chunk manually
                original_tokens = len(content.split())
                chunk = CompressedChunk(
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    compression_ratio=1.0,
                    quality_score=0.8,
                    timestamp=datetime.now(),
                    modality="text",
                    domain="web_qa",
                    compressed_content=content,
                    metadata=item.get("metadata", {})
                )
            else:
                # Intelligent mode: use LLM compression (slow)
                chunk = await kb.compressor.compress_chunk(
                    content=content,
                    domain="web_qa",
                    modality="text",
                    metadata=item.get("metadata", {})
                )

            # Store in KB
            await kb.add_compressed_chunk(chunk)
            
            processed_count += 1
            if processed_count % 10 == 0 or args.fast and processed_count % 500 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / max(0.1, elapsed)
                remaining = (max_samples - processed_count) / max(0.01, rate)
                mode_str = "items/sec" if args.fast else "sec/item"
                display_rate = rate if args.fast else (1/rate if rate > 0 else 0)
                print(f"  [{'⚡' if args.fast else '🧠'}] {processed_count}/{max_samples} nodes | {display_rate:.2f} {mode_str} | ~{remaining/60:.1f} min left")

        except Exception as e:
            print(f"Error processing item: {e}")

    # Save results
    print("\nSaving compressed knowledge base...")
    await kb.save_async()
    
    kb_stats = kb.get_statistics()
    
    print("\n" + "="*50)
    print("✅ GooAQ Ingestion Complete")
    print(f"Processed: {processed_count} items")
    print(f"Time: {time.time() - start_time:.1f} seconds")
    print(f"Knowledge Base: {kb_stats['total_nodes']} total nodes")
    print("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Progress saved up to last checkpoint.")
