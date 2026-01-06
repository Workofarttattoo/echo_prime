import asyncio
import os
import fitz  # PyMuPDF
from typing import List, Dict, Any
from datetime import datetime
from learning.compressed_knowledge_base import CompressedKnowledgeBase, KnowledgeNode
from learning.data_compressor import CompressedChunk
from reasoning.tools.pdf_ingestor import PDFIngestor

async def ingest_to_compressed_kb(file_path: str, kb_path: str = "./massive_kb"):
    """
    Extracts text from PDF and ingests into CompressedKnowledgeBase.
    """
    print(f"🚀 ECH0-PRIME: Ingesting '{os.path.basename(file_path)}' into Compressed Knowledge Base...")
    
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)
        
    kb = CompressedKnowledgeBase(kb_path)
    await kb.load_async()
    
    # 1. Extract Text
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    if not full_text.strip():
        print("ERROR: No text extracted from PDF.")
        return

    # 2. Split and Compress (using FAST mode logic like in GooAQ for speed if many pages)
    # Actually, for a manual, intelligent compression might be better, but let's start with chunks.
    sentences = full_text.split('.')
    chunk_size = 4000
    current_chunk_text = ""
    processed_count = 0
    
    for sentence in sentences:
        if len(current_chunk_text + sentence) > chunk_size and current_chunk_text:
            # Create chunk
            original_tokens = len(current_chunk_text.split())
            chunk = CompressedChunk(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                quality_score=0.9,
                timestamp=datetime.now(),
                modality="text",
                domain="technical_manual",
                compressed_content=current_chunk_text,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "field_manual",
                    "chunk_id": processed_count
                }
            )
            await kb.add_compressed_chunk(chunk)
            processed_count += 1
            current_chunk_text = sentence + "."
        else:
            current_chunk_text += sentence + "."
            
    if current_chunk_text:
        original_tokens = len(current_chunk_text.split())
        chunk = CompressedChunk(
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            quality_score=0.9,
            timestamp=datetime.now(),
            modality="text",
            domain="technical_manual",
            compressed_content=current_chunk_text,
            metadata={
                "source": os.path.basename(file_path),
                "type": "field_manual",
                "chunk_id": processed_count
            }
        )
        await kb.add_compressed_chunk(chunk)
        processed_count += 1

    print(f"✅ Ingested {processed_count} chunks into Compressed KB.")
    await kb.save_async()

def ingest_to_pinecone(file_path: str):
    """
    Uses PDFIngestor to add to Pinecone (Deep Memory).
    """
    print(f"🚀 ECH0-PRIME: Ingesting '{os.path.basename(file_path)}' into Pinecone Deep Memory...")
    ingestor = PDFIngestor()
    result = ingestor.ingest(file_path)
    print(result)

async def main():
    pdf_path = "/Users/noone/Desktop/untitled folder 2/ai LLM/Operator‑bound Private Llm — Field Manual (v1.pdf"
    
    # Run both ingestions
    await ingest_to_compressed_kb(pdf_path)
    ingest_to_pinecone(pdf_path)
    
    print("\n" + "="*50)
    print("✅ Private LLM Field Manual Ingestion Sequence Complete")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())

