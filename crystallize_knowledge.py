#!/usr/bin/env python3
"""
ECH0-PRIME Knowledge Crystallization Script
Uses Advanced Prompt Engineering Masterworks to compress and include knowledge.
"""

import os
import asyncio
import json
import sys  # pyright: ignore[reportUnusedImport]
import logging
from datetime import datetime
from capabilities.prompt_masterworks import PromptMasterworks
from learning.compressed_knowledge_base import CompressedKnowledgeBase, KnowledgeNode  # pyright: ignore[reportUnusedImport]
from learning.data_compressor import CompressedChunk
from reasoning.llm_bridge import OllamaBridge

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crystallization.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

async def crystallize_knowledge(sample_limit=20):  # pyright: ignore[reportMissingParameterType]
    logger.info("💎 Starting Knowledge Crystallization...")
    
    pm = PromptMasterworks()
    kb = CompressedKnowledgeBase("./compressed_kb")
    llm = OllamaBridge(model="llama3.2")
    
    # Ensure KB is loaded
    await kb.load_async()
    
    source_dir = "research_drop/json"
    if not os.path.exists(source_dir):
        logger.error(f"Source directory {source_dir} not found.")
        return

    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} research files. Processing first {sample_limit}...")

    stats = {
        "processed": 0,
        "crystallized": 0,
        "errors": 0,
        "nodes": []
    }

    for i, filename in enumerate(json_files[:sample_limit]):
        file_path = os.path.join(source_dir, filename)
        logger.info(f"Processing: {filename} ({i+1}/{sample_limit})")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)  # pyright: ignore[reportAny]
                content = json.dumps(data, indent=2)
            
            # Use Recursive Compression (Masterwork 9)
            compression_prompt = pm.recursive_compression(content[:4000])  # pyright: ignore[reportAttributeAccessIssue]
            
            # Use LLM Bridge
            compressed_content = await asyncio.get_event_loop().run_in_executor(
                None, llm.query, compression_prompt
            )
            
            if "BRIDGE ERROR" in compressed_content:
                logger.warning(f"LLM Bridge Error for {filename}, skipping.")
                stats["errors"] += 1
                continue

            # Create the compressed chunk
            chunk = CompressedChunk(
                original_tokens=len(content.split()),
                compressed_tokens=len(compressed_content.split()),
                compression_ratio=len(compressed_content.split()) / max(1, len(content.split())),
                quality_score=0.95,
                timestamp=datetime.now(),
                modality="text",
                domain="crystallized_research",
                compressed_content=compressed_content,
                metadata={
                    "source_file": filename,
                    "crystallization_protocol": "Masterwork_9_Recursive_Compression",
                    "original_size": len(content)
                }
            )
            
            # Add to KB
            node_id = await kb.add_compressed_chunk(chunk)
            logger.info(f"Successfully crystallized {filename} into node: {node_id}")
            stats["crystallized"] += 1
            stats["nodes"].append(node_id)
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            stats["errors"] += 1

        stats["processed"] += 1

    # Save the updated KB
    await kb.save_async()
    
    # Final report
    report_path = "crystallization_report.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Crystallization complete. Report saved to {report_path}")
    print(f"✅ Crystallization complete. Processed {stats['processed']} files. See crystallization_report.json for details.")

if __name__ == "__main__":
    asyncio.run(crystallize_knowledge())
