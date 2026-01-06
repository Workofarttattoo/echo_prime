"""
ECH0-PRIME Massive Data Ingestor Tool
Streams data through ECH0 for compression and storage in compressed knowledge base.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import os
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import aiohttp
import aiofiles
from learning.compressed_knowledge_base import CompressedKnowledgeBase, MassiveDataIngestor
from learning.data_compressor import StreamingDataProcessor
from mcp_server.registry import ToolRegistry


@ToolRegistry.register(name="ingest_massive_dataset")
def ingest_massive_dataset(data_source: str, domain: str = "web", batch_size: int = 100) -> str:
    """
    Ingest a massive dataset through ECH0's compression pipeline.

    Args:
        data_source: URL, file path, or API endpoint to ingest from
        domain: Knowledge domain (academic, web, technical, etc.)
        batch_size: Number of items to process simultaneously

    Returns:
        Status message with ingestion statistics
    """
    try:
        # Initialize knowledge base
        kb = CompressedKnowledgeBase("./massive_kb")
        ingestor = MassiveDataIngestor(kb)

        async def run_ingestion():
            # Load existing knowledge
            await kb.load_async()

            # Determine data source type and create appropriate stream
            if data_source.startswith("http"):
                # API or web source
                processor = StreamingDataProcessor()
                data_stream = processor.stream_from_api(data_source, domain=domain)
            elif os.path.isfile(data_source):
                # Local file
                processor = StreamingDataProcessor()
                data_stream = processor.stream_from_file(data_source, domain=domain)
            else:
                return f"Unsupported data source: {data_source}"

            # Ingest the data
            await ingestor.ingest_data_stream(data_stream, domain=domain, batch_size=batch_size)

            # Save the knowledge base
            await kb.save_async()

            # Return statistics
            stats = ingestor.get_ingestion_stats()
            kb_stats = kb.get_statistics()

            return f"""Massive dataset ingestion completed:
- Processed: {stats['total_processed']} items
- Stored: {stats['total_stored']} compressed knowledge nodes
- Errors: {stats['errors']}
- Duration: {stats.get('duration_seconds', 0):.1f} seconds
- Processing rate: {stats.get('processing_rate', 0):.1f} items/sec
- Knowledge base: {kb_stats['total_nodes']} total nodes, {kb_stats['storage_efficiency']}x compression ratio"""

        # Run the async ingestion
        result = asyncio.run(run_ingestion())
        return result

    except Exception as e:
        return f"Ingestion failed: {str(e)}"


@ToolRegistry.register(name="query_compressed_knowledge")
def query_compressed_knowledge(query: str, domain: str = None, limit: int = 5) -> str:
    """
    Query the compressed knowledge base for relevant information.

    Args:
        query: Search query
        domain: Optional domain filter
        limit: Maximum number of results

    Returns:
        Formatted knowledge results
    """
    try:
        async def run_query():
            kb = CompressedKnowledgeBase("./massive_kb")
            await kb.load_async()

            results = await kb.retrieve_knowledge(query, domain=domain, limit=limit)

            if not results:
                return f"No knowledge found for query: {query}"

            formatted_results = []
            for i, node in enumerate(results, 1):
                formatted_results.append(f"""{i}. [{node.domain.upper()}] Quality: {node.quality_score:.2f}
   {node.compressed_content}
   (Accessed {node.access_count} times, {len(node.connections)} connections)""")

            stats = kb.get_statistics()
            summary = f"\nTotal knowledge base: {stats['total_nodes']} nodes, {stats['storage_efficiency']}x compression"

            return "\n\n".join(formatted_results) + summary

        result = asyncio.run(run_query())
        return result

    except Exception as e:
        return f"Query failed: {str(e)}"


@ToolRegistry.register(name="consolidate_domain_knowledge")
def consolidate_domain_knowledge(domain: str) -> str:
    """
    Consolidate knowledge in a specific domain to create higher-level abstractions.

    Args:
        domain: Knowledge domain to consolidate

    Returns:
        Consolidated knowledge summary
    """
    try:
        async def run_consolidation():
            kb = CompressedKnowledgeBase("./massive_kb")
            await kb.load_async()

            consolidated = await kb.consolidate_knowledge(domain)

            return f"Consolidated knowledge for domain '{domain}':\n\n{consolidated}"

        result = asyncio.run(run_consolidation())
        return result

    except Exception as e:
        return f"Consolidation failed: {str(e)}"


@ToolRegistry.register(name="get_knowledge_stats")
def get_knowledge_stats() -> str:
    """
    Get comprehensive statistics about the compressed knowledge base.

    Returns:
        Formatted statistics
    """
    try:
        async def run_stats():
            kb = CompressedKnowledgeBase("./massive_kb")
            await kb.load_async()

            stats = kb.get_statistics()

            output = f"""ECH0-PRIME Compressed Knowledge Base Statistics:

STORAGE METRICS:
- Total Knowledge Nodes: {stats['total_nodes']:,}
- Compressed Tokens: {stats['total_compressed_tokens']:,}
- Original Tokens: {stats['total_original_tokens']:,}
- Storage Efficiency: {stats['storage_efficiency']}x compression
- Average Quality Score: {stats['avg_quality_score']:.3f}
- Average Compression Ratio: {stats['avg_compression_ratio']:.3f}

DOMAIN BREAKDOWN:"""

            for domain_name, domain_stats in stats['domains'].items():
                output += f"""
  {domain_name.upper()}:
    - Nodes: {domain_stats['node_count']:,}
    - Avg Quality: {domain_stats['avg_quality']}
    - Avg Compression: {domain_stats['avg_compression']}
    - Total Tokens: {domain_stats['total_tokens']:,}"""

            output += f"""

SYSTEM STATUS:
- Cache Size: {stats['cache_size']}
- Memory Nodes: {stats['memory_nodes']}
- Domains: {stats['domains_count']}

PROJECTION:
- At current efficiency, could store ~{int(stats['total_original_tokens'] * stats['storage_efficiency']):,} original tokens
- Scaling to 10^15 tokens would require ~{int(1e15 / stats['storage_efficiency']):,} compressed tokens"""

            return output

        result = asyncio.run(run_stats())
        return result

    except Exception as e:
        return f"Stats retrieval failed: {str(e)}"


@ToolRegistry.register(name="optimize_knowledge_base")
def optimize_knowledge_base() -> str:
    """
    Optimize the knowledge base by removing low-quality content and consolidating.

    Returns:
        Optimization results
    """
    try:
        async def run_optimization():
            kb = CompressedKnowledgeBase("./massive_kb")
            await kb.load_async()

            before_stats = kb.get_statistics()

            await kb.optimize_storage()

            after_stats = kb.get_statistics()

            return f"""Knowledge Base Optimization Completed:

BEFORE:
- Nodes: {before_stats['total_nodes']:,}
- Quality: {before_stats['avg_quality_score']:.3f}

AFTER:
- Nodes: {after_stats['total_nodes']:,}
- Quality: {after_stats['avg_quality_score']:.3f}

CHANGES:
- Nodes removed: {before_stats['total_nodes'] - after_stats['total_nodes']:,}
- Quality improvement: {after_stats['avg_quality_score'] - before_stats['avg_quality_score']:.3f}"""

        result = asyncio.run(run_optimization())
        return result

    except Exception as e:
        return f"Optimization failed: {str(e)}"


class MassiveDataStreamer:
    """
    Advanced data streaming system for ingesting massive datasets.
    """

    def __init__(self):
        self.kb = CompressedKnowledgeBase("./massive_kb")
        self.processor = StreamingDataProcessor()

    async def stream_from_huggingface(self, dataset_name: str, config_name: str = None, split: str = "train",
                                    domain: str = "academic", max_samples: int = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream data from HuggingFace datasets.
        """
        try:
            from datasets import load_dataset

            # Use streaming=True to avoid OOM on large datasets
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)

            count = 0
            for example in dataset:
                if max_samples and count >= max_samples:
                    break

                # Extract text content
                content = ""
                if "text" in example:
                    content = example["text"]
                elif "content" in example:
                    content = example["content"]
                elif "article" in example:
                    content = example["article"]
                else:
                    # Try to concatenate all string fields
                    content = " ".join([str(v) for v in example.values() if isinstance(v, str)])

                if content.strip():
                    yield {
                        "content": content,
                        "modality": "text",
                        "domain": domain,
                        "metadata": {
                            "source": "huggingface",
                            "dataset": dataset_name,
                            "split": split,
                            "example_id": count
                        }
                    }
                    count += 1

        except ImportError:
            print("HuggingFace datasets not available")
        except Exception as e:
            print(f"HuggingFace streaming error: {e}")

    async def stream_from_web_archive(self, start_date: str, end_date: str,
                                    domain_filter: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream data from web archives (simulated for now).
        """
        # This would integrate with actual web archive APIs like Common Crawl
        # For now, return sample data
        sample_urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://arxiv.org/abs/1706.03762",
            "https://github.com/openai/gpt-3"
        ]

        for url in sample_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            yield {
                                "content": content[:10000],  # Limit size
                                "modality": "text",
                                "domain": domain_filter or "web",
                                "metadata": {
                                    "source": "web_archive",
                                    "url": url,
                                    "crawl_date": datetime.now().isoformat()
                                }
                            }
            except Exception as e:
                print(f"Error fetching {url}: {e}")

    async def stream_from_api_endpoint(self, api_url: str, api_key: str = None,
                                     data_key: str = "data", domain: str = "api") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream data from API endpoints that return JSON arrays.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Handle different JSON structures
                        if isinstance(data, list):
                            items = data
                        elif isinstance(data, dict) and data_key in data:
                            items = data[data_key]
                        else:
                            items = [data]

                        for item in items:
                            if isinstance(item, dict):
                                content = json.dumps(item)
                            else:
                                content = str(item)

                            yield {
                                "content": content,
                                "modality": "text",
                                "domain": domain,
                                "metadata": {
                                    "source": "api",
                                    "endpoint": api_url,
                                    "timestamp": datetime.now().isoformat()
                                }
                            }

        except Exception as e:
            print(f"API streaming error: {e}")


async def async_stream_huggingface_dataset(dataset_name: str, config_name: str = None, domain: str = "academic",
                                         max_samples: int = 1000) -> str:
    """Async version of HuggingFace streaming tool"""
    streamer = MassiveDataStreamer()
    await streamer.kb.load_async()

    ingestor = MassiveDataIngestor(streamer.kb)

    # Create data stream
    data_stream = streamer.stream_from_huggingface(
        dataset_name, config_name=config_name, domain=domain, max_samples=max_samples
    )

    # Ingest the data
    await ingestor.ingest_data_stream(data_stream, domain=domain)

    # Save results
    await streamer.kb.save_async()

    stats = ingestor.get_ingestion_stats()
    return f"HuggingFace dataset '{dataset_name}' ingested: {stats['total_stored']} knowledge nodes created"


@ToolRegistry.register(name="stream_huggingface_dataset")
def stream_huggingface_dataset(dataset_name: str, config_name: str = None, domain: str = "academic",
                              max_samples: int = 1000) -> str:
    """
    Stream and compress a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "wikitext")
        config_name: Specific config (e.g., "wikitext-103-v1")
        domain: Knowledge domain
        max_samples: Maximum samples to process

    Returns:
        Ingestion results
    """
    try:
        return asyncio.run(async_stream_huggingface_dataset(dataset_name, config_name, domain, max_samples))
    except Exception as e:
        return f"HuggingFace streaming failed: {str(e)}"


@ToolRegistry.register(name="stream_web_content")
def stream_web_content(urls: List[str], domain: str = "web") -> str:
    """
    Stream and compress web content from URLs.

    Args:
        urls: List of URLs to fetch and compress
        domain: Knowledge domain

    Returns:
        Ingestion results
    """
    try:
        async def run_web_streaming():
            streamer = MassiveDataStreamer()
            await streamer.kb.load_async()

            ingestor = MassiveDataIngestor(streamer.kb)

            async def url_stream():
                for url in urls:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url) as response:
                                if response.status == 200:
                                    content = await response.text()
                                    yield {
                                        "content": content,
                                        "modality": "text",
                                        "domain": domain,
                                        "metadata": {"source": "web", "url": url}
                                    }
                    except Exception as e:
                        print(f"Error fetching {url}: {e}")

            await ingestor.ingest_data_stream(url_stream(), domain=domain)
            await streamer.kb.save_async()

            stats = ingestor.get_ingestion_stats()
            return f"Web content ingested: {stats['total_stored']} knowledge nodes from {len(urls)} URLs"

        result = asyncio.run(run_web_streaming())
        return result

    except Exception as e:
        return f"Web streaming failed: {str(e)}"


if __name__ == "__main__":
    # Example usage
    print("ECH0-PRIME Massive Data Ingestor")
    print("Available commands:")
    print("- ingest_massive_dataset <data_source> [domain] [batch_size]")
    print("- query_compressed_knowledge <query> [domain] [limit]")
    print("- consolidate_domain_knowledge <domain>")
    print("- get_knowledge_stats")
    print("- optimize_knowledge_base")
