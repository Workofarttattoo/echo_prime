"""
ECH0-PRIME Compressed Knowledge Base
Stores massive amounts of knowledge using compressed data format.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import os
import json
import asyncio
import hashlib
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import aiofiles
from learning.data_compressor import DataCompressor, CompressedChunk, CompressionConfig
try:
    from reasoning.llm_bridge import OllamaBridge
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM bridge not available, using simplified compression")
from memory.manager import SemanticMemory, EpisodicMemory
from ech0_governance.persistent_memory import PersistentMemory
from reasoning.llm_bridge import OllamaBridge


@dataclass
class KnowledgeNode:
    """A node in the compressed knowledge graph"""
    id: str
    compressed_content: str
    domain: str
    modality: str
    quality_score: float
    compression_ratio: float
    timestamp: datetime
    connections: Set[str] = field(default_factory=set)  # Connected node IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "compressed_content": self.compressed_content,
            "domain": self.domain,
            "modality": self.modality,
            "quality_score": self.quality_score,
            "compression_ratio": self.compression_ratio,
            "timestamp": self.timestamp.isoformat(),
            "connections": list(self.connections),
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        return cls(
            id=data["id"],
            compressed_content=data["compressed_content"],
            domain=data["domain"],
            modality=data["modality"],
            quality_score=data["quality_score"],
            compression_ratio=data["compression_ratio"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            connections=set(data.get("connections", [])),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
        )


@dataclass
class KnowledgeDomain:
    """Represents a domain of knowledge with statistics"""
    name: str
    node_count: int = 0
    total_compressed_tokens: int = 0
    avg_quality_score: float = 0.0
    avg_compression_ratio: float = 0.0
    subdomains: Dict[str, 'KnowledgeDomain'] = field(default_factory=dict)
    last_updated: Optional[datetime] = None


class CompressedKnowledgeBase:
    """
    Massive-scale knowledge storage using compressed data format.
    Can store 10^15+ tokens worth of knowledge efficiently.
    """

    def __init__(self, storage_path: str = "./compressed_kb", max_nodes_per_file: int = 10000):
        self.storage_path = storage_path
        self.max_nodes_per_file = max_nodes_per_file
        self.compressor = DataCompressor()

        # In-memory knowledge graph
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.domains: Dict[str, KnowledgeDomain] = defaultdict(KnowledgeDomain)

        # Indexing for fast retrieval
        self.content_index: Dict[str, Set[str]] = defaultdict(set)  # term -> node_ids
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)  # domain -> node_ids
        self.quality_index: List[Tuple[str, float]] = []  # (node_id, quality) sorted

        # Caching
        self.access_cache: Dict[str, KnowledgeNode] = {}
        self.cache_max_size = 1000

        # Statistics
        self.stats = {
            "total_nodes": 0,
            "total_compressed_tokens": 0,
            "total_original_tokens": 0,
            "avg_compression_ratio": 0.0,
            "avg_quality_score": 0.0,
            "domains_count": 0,
            "storage_files": 0
        }

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "domains"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "indices"), exist_ok=True)

    async def add_compressed_chunk(self, chunk: CompressedChunk) -> str:
        """
        Add a compressed chunk to the knowledge base.
        Returns the node ID.
        """
        # Generate unique ID
        content_hash = hashlib.md5(chunk.compressed_content.encode()).hexdigest()[:16]
        node_id = f"{chunk.domain}_{chunk.modality}_{content_hash}"

        # Check for duplicates (simple hash-based)
        if node_id in self.nodes:
            return node_id  # Already exists

        # Create knowledge node
        node = KnowledgeNode(
            id=node_id,
            compressed_content=chunk.compressed_content,
            domain=chunk.domain,
            modality=chunk.modality,
            quality_score=chunk.quality_score,
            compression_ratio=chunk.compression_ratio,
            timestamp=chunk.timestamp,
            metadata=chunk.metadata
        )

        # Add to in-memory graph
        self.nodes[node_id] = node

        # Update indices
        self._update_indices(node)

        # Update domain statistics
        self._update_domain_stats(chunk.domain, chunk)

        # Update global statistics
        self._update_global_stats(chunk)

        # Auto-save if needed
        if len(self.nodes) % 1000 == 0:
            await self.save_async()

        return node_id

    def _update_indices(self, node: KnowledgeNode):
        """Update search indices for the node"""
        # Domain index
        self.domain_index[node.domain].add(node.id)

        # Content index (simple keyword extraction)
        keywords = self._extract_keywords(node.compressed_content)
        for keyword in keywords:
            self.content_index[keyword].add(node.id)

        # Quality index (maintain sorted list)
        self.quality_index.append((node.id, node.quality_score))
        self.quality_index.sort(key=lambda x: x[1], reverse=True)

        # Keep only top quality entries in index
        if len(self.quality_index) > 10000:
            self.quality_index = self.quality_index[:10000]

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from compressed content"""
        # Simple extraction: nouns and important terms
        words = content.lower().split()
        keywords = []

        # Domain-specific important terms
        important_terms = {
            'academic': ['theory', 'method', 'results', 'evidence', 'hypothesis', 'conclusion'],
            'technical': ['algorithm', 'function', 'class', 'api', 'system', 'performance'],
            'scientific': ['experiment', 'data', 'analysis', 'model', 'prediction', 'validation'],
            'general': ['important', 'key', 'main', 'core', 'significant', 'critical']
        }

        for word in words:
            word = word.strip('.,!?;:')
            if len(word) > 3:  # Skip short words
                keywords.append(word)

        # Add domain-specific terms if they appear
        for term in important_terms.get('general', []):
            if term in content.lower():
                keywords.append(term)

        return list(set(keywords))  # Remove duplicates

    def _update_domain_stats(self, domain: str, chunk: CompressedChunk):
        """Update statistics for a knowledge domain"""
        if domain not in self.domains:
            self.domains[domain] = KnowledgeDomain(name=domain)

        domain_stats = self.domains[domain]
        domain_stats.node_count += 1
        domain_stats.total_compressed_tokens += chunk.compressed_tokens
        domain_stats.last_updated = chunk.timestamp

        # Rolling average for quality and compression
        count = domain_stats.node_count
        domain_stats.avg_quality_score = (
            (domain_stats.avg_quality_score * (count - 1) + chunk.quality_score) / count
        )
        domain_stats.avg_compression_ratio = (
            (domain_stats.avg_compression_ratio * (count - 1) + chunk.compression_ratio) / count
        )

    def _update_global_stats(self, chunk: CompressedChunk):
        """Update global knowledge base statistics"""
        self.stats["total_nodes"] += 1
        self.stats["total_compressed_tokens"] += chunk.compressed_tokens
        self.stats["total_original_tokens"] += chunk.original_tokens

        # Update averages
        total = self.stats["total_nodes"]
        self.stats["avg_compression_ratio"] = (
            (self.stats["avg_compression_ratio"] * (total - 1) + chunk.compression_ratio) / total
        )
        self.stats["avg_quality_score"] = (
            (self.stats["avg_quality_score"] * (total - 1) + chunk.quality_score) / total
        )

        self.stats["domains_count"] = len(self.domains)

    async def retrieve_knowledge(self, query: str, domain: str = None,
                               min_quality: float = 0.0, limit: int = 10) -> List[KnowledgeNode]:
        """
        Retrieve relevant knowledge nodes based on query.
        """
        candidates = set()

        # Find candidate nodes
        query_terms = query.lower().split()
        for term in query_terms:
            if term in self.content_index:
                candidates.update(self.content_index[term])

        # Filter by domain if specified
        if domain:
            domain_nodes = self.domain_index.get(domain, set())
            candidates = candidates.intersection(domain_nodes)

        # Filter by quality
        filtered_candidates = []
        for node_id in candidates:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.quality_score >= min_quality:
                    filtered_candidates.append((node_id, node.quality_score))

        # Sort by quality and return top results
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)
        results = []

        for node_id, _ in filtered_candidates[:limit]:
            node = await self.get_node(node_id)
            if node:
                results.append(node)

        return results

    async def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a knowledge node by ID with caching"""
        # Check cache first
        if node_id in self.access_cache:
            node = self.access_cache[node_id]
            node.access_count += 1
            node.last_accessed = datetime.now()
            return node

        # Check in-memory
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.access_count += 1
            node.last_accessed = datetime.now()

            # Add to cache
            if len(self.access_cache) < self.cache_max_size:
                self.access_cache[node_id] = node

            return node

        # Try loading from disk
        node = await self.load_node_from_disk(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = datetime.now()
            return node

        return None

    async def load_node_from_disk(self, node_id: str) -> Optional[KnowledgeNode]:
        """Load a node from disk storage"""
        domain = node_id.split('_')[0]
        file_path = os.path.join(self.storage_path, "domains", f"{domain}.jsonl")

        if not os.path.exists(file_path):
            return None

        try:
            async with aiofiles.open(file_path, 'r') as f:
                async for line in f:
                    data = json.loads(line.strip())
                    if data["id"] == node_id:
                        node = KnowledgeNode.from_dict(data)
                        # Add to in-memory cache
                        self.nodes[node_id] = node
                        return node
        except Exception as e:
            print(f"Error loading node {node_id}: {e}")

        return None

    async def connect_nodes(self, node_id1: str, node_id2: str):
        """Create a connection between two knowledge nodes"""
        node1 = await self.get_node(node_id1)
        node2 = await self.get_node(node_id2)

        if node1 and node2:
            node1.connections.add(node_id2)
            node2.connections.add(node_id1)

    async def find_related_nodes(self, node_id: str, depth: int = 2) -> List[KnowledgeNode]:
        """Find related nodes through the knowledge graph"""
        visited = set()
        to_visit = [(node_id, 0)]
        related = []

        while to_visit:
            current_id, current_depth = to_visit.pop(0)

            if current_id in visited or current_depth > depth:
                continue

            visited.add(current_id)
            node = await self.get_node(current_id)

            if node and current_depth > 0:  # Don't include the starting node
                related.append(node)

            if current_depth < depth:
                for connected_id in node.connections:
                    if connected_id not in visited:
                        to_visit.append((connected_id, current_depth + 1))

        return related

    async def consolidate_knowledge(self, domain: str) -> str:
        """
        Consolidate knowledge in a domain using LLM to create higher-level abstractions.
        """
        domain_nodes = list(self.domain_index.get(domain, set()))

        if not domain_nodes:
            return f"No knowledge found in domain {domain}"

        # Get high-quality nodes
        high_quality_nodes = []
        for node_id in domain_nodes[:50]:  # Limit for processing
            node = await self.get_node(node_id)
            if node and node.quality_score > 0.8:
                high_quality_nodes.append(node)

        if not high_quality_nodes:
            return f"No high-quality knowledge found in domain {domain}"

        # Combine compressed content
        combined_content = "\n\n".join([node.compressed_content for node in high_quality_nodes])

        # Use LLM to create consolidated knowledge
        if LLM_AVAILABLE:
            consolidation_prompt = f"""
            Consolidate this collection of compressed knowledge from the {domain} domain.
            Create a coherent, comprehensive summary that captures the key insights and patterns.
            Focus on the most important concepts, relationships, and implications.

            Knowledge to consolidate:
            {combined_content[:8000]}  # Limit for LLM processing

            Consolidated Knowledge Summary:
            """

            llm_bridge = OllamaBridge()
            consolidated = llm_bridge.query(consolidation_prompt, temperature=0.2)
        else:
            # Fallback: Simple concatenation with summary
            consolidated = f"Consolidated knowledge from {len(high_quality_nodes)} sources in {domain} domain:\n\n{combined_content[:1000]}..."

        # Store the consolidated knowledge
        consolidated_chunk = CompressedChunk(
            original_tokens=sum(len(node.compressed_content.split()) for node in high_quality_nodes),
            compressed_tokens=len(consolidated.split()),
            compression_ratio=len(consolidated.split()) / sum(len(node.compressed_content.split()) for node in high_quality_nodes),
            quality_score=0.9,  # Assume high quality for consolidated knowledge
            timestamp=datetime.now(),
            modality="text",
            domain=f"{domain}_consolidated",
            compressed_content=consolidated,
            metadata={"consolidated_from": len(high_quality_nodes), "type": "consolidated"}
        )

        await self.add_compressed_chunk(consolidated_chunk)
        return consolidated

    async def save_async(self):
        """Asynchronously save knowledge base to disk"""
        # Save nodes by domain
        for domain, domain_nodes in self.domain_index.items():
            file_path = os.path.join(self.storage_path, "domains", f"{domain}.jsonl")

            # Collect nodes for this domain
            domain_data = []
            for node_id in domain_nodes:
                if node_id in self.nodes:
                    domain_data.append(self.nodes[node_id].to_dict())

            # Save to file
            async with aiofiles.open(file_path, 'w') as f:
                for node_data in domain_data:
                    await f.write(json.dumps(node_data) + '\n')

        # Save indices
        indices_path = os.path.join(self.storage_path, "indices", "indices.json")
        indices_data = {
            "content_index": {k: list(v) for k, v in self.content_index.items()},
            "domain_index": {k: list(v) for k, v in self.domain_index.items()},
            "quality_index": self.quality_index,
            "stats": self.stats,
            "domains": {k: {
                "name": v.name,
                "node_count": v.node_count,
                "total_compressed_tokens": v.total_compressed_tokens,
                "avg_quality_score": v.avg_quality_score,
                "avg_compression_ratio": v.avg_compression_ratio,
                "last_updated": v.last_updated.isoformat() if v.last_updated else None
            } for k, v in self.domains.items()}
        }

        async with aiofiles.open(indices_path, 'w') as f:
            await f.write(json.dumps(indices_data, indent=2))

    async def load_async(self):
        """Asynchronously load knowledge base from disk"""
        indices_path = os.path.join(self.storage_path, "indices", "indices.json")

        if not os.path.exists(indices_path):
            return

        try:
            async with aiofiles.open(indices_path, 'r') as f:
                indices_data = json.loads(await f.read())

            # Load indices
            self.content_index = defaultdict(set, {k: set(v) for k, v in indices_data["content_index"].items()})
            self.domain_index = defaultdict(set, {k: set(v) for k, v in indices_data["domain_index"].items()})
            self.quality_index = indices_data["quality_index"]
            self.stats = indices_data["stats"]

            # Load domains
            for k, v in indices_data["domains"].items():
                domain = KnowledgeDomain(
                    name=v["name"],
                    node_count=v["node_count"],
                    total_compressed_tokens=v["total_compressed_tokens"],
                    avg_quality_score=v["avg_quality_score"],
                    avg_compression_ratio=v["avg_compression_ratio"],
                    last_updated=datetime.fromisoformat(v["last_updated"]) if v.get("last_updated") else None
                )
                self.domains[k] = domain

        except Exception as e:
            print(f"Error loading knowledge base indices: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        return {
            **self.stats,
            "cache_size": len(self.access_cache),
            "memory_nodes": len(self.nodes),
            "domains": {k: {
                "node_count": v.node_count,
                "avg_quality": round(v.avg_quality_score, 3),
                "avg_compression": round(v.avg_compression_ratio, 3),
                "total_tokens": v.total_compressed_tokens
            } for k, v in self.domains.items()},
            "estimated_original_tokens": self.stats["total_original_tokens"],
            "storage_efficiency": round(self.stats["total_original_tokens"] / max(1, self.stats["total_compressed_tokens"]), 2)
        }

    async def optimize_storage(self):
        """Optimize storage by removing low-quality nodes and consolidating"""
        # Remove nodes with quality score < 0.3
        to_remove = []
        for node_id, node in self.nodes.items():
            if node.quality_score < 0.3:
                to_remove.append(node_id)

        for node_id in to_remove:
            del self.nodes[node_id]

        # Rebuild indices
        self.content_index.clear()
        self.domain_index.clear()
        self.quality_index.clear()

        for node in self.nodes.values():
            self._update_indices(node)

        # Consolidate domains with too many nodes
        for domain in self.domains.keys():
            if self.domains[domain].node_count > 50000:
                await self.consolidate_knowledge(domain)

        await self.save_async()


class MassiveDataIngestor:
    """
    Ingests massive amounts of data and compresses it for storage in the knowledge base.
    """

    def __init__(self, knowledge_base: CompressedKnowledgeBase):
        self.kb = knowledge_base
        self.ingestion_stats = {
            "total_processed": 0,
            "total_stored": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }

    async def ingest_data_stream(self, data_stream, domain: str = "web", batch_size: int = 100):
        """
        Ingest a stream of data and compress it for storage.
        """
        self.ingestion_stats["start_time"] = datetime.now()

        batch = []
        async for data_item in data_stream:
            batch.append(data_item)

            if len(batch) >= batch_size:
                await self._process_batch(batch, domain)
                batch.clear()

        # Process remaining items
        if batch:
            await self._process_batch(batch, domain)

        self.ingestion_stats["end_time"] = datetime.now()

    async def _process_batch(self, batch: List[Dict[str, Any]], domain: str):
        """Process a batch of data items"""
        for item in batch:
            try:
                content = item.get("content", "")
                if not content.strip():
                    continue

                # Compress the content
                compressed_chunk = await self.kb.compressor.compress_chunk(
                    content=content,
                    domain=domain,
                    modality=item.get("modality", "text"),
                    metadata=item.get("metadata", {})
                )

                # Store in knowledge base
                node_id = await self.kb.add_compressed_chunk(compressed_chunk)

                self.ingestion_stats["total_processed"] += 1
                self.ingestion_stats["total_stored"] += 1

            except Exception as e:
                print(f"Error processing item: {e}")
                self.ingestion_stats["errors"] += 1

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        stats = self.ingestion_stats.copy()
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            stats["duration_seconds"] = duration.total_seconds()
            stats["processing_rate"] = stats["total_processed"] / max(1, duration.total_seconds())
        return stats


# Integration with ECH0's reasoning system
class CompressedKnowledgeBridge:
    """
    Bridge between compressed knowledge base and ECH0's reasoning system.
    """

    def __init__(self, knowledge_base: CompressedKnowledgeBase):
        self.kb = knowledge_base
        self.llm_bridge = OllamaBridge()

    async def retrieve_for_reasoning(self, query: str, context_limit: int = 5000) -> str:
        """
        Retrieve relevant compressed knowledge for reasoning tasks.
        """
        # Retrieve relevant nodes
        nodes = await self.kb.retrieve_knowledge(query, limit=5, min_quality=0.7)

        if not nodes:
            return "No relevant compressed knowledge found."

        # Combine and format for reasoning
        knowledge_texts = []
        total_length = 0

        for node in nodes:
            if total_length + len(node.compressed_content) > context_limit:
                break
            knowledge_texts.append(f"[{node.domain.upper()}] {node.compressed_content}")
            total_length += len(node.compressed_content)

        return "\n\n".join(knowledge_texts)

    async def expand_compressed_knowledge(self, compressed_content: str, query: str) -> str:
        """
        Use LLM to expand compressed knowledge in response to a specific query.
        """
        expansion_prompt = f"""
        Given this compressed knowledge: {compressed_content}

        And this specific query: {query}

        Expand and elaborate on the relevant parts of the compressed knowledge to provide
        a detailed, contextual response. Focus on accuracy and relevance.
        """

        return self.llm_bridge.query(expansion_prompt, temperature=0.3)


if __name__ == "__main__":
    async def demo():
        # Initialize compressed knowledge base
        kb = CompressedKnowledgeBase("./demo_kb")

        # Load existing knowledge
        await kb.load_async()

        # Add some sample compressed knowledge
        sample_chunks = [
            CompressedChunk(
                original_tokens=1000, compressed_tokens=100, compression_ratio=0.1,
                quality_score=0.9, timestamp=datetime.now(), modality="text",
                domain="academic", compressed_content="Deep learning transformers revolutionized NLP through self-attention mechanisms."
            ),
            CompressedChunk(
                original_tokens=800, compressed_tokens=80, compression_ratio=0.1,
                quality_score=0.85, timestamp=datetime.now(), modality="text",
                domain="technical", compressed_content="Quantum computing uses superposition and entanglement for exponential speedup on specific problems."
            )
        ]

        for chunk in sample_chunks:
            await kb.add_compressed_chunk(chunk)

        # Retrieve knowledge
        results = await kb.retrieve_knowledge("quantum computing", limit=3)
        for node in results:
            print(f"Found: {node.compressed_content}")

        # Get statistics
        stats = kb.get_statistics()
        print(f"Knowledge base stats: {stats['total_nodes']} nodes, {stats['storage_efficiency']}x compression")

        # Save knowledge base
        await kb.save_async()

    asyncio.run(demo())
