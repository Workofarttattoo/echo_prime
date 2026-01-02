"""
ECH0-PRIME Data Compression System
Uses prompt engineering to compress streaming data instead of vectorization.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import os
import json
import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import aiofiles
import aiohttp
from reasoning.llm_bridge import OllamaBridge
try:
    from memory.manager import EpisodicMemory, SemanticMemory  # type: ignore
    memory_available = True
except ImportError:
    memory_available = False
    print("Warning: Memory modules not available, using simplified mode")

try:
    from ech0_governance.persistent_memory import PersistentMemory  # type: ignore
    persistent_memory_available = True
except ImportError:
    persistent_memory_available = False
    print("Warning: Persistent memory not available, using simplified mode")


@dataclass
class CompressionConfig:
    """Configuration for data compression pipeline"""
    max_chunk_size: int = 4000  # tokens
    compression_ratio: float = 0.1  # 10% of original size
    min_compression_ratio: float = 0.05  # minimum compression
    max_compression_ratio: float = 0.3  # maximum compression
    batch_size: int = 10  # process in batches
    quality_threshold: float = 0.7  # minimum quality score
    deduplication_threshold: float = 0.85  # similarity threshold for dedup


@dataclass
class CompressedChunk:
    """Represents a compressed data chunk"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float
    timestamp: datetime
    modality: str  # text, image, audio, video
    domain: str  # academic, web, social, etc.
    compressed_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_url: Optional[str] = None
    checksum: Optional[str] = None


class DataCompressor:
    """
    Compresses streaming data using prompt engineering instead of vectorization.
    Leverages ECH0-PRIME's LLM bridge for intelligent compression.
    """

    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.llm_bridge = OllamaBridge(model="llama3.2")

        # Initialize memory systems if available
        if memory_available:
            self.episodic_memory = EpisodicMemory()
            self.semantic_memory = SemanticMemory()
        else:
            self.episodic_memory = None
            self.semantic_memory = None

        if persistent_memory_available:
            # For demo purposes, create a simple mock
            self.persistent_memory = None  # We'll handle this in store_compressed
        else:
            self.persistent_memory = None

        # Compression statistics
        self.stats = {
            "total_processed": 0,
            "total_compressed": 0,
            "avg_compression_ratio": 0.0,
            "avg_quality_score": 0.0,
            "compression_errors": 0
        }

        # Domain-specific compression prompts
        self.compression_prompts = {
            "academic": self._academic_compression_prompt,
            "web": self._web_compression_prompt,
            "social": self._social_compression_prompt,
            "code": self._code_compression_prompt,
            "news": self._news_compression_prompt,
            "multimodal": self._multimodal_compression_prompt
        }

    def _academic_compression_prompt(self, content: str) -> str:
        """Compress academic/scientific content"""
        return f"""Compress this academic content while preserving:
- Core hypotheses and findings
- Methodological details
- Key evidence and results
- Theoretical contributions
- Important citations

Remove: verbose explanations, redundant details, tangential discussions.

Content: {content[:self.config.max_chunk_size]}

Compressed Summary:"""

    def _web_compression_prompt(self, content: str) -> str:
        """Compress web/general content"""
        return f"""Compress this web content while preserving:
- Main topic and key points
- Important facts and data
- Actionable information
- Unique insights

Remove: advertisements, navigation, boilerplate, redundancy.

Content: {content[:self.config.max_chunk_size]}

Compressed Summary:"""

    def _social_compression_prompt(self, content: str) -> str:
        """Compress social media content"""
        return f"""Compress this social content while preserving:
- Core message or opinion
- Key facts mentioned
- Important context
- Unique perspectives

Remove: emojis, hashtags, @mentions, casual language filler.

Content: {content[:self.config.max_chunk_size]}

Compressed Summary:"""

    def _code_compression_prompt(self, content: str) -> str:
        """Compress code and technical content"""
        return f"""Compress this code/technical content while preserving:
- Core algorithms and logic
- Important functions and classes
- Key design patterns
- Technical specifications
- API signatures

Remove: comments, whitespace, examples, verbose documentation.

Content: {content[:self.config.max_chunk_size]}

Compressed Summary:"""

    def _news_compression_prompt(self, content: str) -> str:
        """Compress news content"""
        return f"""Compress this news content while preserving:
- Who, What, When, Where, Why, How
- Key facts and quotes
- Important context
- Impact and implications

Remove: journalistic style, repetitive phrases, advertisements.

Content: {content[:self.config.max_chunk_size]}

Compressed Summary:"""

    def _multimodal_compression_prompt(self, content: str) -> str:
        """Compress multimodal content (text + images/audio)"""
        return f"""Compress this multimodal content while preserving:
- Visual/audio descriptions and their significance
- Text-visual correlations
- Key multimodal insights
- Cross-modal relationships

Remove: redundant descriptions, technical metadata, file formats.

Content: {content[:self.config.max_chunk_size]}

Compressed Summary:"""

    async def compress_chunk(self, content: str, modality: str = "text",
                           domain: str = "web", metadata: Dict[str, Any] = None) -> CompressedChunk:
        """
        Compress a single chunk of data using prompt engineering.
        """
        start_time = time.time()

        # Select appropriate compression prompt
        compression_prompt = self.compression_prompts.get(domain, self._web_compression_prompt)(content)

        # Use LLM for compression
        compressed_content = await asyncio.get_event_loop().run_in_executor(
            None, self.llm_bridge.query, compression_prompt, None, None, 0.3, 0.9
        )

        if not compressed_content or "BRIDGE ERROR" in compressed_content:
            # Fallback: simple extractive compression
            compressed_content = self._extractive_compress(content)

        # Calculate compression metrics
        original_tokens = len(content.split())
        compressed_tokens = len(compressed_content.split())
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

        # Quality assessment (simplified)
        quality_score = self._assess_compression_quality(content, compressed_content, domain)

        # Adjust compression if outside bounds
        if compression_ratio < self.config.min_compression_ratio:
            compressed_content = self._aggressive_compress(compressed_content)
        elif compression_ratio > self.config.max_compression_ratio:
            compressed_content = self._expand_compress(compressed_content, content)

        chunk = CompressedChunk(
            original_tokens=original_tokens,
            compressed_tokens=len(compressed_content.split()),
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            timestamp=datetime.now(),
            modality=modality,
            domain=domain,
            compressed_content=compressed_content,
            metadata=metadata or {},
            source_url=metadata.get("source_url") if metadata else None
        )

        # Update statistics
        self._update_stats(chunk)

        return chunk

    def _extractive_compress(self, content: str) -> str:
        """Simple extractive compression fallback"""
        sentences = content.split('.')
        # Keep first and last sentences, and any with key terms
        key_sentences = [sentences[0]] if sentences else []
        key_sentences.extend([s for s in sentences[1:-1]
                            if any(term in s.lower() for term in
                                 ['important', 'key', 'significant', 'main', 'core'])])
        if len(sentences) > 1:
            key_sentences.append(sentences[-1])

        return '. '.join(set(key_sentences)).strip()

    def _aggressive_compress(self, content: str) -> str:
        """More aggressive compression"""
        words = content.split()
        # Keep every 3rd word approximately
        compressed = ' '.join(words[::3])
        return compressed if len(compressed) > 10 else content[:200]

    def _expand_compress(self, compressed: str, original: str) -> str:
        """Expand compression if too aggressive"""
        return compressed + " [Note: Compression too aggressive, keeping more detail]"

    def _assess_compression_quality(self, original: str, compressed: str, domain: str) -> float:
        """Simple quality assessment"""
        # Length ratio (too short = bad)
        length_ratio = len(compressed) / len(original)

        # Information density (sentences per word)
        original_sentences = len(original.split('.'))
        compressed_sentences = len(compressed.split('.'))
        density_score = min(1.0, compressed_sentences / max(1, original_sentences))

        # Domain-specific checks
        domain_score = 1.0
        if domain == "academic":
            # Check for academic keywords
            academic_terms = ['hypothesis', 'method', 'results', 'theory', 'evidence']
            found_terms = sum(1 for term in academic_terms if term in compressed.lower())
            domain_score = found_terms / len(academic_terms)

        return (length_ratio * 0.4 + density_score * 0.3 + domain_score * 0.3)

    def _update_stats(self, chunk: CompressedChunk):
        """Update compression statistics"""
        self.stats["total_processed"] += 1
        self.stats["total_compressed"] += chunk.compressed_tokens

        # Rolling average
        total = self.stats["total_processed"]
        self.stats["avg_compression_ratio"] = (
            (self.stats["avg_compression_ratio"] * (total - 1) + chunk.compression_ratio) / total
        )
        self.stats["avg_quality_score"] = (
            (self.stats["avg_quality_score"] * (total - 1) + chunk.quality_score) / total
        )

    async def compress_stream(self, data_stream: AsyncGenerator[Dict[str, Any], None]) -> AsyncGenerator[CompressedChunk, None]:
        """
        Compress a stream of data chunks asynchronously.
        """
        buffer = []
        async for chunk_data in data_stream:
            buffer.append(chunk_data)

            if len(buffer) >= self.config.batch_size:
                # Process batch
                tasks = []
                for data in buffer:
                    task = self.compress_chunk(
                        content=data.get("content", ""),
                        modality=data.get("modality", "text"),
                        domain=data.get("domain", "web"),
                        metadata=data.get("metadata", {})
                    )
                    tasks.append(task)

                # Execute batch compression
                compressed_chunks = await asyncio.gather(*tasks)

                # Yield results
                for chunk in compressed_chunks:
                    if chunk.quality_score >= self.config.quality_threshold:
                        yield chunk
                    else:
                        self.stats["compression_errors"] += 1

                buffer.clear()

        # Process remaining buffer
        if buffer:
            tasks = []
            for data in buffer:
                task = self.compress_chunk(
                    content=data.get("content", ""),
                    modality=data.get("modality", "text"),
                    domain=data.get("domain", "web"),
                    metadata=data.get("metadata", {})
                )
                tasks.append(task)

            compressed_chunks = await asyncio.gather(*tasks)
            for chunk in compressed_chunks:
                if chunk.quality_score >= self.config.quality_threshold:
                    yield chunk

    async def store_compressed(self, chunk: CompressedChunk, use_memory: bool = True):
        """
        Store compressed chunk in ECH0's memory systems.
        """
        if use_memory:
            # Store in episodic memory
            if self.episodic_memory:
                feature_vector = self._chunk_to_vector(chunk)
                self.episodic_memory.store_episode(feature_vector, {
                    "compressed_content": chunk.compressed_content,
                    "metadata": chunk.metadata,
                    "quality_score": chunk.quality_score,
                    "timestamp": chunk.timestamp.isoformat()
                })

            # Store in semantic memory if high quality
            if self.semantic_memory and chunk.quality_score > 0.8:
                self.semantic_memory.store_fact(
                    fact=chunk.compressed_content,
                    confidence=chunk.quality_score,
                    domain=chunk.domain
                )

            # Store in persistent memory
            if self.persistent_memory:
                self.persistent_memory.store(chunk.compressed_content, chunk.metadata)

    def _chunk_to_vector(self, chunk: CompressedChunk) -> np.ndarray:
        """Convert compressed chunk to vector representation"""
        import hashlib
        import numpy as np

        # Simple hash-based vectorization (can be improved with actual embeddings)
        content_hash = hashlib.md5(chunk.compressed_content.encode()).hexdigest()
        vector = np.array([int(content_hash[i:i+2], 16) for i in range(0, len(content_hash), 2)])
        vector = vector / 255.0  # normalize to [0,1]
        return vector[:1024]  # truncate to expected dimension

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get current compression statistics"""
        return self.stats.copy()

    async def adaptive_compress(self, content: str, target_ratio: float = None,
                              domain: str = "web") -> str:
        """
        Adaptively compress content to achieve target compression ratio.
        """
        if target_ratio is None:
            target_ratio = self.config.compression_ratio

        # Try different compression levels
        ratios = [0.05, 0.1, 0.15, 0.2, 0.3]
        best_compression = content
        best_ratio_diff = float('inf')

        for ratio in ratios:
            self.config.compression_ratio = ratio
            chunk = await self.compress_chunk(content, domain=domain)
            ratio_diff = abs(chunk.compression_ratio - target_ratio)

            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_compression = chunk.compressed_content

        return best_compression


class StreamingDataProcessor:
    """
    Processes streaming data sources and compresses them using ECH0-PRIME.
    """

    def __init__(self, compressor: DataCompressor = None):
        self.compressor = compressor or DataCompressor()
        self.active_streams = {}
        self.processed_count = 0

    async def stream_from_api(self, api_url: str, api_key: str = None,
                            domain: str = "web") -> AsyncGenerator[CompressedChunk, None]:
        """
        Stream and compress data from an API endpoint.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(api_url) as response:
                    if response.content_type == 'application/json':
                        data = await response.json()
                        # Process JSON data stream
                        if isinstance(data, list):
                            for item in data:
                                content = json.dumps(item)
                                async def single_item():
                                    yield {
                                        "content": content,
                                        "modality": "text",
                                        "domain": domain,
                                        "metadata": {"source": api_url, "type": "api"}
                                    }
                                async for chunk in self.compressor.compress_stream(single_item()):
                                    yield chunk
                        else:
                            content = json.dumps(data)
                            async def single_item():
                                yield {
                                    "content": content,
                                    "modality": "text",
                                    "domain": domain,
                                    "metadata": {"source": api_url, "type": "api"}
                                }
                            async for chunk in self.compressor.compress_stream(single_item()):
                                yield chunk

                    elif response.content_type.startswith('text/'):
                        text = await response.text()
                        # Split into chunks and process
                        chunks = self._split_text(text, 4000)
                        for i, chunk_text in enumerate(chunks):
                            async def single_chunk():
                                yield {
                                    "content": chunk_text,
                                    "modality": "text",
                                    "domain": domain,
                                    "metadata": {"source": api_url, "chunk_id": i, "type": "api"}
                                }
                            async for chunk in self.compressor.compress_stream(single_chunk()):
                                yield chunk

            except Exception as e:
                print(f"API streaming error: {e}")

    async def stream_from_file(self, file_path: str, domain: str = "web") -> AsyncGenerator[CompressedChunk, None]:
        """
        Stream and compress data from a file.
        """
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Split large files into chunks
            chunks = self._split_text(content, 4000)
            for i, chunk_text in enumerate(chunks):
                async def single_chunk():
                    yield {
                        "content": chunk_text,
                        "modality": "text",
                        "domain": domain,
                        "metadata": {"source": file_path, "chunk_id": i, "type": "file"}
                    }
                async for chunk in self.compressor.compress_stream(single_chunk()):
                    yield chunk

        except Exception as e:
            print(f"File streaming error: {e}")

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks while preserving sentence boundaries"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
            else:
                current_chunk += sentence + "."

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def process_stream(self, stream_id: str, stream_source: AsyncGenerator[CompressedChunk, None]):
        """
        Process a data stream and store compressed results.
        """
        self.active_streams[stream_id] = {"status": "processing", "processed": 0}

        try:
            async for compressed_chunk in stream_source:
                # Store the compressed chunk
                await self.compressor.store_compressed(compressed_chunk)
                self.processed_count += 1
                self.active_streams[stream_id]["processed"] = self.processed_count

        except Exception as e:
            print(f"Stream processing error for {stream_id}: {e}")
            self.active_streams[stream_id]["status"] = "error"
        else:
            self.active_streams[stream_id]["status"] = "completed"

    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get processing status for a stream"""
        return self.active_streams.get(stream_id, {"status": "not_found"})


# Convenience functions for easy use
async def compress_data_stream(data_stream: AsyncGenerator[Dict[str, Any], None],
                             domain: str = "web") -> AsyncGenerator[CompressedChunk, None]:
    """
    High-level function to compress a data stream.
    """
    compressor = DataCompressor()
    async for chunk in compressor.compress_stream(data_stream):
        yield chunk


async def process_api_stream(api_url: str, api_key: str = None, domain: str = "web") -> AsyncGenerator[CompressedChunk, None]:
    """
    Process and compress data from an API.
    """
    processor = StreamingDataProcessor()
    async for chunk in processor.stream_from_api(api_url, api_key, domain):
        yield chunk


if __name__ == "__main__":
    # Example usage
    async def demo():
        compressor = DataCompressor()

        # Example content
        sample_content = """
        The field of artificial intelligence has seen remarkable progress in recent years.
        Large language models have demonstrated impressive capabilities in natural language understanding,
        generation, and reasoning. However, achieving true artificial general intelligence remains
        a significant challenge that requires advances in multiple areas including multimodal learning,
        causal reasoning, and continual learning. The development of systems that can learn from
        diverse data sources while maintaining robustness and safety is crucial for the future of AI.
        """

        # Compress the content
        compressed = await compressor.compress_chunk(sample_content, domain="academic")
        print(f"Original: {compressed.original_tokens} tokens")
        print(f"Compressed: {compressed.compressed_tokens} tokens")
        print(f"Ratio: {compressed.compression_ratio:.2f}")
        print(f"Quality: {compressed.quality_score:.2f}")
        print(f"Compressed content: {compressed.compressed_content}")

    asyncio.run(demo())
