#!/usr/bin/env python3
"""
Semantic Paper Search with Embeddings

Provides advanced semantic search capabilities for downloaded papers using:
- Sentence embeddings for semantic similarity
- Vector similarity search
- Cross-domain concept matching
- Hybrid keyword + semantic search

Optional dependencies:
- sentence-transformers (for embeddings)
- faiss-cpu (for fast vector search)
- numpy (for vector operations)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss-cpu not installed. Install with: pip install faiss-cpu")

from invention_data_indexer import InventionDataIndexer, Paper


class SemanticPaperSearch:
    """
    Semantic search for scientific papers using embeddings
    """

    def __init__(self,
                 data_dir: str = "invention_data",
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str = ".cache/embeddings"):
        """
        Initialize semantic search

        Args:
            data_dir: Directory with downloaded papers
            model_name: Sentence transformer model name
            cache_dir: Directory to cache embeddings
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize indexer
        self.indexer = InventionDataIndexer(data_dir=str(self.data_dir))

        # Initialize embedding model
        if EMBEDDINGS_AVAILABLE:
            print(f"🔧 Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            self.embeddings = None
            self.embedding_map = {}  # arxiv_id -> embedding_index

            # Try to load cached embeddings
            if not self._load_cached_embeddings():
                print("📊 Generating embeddings for all papers...")
                self._generate_embeddings()
                self._save_cached_embeddings()

            # Initialize FAISS index
            if FAISS_AVAILABLE and self.embeddings is not None:
                self._build_faiss_index()
        else:
            self.model = None
            print("⚠️  Falling back to keyword search only")

    def _get_cache_path(self) -> Path:
        """Get path to cached embeddings file"""
        return self.cache_dir / f"embeddings_{len(self.indexer.papers)}.pkl"

    def _load_cached_embeddings(self) -> bool:
        """Load embeddings from cache"""
        cache_file = self._get_cache_path()
        if cache_file.exists():
            try:
                print(f"📂 Loading cached embeddings from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.embedding_map = data['embedding_map']
                print(f"   ✅ Loaded {len(self.embeddings)} embeddings")
                return True
            except Exception as e:
                print(f"   ⚠️  Failed to load cache: {e}")
        return False

    def _save_cached_embeddings(self):
        """Save embeddings to cache"""
        cache_file = self._get_cache_path()
        try:
            print(f"💾 Saving embeddings to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'embedding_map': self.embedding_map
                }, f)
            print(f"   ✅ Saved {len(self.embeddings)} embeddings")
        except Exception as e:
            print(f"   ⚠️  Failed to save cache: {e}")

    def _generate_embeddings(self):
        """Generate embeddings for all papers"""
        if not self.model:
            return

        texts = []
        for i, paper in enumerate(self.indexer.papers):
            # Combine title and abstract
            text = f"{paper.title} {paper.abstract}"
            texts.append(text)
            self.embedding_map[paper.arxiv_id] = i

        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

            if (i // batch_size) % 10 == 0:
                print(f"   Progress: {i}/{len(texts)} papers")

        self.embeddings = np.vstack(all_embeddings)
        print(f"   ✅ Generated {len(self.embeddings)} embeddings")

    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if not FAISS_AVAILABLE or self.embeddings is None:
            return

        print("🔧 Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)

        print(f"   ✅ FAISS index built with {self.faiss_index.ntotal} vectors")

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[Paper, float]]:
        """
        Perform semantic search

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (Paper, similarity_score) tuples
        """
        if not self.model or self.embeddings is None:
            print("⚠️  Embeddings not available, falling back to keyword search")
            papers = self.indexer.search(query, limit=top_k)
            return [(p, p.relevance_score) for p in papers]

        # Encode query
        query_embedding = self.model.encode([query])

        if FAISS_AVAILABLE and hasattr(self, 'faiss_index'):
            # Fast FAISS search
            faiss.normalize_L2(query_embedding)
            distances, indices = self.faiss_index.search(query_embedding, top_k)

            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.indexer.papers):
                    paper = self.indexer.papers[idx]
                    results.append((paper, float(dist)))

            return results
        else:
            # Manual cosine similarity
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                paper = self.indexer.papers[idx]
                similarity = float(similarities[idx])
                results.append((paper, similarity))

            return results

    def hybrid_search(self,
                       query: str,
                       top_k: int = 10,
                       semantic_weight: float = 0.7) -> List[Tuple[Paper, float]]:
        """
        Hybrid search combining semantic and keyword matching

        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic score (0-1)

        Returns:
            List of (Paper, combined_score) tuples
        """
        keyword_weight = 1.0 - semantic_weight

        # Semantic search
        semantic_results = self.semantic_search(query, top_k=top_k * 2)

        # Keyword search
        keyword_results = self.indexer.search(query, limit=top_k * 2)

        # Combine scores
        paper_scores = {}

        # Add semantic scores
        for paper, score in semantic_results:
            paper_scores[paper.arxiv_id] = {
                'paper': paper,
                'semantic': score,
                'keyword': 0.0
            }

        # Add keyword scores
        max_keyword_score = max([p.relevance_score for p in keyword_results]) if keyword_results else 1.0
        for paper in keyword_results:
            if paper.arxiv_id not in paper_scores:
                paper_scores[paper.arxiv_id] = {
                    'paper': paper,
                    'semantic': 0.0,
                    'keyword': 0.0
                }
            # Normalize keyword score
            paper_scores[paper.arxiv_id]['keyword'] = paper.relevance_score / max_keyword_score

        # Calculate combined scores
        results = []
        for arxiv_id, scores in paper_scores.items():
            combined_score = (
                scores['semantic'] * semantic_weight +
                scores['keyword'] * keyword_weight
            )
            results.append((scores['paper'], combined_score))

        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def find_similar_papers(self, arxiv_id: str, top_k: int = 10) -> List[Tuple[Paper, float]]:
        """
        Find papers similar to a given paper

        Args:
            arxiv_id: arXiv ID of the reference paper
            top_k: Number of similar papers to return

        Returns:
            List of (Paper, similarity_score) tuples
        """
        if arxiv_id not in self.embedding_map:
            print(f"⚠️  Paper {arxiv_id} not found")
            return []

        # Get reference paper embedding
        ref_idx = self.embedding_map[arxiv_id]
        ref_embedding = self.embeddings[ref_idx:ref_idx+1]

        if FAISS_AVAILABLE and hasattr(self, 'faiss_index'):
            # FAISS search (skip top result since it's the reference paper)
            distances, indices = self.faiss_index.search(ref_embedding, top_k + 1)

            results = []
            for dist, idx in zip(distances[0][1:], indices[0][1:]):  # Skip first (self)
                if idx < len(self.indexer.papers):
                    paper = self.indexer.papers[idx]
                    results.append((paper, float(dist)))

            return results
        else:
            # Manual similarity
            similarities = np.dot(self.embeddings, ref_embedding.T).flatten()
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip first (self)

            results = []
            for idx in top_indices:
                paper = self.indexer.papers[idx]
                similarity = float(similarities[idx])
                results.append((paper, similarity))

            return results

    def cross_domain_search(self, query: str, min_categories: int = 2, top_k: int = 10) -> List[Tuple[Paper, float]]:
        """
        Search for papers that span multiple domains

        Args:
            query: Search query
            min_categories: Minimum number of arXiv categories per paper
            top_k: Number of results

        Returns:
            List of (Paper, similarity_score) tuples
        """
        # Get more results initially
        results = self.semantic_search(query, top_k=top_k * 3)

        # Filter for cross-domain papers
        cross_domain = [
            (paper, score)
            for paper, score in results
            if len(paper.categories) >= min_categories
        ]

        return cross_domain[:top_k]


def main():
    """Demo usage"""
    print("🔍 Semantic Paper Search Demo\n")

    if not EMBEDDINGS_AVAILABLE:
        print("❌ sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return

    # Initialize search
    search = SemanticPaperSearch()

    if not search.indexer.papers:
        print("\n⚠️  No papers found. Please run:")
        print("    python3 download_invention_data.py --sample")
        return

    # Demo queries
    queries = [
        "quantum computing with superconductors",
        "energy harvesting nanomaterials",
        "3D bioprinting tissue engineering",
        "metamaterial optical computing"
    ]

    print("\n" + "="*80)
    print("SEMANTIC SEARCH DEMO")
    print("="*80)

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = search.semantic_search(query, top_k=3)

        for i, (paper, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {paper.title[:70]}...")
            print(f"     Category: {paper.category_domain}")

    print("\n" + "="*80)
    print("HYBRID SEARCH DEMO")
    print("="*80)

    query = "graphene metamaterial photonics"
    print(f"\nQuery: '{query}'")
    results = search.hybrid_search(query, top_k=5)

    for i, (paper, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {paper.title[:70]}...")
        print(f"     Category: {paper.category_domain}")

    print("\n" + "="*80)
    print("CROSS-DOMAIN SEARCH DEMO")
    print("="*80)

    query = "quantum materials for computing"
    print(f"\nQuery: '{query}'")
    results = search.cross_domain_search(query, min_categories=2, top_k=5)

    for i, (paper, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.3f}] {paper.title[:70]}...")
        print(f"     Categories: {', '.join(paper.categories[:3])}")
        print(f"     Domain: {paper.category_domain}")

    print("\n✅ Semantic search ready!")


if __name__ == "__main__":
    main()
