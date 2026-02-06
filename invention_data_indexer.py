#!/usr/bin/env python3
"""
Invention Data Indexer
Indexes downloaded scientific papers and provides fast retrieval for invention generation

Features:
- Load all downloaded papers from invention_data/
- Create searchable index by category, keywords, and metadata
- Fast retrieval by relevance scoring
- Integration with Echo Prime invention pipeline
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """Represents a scientific paper"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published: str
    categories: List[str]
    pdf_url: str
    category_domain: str  # Our invention category (e.g., "materials_science")
    primary_category: str = ""
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'arxiv_id': self.arxiv_id,
            'published': self.published,
            'categories': self.categories,
            'pdf_url': self.pdf_url,
            'category_domain': self.category_domain,
            'primary_category': self.primary_category,
            'doi': self.doi,
            'journal_ref': self.journal_ref,
            'keywords': self.keywords
        }

    def get_full_text(self) -> str:
        """Get combined title + abstract for search"""
        return f"{self.title} {self.abstract}".lower()


class InventionDataIndexer:
    """
    Indexes and retrieves scientific papers from invention_data directory
    """

    def __init__(self, data_dir: str = "invention_data"):
        self.data_dir = Path(data_dir)
        self.papers: List[Paper] = []
        self.papers_by_category: Dict[str, List[Paper]] = defaultdict(list)
        self.papers_by_keyword: Dict[str, List[Paper]] = defaultdict(list)
        self.index_stats = {
            'total_papers': 0,
            'categories': {},
            'last_indexed': None
        }

        if self.data_dir.exists():
            self._load_all_papers()
        else:
            print(f"⚠️  Warning: {data_dir} directory not found. Run download_invention_data.py first.")

    def _load_all_papers(self):
        """Load all papers from all categories"""
        print(f"📚 Loading papers from {self.data_dir}...")

        for category_dir in self.data_dir.iterdir():
            if not category_dir.is_dir():
                continue

            papers_file = category_dir / "papers.json"
            if not papers_file.exists():
                continue

            category_name = category_dir.name

            try:
                with open(papers_file, 'r') as f:
                    papers_data = json.load(f)

                category_count = 0
                for paper_data in papers_data:
                    paper = Paper(
                        title=paper_data.get('title', ''),
                        authors=paper_data.get('authors', []),
                        abstract=paper_data.get('abstract', ''),
                        arxiv_id=paper_data.get('arxiv_id', ''),
                        published=paper_data.get('published', ''),
                        categories=paper_data.get('categories', []),
                        pdf_url=paper_data.get('pdf_url', ''),
                        category_domain=category_name,
                        primary_category=paper_data.get('primary_category', ''),
                        doi=paper_data.get('doi'),
                        journal_ref=paper_data.get('journal_ref')
                    )

                    # Extract keywords from title and abstract
                    paper.keywords = self._extract_keywords(paper.get_full_text())

                    self.papers.append(paper)
                    self.papers_by_category[category_name].append(paper)

                    # Index by keywords
                    for keyword in paper.keywords:
                        self.papers_by_keyword[keyword].append(paper)

                    category_count += 1

                self.index_stats['categories'][category_name] = category_count
                print(f"  ✅ Loaded {category_count} papers from {category_name}")

            except Exception as e:
                print(f"  ❌ Error loading {category_name}: {e}")

        self.index_stats['total_papers'] = len(self.papers)
        self.index_stats['last_indexed'] = datetime.now().isoformat()

        print(f"\n📊 Total papers indexed: {self.index_stats['total_papers']:,}")

    def _extract_keywords(self, text: str, min_word_len: int = 4, max_keywords: int = 50) -> List[str]:
        """Extract important keywords from text"""
        # Remove special characters and split
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Filter by length and remove common words
        stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were',
                     'their', 'which', 'these', 'such', 'using', 'used', 'also',
                     'show', 'shows', 'shown', 'results', 'paper', 'study'}

        keywords = [w for w in words if len(w) >= min_word_len and w not in stopwords]

        # Count frequency
        keyword_counts = defaultdict(int)
        for kw in keywords:
            keyword_counts[kw] += 1

        # Return top keywords by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, count in sorted_keywords[:max_keywords]]

    def search(self, query: str, limit: int = 10, category: Optional[str] = None) -> List[Paper]:
        """
        Search papers by query string

        Args:
            query: Search query
            limit: Maximum number of results
            category: Optional category filter

        Returns:
            List of relevant papers sorted by relevance
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-z]+\b', query_lower))

        # Choose papers to search
        papers_to_search = (
            self.papers_by_category.get(category, [])
            if category
            else self.papers
        )

        # Score each paper
        scored_papers = []
        for paper in papers_to_search:
            score = self._calculate_relevance(paper, query_lower, query_words)
            if score > 0:
                paper.relevance_score = score
                scored_papers.append(paper)

        # Sort by relevance
        scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)

        return scored_papers[:limit]

    def _calculate_relevance(self, paper: Paper, query: str, query_words: set) -> float:
        """Calculate relevance score for a paper"""
        score = 0.0

        full_text = paper.get_full_text()

        # Exact phrase match (high score)
        if query in full_text:
            score += 10.0

        # Title match (high importance)
        title_lower = paper.title.lower()
        for word in query_words:
            if word in title_lower:
                score += 5.0

        # Abstract match
        abstract_lower = paper.abstract.lower()
        for word in query_words:
            if word in abstract_lower:
                score += 1.0

        # Keyword match
        for word in query_words:
            if word in paper.keywords:
                score += 2.0

        return score

    def get_by_category(self, category: str, limit: Optional[int] = None) -> List[Paper]:
        """Get papers by category"""
        papers = self.papers_by_category.get(category, [])
        return papers[:limit] if limit else papers

    def get_random_sample(self, n: int = 10, category: Optional[str] = None) -> List[Paper]:
        """Get random sample of papers"""
        import random

        papers = (
            self.papers_by_category.get(category, [])
            if category
            else self.papers
        )

        return random.sample(papers, min(n, len(papers)))

    def get_recent_papers(self, n: int = 10, category: Optional[str] = None) -> List[Paper]:
        """Get most recently published papers"""
        papers = (
            self.papers_by_category.get(category, [])
            if category
            else self.papers
        )

        # Sort by published date (descending)
        sorted_papers = sorted(papers, key=lambda p: p.published, reverse=True)
        return sorted_papers[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics"""
        return {
            **self.index_stats,
            'categories_breakdown': {
                cat: len(papers)
                for cat, papers in self.papers_by_category.items()
            },
            'total_keywords': len(self.papers_by_keyword),
            'avg_papers_per_category': (
                self.index_stats['total_papers'] / len(self.papers_by_category)
                if self.papers_by_category else 0
            )
        }

    def export_index(self, output_file: str = "invention_data_index.json"):
        """Export index to JSON for faster loading"""
        index_data = {
            'stats': self.index_stats,
            'papers': [paper.to_dict() for paper in self.papers]
        }

        with open(output_file, 'w') as f:
            json.dump(index_data, f, indent=2)

        print(f"📝 Exported index to {output_file}")

    def search_multi_query(self, queries: List[str], limit_per_query: int = 5) -> List[Paper]:
        """
        Search with multiple queries and combine results

        Args:
            queries: List of search queries
            limit_per_query: Results per query

        Returns:
            Combined list of unique papers
        """
        all_papers = []
        seen_ids = set()

        for query in queries:
            results = self.search(query, limit=limit_per_query)
            for paper in results:
                if paper.arxiv_id not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(paper.arxiv_id)

        return all_papers


def main():
    """Demo usage"""
    print("🔍 Invention Data Indexer Demo\n")

    # Initialize indexer
    indexer = InventionDataIndexer()

    if not indexer.papers:
        print("\n⚠️  No papers found. Please run:")
        print("    python3 download_invention_data.py --sample")
        return

    # Show stats
    stats = indexer.get_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   Total papers: {stats['total_papers']:,}")
    print(f"   Categories: {len(stats['categories_breakdown'])}")
    print(f"   Total keywords: {stats['total_keywords']:,}")
    print(f"\n   Breakdown by category:")
    for cat, count in stats['categories_breakdown'].items():
        print(f"     - {cat}: {count:,} papers")

    # Demo searches
    print("\n🔍 Demo Searches:\n")

    search_queries = [
        "quantum computing",
        "graphene metamaterial",
        "energy storage battery",
        "3D printing nanoscale"
    ]

    for query in search_queries:
        results = indexer.search(query, limit=3)
        print(f"Query: '{query}'")
        print(f"Found {len(results)} results:")
        for i, paper in enumerate(results, 1):
            print(f"  {i}. {paper.title[:80]}...")
            print(f"     Category: {paper.category_domain} | Score: {paper.relevance_score:.1f}")
        print()

    # Export index
    indexer.export_index()

    print("\n✅ Indexer ready for invention generation!")


if __name__ == "__main__":
    main()
