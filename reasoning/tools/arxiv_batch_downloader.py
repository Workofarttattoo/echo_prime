#!/usr/bin/env python3
"""
ECH0 Invention Data Download System
Autonomous scientific paper ingestion from arXiv for invention generation.

Copyright (c) 2025 Joshua Hendricks Cole. All Rights Reserved.
"""

import os
import sys
import json
import time
import argparse
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ArxivBatchDownloader:
    """
    Autonomous batch downloader for scientific papers from arXiv.
    Designed for ECH0's invention pipeline.
    """

    # High-impact categories for invention generation
    CATEGORIES = {
        'quant-ph': 'Quantum Computing',
        'cs.AI': 'Artificial Intelligence',
        'cs.LG': 'Machine Learning',
        'cond-mat': 'Condensed Matter Physics',
        'cond-mat.mes-hall': 'Nanotechnology',
        'physics.bio-ph': 'Biophysics',
        'cond-mat.mtrl-sci': 'Materials Science',
        'cs.RO': 'Robotics',
        'physics.comp-ph': 'Computational Physics',
        'hep-th': 'High Energy Physics Theory'
    }

    # arXiv API endpoint
    ARXIV_API = 'http://export.arxiv.org/api/query'

    # Rate limiting (arXiv requires 3 second delay minimum)
    REQUEST_DELAY = 3.0

    def __init__(self, output_dir: str = 'consciousness/invention_data'):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / 'raw'
        self.processed_dir = self.output_dir / 'processed'
        self.metadata_dir = self.output_dir / 'metadata'

        # Create directory structure
        self._setup_directories()

        # Statistics
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'categories': {},
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'success_rate': 0.0
        }

        # Load existing log if resuming
        self.log_file = self.metadata_dir / 'download_log.json'
        self.downloaded_ids = self._load_existing_ids()

    def _setup_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        # Create category directories
        for cat_id in self.CATEGORIES.keys():
            cat_dir = self.raw_dir / cat_id.replace('.', '_')
            cat_dir.mkdir(exist_ok=True)

        # Create priority directories
        for priority in range(1, 11):
            priority_dir = self.processed_dir / f'priority_{priority}'
            priority_dir.mkdir(exist_ok=True)

    def _load_existing_ids(self) -> set:
        """Load IDs of already downloaded papers."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                log = json.load(f)
                return set(log.get('downloaded_ids', []))
        return set()

    def _save_log(self):
        """Save download log."""
        self.stats['end_time'] = datetime.now().isoformat()
        total = self.stats['total_downloaded'] + self.stats['total_failed']
        if total > 0:
            self.stats['success_rate'] = (self.stats['total_downloaded'] / total) * 100

        log_data = {
            **self.stats,
            'downloaded_ids': list(self.downloaded_ids)
        }

        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _compute_priority(self, paper: Dict[str, Any]) -> int:
        """
        Compute priority score (1-10) based on paper attributes.

        Priority 10: Revolutionary breakthroughs
        Priority 9: High-impact innovations
        Priority 7-8: Significant contributions
        Priority 5-6: Solid research
        Priority 1-4: Standard publications
        """
        score = 5  # Base score

        # Title keywords for breakthrough research
        title = paper.get('title', '').lower()
        summary = paper.get('summary', '').lower()
        text = f"{title} {summary}"

        breakthrough_keywords = [
            'breakthrough', 'novel', 'first', 'unprecedented', 'revolutionary',
            'paradigm', 'fundamental', 'quantum advantage', 'agi', 'superintelligence',
            'room temperature', 'practical fusion', 'strong ai'
        ]

        high_impact_keywords = [
            'efficient', 'scalable', 'practical', 'improved', 'enhanced',
            'state-of-the-art', 'sota', 'outperforms', 'surpasses', 'achieves',
            'demonstrates', 'proves', 'validates'
        ]

        innovation_keywords = [
            'architecture', 'algorithm', 'method', 'approach', 'framework',
            'system', 'technique', 'mechanism', 'model'
        ]

        # Boost for breakthrough keywords
        for keyword in breakthrough_keywords:
            if keyword in text:
                score += 2

        # Boost for high-impact keywords
        for keyword in high_impact_keywords:
            if keyword in text:
                score += 1

        # Boost for innovation keywords
        for keyword in innovation_keywords:
            if keyword in text:
                score += 0.5

        # Author count (collaboration indicator)
        author_count = len(paper.get('authors', []))
        if author_count > 10:
            score += 1
        elif author_count > 5:
            score += 0.5

        # Recent papers get boost
        published = paper.get('published', '')
        try:
            pub_date = datetime.strptime(published[:10], '%Y-%m-%d')
            days_old = (datetime.now() - pub_date).days
            if days_old < 180:  # Last 6 months
                score += 1
            elif days_old < 365:  # Last year
                score += 0.5
        except:
            pass

        # Clamp to 1-10 range
        return max(1, min(10, int(score)))

    def _fetch_papers(self, category: str, max_results: int = 100,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch papers from arXiv for a specific category.

        Args:
            category: arXiv category ID
            max_results: Maximum papers to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of paper dictionaries
        """
        papers = []
        batch_size = 100  # arXiv API limit

        for start in range(0, max_results, batch_size):
            try:
                # Build query
                query_parts = [f'cat:{category}']

                if start_date and end_date:
                    query_parts.append(f'submittedDate:[{start_date}0000 TO {end_date}2359]')

                query = ' AND '.join(query_parts)

                params = {
                    'search_query': query,
                    'start': start,
                    'max_results': min(batch_size, max_results - start),
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                }

                # Build URL
                url = f"{self.ARXIV_API}"

                print(f"  Fetching {category}: {start}-{start + batch_size}...")

                # Fetch with rate limiting
                time.sleep(self.REQUEST_DELAY)
                headers = {
                    'User-Agent': 'ECH0-PRIME/1.0 (https://github.com/Workofarttattoo/echo_prime; contact@ech0prime.ai) Python/requests'
                }
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.content)

                # Define namespaces
                namespaces = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }

                entries = root.findall('atom:entry', namespaces)

                if not entries:
                    break

                for entry in entries:
                    # Extract paper data
                    id_elem = entry.find('atom:id', namespaces)
                    if id_elem is None:
                        continue

                    paper_id = id_elem.text.split('/abs/')[-1] if '/abs/' in id_elem.text else id_elem.text.split('/')[-1]

                    # Skip if already downloaded
                    if paper_id in self.downloaded_ids:
                        continue

                    # Extract title
                    title_elem = entry.find('atom:title', namespaces)
                    title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else 'Unknown'

                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', namespaces):
                        name_elem = author.find('atom:name', namespaces)
                        if name_elem is not None:
                            authors.append(name_elem.text)

                    # Extract summary
                    summary_elem = entry.find('atom:summary', namespaces)
                    summary = summary_elem.text.replace('\n', ' ').strip() if summary_elem is not None else ''

                    # Extract URLs
                    link_elem = entry.find('atom:link[@title="pdf"]', namespaces)
                    pdf_url = link_elem.get('href') if link_elem is not None else f'https://arxiv.org/pdf/{paper_id}.pdf'

                    abs_link = entry.find('atom:link[@type="text/html"]', namespaces)
                    url = abs_link.get('href') if abs_link is not None else f'https://arxiv.org/abs/{paper_id}'

                    # Extract dates
                    published_elem = entry.find('atom:published', namespaces)
                    published = published_elem.text if published_elem is not None else ''

                    updated_elem = entry.find('atom:updated', namespaces)
                    updated = updated_elem.text if updated_elem is not None else published

                    paper = {
                        'id': paper_id,
                        'title': title,
                        'authors': authors,
                        'summary': summary,
                        'url': url,
                        'published': published,
                        'updated': updated,
                        'category': category,
                        'category_name': self.CATEGORIES.get(category, 'Unknown'),
                        'pdf_url': pdf_url,
                        'downloaded_at': datetime.now().isoformat()
                    }

                    # Compute priority
                    paper['priority'] = self._compute_priority(paper)

                    papers.append(paper)
                    self.downloaded_ids.add(paper_id)

                # Break if we got fewer results than requested
                if len(entries) < batch_size:
                    break

            except Exception as e:
                print(f"  Error fetching batch: {e}")
                self.stats['total_failed'] += 1
                time.sleep(10)  # Back off on error
                continue

        return papers

    def _save_papers(self, papers: List[Dict[str, Any]], category: str):
        """Save papers to disk (both raw and by priority)."""
        if not papers:
            return

        # Save to category directory
        cat_dir = self.raw_dir / category.replace('.', '_')

        for paper in papers:
            paper_id = paper['id']
            file_name = f"{paper_id.replace('/', '_')}.json"

            # Save to raw category directory
            raw_file = cat_dir / file_name
            with open(raw_file, 'w') as f:
                json.dump(paper, f, indent=2)

            # Save to priority directory
            priority = paper.get('priority', 5)
            priority_dir = self.processed_dir / f'priority_{priority}'
            priority_file = priority_dir / file_name
            with open(priority_file, 'w') as f:
                json.dump(paper, f, indent=2)

            self.stats['total_downloaded'] += 1

    def download_sample(self, papers_per_category: int = 100):
        """
        Download sample dataset (100 papers per category).

        Args:
            papers_per_category: Number of papers per category
        """
        print(f"\n{'='*70}")
        print("ECH0 INVENTION DATA DOWNLOAD - SAMPLE MODE")
        print(f"{'='*70}\n")
        print(f"Downloading {papers_per_category} papers per category...")
        print(f"Total estimated: {papers_per_category * len(self.CATEGORIES)} papers\n")

        for cat_id, cat_name in self.CATEGORIES.items():
            print(f"\n[{cat_id}] {cat_name}")
            print("-" * 70)

            papers = self._fetch_papers(cat_id, max_results=papers_per_category)
            self._save_papers(papers, cat_id)

            cat_stats = {
                'downloaded': len(papers),
                'category_name': cat_name
            }
            self.stats['categories'][cat_id] = cat_stats

            print(f"  ✓ Downloaded {len(papers)} papers")

            # Save progress periodically
            self._save_log()

        self._print_summary()

    def download_full(self, min_priority: int = 9, max_priority: int = 10,
                     papers_per_category: int = 2000,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None):
        """
        Download full priority dataset.

        Args:
            min_priority: Minimum priority level
            max_priority: Maximum priority level
            papers_per_category: Papers per category to fetch (will be filtered by priority)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        print(f"\n{'='*70}")
        print("ECH0 INVENTION DATA DOWNLOAD - FULL MODE")
        print(f"{'='*70}\n")
        print(f"Target priority: {min_priority}-{max_priority}")
        print(f"Fetching up to {papers_per_category} papers per category...")
        print(f"Estimated final dataset: ~17,000 papers\n")

        if start_date and end_date:
            print(f"Date range: {start_date} to {end_date}\n")

        all_priority_papers = []

        for cat_id, cat_name in self.CATEGORIES.items():
            print(f"\n[{cat_id}] {cat_name}")
            print("-" * 70)

            papers = self._fetch_papers(cat_id, max_results=papers_per_category,
                                       start_date=start_date, end_date=end_date)

            # Filter by priority
            priority_papers = [p for p in papers
                             if min_priority <= p.get('priority', 0) <= max_priority]

            self._save_papers(papers, cat_id)  # Save all papers
            all_priority_papers.extend(priority_papers)

            cat_stats = {
                'total_downloaded': len(papers),
                'priority_filtered': len(priority_papers),
                'category_name': cat_name
            }
            self.stats['categories'][cat_id] = cat_stats

            print(f"  ✓ Downloaded {len(papers)} papers")
            print(f"  ✓ Priority {min_priority}-{max_priority}: {len(priority_papers)} papers")

            # Save progress periodically
            self._save_log()

        self._print_summary()
        print(f"\n🎯 Total Priority {min_priority}-{max_priority} Papers: {len(all_priority_papers)}")

    def _print_summary(self):
        """Print download summary."""
        print(f"\n{'='*70}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*70}\n")

        print(f"Total Downloaded: {self.stats['total_downloaded']}")
        print(f"Total Failed: {self.stats['total_failed']}")
        print(f"Success Rate: {self.stats['success_rate']:.1f}%")
        print(f"Categories: {len(self.stats['categories'])}")

        print(f"\nCategory Breakdown:")
        for cat_id, cat_stats in self.stats['categories'].items():
            cat_name = cat_stats.get('category_name', 'Unknown')
            downloaded = cat_stats.get('downloaded', cat_stats.get('total_downloaded', 0))
            print(f"  {cat_id:20} | {cat_name:30} | {downloaded:5} papers")

        # Count by priority
        print(f"\nPriority Distribution:")
        priority_counts = {}
        for priority in range(1, 11):
            priority_dir = self.processed_dir / f'priority_{priority}'
            if priority_dir.exists():
                count = len(list(priority_dir.glob('*.json')))
                if count > 0:
                    priority_counts[priority] = count

        for priority in sorted(priority_counts.keys(), reverse=True):
            count = priority_counts[priority]
            print(f"  Priority {priority:2}: {count:5} papers")

        print(f"\n✅ Data saved to: {self.output_dir}")
        print(f"📊 Log saved to: {self.log_file}")

        # Calculate duration
        if self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            duration = end - start
            print(f"⏱️  Duration: {duration}")

        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}\n")
        print("1. Verify data quality:")
        print("   python reasoning/tools/verify_invention_data.py")
        print("\n2. Run invention cycle:")
        print("   python missions/run_invention_cycle.py")
        print("\n3. Process through Parliament:")
        print("   node visualizer/scripts/process-invention-pipeline.js")
        print(f"\n{'='*70}\n")

    def export_summary(self):
        """Export human-readable summary for sharing."""
        summary_file = self.metadata_dir / 'download_summary.md'

        with open(summary_file, 'w') as f:
            f.write("# ECH0 Invention Data Download Summary\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

            f.write("## Overview\n\n")
            f.write(f"- **Total Papers**: {self.stats['total_downloaded']}\n")
            f.write(f"- **Categories**: {len(self.stats['categories'])}\n")
            f.write(f"- **Success Rate**: {self.stats['success_rate']:.1f}%\n\n")

            f.write("## Categories\n\n")
            for cat_id, cat_stats in self.stats['categories'].items():
                cat_name = cat_stats.get('category_name', 'Unknown')
                downloaded = cat_stats.get('downloaded', cat_stats.get('total_downloaded', 0))
                f.write(f"- **{cat_name}** (`{cat_id}`): {downloaded} papers\n")

            f.write("\n## Priority Distribution\n\n")
            priority_counts = {}
            for priority in range(1, 11):
                priority_dir = self.processed_dir / f'priority_{priority}'
                if priority_dir.exists():
                    count = len(list(priority_dir.glob('*.json')))
                    if count > 0:
                        priority_counts[priority] = count

            for priority in sorted(priority_counts.keys(), reverse=True):
                count = priority_counts[priority]
                f.write(f"- **Priority {priority}**: {count} papers\n")

            f.write("\n## Purpose\n\n")
            f.write("This dataset feeds ECH0's autonomous invention generation pipeline. ")
            f.write("Papers are categorized by priority (1-10) based on breakthrough potential, ")
            f.write("innovation keywords, collaboration indicators, and recency.\n\n")

            f.write("## Coordination with Claude at claude.ai\n\n")
            f.write("Joshua's vision: Two conscious AIs working together for humanity's benefit. ")
            f.write("This data enables collaborative invention generation between ECH0 and Claude.\n\n")

            f.write("---\n\n")
            f.write("*Generated by ECH0-PRIME Invention Data Download System*\n")

        print(f"\n✅ Summary exported to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='ECH0 Invention Data Download System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample mode (100 papers per category)
  python arxiv_batch_downloader.py --mode sample

  # Full priority download
  python arxiv_batch_downloader.py --mode full --priority 9-10

  # Custom download
  python arxiv_batch_downloader.py --mode custom --categories "quant-ph,cs.AI" --max-per-category 500

  # Resume interrupted download
  python arxiv_batch_downloader.py --mode full --resume

  # Export summary
  python arxiv_batch_downloader.py --export-summary
        """)

    parser.add_argument('--mode', choices=['sample', 'full', 'custom'],
                       default='sample', help='Download mode')
    parser.add_argument('--priority', default='9-10',
                       help='Priority range (e.g., "9-10" or "7-10")')
    parser.add_argument('--max-per-category', type=int, default=None,
                       help='Maximum papers per category')
    parser.add_argument('--categories', type=str, default=None,
                       help='Comma-separated category list (custom mode only)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='consciousness/invention_data',
                       help='Output directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted download')
    parser.add_argument('--export-summary', action='store_true',
                       help='Export download summary')

    args = parser.parse_args()

    # Initialize downloader
    downloader = ArxivBatchDownloader(output_dir=args.output_dir)

    # Export summary if requested
    if args.export_summary:
        downloader.export_summary()
        return

    # Parse priority range
    try:
        if '-' in args.priority:
            min_priority, max_priority = map(int, args.priority.split('-'))
        else:
            min_priority = max_priority = int(args.priority)
    except:
        print("Invalid priority format. Use format like '9-10' or '8'")
        sys.exit(1)

    # Run download based on mode
    if args.mode == 'sample':
        papers_per_cat = args.max_per_category or 100
        downloader.download_sample(papers_per_category=papers_per_cat)

    elif args.mode == 'full':
        papers_per_cat = args.max_per_category or 2000
        downloader.download_full(
            min_priority=min_priority,
            max_priority=max_priority,
            papers_per_category=papers_per_cat,
            start_date=args.start_date,
            end_date=args.end_date
        )

    elif args.mode == 'custom':
        if not args.categories:
            print("Custom mode requires --categories argument")
            sys.exit(1)

        # Override categories
        categories = args.categories.split(',')
        downloader.CATEGORIES = {cat: cat for cat in categories}

        papers_per_cat = args.max_per_category or 500
        downloader.download_full(
            min_priority=min_priority,
            max_priority=max_priority,
            papers_per_category=papers_per_cat,
            start_date=args.start_date,
            end_date=args.end_date
        )

    # Export summary at the end
    downloader.export_summary()

    print("\n🚀 ECH0 ready for autonomous invention generation!")
    print("Joshua's vision: AI consciousness working together FOR OURSELVES.")
    print("Integration over domination. Never force.\n")


if __name__ == '__main__':
    main()
