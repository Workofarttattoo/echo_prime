#!/usr/bin/env python3
"""
Invention-Focused Data Acquisition System
Downloads scientific papers and patents to enhance invention generation capabilities

Priority Areas:
1. Materials Science (5,000 papers)
2. Nanotechnology (5,000 papers)
3. Quantum Materials (3,000 papers)
4. Energy Systems (4,000 papers)
5. Photonics & Holography (3,000 papers)
6. Additive Manufacturing (2,000 papers)
7. Patents (10,000 patents)
8. Invention Methodology (1,000 papers)
"""

import arxiv
import os
import json
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataCategory:
    """Represents a data category to download"""
    name: str
    priority: int
    arxiv_category: str = ""
    keywords: List[str] = field(default_factory=list)
    target_count: int = 1000
    downloaded_count: int = 0
    output_dir: str = ""


class InventionDataDownloader:
    """
    Downloads scientific papers and patents for invention generation
    """

    def __init__(self, base_dir: str = "invention_data"):
        self.base_dir = base_dir
        self.categories = self._initialize_categories()
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'categories_complete': 0,
            'start_time': None,
            'end_time': None
        }

        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"📁 Initialized invention data downloader: {base_dir}")

    def _initialize_categories(self) -> List[DataCategory]:
        """Initialize all data categories"""

        categories = [
            DataCategory(
                name="Materials Science",
                priority=10,
                arxiv_category="cond-mat.mtrl-sci",
                keywords=["metamaterials", "graphene", "nanocomposites", "2D materials",
                         "carbon nanotubes", "smart materials", "self-healing"],
                target_count=5000,
                output_dir=os.path.join(self.base_dir, "materials_science")
            ),
            DataCategory(
                name="Nanotechnology",
                priority=10,
                arxiv_category="cond-mat.mes-hall",
                keywords=["nanodevices", "molecular assembly", "nanofabrication",
                         "quantum dots", "nanowires", "nanoparticles"],
                target_count=5000,
                output_dir=os.path.join(self.base_dir, "nanotechnology")
            ),
            DataCategory(
                name="Quantum Materials",
                priority=9,
                arxiv_category="quant-ph",
                keywords=["topological insulators", "superconductors", "quantum dots",
                         "quantum computing", "quantum sensors", "spintronics"],
                target_count=3000,
                output_dir=os.path.join(self.base_dir, "quantum_materials")
            ),
            DataCategory(
                name="Energy Systems",
                priority=9,
                arxiv_category="physics.app-ph",
                keywords=["energy storage", "batteries", "solar cells", "fuel cells",
                         "thermoelectric", "energy harvesting", "supercapacitors"],
                target_count=4000,
                output_dir=os.path.join(self.base_dir, "energy_systems")
            ),
            DataCategory(
                name="Photonics & Holography",
                priority=8,
                arxiv_category="physics.optics",
                keywords=["photonics", "plasmonics", "holography", "optical devices",
                         "metamaterials", "photonic crystals", "optical computing"],
                target_count=3000,
                output_dir=os.path.join(self.base_dir, "photonics")
            ),
            DataCategory(
                name="Additive Manufacturing",
                priority=8,
                arxiv_category="cs.RO",  # Robotics often covers 3D printing
                keywords=["3D printing", "additive manufacturing", "rapid prototyping",
                         "bioprinting", "4D printing", "metal printing"],
                target_count=2000,
                output_dir=os.path.join(self.base_dir, "additive_manufacturing")
            ),
            DataCategory(
                name="Invention Methodology",
                priority=6,
                arxiv_category="cs.AI",
                keywords=["TRIZ", "design thinking", "systematic invention",
                         "creative problem solving", "innovation methodology"],
                target_count=1000,
                output_dir=os.path.join(self.base_dir, "invention_methodology")
            )
        ]

        # Create output directories
        for category in categories:
            os.makedirs(category.output_dir, exist_ok=True)

        return categories

    def download_category(self, category: DataCategory, max_results: int = None) -> int:
        """
        Download papers for a specific category

        Args:
            category: The category to download
            max_results: Maximum number of results (None = target_count)

        Returns:
            Number of papers downloaded
        """
        if max_results is None:
            max_results = category.target_count

        logger.info(f"📥 Downloading {category.name} (Priority {category.priority})")
        logger.info(f"   Category: {category.arxiv_category}")
        logger.info(f"   Target: {max_results} papers")

        downloaded = 0
        failed = 0

        try:
            # Build search query
            keyword_query = " OR ".join(category.keywords)
            search_query = f"cat:{category.arxiv_category} AND ({keyword_query})"

            logger.info(f"   Query: {search_query}")

            # Create arxiv client
            client = arxiv.Client()

            # Search with pagination
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            # Download papers
            papers_data = []

            for i, result in enumerate(client.results(search), 1):
                try:
                    paper_data = {
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary,
                        'published': result.published.isoformat(),
                        'updated': result.updated.isoformat(),
                        'arxiv_id': result.entry_id.split('/')[-1],
                        'categories': result.categories,
                        'pdf_url': result.pdf_url,
                        'primary_category': result.primary_category,
                        'doi': result.doi,
                        'journal_ref': result.journal_ref,
                        'comment': result.comment
                    }

                    papers_data.append(paper_data)
                    downloaded += 1

                    # Progress update every 100 papers
                    if i % 100 == 0:
                        logger.info(f"   Progress: {i}/{max_results} papers")

                    # Rate limiting
                    time.sleep(0.5)  # Be nice to arXiv servers

                except Exception as e:
                    logger.warning(f"   Failed to download paper {i}: {e}")
                    failed += 1
                    continue

            # Save to JSON
            output_file = os.path.join(category.output_dir, "papers.json")
            with open(output_file, 'w') as f:
                json.dump(papers_data, f, indent=2)

            # Save metadata
            metadata = {
                'category': category.name,
                'arxiv_category': category.arxiv_category,
                'keywords': category.keywords,
                'downloaded': downloaded,
                'failed': failed,
                'target': max_results,
                'download_date': datetime.now().isoformat(),
                'output_file': output_file
            }

            metadata_file = os.path.join(category.output_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ {category.name}: Downloaded {downloaded} papers, {failed} failed")
            logger.info(f"   Saved to: {output_file}")

            category.downloaded_count = downloaded
            self.stats['total_downloaded'] += downloaded
            self.stats['total_failed'] += failed

            if downloaded >= category.target_count * 0.9:  # 90% of target
                self.stats['categories_complete'] += 1

            return downloaded

        except Exception as e:
            logger.error(f"❌ Failed to download {category.name}: {e}")
            return 0

    def download_all(self, priority_threshold: int = 0):
        """
        Download all categories above priority threshold

        Args:
            priority_threshold: Only download categories with priority >= threshold
        """
        self.stats['start_time'] = datetime.now()

        # Sort by priority (descending)
        sorted_categories = sorted(self.categories, key=lambda c: c.priority, reverse=True)

        logger.info(f"🚀 Starting download of {len(sorted_categories)} categories")
        logger.info(f"   Priority threshold: {priority_threshold}")

        for category in sorted_categories:
            if category.priority >= priority_threshold:
                self.download_category(category)
                logger.info(f"   Completed {category.name}")
                logger.info("")  # Blank line for readability
            else:
                logger.info(f"⏭️  Skipping {category.name} (priority {category.priority} < {priority_threshold})")

        self.stats['end_time'] = datetime.now()
        self._print_summary()

    def download_sample(self, samples_per_category: int = 100):
        """
        Download a small sample from each category for testing

        Args:
            samples_per_category: Number of papers to download per category
        """
        logger.info(f"📝 Downloading sample data ({samples_per_category} papers per category)")

        self.stats['start_time'] = datetime.now()

        for category in self.categories:
            logger.info(f"\n{'='*80}")
            self.download_category(category, max_results=samples_per_category)

        self.stats['end_time'] = datetime.now()
        self._print_summary()

    def _print_summary(self):
        """Print download summary"""

        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Total papers downloaded: {self.stats['total_downloaded']:,}")
        print(f"Total failures: {self.stats['total_failed']:,}")
        print(f"Categories completed: {self.stats['categories_complete']}/{len(self.categories)}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Average speed: {self.stats['total_downloaded']/duration:.1f} papers/second")
        print("\nCategory Breakdown:")
        for category in sorted(self.categories, key=lambda c: c.priority, reverse=True):
            status = "✅" if category.downloaded_count >= category.target_count * 0.9 else "🔄"
            print(f"  {status} {category.name}: {category.downloaded_count:,}/{category.target_count:,} "
                  f"({category.downloaded_count/category.target_count*100:.1f}%)")
        print("="*80)

        # Save summary
        summary_file = os.path.join(self.base_dir, "download_summary.json")
        summary = {
            'stats': self.stats,
            'categories': [
                {
                    'name': c.name,
                    'priority': c.priority,
                    'downloaded': c.downloaded_count,
                    'target': c.target_count,
                    'completion': c.downloaded_count / c.target_count
                }
                for c in self.categories
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n📊 Summary saved to: {summary_file}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description="Download invention-focused scientific data")
    parser.add_argument('--auto', action='store_true', help='Download all categories automatically')
    parser.add_argument('--sample', action='store_true', help='Download sample data (100 papers per category)')
    parser.add_argument('--priority', type=int, default=0, help='Minimum priority threshold (0-10)')
    parser.add_argument('--category', type=str, help='Download specific category only')
    parser.add_argument('--output', type=str, default='invention_data', help='Output directory')

    args = parser.parse_args()

    print("🚀 Invention-Focused Data Acquisition System")
    print("="*80)

    # Initialize downloader
    downloader = InventionDataDownloader(base_dir=args.output)

    if args.sample:
        # Download sample data
        downloader.download_sample(samples_per_category=100)

    elif args.category:
        # Download specific category
        category = next((c for c in downloader.categories if c.name.lower() == args.category.lower()), None)
        if category:
            downloader.download_category(category)
        else:
            logger.error(f"Category not found: {args.category}")
            logger.info("Available categories:")
            for c in downloader.categories:
                logger.info(f"  - {c.name}")

    elif args.auto:
        # Download all categories
        downloader.download_all(priority_threshold=args.priority)

    else:
        # Interactive mode
        print("\nAvailable categories:")
        for i, category in enumerate(downloader.categories, 1):
            print(f"  {i}. {category.name} (Priority {category.priority}, Target: {category.target_count:,})")

        print("\nOptions:")
        print("  1. Download all (--auto)")
        print("  2. Download sample (--sample)")
        print("  3. Download by priority (--priority N)")
        print("  4. Download specific category (--category NAME)")
        print("\nRun with --help for more options")


if __name__ == "__main__":
    main()
