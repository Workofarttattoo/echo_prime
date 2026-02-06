#!/usr/bin/env python3
"""
Enhanced Invention Cycle with Downloaded Paper Integration

Uses the full invention_data dataset (23,000+ papers) instead of live arXiv queries
Provides richer context for higher-quality inventions
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.llm_bridge import OllamaBridge
from ech0_governance.evaluators import Parliament

# Import the indexer
try:
    from invention_data_indexer import InventionDataIndexer
except ImportError:
    print("⚠️  invention_data_indexer.py not found in path")
    InventionDataIndexer = None


class EnhancedInventionCycle:
    """
    Enhanced invention generation using full paper dataset
    """

    def __init__(self, llm_model: str = "ech0-unified-14b-enhanced"):
        self.llm = OllamaBridge(model=llm_model)
        self.parliament = Parliament(self.llm)

        # Initialize paper indexer
        if InventionDataIndexer:
            self.indexer = InventionDataIndexer()
            if not self.indexer.papers:
                print("⚠️  No papers loaded. Run: python3 download_invention_data.py --sample")
                self.indexer = None
        else:
            self.indexer = None
            print("⚠️  Using fallback mode without paper database")

    def get_papers_for_context(self,
                                 queries: List[str] = None,
                                 categories: List[str] = None,
                                 n_papers: int = 20) -> List[Dict[str, Any]]:
        """
        Get papers for invention context

        Args:
            queries: Search queries for relevant papers
            categories: Category filters
            n_papers: Number of papers to retrieve

        Returns:
            List of paper dictionaries
        """
        if not self.indexer:
            print("⚠️  No indexer available, using empty context")
            return []

        papers = []

        # Multi-query search
        if queries:
            papers = self.indexer.search_multi_query(queries, limit_per_query=n_papers // len(queries))

        # Category-based retrieval
        elif categories:
            for category in categories:
                cat_papers = self.indexer.get_random_sample(
                    n=n_papers // len(categories),
                    category=category
                )
                papers.extend(cat_papers)

        # Random sample across all categories
        else:
            papers = self.indexer.get_random_sample(n=n_papers)

        return [p.to_dict() for p in papers]

    def format_papers_context(self, papers: List[Dict[str, Any]], max_abstract_len: int = 200) -> str:
        """Format papers into context string for LLM"""
        lines = []
        for i, paper in enumerate(papers, 1):
            abstract = paper['abstract'][:max_abstract_len] + "..."
            lines.append(
                f"{i}. {paper['title']}\n"
                f"   Category: {paper['category_domain']} | arXiv: {paper['arxiv_id']}\n"
                f"   Abstract: {abstract}\n"
            )
        return "\n".join(lines)

    def generate_inventions(self,
                             focus_area: str = "cross-domain innovation",
                             queries: List[str] = None,
                             categories: List[str] = None,
                             n_papers: int = 20,
                             n_inventions: int = 10) -> str:
        """
        Generate inventions from papers

        Args:
            focus_area: Area to focus invention generation
            queries: Search queries for papers
            categories: Category filters
            n_papers: Number of papers to use as context
            n_inventions: Number of inventions to generate

        Returns:
            JSON string of inventions
        """
        print(f"\n🚀 Enhanced Invention Generation")
        print(f"   Focus: {focus_area}")
        print(f"   Papers: {n_papers}")
        print(f"   Target inventions: {n_inventions}")

        # Get papers
        print(f"\n[PHASE 1] Retrieving papers...")
        papers = self.get_papers_for_context(queries, categories, n_papers)

        if not papers:
            print("⚠️  No papers available")
            return json.dumps([])

        print(f"   Retrieved {len(papers)} papers")
        print(f"   Categories: {set(p['category_domain'] for p in papers)}")

        # Format context
        context = self.format_papers_context(papers)

        # Generate inventions
        print(f"\n[PHASE 2] Generating {n_inventions} inventions...")

        prompt = f"""You are ECH0 Prime, an advanced AI invention system. Analyze these scientific papers and generate {n_inventions} breakthrough invention concepts.

FOCUS AREA: {focus_area}

SCIENTIFIC PAPERS:
{context}

TASK: Synthesize these papers into {n_inventions} novel, ground-breaking invention concepts.

For each invention, provide:
1. title: Clear, descriptive title
2. scientific_principle: The core scientific breakthrough (intersection of fields/concepts)
3. category: Primary invention domain (Materials Science, Nanotechnology, Quantum, Energy, Photonics, Additive Manufacturing, or Methodology)
4. proof_of_concept: Concrete steps to build a working prototype
5. required_resources: List of materials, equipment, and expertise needed
6. success_criteria: Measurable outcomes to validate the invention
7. estimated_timeline: Development timeframe
8. expected_impact: Transformative potential and applications
9. confidence_pct: Your confidence this will work (0-100)

Output ONLY valid JSON in this exact format:
[
  {{
    "title": "...",
    "scientific_principle": "...",
    "category": "...",
    "proof_of_concept": "...",
    "required_resources": ["...", "..."],
    "success_criteria": ["...", "..."],
    "estimated_timeline": "...",
    "expected_impact": "...",
    "confidence_pct": 85
  }}
]

Generate creative, feasible inventions that combine insights from multiple papers."""

        raw_output = self.llm.query(prompt)

        # Try to extract JSON
        try:
            # Look for JSON array in output
            start = raw_output.find('[')
            end = raw_output.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = raw_output[start:end]
                inventions = json.loads(json_str)
                print(f"   ✅ Generated {len(inventions)} inventions")
                return json.dumps(inventions, indent=2)
        except Exception as e:
            print(f"   ⚠️  JSON parsing failed: {e}")

        return raw_output

    def filter_by_parliament(self, inventions_json: str) -> str:
        """Filter inventions through Parliament governance"""
        print(f"\n[PHASE 3] Parliament Review...")

        try:
            inventions = json.loads(inventions_json)
        except:
            print("⚠️  Invalid JSON, skipping parliament filter")
            return inventions_json

        # Parliament scoring
        filtered = []
        for inv in inventions:
            # Simple scoring based on confidence and criteria
            confidence = inv.get('confidence_pct', 0)
            success_criteria_count = len(inv.get('success_criteria', []))
            resource_count = len(inv.get('required_resources', []))

            # Parliament prefers high confidence, clear success criteria, reasonable resources
            score = confidence
            score += success_criteria_count * 5
            score -= max(0, resource_count - 5) * 2  # Penalize resource-heavy inventions

            inv['parliament_score'] = score

            # Filter: Keep if score > 70
            if score >= 70:
                filtered.append(inv)

        print(f"   Parliament approved: {len(filtered)}/{len(inventions)} inventions")
        return json.dumps(filtered, indent=2)

    def run_full_cycle(self,
                        focus_area: str = "cross-domain innovation",
                        queries: List[str] = None,
                        categories: List[str] = None,
                        output_file: str = "ech0_enhanced_inventions.json") -> str:
        """
        Run complete invention generation cycle

        Args:
            focus_area: Innovation focus area
            queries: Search queries for papers
            categories: Category filters
            output_file: Output JSON file

        Returns:
            Path to output file
        """
        print("\n" + "="*80)
        print("ECH0 ENHANCED INVENTION GENERATION CYCLE")
        print("="*80)

        # Generate
        inventions = self.generate_inventions(
            focus_area=focus_area,
            queries=queries,
            categories=categories
        )

        # Filter
        filtered = self.filter_by_parliament(inventions)

        # Save
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write(filtered)

        print(f"\n✅ Invention cycle complete!")
        print(f"   Output: {output_path}")
        print(f"   Size: {len(json.loads(filtered))} inventions")

        return str(output_path)


def main():
    """Main execution"""

    # Example 1: Cross-domain search
    print("\n" + "="*80)
    print("EXAMPLE 1: Cross-Domain Innovation (Quantum + Materials)")
    print("="*80)

    cycle = EnhancedInventionCycle()

    cycle.run_full_cycle(
        focus_area="Quantum materials for next-generation computing",
        queries=["quantum computing", "superconductors", "metamaterials", "photonics"],
        output_file="inventions_quantum_materials.json"
    )

    # Example 2: Energy innovation
    print("\n" + "="*80)
    print("EXAMPLE 2: Energy Storage Innovation")
    print("="*80)

    cycle.run_full_cycle(
        focus_area="Revolutionary energy storage and harvesting",
        categories=["energy_systems", "materials_science"],
        output_file="inventions_energy.json"
    )

    # Example 3: Nanotechnology applications
    print("\n" + "="*80)
    print("EXAMPLE 3: Nanotechnology Applications")
    print("="*80)

    cycle.run_full_cycle(
        focus_area="Medical and manufacturing nanotechnology",
        categories=["nanotechnology", "additive_manufacturing"],
        output_file="inventions_nanotech.json"
    )

    print("\n" + "="*80)
    print("ALL CYCLES COMPLETE")
    print("="*80)
    print("\nGenerated invention files:")
    print("  - inventions_quantum_materials.json")
    print("  - inventions_energy.json")
    print("  - inventions_nanotech.json")


if __name__ == "__main__":
    main()
