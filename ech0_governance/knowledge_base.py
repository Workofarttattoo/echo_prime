#!/usr/bin/env python3
"""
ECH0-PRIME Knowledge Base Integration
Enhanced knowledge management system based on ech0 training datasets.
Provides domain-specific knowledge retrieval and reasoning augmentation.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import random
from datetime import datetime

from reasoning.llm_bridge import OllamaBridge


class KnowledgeBase:
    """
    Comprehensive knowledge base system for ECH0-PRIME.
    Integrates multiple domain-specific knowledge sources for enhanced reasoning.
    """

    def __init__(self, data_directory: str = "training_data"):
        self.data_directory = Path(data_directory)
        self.domains = {}
        self.llm_bridge = OllamaBridge(model="llama3.2")
        self.knowledge_index = {}

        # Load all available knowledge domains
        self._load_knowledge_domains()

        print(f"🧠 Knowledge Base loaded with {len(self.domains)} domains")

    def _load_knowledge_domains(self):
        """Load knowledge from training data files"""
        if not self.data_directory.exists():
            print(f"⚠️ Knowledge data directory not found: {self.data_directory}")
            return

        # Domain mapping from filenames
        domain_mapping = {
            "reasoning_dataset.json": "reasoning",
            "creativity_dataset.json": "creativity",
            "law_dataset.json": "law",
            "crypto_dataset.json": "cryptocurrency",
            "ai_ml_dataset.json": "artificial_intelligence",
            "materials_science_dataset.json": "materials_science",
            "advanced_software_dataset.json": "software_engineering",
            "prompt_engineering_dataset.json": "prompt_engineering",
            "stock_prediction_dataset.json": "finance",
            "court_prediction_dataset.json": "legal"
        }

        for file_path in self.data_directory.glob("*.json"):
            domain_name = domain_mapping.get(file_path.name)
            if domain_name:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Process and index the knowledge
                    self.domains[domain_name] = self._process_domain_data(data, domain_name)
                    print(f"  ✅ Loaded {domain_name}: {len(data)} knowledge items")

                except Exception as e:
                    print(f"  ❌ Failed to load {file_path.name}: {e}")

    def _process_domain_data(self, data: List[Dict], domain: str) -> Dict[str, Any]:
        """Process raw domain data into structured knowledge"""
        processed = {
            "entries": data,
            "categories": {},
            "concepts": set(),
            "index": {},
            "metadata": {
                "total_entries": len(data),
                "last_updated": datetime.now().isoformat(),
                "domain": domain
            }
        }

        # Build category index
        for entry in data:
            category = entry.get("category", "general")
            if category not in processed["categories"]:
                processed["categories"][category] = []
            processed["categories"][category].append(entry)

            # Extract concepts from instruction and output
            instruction_concepts = self._extract_concepts(entry.get("instruction", ""))
            output_concepts = self._extract_concepts(entry.get("output", ""))
            processed["concepts"].update(instruction_concepts + output_concepts)

            # Build search index
            key = f"{entry.get('instruction', '')} {entry.get('output', '')}".lower()
            processed["index"][key] = entry

        processed["concepts"] = list(processed["concepts"])
        return processed

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction - could be enhanced with NLP
        words = text.lower().split()
        concepts = []

        # Look for technical terms, proper nouns, and key phrases
        for i, word in enumerate(words):
            # Capitalized words (potential proper nouns)
            if word[0].isupper() and len(word) > 3:
                concepts.append(word)

            # Technical terms (underscores, hyphens)
            if '_' in word or '-' in word:
                concepts.append(word)

            # Multi-word concepts
            if i < len(words) - 1:
                bigram = f"{word} {words[i+1]}"
                if len(bigram) > 6:  # Avoid very short phrases
                    concepts.append(bigram)

        return list(set(concepts))[:5]  # Limit concepts per text

    def query_knowledge(self, query: str, domain: str = None,
                        max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information.

        Args:
            query: Search query
            domain: Specific domain to search (optional)
            max_results: Maximum number of results to return

        Returns:
            List of relevant knowledge entries
        """
        results = []
        query_lower = query.lower()

        # Determine which domains to search
        domains_to_search = [domain] if domain else list(self.domains.keys())

        for domain_name in domains_to_search:
            if domain_name not in self.domains:
                continue

            domain_data = self.domains[domain_name]

            # Search through indexed entries
            for index_key, entry in domain_data["index"].items():
                if query_lower in index_key:
                    relevance_score = self._calculate_relevance(query_lower, index_key)
                    results.append({
                        "entry": entry,
                        "domain": domain_name,
                        "relevance_score": relevance_score,
                        "matched_text": index_key[:100] + "..." if len(index_key) > 100 else index_key
                    })

        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text"""
        query_words = set(query.split())
        text_words = set(text.split())

        # Jaccard similarity
        intersection = len(query_words & text_words)
        union = len(query_words | text_words)

        if union == 0:
            return 0.0

        return intersection / union

    def get_domain_expertise(self, domain: str) -> Dict[str, Any]:
        """Get comprehensive information about a domain"""
        if domain not in self.domains:
            return {"error": f"Domain '{domain}' not found"}

        domain_data = self.domains[domain]

        return {
            "domain": domain,
            "total_entries": domain_data["metadata"]["total_entries"],
            "categories": list(domain_data["categories"].keys()),
            "key_concepts": domain_data["concepts"][:20],  # Top 20 concepts
            "sample_entries": random.sample(domain_data["entries"], min(3, len(domain_data["entries"]))),
            "last_updated": domain_data["metadata"]["last_updated"]
        }

    def enhance_reasoning_with_knowledge(self, reasoning_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance reasoning with relevant knowledge from the knowledge base.

        Args:
            reasoning_context: Current reasoning context

        Returns:
            Enhanced reasoning context with relevant knowledge
        """
        query = reasoning_context.get("query", "")
        domain = reasoning_context.get("domain", "reasoning")

        # Query knowledge base
        relevant_knowledge = self.query_knowledge(query, domain, max_results=3)

        # Integrate knowledge into reasoning context
        if relevant_knowledge:
            knowledge_context = []
            for item in relevant_knowledge:
                entry = item["entry"]
                knowledge_context.append({
                    "instruction": entry.get("instruction", ""),
                    "output": entry.get("output", ""),
                    "domain": item["domain"],
                    "category": entry.get("category", ""),
                    "relevance": item["relevance_score"]
                })

            reasoning_context["relevant_knowledge"] = knowledge_context

            # Generate knowledge-enhanced reasoning prompt
            knowledge_summary = self._summarize_relevant_knowledge(knowledge_context)
            reasoning_context["knowledge_enhanced_prompt"] = f"""
Based on relevant knowledge from the {domain} domain:

{knowledge_summary}

Original reasoning task: {query}
"""
        else:
            reasoning_context["relevant_knowledge"] = []
            reasoning_context["knowledge_enhanced_prompt"] = f"No directly relevant knowledge found for: {query}"

        return reasoning_context

    def _summarize_relevant_knowledge(self, knowledge_items: List[Dict]) -> str:
        """Summarize relevant knowledge items"""
        if not knowledge_items:
            return "No relevant knowledge available."

        summaries = []
        for item in knowledge_items:
            summary = f"• {item['instruction'][:50]}... → {item['output'][:100]}..."
            summaries.append(summary)

        return "\n".join(summaries)

    def add_knowledge_entry(self, domain: str, entry: Dict[str, Any]):
        """Add a new knowledge entry to the knowledge base"""
        if domain not in self.domains:
            self.domains[domain] = {
                "entries": [],
                "categories": {},
                "concepts": set(),
                "index": {},
                "metadata": {
                    "total_entries": 0,
                    "last_updated": datetime.now().isoformat(),
                    "domain": domain
                }
            }

        # Add entry
        self.domains[domain]["entries"].append(entry)
        self.domains[domain]["metadata"]["total_entries"] += 1
        self.domains[domain]["metadata"]["last_updated"] = datetime.now().isoformat()

        # Update index
        key = f"{entry.get('instruction', '')} {entry.get('output', '')}".lower()
        self.domains[domain]["index"][key] = entry

        # Update categories
        category = entry.get("category", "general")
        if category not in self.domains[domain]["categories"]:
            self.domains[domain]["categories"][category] = []
        self.domains[domain]["categories"][category].append(entry)

        print(f"✅ Added knowledge entry to {domain} domain")

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base"""
        total_entries = sum(domain_data["metadata"]["total_entries"]
                          for domain_data in self.domains.values())

        domain_stats = {}
        for domain_name, domain_data in self.domains.items():
            domain_stats[domain_name] = {
                "entries": domain_data["metadata"]["total_entries"],
                "categories": len(domain_data["categories"]),
                "concepts": len(domain_data["concepts"])
            }

        return {
            "total_domains": len(self.domains),
            "total_entries": total_entries,
            "domain_breakdown": domain_stats,
            "domains_list": list(self.domains.keys()),
            "last_updated": datetime.now().isoformat()
        }

    def export_knowledge_base(self, format: str = "json") -> str:
        """Export the knowledge base in various formats"""
        if format == "json":
            return json.dumps({
                "domains": self.domains,
                "statistics": self.get_knowledge_statistics()
            }, indent=2, default=str)
        elif format == "markdown":
            return self._export_markdown()
        else:
            return str(self.get_knowledge_statistics())

    def _export_markdown(self) -> str:
        """Export knowledge base as markdown"""
        md = ["# ECH0-PRIME Knowledge Base\n"]

        stats = self.get_knowledge_statistics()
        md.append(f"**Total Domains:** {stats['total_domains']}")
        md.append(f"**Total Entries:** {stats['total_entries']}\n")

        for domain_name, domain_info in stats['domain_breakdown'].items():
            md.append(f"## {domain_name.title()}")
            md.append(f"- Entries: {domain_info['entries']}")
            md.append(f"- Categories: {domain_info['categories']}")
            md.append(f"- Concepts: {domain_info['concepts']}\n")

        return "\n".join(md)


# Global knowledge base instance
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBase:
    """Get the global knowledge base instance"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base
