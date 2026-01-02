#!/usr/bin/env python3
"""
ECH0-PRIME Training Data Integration System
Integrates the complete ECH0 training datasets into ECH0-PRIME's learning capabilities.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import os
import json
import asyncio
import random
import statistics
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

try:
    from learning.compressed_knowledge_base import CompressedKnowledgeBase
    compressed_kb_available = True
except ImportError:
    compressed_kb_available = False
    print("Warning: CompressedKnowledgeBase not available, using simplified mode")

try:
    from learning.data_compressor import DataCompressor
    data_compressor_available = True
except ImportError:
    data_compressor_available = False
    print("Warning: DataCompressor not available, using simplified mode")


@dataclass
class TrainingSample:
    """Represents a single training sample from the ECH0 datasets."""
    instruction: str
    input: str
    output: str
    domain: str
    category: str
    difficulty: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_prompt(self) -> str:
        """Generate the full prompt for this sample."""
        if self.input:
            return f"{self.instruction}\n\nInput: {self.input}"
        return self.instruction

    @property
    def quality_score(self) -> float:
        """Calculate a quality score based on metadata and content."""
        score = 0.5  # Base score

        # Difficulty bonus
        difficulty_scores = {"easy": 0.1, "medium": 0.2, "hard": 0.3, "expert": 0.4}
        score += difficulty_scores.get(self.difficulty, 0)

        # Grounded bonus
        if self.metadata.get("grounded", False):
            score += 0.2

        # Source quality bonus
        if "verified" in self.metadata.get("source", ""):
            score += 0.2

        return min(score, 1.0)


@dataclass
class DatasetStats:
    """Statistics for a training dataset."""
    name: str
    total_samples: int = 0
    domains: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)
    difficulties: Dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    load_timestamp: Optional[datetime] = None


class ECH0TrainingDataManager:
    """
    Manages the integration of ECH0 training datasets into ECH0-PRIME.
    Provides access to all training samples with advanced filtering and retrieval.
    """

    def __init__(self, ech0_data_path: str = "/Users/noone/ech0/ech0_training_data"):
        self.ech0_data_path = Path(ech0_data_path)
        self.datasets: Dict[str, List[TrainingSample]] = {}
        self.dataset_stats: Dict[str, DatasetStats] = {}
        self.domain_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.category_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.difficulty_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # Integration with ECH0-PRIME systems
        self.compressed_kb = CompressedKnowledgeBase() if compressed_kb_available else None
        self.data_compressor = DataCompressor() if data_compressor_available else None

        # Loading status
        self.loaded_datasets: set = set()

    async def load_all_datasets(self) -> Dict[str, DatasetStats]:
        """
        Load all available ECH0 training datasets asynchronously.
        """
        if not self.ech0_data_path.exists():
            raise FileNotFoundError(f"ECH0 training data path not found: {self.ech0_data_path}")

        json_files = list(self.ech0_data_path.glob("*.json"))
        print(f"Found {len(json_files)} dataset files to load")

        # Load datasets concurrently
        tasks = [self.load_dataset(json_file) for json_file in json_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_loads = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"Error loading dataset: {result}")
            else:
                successful_loads += 1

        print(f"Successfully loaded {successful_loads}/{len(json_files)} datasets")
        return self.dataset_stats

    async def load_dataset(self, file_path: Path) -> DatasetStats:
        """
        Load a single dataset file and build indices.
        """
        dataset_name = file_path.stem
        print(f"Loading dataset: {dataset_name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            samples = []
            domains = defaultdict(int)
            categories = defaultdict(int)
            difficulties = defaultdict(int)
            quality_scores = []

            for i, item in enumerate(data):
                sample = TrainingSample(
                    instruction=item.get("instruction", ""),
                    input=item.get("input", ""),
                    output=item.get("output", ""),
                    domain=item.get("domain", "unknown"),
                    category=item.get("category", "unknown"),
                    difficulty=item.get("difficulty", "unknown"),
                    metadata=item.get("metadata", {})
                )

                samples.append(sample)

                # Update statistics
                domains[sample.domain] += 1
                categories[sample.category] += 1
                difficulties[sample.difficulty] += 1
                quality_scores.append(sample.quality_score)

                # Build indices
                self.domain_index[sample.domain].append((dataset_name, i))
                self.category_index[sample.category].append((dataset_name, i))
                self.difficulty_index[sample.difficulty].append((dataset_name, i))

            # Store dataset
            self.datasets[dataset_name] = samples

            # Create stats
            stats = DatasetStats(
                name=dataset_name,
                total_samples=len(samples),
                domains=dict(domains),
                categories=dict(categories),
                difficulties=dict(difficulties),
                avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
                load_timestamp=datetime.now()
            )

            self.dataset_stats[dataset_name] = stats
            self.loaded_datasets.add(dataset_name)

            print(f"Loaded {len(samples)} samples from {dataset_name}")
            return stats

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            raise

    def get_samples_by_domain(self, domain: str, limit: Optional[int] = None) -> List[TrainingSample]:
        """Get all samples for a specific domain."""
        samples = []
        for dataset_name, sample_idx in self.domain_index[domain]:
            if dataset_name in self.datasets:
                samples.append(self.datasets[dataset_name][sample_idx])

        if limit:
            samples = samples[:limit]
        return samples

    def get_samples_by_category(self, category: str, limit: Optional[int] = None) -> List[TrainingSample]:
        """Get all samples for a specific category."""
        samples = []
        for dataset_name, sample_idx in self.category_index[category]:
            if dataset_name in self.datasets:
                samples.append(self.datasets[dataset_name][sample_idx])

        if limit:
            samples = samples[:limit]
        return samples

    def get_samples_by_difficulty(self, difficulty: str, limit: Optional[int] = None) -> List[TrainingSample]:
        """Get all samples for a specific difficulty level."""
        samples = []
        for dataset_name, sample_idx in self.difficulty_index[difficulty]:
            if dataset_name in self.datasets:
                samples.append(self.datasets[dataset_name][sample_idx])

        if limit:
            samples = samples[:limit]
        return samples

    def search_samples(self, query: str, limit: Optional[int] = None) -> List[TrainingSample]:
        """Search samples by instruction or output content."""
        query_lower = query.lower()
        matching_samples = []

        for dataset_samples in self.datasets.values():
            for sample in dataset_samples:
                if (query_lower in sample.instruction.lower() or
                    query_lower in sample.output.lower() or
                    query_lower in sample.category.lower() or
                    query_lower in sample.domain.lower()):
                    matching_samples.append(sample)

        if limit:
            matching_samples = matching_samples[:limit]
        return matching_samples

    def get_random_samples(self, count: int = 10, domain_filter: Optional[str] = None) -> List[TrainingSample]:
        """Get random samples, optionally filtered by domain."""
        all_samples = []
        if domain_filter:
            all_samples = self.get_samples_by_domain(domain_filter)
        else:
            for dataset_samples in self.datasets.values():
                all_samples.extend(dataset_samples)

        if len(all_samples) <= count:
            return all_samples

        return random.sample(all_samples, min(count, len(all_samples)))

    async def integrate_with_compressed_kb(self, sample: TrainingSample) -> Optional[str]:
        """
        Integrate a training sample with the compressed knowledge base.
        """
        if not self.compressed_kb:
            return None

        try:
            # Compress the sample content
            compressed_data = await self.data_compressor.compress_chunk(
                content=f"{sample.instruction}\n{sample.output}",
                domain=sample.domain,
                metadata={
                    "category": sample.category,
                    "difficulty": sample.difficulty,
                    "source": "ech0_training_data",
                    **sample.metadata
                }
            )

            # Store in compressed knowledge base
            await self.compressed_kb.store_compressed_sample(
                compressed_data,
                domain=sample.domain,
                category=sample.category
            )

            return compressed_data.compressed_content

        except Exception as e:
            print(f"Error integrating sample with compressed KB: {e}")
            return None

    async def create_training_batch(self, batch_size: int = 32,
                                  domains: Optional[List[str]] = None,
                                  difficulties: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Create a training batch suitable for fine-tuning.
        """
        # Filter samples based on criteria
        candidate_samples = []
        for dataset_samples in self.datasets.values():
            for sample in dataset_samples:
                if domains and sample.domain not in domains:
                    continue
                if difficulties and sample.difficulty not in difficulties:
                    continue
                candidate_samples.append(sample)

        # Select random batch
        if len(candidate_samples) <= batch_size:
            selected_samples = candidate_samples
        else:
            selected_samples = random.sample(candidate_samples, min(batch_size, len(candidate_samples)))

        # Format for training
        training_batch = []
        for sample in selected_samples:
            training_batch.append({
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output,
                "domain": sample.domain,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "quality_score": sample.quality_score
            })

        return training_batch

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all loaded datasets."""
        if not self.dataset_stats:
            return {"status": "no_datasets_loaded"}

        total_samples = sum(stats.total_samples for stats in self.dataset_stats.values())

        # Aggregate domain statistics
        all_domains = defaultdict(int)
        all_categories = defaultdict(int)
        all_difficulties = defaultdict(int)

        for stats in self.dataset_stats.values():
            for domain, count in stats.domains.items():
                all_domains[domain] += count
            for category, count in stats.categories.items():
                all_categories[category] += count
            for difficulty, count in stats.difficulties.items():
                all_difficulties[difficulty] += count

        return {
            "total_datasets": len(self.dataset_stats),
            "total_samples": total_samples,
            "loaded_datasets": list(self.loaded_datasets),
            "domains": dict(all_domains),
            "categories": dict(all_categories),
            "difficulties": dict(all_difficulties),
            "avg_quality_score": statistics.mean([stats.avg_quality_score for stats in self.dataset_stats.values()]) if self.dataset_stats else 0.0,
            "compressed_kb_integrated": self.compressed_kb is not None,
            "data_compressor_integrated": self.data_compressor is not None
        }

    def export_training_data(self, output_path: str,
                            format: str = "jsonl",
                            domains: Optional[List[str]] = None) -> str:
        """
        Export training data in various formats for fine-tuning.
        """
        output_file = f"{output_path}/ech0_training_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

        samples_to_export = []
        for dataset_samples in self.datasets.values():
            for sample in dataset_samples:
                if domains and sample.domain not in domains:
                    continue
                samples_to_export.append(sample)

        if format == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples_to_export:
                    f.write(json.dumps({
                        "instruction": sample.instruction,
                        "input": sample.input,
                        "output": sample.output,
                        "domain": sample.domain,
                        "category": sample.category,
                        "difficulty": sample.difficulty
                    }, ensure_ascii=False) + '\n')

        elif format == "json":
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_samples": len(samples_to_export),
                    "domains": list(set(s.domain for s in samples_to_export)),
                    "source": "ECH0-PRIME Training Data Integration"
                },
                "samples": [
                    {
                        "instruction": s.instruction,
                        "input": s.input,
                        "output": s.output,
                        "domain": s.domain,
                        "category": s.category,
                        "difficulty": s.difficulty,
                        "quality_score": s.quality_score
                    }
                    for s in samples_to_export
                ]
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"Exported {len(samples_to_export)} samples to {output_file}")
        return output_file


class ECH0TrainingOrchestrator:
    """
    Orchestrates training workflows using the integrated ECH0 datasets.
    """

    def __init__(self, data_manager: ECH0TrainingDataManager):
        self.data_manager = data_manager
        self.active_workflows: Dict[str, Any] = {}

    async def create_domain_specific_training(self, domain: str,
                                           batch_size: int = 32,
                                           epochs: int = 3) -> Dict[str, Any]:
        """
        Create a domain-specific training workflow.
        """
        samples = self.data_manager.get_samples_by_domain(domain, limit=batch_size * epochs)

        if not samples:
            return {"error": f"No samples found for domain: {domain}"}

        workflow = {
            "domain": domain,
            "total_samples": len(samples),
            "batch_size": batch_size,
            "epochs": epochs,
            "estimated_batches": len(samples) // batch_size,
            "quality_distribution": {
                "high": len([s for s in samples if s.quality_score > 0.8]),
                "medium": len([s for s in samples if 0.6 <= s.quality_score <= 0.8]),
                "low": len([s for s in samples if s.quality_score < 0.6])
            },
            "categories": list(set(s.category for s in samples))
        }

        workflow_id = f"{domain}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = workflow

        return {
            "workflow_id": workflow_id,
            "workflow": workflow,
            "ready_to_train": True
        }

    async def generate_curriculum(self, domains: List[str],
                                difficulty_progression: List[str] = None) -> Dict[str, Any]:
        """
        Generate a learning curriculum across multiple domains.
        """
        if difficulty_progression is None:
            difficulty_progression = ["easy", "medium", "hard", "expert"]

        curriculum = {}
        total_samples = 0

        for domain in domains:
            domain_curriculum = {}
            for difficulty in difficulty_progression:
                samples = self.data_manager.get_samples_by_difficulty(difficulty)
                domain_samples = [s for s in samples if s.domain == domain]

                domain_curriculum[difficulty] = {
                    "sample_count": len(domain_samples),
                    "categories": list(set(s.category for s in domain_samples)),
                    "avg_quality": statistics.mean([s.quality_score for s in domain_samples]) if domain_samples else 0.0
                }

                total_samples += len(domain_samples)

            curriculum[domain] = domain_curriculum

        return {
            "curriculum": curriculum,
            "total_samples": total_samples,
            "difficulty_levels": difficulty_progression,
            "domains_covered": domains,
            "estimated_training_time": f"{total_samples * 0.1:.1f} hours (estimated)"
        }


# Convenience functions
async def initialize_ech0_training_integration(ech0_path: str = "/Users/noone/ech0/ech0_training_data") -> ECH0TrainingDataManager:
    """
    Initialize and load all ECH0 training data into ECH0-PRIME.
    """
    print("Initializing ECH0 Training Data Integration...")
    manager = ECH0TrainingDataManager(ech0_path)
    await manager.load_all_datasets()

    stats = manager.get_statistics_summary()
    print(f"✅ Integration Complete: {stats['total_samples']} samples from {stats['total_datasets']} datasets")
    print(f"📊 Domains: {', '.join(stats['domains'].keys())}")
    print(f"🎯 Categories: {len(stats['categories'])} total")
    print(f"📈 Quality Score: {stats['avg_quality_score']:.2f}")

    return manager


def create_training_export(manager: ECH0TrainingDataManager,
                           domains: List[str] = None,
                           output_dir: str = "/tmp") -> str:
    """
    Export training data for external fine-tuning.
    """
    return manager.export_training_data(output_dir, format="jsonl", domains=domains)


if __name__ == "__main__":
    # Example usage
    async def demo():
        manager = await initialize_ech0_training_integration()

        # Get some creative samples
        creative_samples = manager.get_samples_by_domain("creativity", limit=5)
        print(f"\n🎨 Sample Creative Training Data:")
        for i, sample in enumerate(creative_samples[:3]):
            print(f"{i+1}. {sample.instruction[:100]}...")
            print(f"   Quality: {sample.quality_score:.2f}, Category: {sample.category}")

        # Create a training batch
        batch = await manager.create_training_batch(batch_size=10, domains=["ai_ml", "creativity"])
        print(f"\n📚 Created training batch with {len(batch)} samples")

        # Generate curriculum
        orchestrator = ECH0TrainingOrchestrator(manager)
        curriculum = await orchestrator.generate_curriculum(["ai_ml", "creativity"])
        print(f"\n📖 Generated curriculum covering {curriculum['total_samples']} samples")

    asyncio.run(demo())
