# 🔗 Invention Data Integration Guide

Complete guide for integrating downloaded scientific papers with Echo Prime's invention generation system.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Quick Start](#quick-start)
4. [Usage Examples](#usage-examples)
5. [Advanced Features](#advanced-features)
6. [Workflow Examples](#workflow-examples)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This integration connects your downloaded scientific papers (from `download_invention_data.py`) with Echo Prime's invention generation system, enabling:

- **Large-scale paper indexing** (23,000+ papers)
- **Semantic search** with embeddings
- **Cross-domain invention generation**
- **Enhanced context** for higher-quality inventions

---

## System Components

### 1. **invention_data_indexer.py**
Fast keyword-based indexing and retrieval

**Features**:
- Loads all papers from `invention_data/`
- Keyword extraction and indexing
- Multi-query search
- Category filtering
- Statistics and export

### 2. **semantic_paper_search.py**
Advanced semantic search with embeddings

**Features**:
- Sentence embeddings for similarity
- FAISS vector search
- Hybrid keyword + semantic search
- Cross-domain discovery
- Similar paper finder

### 3. **missions/enhanced_invention_cycle.py**
Integrated invention generation pipeline

**Features**:
- Uses full paper dataset (not live arXiv)
- Multi-query paper retrieval
- Category-based focusing
- Parliament governance
- Batch invention generation

---

## Quick Start

### Step 1: Download Papers

If you haven't already:

```bash
# Install dependencies
pip install arxiv

# Download sample data
python3 download_invention_data.py --sample

# Or download high-priority categories
python3 download_invention_data.py --auto --priority 9
```

This creates `invention_data/` with papers organized by category.

---

### Step 2: Index Papers

```python
from invention_data_indexer import InventionDataIndexer

# Initialize indexer
indexer = InventionDataIndexer()

# View stats
stats = indexer.get_stats()
print(f"Total papers: {stats['total_papers']}")

# Search
results = indexer.search("quantum computing", limit=5)
for paper in results:
    print(f"- {paper.title}")
```

---

### Step 3: Generate Inventions

```python
from missions.enhanced_invention_cycle import EnhancedInventionCycle

# Initialize
cycle = EnhancedInventionCycle()

# Generate inventions
cycle.run_full_cycle(
    focus_area="Quantum materials for computing",
    queries=["quantum", "superconductor", "metamaterial"],
    output_file="my_inventions.json"
)
```

Output: `my_inventions.json` with 5-10 invention concepts

---

### Step 4: Semantic Search (Optional)

For better paper discovery:

```bash
# Install semantic search dependencies
pip install sentence-transformers faiss-cpu
```

```python
from semantic_paper_search import SemanticPaperSearch

# Initialize (generates embeddings on first run)
search = SemanticPaperSearch()

# Semantic search
results = search.semantic_search("energy harvesting nanomaterials", top_k=10)
for paper, score in results:
    print(f"[{score:.3f}] {paper.title}")
```

---

## Usage Examples

### Example 1: Keyword Search

```python
from invention_data_indexer import InventionDataIndexer

indexer = InventionDataIndexer()

# Simple search
papers = indexer.search("graphene", limit=10)

# Search with category filter
papers = indexer.search("battery", limit=10, category="energy_systems")

# Multi-query search
papers = indexer.search_multi_query(
    queries=["quantum", "superconductor", "topological"],
    limit_per_query=5
)

# Get recent papers
papers = indexer.get_recent_papers(n=10, category="materials_science")

# Random sample
papers = indexer.get_random_sample(n=20)
```

---

### Example 2: Generate Inventions from Specific Categories

```python
from missions.enhanced_invention_cycle import EnhancedInventionCycle

cycle = EnhancedInventionCycle()

# Energy innovations
cycle.run_full_cycle(
    focus_area="Revolutionary energy storage",
    categories=["energy_systems", "materials_science"],
    output_file="energy_inventions.json"
)

# Nanotechnology + Manufacturing
cycle.run_full_cycle(
    focus_area="Nanoscale manufacturing techniques",
    categories=["nanotechnology", "additive_manufacturing"],
    output_file="nano_manufacturing.json"
)

# Cross-domain quantum applications
cycle.run_full_cycle(
    focus_area="Quantum sensing and computing",
    categories=["quantum_materials", "photonics"],
    output_file="quantum_applications.json"
)
```

---

### Example 3: Semantic Search

```python
from semantic_paper_search import SemanticPaperSearch

search = SemanticPaperSearch()

# Semantic search (understands meaning, not just keywords)
results = search.semantic_search(
    "materials that conduct electricity without resistance",
    top_k=10
)

# Hybrid search (combines semantic + keywords)
results = search.hybrid_search(
    "3D printing living tissue",
    top_k=10,
    semantic_weight=0.7  # 70% semantic, 30% keyword
)

# Find similar papers
results = search.find_similar_papers(
    arxiv_id="2501.12345",
    top_k=10
)

# Cross-domain papers (spans multiple fields)
results = search.cross_domain_search(
    "quantum materials",
    min_categories=2,
    top_k=10
)
```

---

### Example 4: Custom Invention Pipeline

```python
from invention_data_indexer import InventionDataIndexer
from missions.enhanced_invention_cycle import EnhancedInventionCycle

# Initialize
indexer = InventionDataIndexer()
cycle = EnhancedInventionCycle()

# Find papers about a specific topic
papers = indexer.search("self-healing materials", limit=20)
print(f"Found {len(papers)} papers about self-healing materials")

# Generate inventions from those papers
queries = ["self-healing", "smart materials", "damage detection"]
cycle.run_full_cycle(
    focus_area="Self-healing infrastructure materials",
    queries=queries,
    output_file="self_healing_inventions.json"
)
```

---

### Example 5: Batch Invention Generation

```python
from missions.enhanced_invention_cycle import EnhancedInventionCycle

cycle = EnhancedInventionCycle()

# Define multiple invention focuses
focuses = [
    {
        "name": "Quantum Computing Hardware",
        "queries": ["quantum", "qubit", "superconductor", "cryogenic"],
        "output": "inventions_quantum_hw.json"
    },
    {
        "name": "Energy Harvesting",
        "queries": ["energy harvesting", "thermoelectric", "piezoelectric"],
        "output": "inventions_energy_harvest.json"
    },
    {
        "name": "Medical Nanotechnology",
        "queries": ["nanoparticle", "drug delivery", "biosensor"],
        "output": "inventions_medical_nano.json"
    }
]

# Generate inventions for each focus
for focus in focuses:
    print(f"\nGenerating: {focus['name']}")
    cycle.run_full_cycle(
        focus_area=focus['name'],
        queries=focus['queries'],
        output_file=focus['output']
    )
```

---

## Advanced Features

### Feature 1: Export Index for Fast Loading

```python
from invention_data_indexer import InventionDataIndexer

# First time: Load and export
indexer = InventionDataIndexer()
indexer.export_index("fast_index.json")

# Later: Load from exported index (much faster)
import json
with open("fast_index.json") as f:
    index_data = json.load(f)
# Use index_data directly
```

---

### Feature 2: Category Statistics

```python
from invention_data_indexer import InventionDataIndexer

indexer = InventionDataIndexer()
stats = indexer.get_stats()

print(f"Total papers: {stats['total_papers']:,}")
print(f"Categories: {len(stats['categories_breakdown'])}")
print(f"Avg papers/category: {stats['avg_papers_per_category']:.0f}")

print("\nBreakdown:")
for category, count in stats['categories_breakdown'].items():
    print(f"  {category}: {count:,} papers")
```

---

### Feature 3: Generate Embeddings Once, Use Many Times

```python
from semantic_paper_search import SemanticPaperSearch

# First run: Generates and caches embeddings
search = SemanticPaperSearch()  # Takes 5-10 minutes

# Subsequent runs: Loads from cache instantly
search2 = SemanticPaperSearch()  # Takes <1 second
```

Embeddings are cached in `.cache/embeddings/`

---

### Feature 4: Custom LLM Models

```python
from missions.enhanced_invention_cycle import EnhancedInventionCycle

# Use different model
cycle = EnhancedInventionCycle(llm_model="llama3")

# Or custom model
cycle = EnhancedInventionCycle(llm_model="your-custom-model")
```

---

## Workflow Examples

### Workflow 1: Daily Invention Generation

```bash
#!/bin/bash
# daily_inventions.sh

echo "=== Daily Invention Generation ==="

# 1. Check for new papers (if needed)
# python3 download_invention_data.py --category "Materials Science"

# 2. Generate inventions
python3 << EOF
from missions.enhanced_invention_cycle import EnhancedInventionCycle
import datetime

cycle = EnhancedInventionCycle()

date_str = datetime.datetime.now().strftime("%Y-%m-%d")
output_file = f"inventions_{date_str}.json"

cycle.run_full_cycle(
    focus_area="Cross-domain breakthrough inventions",
    output_file=output_file
)

print(f"\\n✅ Generated: {output_file}")
EOF
```

---

### Workflow 2: Focused Research Sprint

```python
# research_sprint.py
from invention_data_indexer import InventionDataIndexer
from semantic_paper_search import SemanticPaperSearch
from missions.enhanced_invention_cycle import EnhancedInventionCycle

# Define research focus
FOCUS = "Room-temperature superconductors"
QUERIES = ["superconductor", "high temperature", "critical temperature", "zero resistance"]

# Initialize systems
indexer = InventionDataIndexer()
search = SemanticPaperSearch()
cycle = EnhancedInventionCycle()

# Phase 1: Discover relevant papers
print(f"📚 Researching: {FOCUS}")
papers = []
for query in QUERIES:
    results = search.semantic_search(query, top_k=10)
    papers.extend([paper for paper, score in results])

print(f"Found {len(papers)} relevant papers")

# Phase 2: Analyze paper trends
categories = {}
for paper in papers:
    cat = paper.category_domain
    categories[cat] = categories.get(cat, 0) + 1

print("\nPaper distribution:")
for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat}: {count} papers")

# Phase 3: Generate inventions
cycle.run_full_cycle(
    focus_area=FOCUS,
    queries=QUERIES,
    output_file="superconductor_inventions.json"
)

print("\n✅ Research sprint complete!")
```

---

### Workflow 3: Patent Analysis Pipeline

```python
# patent_analysis.py
from invention_data_indexer import InventionDataIndexer
import json
from collections import defaultdict

indexer = InventionDataIndexer()

# Analyze paper keywords to find patentable concepts
print("🔍 Analyzing papers for patent opportunities...")

keyword_papers = defaultdict(list)
for paper in indexer.papers:
    for keyword in paper.keywords[:10]:  # Top 10 keywords
        keyword_papers[keyword].append(paper.arxiv_id)

# Find high-frequency keywords (potential patent areas)
top_keywords = sorted(
    keyword_papers.items(),
    key=lambda x: len(x[1]),
    reverse=True
)[:50]

print("\nTop patent opportunity areas:")
for keyword, papers in top_keywords[:20]:
    print(f"  {keyword}: {len(papers)} papers")

# Generate inventions in top areas
from missions.enhanced_invention_cycle import EnhancedInventionCycle

cycle = EnhancedInventionCycle()
top_queries = [kw for kw, _ in top_keywords[:10]]

cycle.run_full_cycle(
    focus_area="High-impact patentable inventions",
    queries=top_queries,
    output_file="patent_inventions.json"
)
```

---

## API Reference

### InventionDataIndexer

```python
class InventionDataIndexer:
    def __init__(self, data_dir: str = "invention_data")

    def search(self, query: str, limit: int = 10, category: Optional[str] = None) -> List[Paper]

    def search_multi_query(self, queries: List[str], limit_per_query: int = 5) -> List[Paper]

    def get_by_category(self, category: str, limit: Optional[int] = None) -> List[Paper]

    def get_random_sample(self, n: int = 10, category: Optional[str] = None) -> List[Paper]

    def get_recent_papers(self, n: int = 10, category: Optional[str] = None) -> List[Paper]

    def get_stats(self) -> Dict[str, Any]

    def export_index(self, output_file: str = "invention_data_index.json")
```

---

### SemanticPaperSearch

```python
class SemanticPaperSearch:
    def __init__(self, data_dir: str = "invention_data", model_name: str = "all-MiniLM-L6-v2")

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[Paper, float]]

    def hybrid_search(self, query: str, top_k: int = 10, semantic_weight: float = 0.7) -> List[Tuple[Paper, float]]

    def find_similar_papers(self, arxiv_id: str, top_k: int = 10) -> List[Tuple[Paper, float]]

    def cross_domain_search(self, query: str, min_categories: int = 2, top_k: int = 10) -> List[Tuple[Paper, float]]
```

---

### EnhancedInventionCycle

```python
class EnhancedInventionCycle:
    def __init__(self, llm_model: str = "ech0-unified-14b-enhanced")

    def run_full_cycle(self,
                        focus_area: str = "cross-domain innovation",
                        queries: List[str] = None,
                        categories: List[str] = None,
                        output_file: str = "ech0_enhanced_inventions.json") -> str
```

---

## Troubleshooting

### Issue: "No papers found"

**Solution**:
```bash
# Run the download script first
python3 download_invention_data.py --sample
```

---

### Issue: "sentence-transformers not installed"

**Solution**:
```bash
pip install sentence-transformers faiss-cpu numpy
```

---

### Issue: "Embedding generation is slow"

**Expected**: First run takes 5-10 minutes to generate embeddings for all papers.
**Cached**: Subsequent runs are instant (<1 second).

**Speed up**:
```python
# Use smaller model
search = SemanticPaperSearch(model_name="paraphrase-MiniLM-L3-v2")
```

---

### Issue: "Out of memory during embedding generation"

**Solution**:
```python
# Generate embeddings in smaller batches
# Edit semantic_paper_search.py, line ~150:
batch_size = 16  # Reduce from 32
```

---

### Issue: "No Ollama models found"

**Solution**:
```bash
# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai

# Start Ollama
ollama serve &

# Pull model
ollama pull ech0-unified-14b-enhanced
# or
ollama pull llama3
```

---

## Performance Tips

1. **Use cached embeddings**: Run `SemanticPaperSearch()` once, reuse many times
2. **Export index**: Use `indexer.export_index()` for faster subsequent loads
3. **Batch processing**: Generate multiple invention sets in one script run
4. **Category filtering**: Search within specific categories for faster results
5. **Limit results**: Use smaller `limit` values for faster searches

---

## Next Steps

- **Create custom workflows** tailored to your research interests
- **Build a web interface** for exploring papers and inventions
- **Integrate with existing tools** (Jupyter notebooks, dashboards)
- **Automate daily invention generation**
- **Analyze invention trends** across categories

---

**Ready to Generate Breakthrough Inventions!**

All systems operational. Start with the Quick Start examples above.

**Questions?** Check INVENTION_DATA_GUIDE.md for download details.
