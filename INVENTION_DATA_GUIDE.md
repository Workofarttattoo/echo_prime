# 📚 Invention Data Acquisition - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Category Descriptions](#category-descriptions)
4. [Installation](#installation)
5. [Usage Methods](#usage-methods)
6. [Data Formats](#data-formats)
7. [Integration with Echo Prime](#integration-with-echo-prime)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)
10. [Patent Data Acquisition](#patent-data-acquisition)

---

## Overview

The Invention Data Acquisition System is designed to download and organize scientific papers and patents across 7 high-priority invention domains. This data feeds Echo Prime's invention generation capabilities.

### Goals
- Acquire **23,000+ research papers** from arXiv
- Acquire **10,000+ patents** from Google Patents/USPTO
- Organize by invention domain and priority
- Enable rapid knowledge retrieval for invention generation

### Key Features
- Priority-based downloading (Priority 1-10)
- Automatic metadata extraction
- Progress tracking and resumption
- Rate limiting and error handling
- Multiple download methods (Python, Shell, Manual)

---

## System Architecture

```
download_invention_data.py          # Main Python script
  ├─ InventionDataDownloader        # Core downloader class
  │   ├─ _initialize_categories()   # Configure 7 categories
  │   ├─ download_category()        # Download single category
  │   ├─ download_all()             # Download all categories
  │   └─ download_sample()          # Sample mode for testing
  └─ DataCategory                   # Category dataclass

scripts/download_invention_data.sh  # Shell script alternative
  ├─ check_dependencies()
  ├─ download_category()
  └─ print_summary()

invention_data/                     # Output directory
  ├─ materials_science/
  ├─ nanotechnology/
  ├─ quantum_materials/
  ├─ energy_systems/
  ├─ photonics/
  ├─ additive_manufacturing/
  ├─ invention_methodology/
  └─ download_summary.json
```

---

## Category Descriptions

### 1. Materials Science (Priority 10)
**Target**: 5,000 papers | **arXiv**: cond-mat.mtrl-sci

Revolutionary materials that enable breakthrough inventions.

**Keywords**:
- Metamaterials
- Graphene and 2D materials
- Nanocomposites
- Carbon nanotubes
- Smart materials
- Self-healing materials

**Invention Examples**:
- Transparent aluminum armor
- Self-healing phone screens
- Ultra-lightweight construction materials
- Programmable matter

**Why Priority 10?**
Materials are the foundation of physical inventions. New materials enable entirely new classes of devices.

---

### 2. Nanotechnology (Priority 10)
**Target**: 5,000 papers | **arXiv**: cond-mat.mes-hall

Nanoscale engineering for molecular-level control.

**Keywords**:
- Nanodevices
- Molecular assembly
- Nanofabrication
- Quantum dots
- Nanowires
- Nanoparticles

**Invention Examples**:
- Molecular manufacturing systems
- Nano-robots for medicine
- Ultra-efficient solar panels
- Quantum dot displays

**Why Priority 10?**
Nanotechnology enables precision manufacturing at atomic scales, opening entirely new invention domains.

---

### 3. Quantum Materials (Priority 9)
**Target**: 3,000 papers | **arXiv**: quant-ph

Materials with quantum properties for next-gen computing and sensing.

**Keywords**:
- Topological insulators
- Superconductors
- Quantum computing
- Quantum sensors
- Spintronics

**Invention Examples**:
- Room-temperature superconductors
- Quantum computers
- Ultra-sensitive medical sensors
- Quantum communication devices

**Why Priority 9?**
Quantum materials will power the next computing revolution.

---

### 4. Energy Systems (Priority 9)
**Target**: 4,000 papers | **arXiv**: physics.app-ph

Energy storage, generation, and harvesting technologies.

**Keywords**:
- Energy storage
- Batteries
- Solar cells
- Fuel cells
- Thermoelectric
- Energy harvesting
- Supercapacitors

**Invention Examples**:
- Solid-state batteries
- Wireless power transmission
- Waste heat harvesting
- Grid-scale energy storage

**Why Priority 9?**
Energy is the bottleneck for most breakthrough inventions.

---

### 5. Photonics & Holography (Priority 8)
**Target**: 3,000 papers | **arXiv**: physics.optics

Light-based technologies for computing, displays, and sensing.

**Keywords**:
- Photonics
- Plasmonics
- Holography
- Optical devices
- Photonic crystals
- Optical computing

**Invention Examples**:
- Holographic displays
- Optical computers
- Light-based internet
- Invisibility cloaking

**Why Priority 8?**
Photonics enables high-bandwidth communication and novel display technologies.

---

### 6. Additive Manufacturing (Priority 8)
**Target**: 2,000 papers | **arXiv**: cs.RO

3D printing and rapid prototyping technologies.

**Keywords**:
- 3D printing
- Additive manufacturing
- Rapid prototyping
- Bioprinting
- 4D printing
- Metal printing

**Invention Examples**:
- On-demand manufacturing
- Bioprinted organs
- Self-assembling structures
- Custom prosthetics

**Why Priority 8?**
Enables rapid iteration and distributed manufacturing of inventions.

---

### 7. Invention Methodology (Priority 6)
**Target**: 1,000 papers | **arXiv**: cs.AI

Systematic approaches to invention and problem-solving.

**Keywords**:
- TRIZ
- Design thinking
- Systematic invention
- Creative problem solving
- Innovation methodology

**Invention Examples**:
- AI-powered invention systems
- Automated patent generation
- Design optimization tools
- Innovation frameworks

**Why Priority 6?**
Meta-knowledge about how to invent better.

---

## Installation

### Prerequisites

**macOS/Linux**:
```bash
# Install Python 3.10+
python3 --version  # Should be 3.10 or higher

# Install pip (if not already installed)
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip curl jq
```

### Install arxiv Library

```bash
# Install via pip
pip install arxiv

# Or install in a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install arxiv
```

### Verify Installation

```bash
python3 -c "import arxiv; print('arXiv library installed successfully')"
```

---

## Usage Methods

### Method 1: Python Script (Recommended)

#### Sample Mode (Testing)
Download 100 papers per category to test the system:

```bash
python3 download_invention_data.py --sample
```

**Output**: ~700 papers in ~5-10 minutes

#### Priority 9-10 (High-Priority Categories)
Download only the highest-priority categories:

```bash
python3 download_invention_data.py --auto --priority 9
```

**Output**: 17,000 papers (Materials, Nanotechnology, Quantum, Energy)

#### All Categories
Download all 23,000 papers:

```bash
python3 download_invention_data.py --auto
```

**Estimated Time**: 6-8 hours with rate limiting

#### Specific Category
Download one category only:

```bash
python3 download_invention_data.py --category "Materials Science"
```

#### Custom Output Directory
```bash
python3 download_invention_data.py --sample --output /path/to/data
```

---

### Method 2: Shell Script

For systems without Python arxiv library:

```bash
# Make executable
chmod +x scripts/download_invention_data.sh

# Sample mode
./scripts/download_invention_data.sh --sample

# Full download
./scripts/download_invention_data.sh --full

# Custom parameters
MAX_RESULTS=500 OUTPUT_DIR=my_data ./scripts/download_invention_data.sh
```

---

### Method 3: Manual Download

#### From arXiv Website
1. Visit https://arxiv.org/
2. Use advanced search: https://arxiv.org/search/advanced
3. Select category (e.g., cond-mat.mtrl-sci)
4. Add keywords from category descriptions
5. Download metadata as CSV
6. Convert to JSON format

#### arXiv Bulk Data Access
For large-scale downloads:

```bash
# Install AWS CLI
pip install awscli

# Download arXiv bulk metadata
aws s3 sync s3://arxiv/src/ ./arxiv_src/ --no-sign-request

# Filter by category
python3 -c "
import tarfile
import json

# Extract and filter papers
# (See arXiv bulk data documentation)
"
```

Reference: https://arxiv.org/help/bulk_data

---

## Data Formats

### papers.json Structure

Each category produces a `papers.json` file with this structure:

```json
[
  {
    "title": "Novel Graphene-Based Metamaterial for Optical Computing",
    "authors": ["Jane Smith", "John Doe"],
    "abstract": "We present a novel graphene-based metamaterial...",
    "published": "2025-01-15T12:00:00",
    "updated": "2025-01-20T15:30:00",
    "arxiv_id": "2501.12345",
    "categories": ["cond-mat.mtrl-sci", "physics.optics"],
    "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
    "primary_category": "cond-mat.mtrl-sci",
    "doi": "10.1234/example.doi",
    "journal_ref": "Nature Materials 2025",
    "comment": "10 pages, 5 figures"
  }
]
```

### metadata.json Structure

Each category also produces a `metadata.json`:

```json
{
  "category": "Materials Science",
  "arxiv_category": "cond-mat.mtrl-sci",
  "keywords": ["metamaterials", "graphene", "nanocomposites"],
  "downloaded": 5000,
  "failed": 12,
  "target": 5000,
  "download_date": "2026-01-27T10:30:00",
  "output_file": "invention_data/materials_science/papers.json"
}
```

### download_summary.json Structure

Overall summary of all downloads:

```json
{
  "stats": {
    "total_downloaded": 23000,
    "total_failed": 45,
    "categories_complete": 7,
    "start_time": "2026-01-27T08:00:00",
    "end_time": "2026-01-27T16:00:00"
  },
  "categories": [
    {
      "name": "Materials Science",
      "priority": 10,
      "downloaded": 5000,
      "target": 5000,
      "completion": 1.0
    }
  ],
  "timestamp": "2026-01-27T16:00:00"
}
```

---

## Integration with Echo Prime

### Step 1: Index the Papers

Create a searchable index:

```python
import json
from pathlib import Path

# Load all papers
all_papers = []
for category_dir in Path('invention_data').iterdir():
    if category_dir.is_dir():
        papers_file = category_dir / 'papers.json'
        if papers_file.exists():
            with open(papers_file) as f:
                papers = json.load(f)
                for paper in papers:
                    paper['category'] = category_dir.name
                all_papers.extend(papers)

print(f"Loaded {len(all_papers)} papers")
```

### Step 2: Generate Embeddings

For semantic search:

```python
# Using sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = []
for paper in all_papers:
    text = f"{paper['title']} {paper['abstract']}"
    embedding = model.encode(text)
    embeddings.append(embedding)
```

### Step 3: Connect to Invention Pipeline

Add to `missions/run_invention_cycle.py`:

```python
from pathlib import Path
import json

class InventionDataRetriever:
    def __init__(self, data_dir='invention_data'):
        self.data_dir = Path(data_dir)
        self.papers = self._load_all_papers()

    def _load_all_papers(self):
        all_papers = []
        for category_dir in self.data_dir.iterdir():
            if category_dir.is_dir():
                papers_file = category_dir / 'papers.json'
                if papers_file.exists():
                    with open(papers_file) as f:
                        papers = json.load(f)
                        for paper in papers:
                            paper['category'] = category_dir.name
                        all_papers.extend(papers)
        return all_papers

    def search(self, query, limit=10):
        # Simple keyword search (can be enhanced with embeddings)
        results = []
        query_lower = query.lower()
        for paper in self.papers:
            if (query_lower in paper['title'].lower() or
                query_lower in paper['abstract'].lower()):
                results.append(paper)
        return results[:limit]
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'arxiv'"

**Solution**:
```bash
pip install arxiv
```

### Issue: "Rate limited by arXiv"

**Solution**:
The script includes automatic rate limiting (0.5s between requests). If you still get rate limited:
1. Reduce batch size
2. Download one category at a time
3. Wait 1 hour and retry

### Issue: "Connection timeout"

**Solution**:
```bash
# Increase timeout in download script
# Edit download_invention_data.py:
client = arxiv.Client(
    page_size=100,
    delay_seconds=1.0,  # Increase from 0.5
    num_retries=5
)
```

### Issue: "Disk space full"

**Solution**:
```bash
# Check disk space
df -h

# Download to external drive
python3 download_invention_data.py --output /mnt/external/invention_data --sample

# Or download metadata only (modify script to skip PDF downloads)
```

### Issue: "Papers.json is empty"

**Possible Causes**:
1. Network connection issues
2. arXiv API changes
3. Invalid search query

**Debugging**:
```bash
# Test with simple query
python3 -c "
import arxiv
search = arxiv.Search(query='quantum computing', max_results=5)
for result in arxiv.Client().results(search):
    print(result.title)
"
```

---

## Advanced Topics

### Custom Categories

Add your own category:

```python
# In download_invention_data.py, add to _initialize_categories():
DataCategory(
    name="Your Custom Category",
    priority=7,
    arxiv_category="your.category",
    keywords=["keyword1", "keyword2"],
    target_count=1000,
    output_dir=os.path.join(self.base_dir, "custom_category")
)
```

### Date Range Filtering

Modify search query to filter by date:

```python
# In download_category method:
from datetime import datetime, timedelta

# Last 2 years only
cutoff_date = datetime.now() - timedelta(days=730)
search_query = f"cat:{category.arxiv_category} AND submittedDate:[{cutoff_date.strftime('%Y%m%d')} TO *]"
```

### Parallel Downloads

Speed up downloads with multiprocessing:

```python
from multiprocessing import Pool

def download_wrapper(category):
    downloader = InventionDataDownloader()
    return downloader.download_category(category)

# In download_all method:
with Pool(4) as pool:
    pool.map(download_wrapper, self.categories)
```

### Resume Interrupted Downloads

```python
# Check what's already downloaded
def _get_downloaded_count(self, category):
    output_file = os.path.join(category.output_dir, "papers.json")
    if os.path.exists(output_file):
        with open(output_file) as f:
            return len(json.load(f))
    return 0

# In download_category:
already_downloaded = self._get_downloaded_count(category)
remaining = max_results - already_downloaded
```

---

## Patent Data Acquisition

### Google Patents

**Manual Search**:
1. Visit https://patents.google.com/
2. Search by keywords from categories
3. Filter by date (last 5 years recommended)
4. Export results as CSV
5. Convert to JSON format

**Example Searches**:
- "metamaterial optical" (Materials Science)
- "nanodevice fabrication" (Nanotechnology)
- "quantum computing qubit" (Quantum Materials)
- "energy harvesting battery" (Energy Systems)
- "3D printing bioprinting" (Additive Manufacturing)

### USPTO Bulk Data

**Download USPTO Data**:
```bash
# Visit USPTO bulk data portal
# https://bulkdata.uspto.gov/

# Download patent grants
wget https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/2025/

# Parse XML to JSON
python3 parse_uspto.py
```

### Patent Parsing Script

```python
# parse_patents.py
import xml.etree.ElementTree as ET
import json
from pathlib import Path

def parse_uspto_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    patents = []
    for patent in root.findall('.//us-patent-grant'):
        data = {
            'patent_id': patent.find('.//doc-number').text,
            'title': patent.find('.//invention-title').text,
            'abstract': patent.find('.//abstract/p').text,
            'date': patent.find('.//publication-reference/document-id/date').text,
            'inventors': [inv.text for inv in patent.findall('.//inventor/addressbook/last-name')],
            'assignees': [ass.text for ass in patent.findall('.//assignee/addressbook/orgname')]
        }
        patents.append(data)

    return patents

# Usage
patents = parse_uspto_xml('uspto_2025.xml')
with open('patents.json', 'w') as f:
    json.dump(patents, f, indent=2)
```

### Organizing Patent Data

```
invention_data/
└── patents/
    ├── materials_science_patents.json
    ├── nanotechnology_patents.json
    ├── quantum_materials_patents.json
    ├── energy_systems_patents.json
    ├── photonics_patents.json
    ├── additive_manufacturing_patents.json
    └── metadata.json
```

---

## Performance Optimization

### Disk I/O Optimization

```bash
# Use SSD for faster writes
# Mount temporary directory on RAM
sudo mount -t tmpfs -o size=4G tmpfs /tmp/papers_temp

# Download to temp, then move to final location
python3 download_invention_data.py --output /tmp/papers_temp
mv /tmp/papers_temp invention_data/
```

### Network Optimization

```python
# Use connection pooling
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=5, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

---

## Best Practices

1. **Start with sample mode** to verify everything works
2. **Download high-priority categories first** (Priority 9-10)
3. **Monitor disk space** - 23,000 papers = ~500MB JSON + metadata
4. **Respect arXiv rate limits** - don't modify the sleep delays
5. **Keep backups** - copy `invention_data/` to external storage
6. **Document your changes** - if you modify categories or queries
7. **Validate data** - check that JSON files are well-formed
8. **Version control** - track which papers were downloaded when

---

## Future Enhancements

### Planned Features
- [ ] Automatic embedding generation
- [ ] Semantic search interface
- [ ] Citation graph analysis
- [ ] Automatic invention suggestion from papers
- [ ] Integration with visualization dashboard
- [ ] PDF full-text extraction
- [ ] Multi-language support
- [ ] Real-time paper monitoring

### Contributing

To add new categories or improve the system:

1. Fork the repository
2. Add your category to `_initialize_categories()`
3. Test with sample mode
4. Submit pull request with documentation

---

## Resources

- **arXiv API Documentation**: https://arxiv.org/help/api
- **arXiv Category Taxonomy**: https://arxiv.org/category_taxonomy
- **arXiv Bulk Data**: https://arxiv.org/help/bulk_data
- **Google Patents**: https://patents.google.com/
- **USPTO Bulk Data**: https://bulkdata.uspto.gov/
- **Python arxiv Library**: https://pypi.org/project/arxiv/

---

## Support

For issues or questions:

1. Check this guide's Troubleshooting section
2. Review the INVENTION_DATA_QUICKSTART.md
3. Check the logs in `invention_data/download.log`
4. File an issue on the repository

---

**Document Version**: 1.0
**Last Updated**: 2026-01-27
**Author**: Echo Prime Development Team
**License**: See LICENSE file in repository
