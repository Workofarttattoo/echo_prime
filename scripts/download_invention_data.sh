#!/bin/bash
#
# Invention Data Download Shell Script
# Alternative method for downloading invention data when Python arxiv library is not available
# Uses curl to interact with arXiv API directly
#

set -e  # Exit on error

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-invention_data}"
LOG_FILE="${OUTPUT_DIR}/download.log"
MAX_RESULTS="${MAX_RESULTS:-100}"  # Default to sample mode

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=()

    for cmd in curl jq python3; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Install with: sudo apt-get install ${missing_deps[*]} (Ubuntu/Debian)"
        log_info "Or: brew install ${missing_deps[*]} (macOS)"
        exit 1
    fi

    log_success "All dependencies installed"
}

# Create directory structure
setup_directories() {
    log_info "Setting up directory structure..."

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/materials_science"
    mkdir -p "$OUTPUT_DIR/nanotechnology"
    mkdir -p "$OUTPUT_DIR/quantum_materials"
    mkdir -p "$OUTPUT_DIR/energy_systems"
    mkdir -p "$OUTPUT_DIR/photonics"
    mkdir -p "$OUTPUT_DIR/additive_manufacturing"
    mkdir -p "$OUTPUT_DIR/invention_methodology"

    log_success "Directories created"
}

# Download category using arXiv API
download_category() {
    local name="$1"
    local arxiv_cat="$2"
    local keywords="$3"
    local target="$4"
    local output_subdir="$5"

    log_info "Downloading $name..."
    log_info "  Category: $arxiv_cat"
    log_info "  Target: $target papers"

    local output_file="$OUTPUT_DIR/$output_subdir/papers.json"
    local temp_file="$OUTPUT_DIR/$output_subdir/papers_temp.xml"

    # Build query
    local keyword_query=$(echo "$keywords" | sed 's/,/ OR /g')
    local query="cat:${arxiv_cat}+AND+(${keyword_query})"

    # URL encode the query
    query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$query'))")

    # arXiv API URL
    local url="http://export.arxiv.org/api/query?search_query=${query}&start=0&max_results=${target}&sortBy=submittedDate&sortOrder=descending"

    log_info "  Fetching from arXiv API..."

    # Download with curl
    if curl -s "$url" -o "$temp_file"; then
        log_success "  Downloaded XML response"

        # Parse XML to JSON using Python
        python3 << EOF > "$output_file"
import xml.etree.ElementTree as ET
import json
import sys

try:
    tree = ET.parse('$temp_file')
    root = tree.getroot()

    # Namespace for arXiv API
    ns = {'atom': 'http://www.w3.org/2005/Atom',
          'arxiv': 'http://arxiv.org/schemas/atom'}

    papers = []
    for entry in root.findall('atom:entry', ns):
        paper = {
            'title': entry.find('atom:title', ns).text.strip(),
            'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
            'abstract': entry.find('atom:summary', ns).text.strip(),
            'published': entry.find('atom:published', ns).text,
            'updated': entry.find('atom:updated', ns).text,
            'arxiv_id': entry.find('atom:id', ns).text.split('/')[-1],
            'pdf_url': next((link.get('href') for link in entry.findall('atom:link', ns) if link.get('title') == 'pdf'), None),
            'categories': [cat.get('term') for cat in entry.findall('atom:category', ns)],
            'primary_category': entry.find('arxiv:primary_category', ns).get('term') if entry.find('arxiv:primary_category', ns) is not None else None
        }
        papers.append(paper)

    print(json.dumps(papers, indent=2))
    sys.exit(0)

except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)
EOF

        if [ $? -eq 0 ]; then
            local count=$(jq '. | length' "$output_file")
            log_success "  Saved $count papers to $output_file"

            # Save metadata
            cat > "$OUTPUT_DIR/$output_subdir/metadata.json" << EOF
{
  "category": "$name",
  "arxiv_category": "$arxiv_cat",
  "keywords": "$keywords",
  "downloaded": $count,
  "target": $target,
  "download_date": "$(date -Iseconds)",
  "output_file": "$output_file"
}
EOF

        else
            log_error "  Failed to parse XML response"
        fi

        # Cleanup
        rm -f "$temp_file"
    else
        log_error "  Failed to download from arXiv API"
    fi

    # Rate limiting
    sleep 2
}

# Main download function
download_all() {
    log_info "Starting invention data download"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Max results per category: $MAX_RESULTS"
    log_info ""

    # Materials Science (Priority 10)
    download_category \
        "Materials Science" \
        "cond-mat.mtrl-sci" \
        "metamaterials,graphene,nanocomposites,2D materials,carbon nanotubes,smart materials,self-healing" \
        "$MAX_RESULTS" \
        "materials_science"

    # Nanotechnology (Priority 10)
    download_category \
        "Nanotechnology" \
        "cond-mat.mes-hall" \
        "nanodevices,molecular assembly,nanofabrication,quantum dots,nanowires,nanoparticles" \
        "$MAX_RESULTS" \
        "nanotechnology"

    # Quantum Materials (Priority 9)
    download_category \
        "Quantum Materials" \
        "quant-ph" \
        "topological insulators,superconductors,quantum dots,quantum computing,quantum sensors,spintronics" \
        "$MAX_RESULTS" \
        "quantum_materials"

    # Energy Systems (Priority 9)
    download_category \
        "Energy Systems" \
        "physics.app-ph" \
        "energy storage,batteries,solar cells,fuel cells,thermoelectric,energy harvesting,supercapacitors" \
        "$MAX_RESULTS" \
        "energy_systems"

    # Photonics & Holography (Priority 8)
    download_category \
        "Photonics & Holography" \
        "physics.optics" \
        "photonics,plasmonics,holography,optical devices,metamaterials,photonic crystals,optical computing" \
        "$MAX_RESULTS" \
        "photonics"

    # Additive Manufacturing (Priority 8)
    download_category \
        "Additive Manufacturing" \
        "cs.RO" \
        "3D printing,additive manufacturing,rapid prototyping,bioprinting,4D printing,metal printing" \
        "$MAX_RESULTS" \
        "additive_manufacturing"

    # Invention Methodology (Priority 6)
    download_category \
        "Invention Methodology" \
        "cs.AI" \
        "TRIZ,design thinking,systematic invention,creative problem solving,innovation methodology" \
        "$MAX_RESULTS" \
        "invention_methodology"
}

# Print summary
print_summary() {
    log_info ""
    log_info "================================"
    log_info "DOWNLOAD SUMMARY"
    log_info "================================"

    local total=0
    for dir in "$OUTPUT_DIR"/*/; do
        if [ -f "$dir/papers.json" ]; then
            local count=$(jq '. | length' "$dir/papers.json")
            local name=$(basename "$dir")
            log_info "  $name: $count papers"
            total=$((total + count))
        fi
    done

    log_info "================================"
    log_info "Total papers downloaded: $total"
    log_info "================================"
}

# Main execution
main() {
    echo "🚀 Invention Data Download System"
    echo "=================================="
    echo ""

    check_dependencies
    setup_directories
    download_all
    print_summary

    log_success "Download complete!"
    log_info "Data saved to: $OUTPUT_DIR"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            MAX_RESULTS=100
            shift
            ;;
        --full)
            MAX_RESULTS=5000
            shift
            ;;
        --max-results)
            MAX_RESULTS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --sample           Download sample (100 papers per category)"
            echo "  --full             Download full dataset (5000 papers per category)"
            echo "  --max-results N    Download N papers per category"
            echo "  --output DIR       Output directory (default: invention_data)"
            echo "  --help             Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
