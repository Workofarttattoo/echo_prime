#!/bin/bash
# ECH0-PRIME Unified Upload Script: GitHub + HuggingFace
# Uploads, commits, and merges changes to both GitHub and HuggingFace repositories

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🚀 ECH0-PRIME UNIFIED UPLOAD: GitHub + HuggingFace${NC}"
echo -e "${PURPLE}===================================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run command with error handling
run_cmd() {
    local description="$1"
    shift
    echo -e "${BLUE}🔧 $description${NC}"
    echo -e "${BLUE}Command: $@${NC}"
    if "$@"; then
        echo -e "${GREEN}✅ $description - Success${NC}"
        return 0
    else
        echo -e "${RED}❌ $description - Failed${NC}"
        return 1
    fi
}

# ========================================================================================
# STEP 1: GENERATE LATEST SUPREMACY REPORT
# ========================================================================================

echo -e "${YELLOW}Step 1: Generating Latest Supremacy Report${NC}"
if run_cmd "Generating final supremacy report" python3 scripts/generate_final_supremacy_report.py; then
    echo -e "${GREEN}✅ Latest benchmark results generated${NC}"
else
    echo -e "${YELLOW}⚠️ Supremacy report generation had issues, continuing...${NC}"
fi

# ========================================================================================
# STEP 2: GIT COMMIT AND PUSH TO GITHUB
# ========================================================================================

echo -e "${YELLOW}Step 2: Committing and Pushing to GitHub${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Not in a git repository. Cannot proceed with GitHub operations.${NC}"
    exit 1
fi

# Check git status
echo -e "${BLUE}Checking git status...${NC}"
GIT_STATUS=$(git status --porcelain)
if [ -z "$GIT_STATUS" ]; then
    echo -e "${YELLOW}⚠️ No changes to commit. Repository is up to date.${NC}"
else
    echo -e "${BLUE}Found changes to commit:${NC}"
    echo "$GIT_STATUS"
fi

# Add all changes
if run_cmd "Adding all files to git" git add .; then
    # Create meaningful commit message with current timestamp and supremacy score
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    if [ -f "FINAL_SUPREMACY_REPORT.md" ]; then
        SCORE_LINE=$(grep "AI suite overall score" FINAL_SUPREMACY_REPORT.md | head -1)
        if [ ! -z "$SCORE_LINE" ]; then
            SCORE=$(echo "$SCORE_LINE" | grep -o "[0-9]*\.[0-9]*%")
            COMMIT_MSG="Update ECH0-PRIME: $SCORE supremacy - $TIMESTAMP"
        else
            COMMIT_MSG="Update ECH0-PRIME - $TIMESTAMP"
        fi
    else
        COMMIT_MSG="Update ECH0-PRIME - $TIMESTAMP"
    fi

    echo -e "${BLUE}Commit message: $COMMIT_MSG${NC}"

    # Commit changes
    if run_cmd "Committing changes" git commit -m "$COMMIT_MSG"; then
        echo -e "${GREEN}✅ Changes committed successfully${NC}"
    else
        echo -e "${YELLOW}⚠️ No changes to commit or commit failed${NC}"
    fi

    # Push to GitHub
    if run_cmd "Pushing to GitHub" git push origin main; then
        echo -e "${GREEN}✅ Successfully pushed to GitHub${NC}"
        echo -e "${GREEN}🌐 GitHub Repository: https://github.com/Workofarttattoo/echo_prime${NC}"
    else
        echo -e "${RED}❌ Failed to push to GitHub${NC}"
        exit 1
    fi
fi

# ========================================================================================
# STEP 3: UPLOAD TO HUGGINGFACE
# ========================================================================================

echo -e "${YELLOW}Step 3: Uploading to HuggingFace${NC}"

# Check if HuggingFace CLI is available
if ! command_exists huggingface-cli && ! command_exists hf; then
    echo -e "${YELLOW}Installing HuggingFace CLI...${NC}"
    pip install huggingface_hub[cli]
fi

# Determine which HF command to use
if command_exists hf; then
    HF_CMD="hf"
elif command_exists huggingface-cli; then
    HF_CMD="huggingface-cli"
else
    echo -e "${RED}❌ HuggingFace CLI not found. Please install it with: pip install huggingface_hub[cli]${NC}"
    exit 1
fi

echo -e "${BLUE}Using HF command: $HF_CMD${NC}"

# Check HuggingFace authentication
echo -e "${BLUE}Checking HuggingFace authentication...${NC}"
if ! $HF_CMD auth whoami > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Not authenticated with HuggingFace${NC}"
    echo -e "${BLUE}Please run: $HF_CMD auth login${NC}"
    echo -e "${BLUE}Then re-run this script.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ HuggingFace authentication confirmed${NC}"

# Repository info
HF_USER=$($HF_CMD auth whoami 2>/dev/null | head -n 1)
if [ -z "$HF_USER" ]; then
    echo -e "${YELLOW}⚠️ Could not detect HF username, using 'noone' as default${NC}"
    HF_USER="noone"
fi

REPO_NAME="ech0-prime-agi"
FULL_REPO_NAME="${HF_USER}/${REPO_NAME}"

echo -e "${BLUE}Target repository: https://huggingface.co/${FULL_REPO_NAME}${NC}"

# Create repository if it doesn't exist
echo -e "${BLUE}Ensuring repository exists...${NC}"
$HF_CMD repo create "$REPO_NAME" --type model --organization "$HF_USER" 2>/dev/null || echo "Repository already exists"

# Upload files using the comprehensive upload script
echo -e "${BLUE}📤 Uploading to HuggingFace using comprehensive script...${NC}"

# Use the existing setup_huggingface_repo.sh script if available
if [ -f "setup_huggingface_repo.sh" ]; then
    echo -e "${BLUE}Using existing setup_huggingface_repo.sh script...${NC}"
    chmod +x setup_huggingface_repo.sh
    if ./setup_huggingface_repo.sh; then
        echo -e "${GREEN}✅ HuggingFace upload completed successfully${NC}"
    else
        echo -e "${YELLOW}⚠️ Comprehensive upload script failed, trying simple upload...${NC}"

        # Fallback to simple upload
        echo -e "${BLUE}📤 Performing simple upload...${NC}"

        # Upload core documentation
        $HF_CMD upload "$FULL_REPO_NAME" README.md README.md --quiet || echo "README upload failed"
        $HF_CMD upload "$FULL_REPO_NAME" model_card.md model_card.md --quiet || echo "model_card upload failed"
        $HF_CMD upload "$FULL_REPO_NAME" requirements.txt requirements.txt --quiet || echo "requirements upload failed"

        # Upload latest benchmark results
        if [ -f "verified_benchmark_results.json" ]; then
            $HF_CMD upload "$FULL_REPO_NAME" verified_benchmark_results.json verified_benchmark_results.json --quiet || echo "benchmark results upload failed"
        fi

        if [ -f "FINAL_SUPREMACY_REPORT.md" ]; then
            $HF_CMD upload "$FULL_REPO_NAME" FINAL_SUPREMACY_REPORT.md FINAL_SUPREMACY_REPORT.md --quiet || echo "supremacy report upload failed"
        fi

        echo -e "${GREEN}✅ Simple HuggingFace upload completed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ setup_huggingface_repo.sh not found, performing basic upload${NC}"

    # Basic upload of essential files
    $HF_CMD upload "$FULL_REPO_NAME" README.md README.md --quiet || echo "README upload failed"
    $HF_CMD upload "$FULL_REPO_NAME" model_card.md model_card.md --quiet || echo "model_card upload failed"
    $HF_CMD upload "$FULL_REPO_NAME" requirements.txt requirements.txt --quiet || echo "requirements upload failed"

    if [ -f "verified_benchmark_results.json" ]; then
        $HF_CMD upload "$FULL_REPO_NAME" verified_benchmark_results.json verified_benchmark_results.json --quiet || echo "benchmark results upload failed"
    fi

    if [ -f "FINAL_SUPREMACY_REPORT.md" ]; then
        $HF_CMD upload "$FULL_REPO_NAME" FINAL_SUPREMACY_REPORT.md FINAL_SUPREMACY_REPORT.md --quiet || echo "supremacy report upload failed"
    fi

    echo -e "${GREEN}✅ Basic HuggingFace upload completed${NC}"
fi

# ========================================================================================
# FINAL SUMMARY
# ========================================================================================

echo -e "${PURPLE}===================================================${NC}"
echo -e "${GREEN}🎉 ECH0-PRIME UNIFIED UPLOAD COMPLETE!${NC}"
echo -e "${PURPLE}===================================================${NC}"

if [ -f "FINAL_SUPREMACY_REPORT.md" ]; then
    echo -e "${BLUE}📊 Latest Supremacy Report:${NC}"
    grep -E "(AI suite overall score|HLE percent score|wisdom processed)" FINAL_SUPREMACY_REPORT.md || echo "Could not extract metrics"
fi

echo
echo -e "${GREEN}📍 Repositories:${NC}"
echo -e "${GREEN}  GitHub: https://github.com/Workofarttattoo/echo_prime${NC}"
echo -e "${GREEN}  HuggingFace: https://huggingface.co/${FULL_REPO_NAME}${NC}"

echo
echo -e "${BLUE}🚀 ECH0-PRIME is now live on both platforms!${NC}"
echo -e "${PURPLE}===================================================${NC}"
