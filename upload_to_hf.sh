#!/bin/bash
# Simple ECH0-PRIME Hugging Face Upload Script
# Run this after authenticating with: hf auth login

set -e

echo "🚀 ECH0-PRIME Hugging Face Upload"
echo "================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check authentication
echo -e "${BLUE}Checking authentication...${NC}"
if ! hf auth whoami > /dev/null 2>&1; then
    echo -e "${RED}❌ Not authenticated with Hugging Face${NC}"
    echo -e "${YELLOW}Please run: hf auth login${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Authentication confirmed${NC}"

# Repository info - use authenticated user's namespace
REPO_NAME="workofarttattoo/ech0-prime-agi"
echo -e "${BLUE}Target repository: https://huggingface.co/${REPO_NAME}${NC}"

# Create repository if needed
echo -e "${BLUE}Ensuring repository exists...${NC}"
hf repo create ech0-prime-agi --type model 2>/dev/null || echo "Repository already exists"

# Upload files
echo -e "${BLUE}📤 Uploading files...${NC}"

# Core files
echo -e "${GREEN}Uploading documentation...${NC}"
hf upload $REPO_NAME README.md --quiet || echo "README upload failed"
hf upload $REPO_NAME model_card.md --quiet || echo "model_card upload failed"
hf upload $REPO_NAME HUGGINGFACE_README.md --quiet || echo "HUGGINGFACE_README upload failed"
hf upload $REPO_NAME requirements.txt --quiet || echo "requirements upload failed"
hf upload $REPO_NAME verified_benchmark_results.json --quiet || echo "supremacy results upload failed"
hf upload $REPO_NAME FINAL_SUPREMACY_REPORT.md --quiet || echo "supremacy report upload failed"

# Visual assets
echo -e "${GREEN}Uploading visual assets...${NC}"
hf upload $REPO_NAME architecture_diagram.png --quiet || echo "architecture diagram upload failed"
hf upload $REPO_NAME consciousness_metrics.png --quiet || echo "consciousness metrics upload failed"
hf upload $REPO_NAME banner.png --quiet || echo "banner upload failed"
hf upload $REPO_NAME example_outputs.png --quiet || echo "example outputs upload failed"

echo -e "${GREEN}✅ Upload complete!${NC}"
echo -e "${GREEN}🌐 Repository: https://huggingface.co/${REPO_NAME}${NC}"
echo -e "${GREEN}🎉 ECH0-PRIME is now live on Hugging Face!${NC}"
