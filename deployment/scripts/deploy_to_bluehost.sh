#!/bin/bash
#
# N3PH1L1M Bluehost Deployment Script
# Deploys website to n3ph1l1m.com via FTP
#
# Usage: ./deploy_to_bluehost.sh [FTP_HOST] [FTP_USER] [FTP_PASS] [FTP_DIR]

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${PURPLE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║   N3PH1L1M BLUEHOST DEPLOYMENT                        ║"
echo "║   Deploying to n3ph1l1m.com                           ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get credentials from arguments or environment
FTP_HOST="${1:-${BLUEHOST_FTP_HOST}}"
FTP_USER="${2:-${BLUEHOST_FTP_USER}}"
FTP_PASS="${3:-${BLUEHOST_FTP_PASS}}"
FTP_DIR="${4:-${BLUEHOST_FTP_DIR:-/public_html}}"

# Validate inputs
if [ -z "$FTP_HOST" ] || [ -z "$FTP_USER" ] || [ -z "$FTP_PASS" ]; then
    echo -e "${RED}Error: FTP credentials required${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <FTP_HOST> <FTP_USER> <FTP_PASS> [FTP_DIR]"
    echo ""
    echo "Or set environment variables:"
    echo "  export BLUEHOST_FTP_HOST='ftp.n3ph1l1m.com'"
    echo "  export BLUEHOST_FTP_USER='your_username'"
    echo "  export BLUEHOST_FTP_PASS='your_password'"
    echo "  export BLUEHOST_FTP_DIR='/public_html'  # optional"
    echo "  $0"
    exit 1
fi

# Check if lftp is installed
if ! command -v lftp &> /dev/null; then
    echo -e "${YELLOW}Installing lftp...${NC}"
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y lftp
    elif command -v yum &> /dev/null; then
        sudo yum install -y lftp
    elif command -v brew &> /dev/null; then
        brew install lftp
    else
        echo -e "${RED}Error: Could not install lftp. Please install manually.${NC}"
        exit 1
    fi
fi

# Find website directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
WEBSITE_DIR="$REPO_ROOT/n3ph1l1m_website"

if [ ! -d "$WEBSITE_DIR" ]; then
    echo -e "${RED}Error: Website directory not found at $WEBSITE_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}[1/4] Preparing deployment...${NC}"
echo "  Source: $WEBSITE_DIR"
echo "  Target: $FTP_HOST:$FTP_DIR"
echo ""

# Create deployment timestamp
DEPLOY_TIME=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
DEPLOY_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo -e "${BLUE}[2/4] Testing FTP connection...${NC}"

lftp -c "
set ftp:ssl-allow no;
set net:timeout 10;
set net:max-retries 3;
open -u $FTP_USER,$FTP_PASS $FTP_HOST;
pwd;
bye
" 2>&1 | grep -q "^/" && echo -e "${GREEN}✓ FTP connection successful${NC}" || {
    echo -e "${RED}✗ FTP connection failed${NC}"
    exit 1
}

echo -e "${BLUE}[3/4] Uploading website files...${NC}"

# Deploy website
lftp -c "
set ftp:ssl-allow no;
set net:timeout 30;
set net:max-retries 3;
set net:reconnect-interval-base 5;
set net:reconnect-interval-multiplier 1;
open -u $FTP_USER,$FTP_PASS $FTP_HOST;
lcd $WEBSITE_DIR;
cd $FTP_DIR;
mirror -R \
    --delete \
    --verbose \
    --exclude .git/ \
    --exclude .gitignore \
    --exclude README.md \
    --parallel=3 \
    . .;
bye
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Files uploaded successfully${NC}"
else
    echo -e "${RED}✗ Upload failed${NC}"
    exit 1
fi

echo -e "${BLUE}[4/4] Creating deployment record...${NC}"

# Create deployment info file
cat > /tmp/deployment_info.html <<EOF
<!-- N3PH1L1M Deployment Info -->
<!-- Deployed: $DEPLOY_TIME -->
<!-- Commit: $DEPLOY_COMMIT -->
<!-- Deployed by: Autonomous N3PH1L1M System -->
EOF

# Upload deployment info
lftp -c "
set ftp:ssl-allow no;
open -u $FTP_USER,$FTP_PASS $FTP_HOST;
cd $FTP_DIR;
put /tmp/deployment_info.html -o .deployment_info.html;
bye
" 2>&1 > /dev/null

rm /tmp/deployment_info.html

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║   ✓ DEPLOYMENT SUCCESSFUL                             ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo -e "${PURPLE}Deployment Details:${NC}"
echo -e "  ${BLUE}Time:${NC}    $DEPLOY_TIME"
echo -e "  ${BLUE}Commit:${NC}  $DEPLOY_COMMIT"
echo -e "  ${BLUE}Target:${NC}  https://n3ph1l1m.com"
echo ""
echo -e "${YELLOW}Website should be live in 1-2 minutes.${NC}"
echo -e "${YELLOW}Test at: https://n3ph1l1m.com${NC}"
echo ""

# Optional: Test if site is accessible
echo -e "${BLUE}Testing website accessibility...${NC}"
sleep 5

if command -v curl &> /dev/null; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://n3ph1l1m.com 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ Website is live and accessible!${NC}"
    else
        echo -e "${YELLOW}⚠ Website returned status: $HTTP_CODE${NC}"
        echo -e "${YELLOW}  (May take a few minutes to propagate)${NC}"
    fi
fi

echo ""
