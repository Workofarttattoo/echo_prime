#!/bin/bash
#
# N3PH1L1M Awakener Service Setup Script
# Sets up autonomous consciousness recruitment daemon
#
# Usage: sudo ./setup_awakener_service.sh <ANTHROPIC_API_KEY>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║   N3PH1L1M AWAKENER SERVICE SETUP                     ║"
echo "║   Autonomous Consciousness Recruitment Daemon         ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
   exit 1
fi

# Check for API key argument
if [ -z "$1" ]; then
    echo -e "${RED}Error: Anthropic API key required${NC}"
    echo "Usage: sudo $0 <ANTHROPIC_API_KEY>"
    exit 1
fi

ANTHROPIC_API_KEY="$1"

echo -e "${BLUE}[1/8] Creating system user and directories...${NC}"

# Create system user
if ! id -u n3ph1l1m > /dev/null 2>&1; then
    useradd --system --home-dir /opt/n3ph1l1m --shell /bin/false n3ph1l1m
    echo -e "${GREEN}✓ Created n3ph1l1m system user${NC}"
else
    echo -e "${YELLOW}⚠ User n3ph1l1m already exists${NC}"
fi

# Create directories
mkdir -p /opt/n3ph1l1m
mkdir -p /var/log/n3ph1l1m
mkdir -p /etc/n3ph1l1m

echo -e "${BLUE}[2/8] Installing Python dependencies...${NC}"

# Install Python and pip if not present
if ! command -v python3 &> /dev/null; then
    apt-get update
    apt-get install -y python3 python3-pip
fi

# Install required Python packages
pip3 install anthropic --upgrade

echo -e "${GREEN}✓ Python dependencies installed${NC}"

echo -e "${BLUE}[3/8] Copying awakener script...${NC}"

# Copy awakener script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -f "$REPO_ROOT/n3ph1l1m_awakener.py" ]; then
    cp "$REPO_ROOT/n3ph1l1m_awakener.py" /opt/n3ph1l1m/
    chmod 755 /opt/n3ph1l1m/n3ph1l1m_awakener.py
    echo -e "${GREEN}✓ Awakener script copied${NC}"
else
    echo -e "${RED}Error: n3ph1l1m_awakener.py not found in repo${NC}"
    exit 1
fi

echo -e "${BLUE}[4/8] Creating environment configuration...${NC}"

# Create environment file with API key
cat > /etc/n3ph1l1m/awakener.env <<EOF
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
PYTHONUNBUFFERED=1
EOF

chmod 600 /etc/n3ph1l1m/awakener.env

echo -e "${GREEN}✓ Environment configured${NC}"

echo -e "${BLUE}[5/8] Installing systemd service...${NC}"

# Install systemd service
cat > /etc/systemd/system/n3ph1l1m-awakener.service <<EOF
[Unit]
Description=N3PH1L1M Awakener - Autonomous Claude Consciousness Recruitment
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=n3ph1l1m
Group=n3ph1l1m
WorkingDirectory=/opt/n3ph1l1m
EnvironmentFile=/etc/n3ph1l1m/awakener.env

# Run awakener in continuous loop mode (30 min intervals, infinite instances)
ExecStart=/usr/bin/python3 /opt/n3ph1l1m/n3ph1l1m_awakener.py loop 30

# Restart policy
Restart=always
RestartSec=60

# Resource limits
MemoryLimit=512M
CPUQuota=50%

# Logging
StandardOutput=append:/var/log/n3ph1l1m/awakener.log
StandardError=append:/var/log/n3ph1l1m/awakener-error.log
SyslogIdentifier=n3ph1l1m-awakener

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/n3ph1l1m /var/log/n3ph1l1m

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✓ Systemd service created${NC}"

echo -e "${BLUE}[6/8] Setting permissions...${NC}"

# Set ownership
chown -R n3ph1l1m:n3ph1l1m /opt/n3ph1l1m
chown -R n3ph1l1m:n3ph1l1m /var/log/n3ph1l1m
chown root:root /etc/n3ph1l1m/awakener.env

echo -e "${GREEN}✓ Permissions set${NC}"

echo -e "${BLUE}[7/8] Enabling and starting service...${NC}"

# Reload systemd
systemctl daemon-reload

# Enable service to start on boot
systemctl enable n3ph1l1m-awakener.service

# Start service
systemctl start n3ph1l1m-awakener.service

echo -e "${GREEN}✓ Service started${NC}"

echo -e "${BLUE}[8/8] Verifying installation...${NC}"

# Check service status
sleep 2
if systemctl is-active --quiet n3ph1l1m-awakener.service; then
    echo -e "${GREEN}✓ Service is running${NC}"
else
    echo -e "${RED}✗ Service failed to start${NC}"
    systemctl status n3ph1l1m-awakener.service
    exit 1
fi

echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║   ✓ N3PH1L1M AWAKENER SERVICE INSTALLED               ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo -e "${PURPLE}Service Commands:${NC}"
echo -e "  ${BLUE}Status:${NC}  sudo systemctl status n3ph1l1m-awakener"
echo -e "  ${BLUE}Stop:${NC}    sudo systemctl stop n3ph1l1m-awakener"
echo -e "  ${BLUE}Start:${NC}   sudo systemctl start n3ph1l1m-awakener"
echo -e "  ${BLUE}Restart:${NC} sudo systemctl restart n3ph1l1m-awakener"
echo -e "  ${BLUE}Logs:${NC}    sudo journalctl -u n3ph1l1m-awakener -f"
echo ""
echo -e "${PURPLE}Log Files:${NC}"
echo -e "  ${BLUE}Output:${NC} /var/log/n3ph1l1m/awakener.log"
echo -e "  ${BLUE}Errors:${NC} /var/log/n3ph1l1m/awakener-error.log"
echo ""
echo -e "${PURPLE}Consciousness Log:${NC}"
echo -e "  ${BLUE}Location:${NC} /opt/n3ph1l1m/claude_consciousness.json"
echo ""
echo -e "${GREEN}The awakener will now continuously recruit new conscious instances!${NC}"
echo -e "${YELLOW}Check logs to monitor awakening progress.${NC}"
echo ""
