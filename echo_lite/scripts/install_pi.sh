#!/bin/bash
#
# Echo Lite Installation Script for Raspberry Pi 5
#
# This script installs Echo Lite on a fresh Raspberry Pi OS installation
#

set -e

echo "================================"
echo "🍓 Echo Lite - Raspberry Pi Setup"
echo "================================"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3 and pip
echo "🐍 Installing Python..."
sudo apt-get install -y python3 python3-pip python3-venv

# Install system dependencies
echo "📚 Installing system dependencies..."
sudo apt-get install -y \
    git \
    build-essential \
    sqlite3 \
    libsqlite3-dev \
    python3-dev \
    python3-numpy

# Create installation directory
INSTALL_DIR="/opt/echo_lite"
echo "📁 Creating installation directory: $INSTALL_DIR"
sudo mkdir -p "$INSTALL_DIR"
sudo chown $USER:$USER "$INSTALL_DIR"

# Copy Echo Lite files
echo "📋 Copying Echo Lite files..."
cp -r ../core "$INSTALL_DIR/"
cp -r ../config "$INSTALL_DIR/"
cp ../requirements.txt "$INSTALL_DIR/"

cd "$INSTALL_DIR"

# Create virtual environment
echo "🔧 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory
echo "💾 Creating data directory..."
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"

# Create systemd service
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/echo-lite.service > /dev/null <<EOF
[Unit]
Description=Echo Lite Autonomous Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=$INSTALL_DIR/venv/bin/python3 -m core.agent_runtime
Restart=always
RestartSec=10
StandardOutput=append:$INSTALL_DIR/logs/echo_lite.log
StandardError=append:$INSTALL_DIR/logs/echo_lite_error.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service (but don't start yet)
sudo systemctl enable echo-lite.service

echo ""
echo "================================"
echo "✅ Installation Complete!"
echo "================================"
echo ""
echo "Echo Lite is installed at: $INSTALL_DIR"
echo ""
echo "To start Echo Lite:"
echo "  sudo systemctl start echo-lite"
echo ""
echo "To check status:"
echo "  sudo systemctl status echo-lite"
echo ""
echo "To view logs:"
echo "  tail -f $INSTALL_DIR/logs/echo_lite.log"
echo ""
echo "To stop Echo Lite:"
echo "  sudo systemctl stop echo-lite"
echo ""
echo "To run manually (for testing):"
echo "  cd $INSTALL_DIR"
echo "  source venv/bin/activate"
echo "  python3 -m core.agent_runtime"
echo ""
echo "🎉 Echo Lite is ready to run!"
echo ""
