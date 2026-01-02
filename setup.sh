#!/bin/bash
# ECH0-PRIME Setup Script
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e  # Exit on error

echo "=================================================="
echo "ECH0-PRIME Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION detected"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo ""
    read -p "⚠️  Virtual environment already exists. Recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old virtual environment..."
        rm -rf venv
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q

# Install package in editable mode
echo ""
echo "🔧 Installing echo-prime in editable mode..."
pip install -e . -q

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
else
    echo ""
    echo "ℹ️  .env file already exists (keeping existing configuration)"
fi

# Create required directories
echo ""
echo "📁 Creating required directories..."
mkdir -p sensory_input audio_input memory_data dashboard/data
echo "✓ Directories created"

# Check for Ollama
echo ""
echo "🔍 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "✓ Ollama service is running"
    else
        echo "⚠️  Ollama is installed but not running"
        echo "   Start it with: ollama serve"
    fi

    # Check for llama3.2 model
    if ollama list | grep -q "llama3.2"; then
        echo "✓ llama3.2 model is available"
    else
        echo "⚠️  llama3.2 model not found"
        echo "   Pull it with: ollama pull llama3.2"
    fi
else
    echo "⚠️  Ollama not found"
    echo "   Install with: brew install ollama"
fi

# Test imports
echo ""
echo "🧪 Testing module imports..."
if python -c "import core, memory, learning, reasoning, safety, training" 2>/dev/null; then
    echo "✓ All modules import successfully"
else
    echo "❌ Module import failed"
    exit 1
fi

# Run basic test
echo ""
echo "🧪 Running Phase 1 verification..."
if python tests/test_phase_1.py > /dev/null 2>&1; then
    echo "✓ Phase 1 tests passed"
else
    echo "⚠️  Phase 1 tests failed (may be expected in some configurations)"
fi

echo ""
echo "=================================================="
echo "✅ ECH0-PRIME Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Ensure Ollama is running: ollama serve"
echo "3. Run ECH0-PRIME: python main_orchestrator.py"
echo ""
echo "Optional:"
echo "- Launch dashboard: cd dashboard/v2 && npm install && npm run dev"
echo "- Read README.md for usage instructions"
echo "- Check INSTALL.md for troubleshooting"
echo ""
