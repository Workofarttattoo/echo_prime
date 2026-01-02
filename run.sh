#!/bin/bash
# ECH0-PRIME Quick Run Script
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "🚀 Starting ECH0-PRIME..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import numpy" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt -q
    pip install -e . -q
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama not running. Starting Ollama..."
    echo "   (You may need to run 'ollama serve' in another terminal)"
    echo ""
fi

# Run ECH0-PRIME
echo "✓ Launching ECH0-PRIME..."
echo ""
python main_orchestrator.py
