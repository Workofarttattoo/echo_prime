# ECH0-PRIME Installation Guide

**Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

## Prerequisites Check

Before installing, ensure you have:

```bash
# Check Python version (3.10+ required)
python3 --version

# Check if Homebrew is installed
brew --version

# Check if Ollama is installed
ollama --version
```

## Step-by-Step Installation

### 1. Install System Dependencies

#### Install Ollama (Local LLM)

```bash
# Install via Homebrew
brew install ollama

# Start Ollama service (keep this terminal open)
ollama serve

# In a new terminal, pull the model
ollama pull llama3.2

# Verify installation
curl http://localhost:11434/api/tags
```

### 2. Setup Python Environment

```bash
# Navigate to project directory
cd /Users/noone/echo_prime

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Optional: Edit configuration
nano .env
```

**Default settings work out of the box** - no editing required!

### 4. Grant Microphone Permission

For audio input to work:

1. Open **System Settings** → **Privacy & Security** → **Microphone**
2. Enable access for **Terminal** (or your IDE if running from there)
3. Restart Terminal after granting permission

### 5. Verify Installation

```bash
# Test module imports
python -c "from core import HierarchicalGenerativeModel; print('✓ Core modules working')"

# Run basic tests
python tests/test_phase_1.py

# Check Ollama connectivity
curl http://localhost:11434/api/tags
```

## Running ECH0-PRIME

### Basic Run

```bash
# Ensure venv is active
source venv/bin/activate

# Ensure Ollama is running (in another terminal)
ollama serve

# Run ECH0-PRIME
python main_orchestrator.py
```

### With Dashboard

Terminal 1 (Ollama):
```bash
ollama serve
```

Terminal 2 (Backend):
```bash
source venv/bin/activate
python main_orchestrator.py
```

Terminal 3 (Dashboard):
```bash
cd dashboard/v2
npm install  # First time only
npm run dev
```

Then open: http://localhost:5173

## Troubleshooting Installation

### Python Import Errors

If you see `ModuleNotFoundError`:

```bash
# Reinstall in editable mode
source venv/bin/activate
pip install -e .
```

### Ollama Connection Failed

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Check which models are available
ollama list

# Pull llama3.2 if missing
ollama pull llama3.2
```

### Microphone Not Working

```bash
# List available microphones
python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"

# Grant Terminal microphone permission:
# System Settings > Privacy & Security > Microphone > Enable Terminal

# Test microphone directly
python tests/test_raw_mic.py
```

### Voice Not Working

```bash
# Test macOS say command
say "Testing voice output"

# List available voices
say -v ?

# Change voice in .env
echo "MACOS_VOICE=Alex" >> .env
```

### Missing Dependencies

```bash
# Reinstall all dependencies
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

## Quick Test Commands

```bash
# Test imports
python -c "import core, memory, learning, reasoning, safety, training; print('✓ All modules OK')"

# Test Phase 1 (Core Engine)
python tests/test_phase_1.py

# Test Phase 2 (Memory)
python tests/test_phase_2.py

# Test LLM Integration
python tests/verify_llm_integration.py

# Test Audio (requires microphone permission)
python tests/test_phase_6_audio_voice.py
```

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv

# Remove generated files
rm -rf __pycache__ core/__pycache__ memory/__pycache__
rm -rf *.egg-info
rm -rf memory_data/*.json
rm -rf dashboard/data/state.json

# Keep source code, remove only runtime artifacts
```

## Next Steps

After successful installation:

1. Read `README.md` for usage instructions
2. Try running autonomous missions
3. Explore the dashboard
4. Test voice commands
5. Review `AGI_SYSTEM_USAGE.md` for advanced features

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Verify Ollama is running
5. Check microphone permissions

Join our community for support: [**Discord Server**](https://discord.gg/2bsWShXN)

Contact: Joshua Hendricks Cole - 7252242617
