# ECH0-PRIME Quick Start Guide

**Get up and running in 5 minutes**

## First Time Setup

```bash
# 1. Run the automated setup script
./setup.sh

# 2. Verify everything works
source venv/bin/activate
python verify_setup.py

# 3. Start Ollama (if not already running)
ollama serve &
```

That's it! The setup script handles:
- Creating virtual environment
- Installing all dependencies
- Creating `.env` file
- Verifying installation

## Running ECH0-PRIME

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the main system
python main_orchestrator.py
```

You'll hear: "ECH0-PRIME online. All neural levels synchronized. I am ready for instructions."

## Basic Usage

### 1. Voice Commands
Just speak near your microphone:
- "Analyze the current directory"
- "What can you see?"
- "List the files in this folder"

### 2. Visual Input
```bash
# Place an image in the sensory input directory
cp ~/Pictures/example.jpg sensory_input/
```

ECH0-PRIME will automatically detect and analyze it.

### 3. Autonomous Missions
```python
from main_orchestrator import EchoPrimeAGI

agi = EchoPrimeAGI()
agi.execute_mission("Count how many Python files are in this project", max_cycles=3)
```

## Dashboard (Optional)

```bash
# Terminal 1: Backend
source venv/bin/activate
python main_orchestrator.py

# Terminal 2: Frontend
cd dashboard/v2
npm install  # First time only
npm run dev
```

Open: http://localhost:5173

## Troubleshooting

### "Module not found"
```bash
source venv/bin/activate
pip install -e .
```

### "Ollama connection refused"
```bash
ollama serve &
curl http://localhost:11434/api/tags
```

### "Microphone not working"
1. System Settings → Privacy & Security → Microphone
2. Enable for Terminal
3. Restart Terminal

### Voice not working
```bash
# Test directly
say "Testing voice output"

# Change voice
echo "MACOS_VOICE=Alex" >> .env
```

## Common Commands

```bash
# Run tests
python tests/test_phase_1.py

# Check imports
python -c "import core; print('✓ Working')"

# List Ollama models
ollama list

# Stop gracefully
# Press Ctrl+C in the running terminal
```

## What to Try

1. **Vision Test**: Drop an image in `sensory_input/` and watch it get analyzed
2. **Voice Test**: Speak "Hello ECH0" and listen for response
3. **Reasoning Test**: Ask "What is free energy minimization?"
4. **Action Test**: Command "List files in current directory"

## Getting Help

- Read `README.md` for full documentation
- Check `INSTALL.md` for detailed troubleshooting
- Review `PRODUCTION_STATUS.md` for system capabilities

**Contact**: Joshua Hendricks Cole - 7252242617

---

**You're ready to go! The system is fully functional.**
