# ECH0-PRIME: Cognitive-Synthetic Architecture

**Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**

A production-grade Cognitive-Synthetic Architecture (CSA) featuring hierarchical generative models, quantum attention mechanisms, and autonomous reasoning capabilities.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running](#running)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ECH0-PRIME implements a five-level cortical hierarchy with predictive coding, free-energy minimization, and optional quantum attention circuits. It integrates with local LLMs (Ollama), cloud providers (OpenAI, Anthropic, Together AI), and provides a real-time monitoring dashboard.

### Key Capabilities

| Domain | Features |
|---|---|
| **Cognitive Engine** | 5-level cortical hierarchy, free-energy minimization, variational inference |
| **Quantum Attention** | VQE-optimized variational quantum circuits (Qiskit) |
| **Reasoning** | Probabilistic, causal, analogical, and neuro-symbolic reasoning |
| **LLM Integration** | Local Ollama, OpenAI, Anthropic, Together AI bridges |
| **Memory** | FAISS vector store, episodic/semantic consolidation, knowledge graph |
| **Multi-Agent** | Hive-mind swarm intelligence, consensus mechanisms |
| **Safety** | Constitutional AI alignment, multi-layer value checks |
| **Voice / Vision** | ElevenLabs TTS, Apple Intelligence bridge, vision processing |
| **Dashboard** | Real-time React UI with WebSocket state streaming |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ECH0-PRIME AGI                        │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  Meta    │Prefrontal│Associative│Perceptual│  Sensory    │  ← cortical levels
│  (L4)    │  (L3)    │  (L2)    │  (L1)    │  (L0)       │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│  Free Energy Engine  │  Global Workspace  │  Attention   │
├──────────────────────┴─────────────────────┴─────────────┤
│   Memory Manager  │  Safety Orchestrator  │  LLM Bridge  │
├───────────────────┴───────────────────────┴──────────────┤
│   Dashboard API (FastAPI)  │  Gradio HF Space Interface  │
└────────────────────────────┴─────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Workofarttattoo/echo_prime.git
cd echo_prime

# 2. Copy env and configure
cp .env.example .env
# Edit .env — set at minimum: ECH0_PHASE, LLM_PROVIDER

# 3. Start with Docker
make build
make up

# 4. Open the dashboard
open http://localhost:8000
```

---

## Installation

### Prerequisites

- **Python** 3.10+
- **Docker** & Docker Compose (for containerized deployment)
- **Ollama** (optional — for local LLM inference)
- **Node.js 18+** (optional — only for building dashboard v2)

### Local Install

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install dev tools (optional)
pip install pytest black flake8 mypy
```

### Optional Dependencies (in `pyproject.toml`)

```bash
pip install ".[api]"       # FastAPI, uvicorn, pydantic
pip install ".[cloud]"     # OpenAI, Anthropic
pip install ".[database]"  # PostgreSQL, Redis
pip install ".[dev]"       # pytest, black, flake8, mypy
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

See [Environment Variables](#environment-variables) for the full list.

---

## Running

### Application Modes

ECH0-PRIME supports three modes controlled by the `ECH0_MODE` env var:

| Mode | Description | Default Port |
|---|---|---|
| `orchestrator` | Full cognitive engine (default) | — |
| `dashboard` | Dashboard API + WebSocket server | 8000 |
| `gradio` | Gradio chat interface (HF Spaces) | 7860 |

```bash
# Run the orchestrator
ECH0_MODE=orchestrator python app.py

# Run the dashboard API
ECH0_MODE=dashboard python app.py

# Run the Gradio demo
ECH0_MODE=gradio python app.py
```

### Using Make

```bash
make dev            # orchestrator
make dev-dashboard  # dashboard
make dev-gradio     # gradio
```

---

## Docker Deployment

### Build & Run

```bash
make build          # Build images
make up             # Start services (detached)
make logs           # Tail logs
make down           # Stop everything
```

### Production Deploy

```bash
./scripts/deploy.sh           # Full deploy with health checks
./scripts/deploy.sh --build   # Force-rebuild images first
./scripts/deploy.sh --dry-run # Preview without executing
```

### Services

| Service | Container | Port | Description |
|---|---|---|---|
| `echo-prime` | echo-prime-core | 7860 | Main cognitive engine |
| `dashboard` | echo-prime-dashboard | 8000 | Dashboard API |
| `redis` | echo-prime-redis | 6379 | Cache / session store |

---

## API Reference

### Dashboard API (`/api/…`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | System status & active model |
| `GET` | `/api/consciousness` | IIT Phi value and consciousness metrics |
| `GET` | `/api/evolution-units` | System evolution metrics |
| `GET` | `/api/missions` | Active missions and objectives |
| `GET` | `/api/autonomous-activity` | Autonomous operation logs |
| `POST` | `/api/chat` | Send a chat message to the reasoning engine |
| `POST` | `/api/evaluate-repo` | Evaluate & improve a GitHub repository |

### WebSocket

| Endpoint | Description |
|---|---|
| `ws://HOST:PORT/ws` | Real-time state stream (JSON frames) |

### Example

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'

# Consciousness metrics
curl http://localhost:8000/api/consciousness
```

---

## Testing

```bash
make test          # Run full test suite
make test-basic    # Smoke tests only
make lint          # Lint with flake8
make format        # Auto-format with black
make typecheck     # mypy type-checking
```

Tests live in `tests/` and follow `test_*.py` naming. The suite covers:

- Basic math & reasoning (`test_ech0_basic.py`)
- Phase validation (`test_phase_1.py` … `test_phase_6_audio_voice.py`)
- Integration tests (`test_integration.py`)
- LLM reasoning (`test_llm_reasoning.py`)
- Hive mind (`test_hive_mind.py`)

---

## Project Structure

```
echo_prime/
├── app.py                  # Unified entry point (orchestrator / gradio / dashboard)
├── main_orchestrator.py    # EchoPrimeAGI — core cognitive loop
├── dashboard_server.py     # FastAPI dashboard with chat + WS
├── Dockerfile              # Multi-stage production build
├── docker-compose.yml      # Full service stack
├── Makefile                # Build, test, deploy commands
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata & tool config
├── .env.example            # Environment variable template
│
├── core/                   # Cognitive engine, attention, config
│   ├── engine.py           #   Hierarchical generative model
│   ├── attention.py        #   Quantum attention head
│   ├── config.py           #   Dimension profiles
│   ├── api_service.py      #   Internal API server
│   ├── vision_bridge.py    #   Vision processing
│   ├── voice_bridge.py     #   Voice synthesis
│   └── license_manager.py  #   License & code protection
│
├── reasoning/              # Reasoning subsystems
│   ├── orchestrator.py     #   Multi-strategy reasoning router
│   ├── llm_bridge.py       #   Ollama & Together AI bridges
│   ├── probabilistic.py    #   Bayesian reasoning
│   ├── causal_discovery.py #   Causal inference
│   └── tools/              #   ArXiv, QuLab, Pinecone bridges
│
├── learning/               # Meta-learning, transfer, architecture search
├── memory/                 # FAISS-backed episodic & semantic memory
├── safety/                 # Constitutional AI alignment
├── agents/                 # Multi-agent collaboration
├── capabilities/           # Creativity, math, scientific discovery
├── ech0_governance/        # Knowledge graph, evaluators, persistent memory
├── mcp_server/             # Tool registry & discovery
├── infrastructure/         # Distributed processing, monitoring
├── missions/               # Autonomous goal pursuit
├── training/               # AGI training pipeline
├── quantum_attention/      # Quantum circuit bridge
├── research/               # Consciousness tracker, philosophy engine
├── code_evaluation/        # Autonomous code analysis
│
├── dashboard/              # Legacy HTML dashboard
├── dashboard-v3/           # v3 dashboard (vanilla JS)
├── dashboard/v2/           # React + Vite dashboard
│
├── scripts/                # Operational scripts
│   ├── deploy.sh           #   Production deployment
│   └── benchmarks/         #   Benchmark runners
│
├── tests/                  # Test suite
├── docs/                   # Guides & documentation
├── deployment/             # Systemd units, deploy scripts
├── external/               # Vendored libraries (JetStream, etc.)
└── hf_space/               # HuggingFace Spaces Gradio app
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ECH0_PHASE` | ✅ | `2` | System phase (1/2/3) |
| `ECH0_LIGHTWEIGHT` | | `0` | Skip heavy models (`1` for testing) |
| `ECH0_MODE` | | `orchestrator` | App mode: orchestrator / dashboard / gradio |
| `ECH0_DIM_PROFILE` | | `lite` | Dimension profile: full / balanced / lite |
| `LLM_PROVIDER` | | `ollama` | LLM backend |
| `OLLAMA_BASE_URL` | | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | | `llama3.2` | Ollama model name |
| `OPENAI_API_KEY` | | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | | — | Anthropic API key |
| `TOGETHER_API_KEY` | | — | Together AI key |
| `ELEVENLABS_API_KEY` | | — | ElevenLabs TTS key |
| `PINECONE_API_KEY` | | — | Pinecone vector DB key |
| `DATABASE_URL` | | — | PostgreSQL connection string |
| `REDIS_URL` | | `redis://redis:6379/0` | Redis URL |
| `DASHBOARD_PORT` | | `8000` | Dashboard API port |
| `GRADIO_PORT` | | `7860` | Gradio UI port |
| `HF_TOKEN` | | — | HuggingFace token |
| `ECH0_LICENSE_SECRET` | ✅ (prod) | — | License signing secret |

---

## Contributing

This is proprietary software. Contact Joshua Hendricks Cole for licensing and contribution agreements.

---

## License

**Proprietary** — Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Unauthorized copying, modification, distribution, or use of this software is strictly prohibited. See `pyproject.toml` for details.
