# ===========================================================================
# ECH0-PRIME: Cognitive-Synthetic Architecture — Production Dockerfile
# Multi-stage build: deps → runtime
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
# All Rights Reserved. PATENT PENDING.
# ===========================================================================

# ---------- Stage 1: dependency builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile wheels (numpy, torch cpu, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

LABEL maintainer="Joshua Hendricks Cole <josh@corporationoflight.com>"
LABEL org.opencontainers.image.description="ECH0-PRIME Cognitive-Synthetic AGI Architecture"

WORKDIR /app

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        portaudio19-dev \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source — order from least- to most-frequently changed
# so Docker layer caching works well.
COPY pyproject.toml .
COPY core/           core/
COPY reasoning/      reasoning/
COPY learning/       learning/
COPY memory/         memory/
COPY safety/         safety/
COPY agents/         agents/
COPY training/       training/
COPY capabilities/   capabilities/
COPY ech0_governance/ ech0_governance/
COPY mcp_server/     mcp_server/
COPY infrastructure/ infrastructure/
COPY missions/       missions/
COPY quantum_attention/ quantum_attention/
COPY research/       research/
COPY code_evaluation/ code_evaluation/
COPY dashboard/      dashboard/
COPY dashboard-v3/   dashboard-v3/
COPY dashboard_server.py .
COPY main_orchestrator.py .
COPY app.py .
COPY memory_data/    memory_data/
COPY data/           data/

# Create runtime directories
RUN mkdir -p memory_data checkpoints sensory_input optimization_state \
             audio_input logs

# Non-root user for security
RUN groupadd -r ech0 && useradd -r -g ech0 -d /app ech0 \
    && chown -R ech0:ech0 /app
USER ech0

# Expose ports: Gradio / HF Spaces | Dashboard API
EXPOSE 7860 8000

# Healthcheck — hit the dashboard health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]

# Default: launch in orchestrator mode
# Override with ECH0_MODE=gradio or ECH0_MODE=dashboard
ENV ECH0_MODE=orchestrator
CMD ["python", "app.py"]
