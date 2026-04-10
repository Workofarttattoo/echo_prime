# ECH0-PRIME: Cognitive-Synthetic Architecture
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (core modules only, not external/)
COPY core/ core/
COPY reasoning/ reasoning/
COPY learning/ learning/
COPY memory/ memory/
COPY safety/ safety/
COPY agents/ agents/
COPY training/ training/
COPY main_orchestrator.py .
COPY app.py .
COPY dashboard_server.py .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p memory_data checkpoints sensory_input optimization_state

# Expose ports (dashboard + API)
EXPOSE 7860 8000

# Default: run the main orchestrator
CMD ["python", "main_orchestrator.py"]
