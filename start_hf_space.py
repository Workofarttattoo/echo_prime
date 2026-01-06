#!/usr/bin/env python3
"""
ECH0-PRIME Hugging Face Space Entry Point
Orchestrates the backend, the dashboard, and local Ollama in a single container.
"""

import os
import sys
import threading
import time
import subprocess
import signal

def start_backend():
    print("🚀 Starting ECH0-PRIME API Server (FastAPI)...")
    # Hugging Face Spaces expect the app on port 7860
    os.environ["PORT"] = "7860"
    from core.api_service import start_api_server
    try:
        start_api_server(port=7860)
    except Exception as e:
        print(f"❌ Backend server failed: {e}")

def start_orchestrator():
    print("🧠 Starting ECH0-PRIME Unified Orchestrator...")
    # Phase 2, heavy-but-runnable dims, hosted LLM
    os.environ["ECH0_PHASE"] = "2"
    os.environ["ECH0_LIGHTWEIGHT"] = "0"
    os.environ.setdefault("ECH0_LLM_PROVIDER", "hf")
    # Default HF model can be overridden via ECH0_LLM_MODEL
    os.environ.setdefault("ECH0_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # We run the orchestrator in a loop to keep it alive
    while True:
        try:
            from main_orchestrator import boot_system
            boot_system()
        except Exception as e:
            print(f"⚠️ Orchestrator encountered an error: {e}")
            print("Restarting in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    print("🌟 ECH0-PRIME WORLD RELEASE: Hugging Face Edition 🌟")
    print("=" * 60)

    # 1. Create necessary directories if they don't exist
    for d in ["sensory_input", "logs", "backups", "memory_data", "checkpoints"]:
        os.makedirs(d, exist_ok=True)

    # 2. Start the API server in a separate thread
    # This serves the dashboard and handles WebSocket connections
    api_thread = threading.Thread(target=start_backend, daemon=True)
    api_thread.start()

    # 3. Wait a moment for the API to initialize
    time.sleep(5)

    # 4. Start the main cognitive orchestrator in the main thread
    # This runs the 5-level cognitive cycles
    try:
        start_orchestrator()
    except KeyboardInterrupt:
        print("\n🛑 ECH0-PRIME shutting down.")
        sys.exit(0)

