#!/usr/bin/env python3
"""
ECH0-PRIME Application Entry Point

Launches either the Gradio HuggingFace Space demo or the main orchestrator
depending on the ECH0_MODE environment variable.

Modes:
  - "gradio" (default): Launch the Gradio chat interface (HF Spaces)
  - "orchestrator":     Launch the full cognitive orchestrator
  - "dashboard":        Launch the dashboard API server

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def launch_gradio():
    """Launch the Gradio demo interface (for HuggingFace Spaces)."""
    try:
        from hf_space.app import demo
    except ImportError:
        # Fallback: build a minimal Gradio interface inline
        import gradio as gr  # type: ignore[import-untyped]

        def _chat(message: str, history: list) -> str:
            return "ECH0-PRIME is running but the full Gradio module is unavailable."

        demo = gr.ChatInterface(fn=_chat, title="ECH0-PRIME")

    demo.queue(max_size=20)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860"))),
        share=False,
        show_error=True,
    )


def launch_orchestrator():
    """Launch the full cognitive orchestrator."""
    from main_orchestrator import EchoPrimeAGI

    agi = EchoPrimeAGI()
    agi.run()


def launch_dashboard():
    """Launch the dashboard API server."""
    from core.api_service import start_api_server

    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    start_api_server(port=port)


if __name__ == "__main__":
    mode = os.getenv("ECH0_MODE", "gradio").lower()

    launchers = {
        "gradio": launch_gradio,
        "orchestrator": launch_orchestrator,
        "dashboard": launch_dashboard,
    }

    launcher = launchers.get(mode)
    if launcher is None:
        print(f"❌ Unknown ECH0_MODE: {mode!r}. Choose from: {', '.join(launchers)}")
        sys.exit(1)

    print(f"🚀 ECH0-PRIME starting in {mode!r} mode …")
    launcher()
