#!/usr/bin/env python3
"""
HuggingFace Spaces app.py entry point for Kairos
Minimal launcher that imports and runs the main application
"""

from huggingface_app import create_gradio_interface
import os

# Set environment variables for Spaces
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_SERVER_PORT", os.getenv("PORT", "7860"))

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()

    # Launch with Spaces optimizations
    interface.queue(max_size=20)  # Allow more concurrent users
    interface.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=False,  # Spaces handles sharing
        show_error=True,
        favicon_path=None,  # Use default favicon
        auth=None,  # No authentication for public demo
        max_threads=4  # Limit threads for stability
    )
