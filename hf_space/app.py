# app.py – Gradio demo for the ECH0‑PRIME Hugging Face Space

import os
from dotenv import load_dotenv
import gradio as gr
import requests
from typing import Optional

# ------------------------------------------------------------
# Load environment variables (TOGETHER_API_KEY, optional MODEL)
# ------------------------------------------------------------
load_dotenv()

class TogetherBridge:
    """Thin wrapper around the Together AI chat completion endpoint."""
    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-34B-Instruct-Turbo"):
        self.model = model
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.api_url = "https://api.together.xyz/v1/chat/completions"

    def query(self, prompt: str, system: Optional[str] = None) -> str:
        if not self.api_key:
            return "❌ ERROR – TOGETHER_API_KEY not set."
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ TOGETHER API ERROR: {e}"

# ------------------------------------------------------------
# Choose model – can be overridden via env var `HF_SPACE_MODEL`
# ------------------------------------------------------------
default_model = os.getenv("HF_SPACE_MODEL", "meta-llama/Meta-Llama-3.1-34B-Instruct-Turbo")
bridge = TogetherBridge(model=default_model)

# ------------------------------------------------------------
# System prompt that gives ECH0‑PRIME its persona
# ------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are ECH0‑PRIME, a Frontier AGI system. "
    "You have access to real‑time tools (QuLab, Arxiv, Python execution). "
    "When a user asks for a tool, actually invoke it via the ReasoningOrchestrator, "
    "do not just describe the tool."
)

def chat(user_msg: str, history: list):
    """Gradio callback – sends the user message to the model and appends the reply."""
    response = bridge.query(user_msg, system=SYSTEM_PROMPT)
    history.append((user_msg, response))
    return "", history

# ------------------------------------------------------------
# Build the Gradio UI – sleek, glass‑morphic style
# ------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        "# 🤖 ECH0‑PRIME – Autonomous AGI Demo\n\n"
        "*Powered by the 34B Together model (or any model you set via `HF_SPACE_MODEL`).*\n\n"
        "Ask it to run simulations, fetch papers, or just chat!"
    )
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your request here...", label="Message")
    send = gr.Button("Send", variant="primary")
    send.click(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    # When run locally (e.g., `python app.py`) launch on a local port.
    demo.queue().launch()
