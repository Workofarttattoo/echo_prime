# app.py - Gradio demo for ECH0‑PRIME

import os
from dotenv import load_dotenv
import gradio as gr
import requests
import json
from typing import Optional

# Load environment variables (TOGETHER_API_KEY, etc.)
load_dotenv()

class TogetherBridge:
    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
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

bridge = TogetherBridge()

system_prompt = (
    "You are ECH0‑PRIME, a Frontier AGI system. "
    "You have access to real‑time tools (QuLab, Arxiv, Python execution). "
    "When a user asks for a tool, actually invoke it via the ReasoningOrchestrator, not just describe it."
)

def chat(user_msg: str, history: list):
    response = bridge.query(user_msg, system=system_prompt)
    history.append((user_msg, response))
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("""# 🤖 ECH0‑PRIME – Autonomous AGI Demo\n*Powered by the 70B Together model*\nAsk it to run simulations, fetch papers, or just chat!""")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your request here...", label="Message")
    send = gr.Button("Send", variant="primary")
    send.click(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.queue().launch()
