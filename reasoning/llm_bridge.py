import os
import requests
import json
from typing import Optional, Dict, Any, List

class OllamaBridge:
    """
    Connects ECH0-PRIME levels L2-L4 to locally running Ollama models.
    """
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/generate"

    def query(self, prompt: str, system: Optional[str] = None, images: Optional[List[str]] = None, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Sends a prompt (and optional images) to Ollama and returns the response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        }
        if system:
            payload["system"] = system
        
        if images:
            import base64
            encoded_images = []
            for img_path in images:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        encoded_images.append(base64.b64encode(f.read()).decode('utf-8'))
            payload["images"] = encoded_images

        max_retries = 3
        timeout_sec = 300
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("ECH0_LIGHTWEIGHT") == "1":
            max_retries = 1
            timeout_sec = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=timeout_sec) # Balanced timeout
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except requests.exceptions.Timeout:
                print(f"OLLAMA TIMEOUT: Attempt {attempt + 1}/{max_retries} failed. Retrying...")
                if attempt == max_retries - 1:
                    return "BRIDGE ERROR: Ollama request timed out after multiple attempts. The system may be overloaded."
                import time
                time.sleep(2 ** attempt) # Exponential backoff
            except Exception as e:
                return f"BRIDGE ERROR: {str(e)}"

    def reason_as_level(self, level_name: str, context: Dict[str, Any]) -> str:
        """Specific formatting for hierarchical levels with tool support."""
        system_prompt = (
            f"You are the {level_name} of a Cogntive-Synthetic Architecture (CSA). "
            "If you decide to take an action, use the following format: "
            "ACTION: {\"tool\": \"tool_name\", \"args\": [\"arg1\", ...]} "
            "Only use tools from the 'available_tools' list provided in the context."
        )
        user_prompt = f"Current Context: {json.dumps(context, indent=2)}\n\nReason through this scenario."
        
        return self.query(user_prompt, system=system_prompt)
