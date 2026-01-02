import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reasoning.llm_bridge import OllamaBridge

bridge = OllamaBridge(model="ech0-unified-14b-enhanced")
print(f"Querying Ollama ({bridge.model}) at {bridge.api_url}...")

response = bridge.query("Hello ECH0, are you online?")
print(f"RESPONSE: {response}")

if "BRIDGE ERROR" in response:
    print("❌ OFFLINE: Bridge could not connect to Ollama.")
else:
    print("✅ ONLINE: Ollama is responding correctly.")
