import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.voice_bridge import VoiceBridge
from dotenv import load_dotenv

load_dotenv()

print("Initializing VoiceBridge with Jessica (cgSgspJ2msm6clMCkdW9)...")
# Using the same initialization as main_orchestrator.py
voice_bridge = VoiceBridge(voice="Samantha", eleven_voice_id="cgSgspJ2msm6clMCkdW9")

print(f"Check: Use ElevenLabs? {voice_bridge.use_eleven}")
print(f"Check: Voice ID? {getattr(voice_bridge, 'eleven_voice_id', 'N/A')}")

text = "System check complete. Voice output is now active using the Jessica voice. I am online and ready."
print(f"Speaking: '{text}'")

voice_bridge.speak(text, async_mode=False)

print("Verification complete.")
