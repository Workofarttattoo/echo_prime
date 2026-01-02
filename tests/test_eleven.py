import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.voice_bridge import VoiceBridge

load_dotenv()

print("Testing ECH0 VoiceBridge (ElevenLabs -> afplay fallback)...")
# Using Ms. Walker (Southern US)
vb = VoiceBridge(voice="Samantha", eleven_voice_id="DLsHlh26Ugcm6ELvS0qi")

if vb.use_eleven:
    print("✅ ElevenLabs initialized.")
    print("🔊 Speaking test sentence...")
    vb.speak("Hello Joshua. This is Ms. Walker speaking with a Southern accent from the ElevenLabs server.", async_mode=False)
    print("Test complete.")
else:
    print("❌ ElevenLabs failed to initialize. Check your API key or HAS_ELEVEN status.")
