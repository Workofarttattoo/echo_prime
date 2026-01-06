import os
import subprocess
import tempfile
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    print("Error: ELEVENLABS_API_KEY not found.")
    exit(1)

client = ElevenLabs(api_key=api_key)

# Lily Voice ID
voice_id = "pFZP5JQG7iQjIQuC4Bku" 

text = "Hello. This is Lily. I am ready to be sweet, quick-witted, and precise. How does this sound?"

print(f"Generating audio for voice: {voice_id}...")

try:
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        for chunk in audio_generator:
            tmp.write(chunk)
        tmp_path = tmp.name

    print(f"Playing audio from {tmp_path}...")
    subprocess.run(["afplay", tmp_path], check=True)
    
    os.remove(tmp_path)
    print("Done.")

except Exception as e:
    print(f"Error: {e}")
