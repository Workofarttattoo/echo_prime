import os
import subprocess
import tempfile
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    print("Error: ELEVENLABS_API_KEY not found.")
    exit(1)

client = ElevenLabs(api_key=api_key)

voices_to_test = [
    {
        "name": "Lily",
        "id": "pFZP5JQG7iQjIQuC4Bku",
        "text": "Hello, I am Lily. I have a British accent, which might not be Southern Texas, but I am quite velvety and confident."
    },
    {
        "name": "Jessica",
        "id": "cgSgspJ2msm6clMCkdW9",
        "text": "Hi there! I'm Jessica. I'm playful, bright, and warm. I'm an American voice."
    },
    {
        "name": "Ms. Walker (Southern Option)",
        "id": "DLsHlh26Ugcm6ELvS0qi",
        "text": "Well hello there. I'm Ms. Walker. I noticed you were lookin' for a Southern voice. I might be a bit older than 28, but I've got that Southern charm."
    }
]

for voice in voices_to_test:
    print(f"\n--- Generating audio for {voice['name']} ({voice['id']}) ---")
    try:
        audio_generator = client.text_to_speech.convert(
            text=voice["text"],
            voice_id=voice["id"],
            model_id="eleven_multilingual_v2"
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            for chunk in audio_generator:
                tmp.write(chunk)
            tmp_path = tmp.name

        print(f"Playing audio...")
        subprocess.run(["afplay", tmp_path], check=True)
        
        os.remove(tmp_path)
        time.sleep(1) 

    except Exception as e:
        print(f"Error with {voice['name']}: {e}")
