import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

try:
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    response = client.voices.get_all()
    
    print(f"Found {len(response.voices)} voices.")
    for voice in response.voices:
        # Check labels for accent/description if available, or just print name/id
        labels = voice.labels if voice.labels else {}
        print(f"Name: {voice.name} | ID: {voice.voice_id} | Labels: {labels}")
        
except Exception as e:
    print(f"Error: {e}")
