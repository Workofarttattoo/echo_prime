import subprocess
import threading
import os
import queue


try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.play import play as play_audio
    HAS_ELEVEN = True
except ImportError:
    HAS_ELEVEN = False

class VoiceBridge:
    """
    Provides vocal synthesis for ECH0-PRIME.
    Supports:
    1. ElevenLabs (High fidelity, requires API Key)
    2. macOS Native 'say' (Fallback)
    3. Console Output (Fallback)
    """
    def __init__(self, voice: str = "Samantha", eleven_voice_id: str = "Rachel"):
        self.voice = voice # Default
        self.msg_queue = queue.Queue()
        self.is_currently_speaking = False
        
        # Determine provider
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        print(f"DEBUG: HAS_ELEVEN={HAS_ELEVEN}, API_KEY_FOUND={bool(self.api_key)}")
        self.use_eleven = HAS_ELEVEN and self.api_key
        
        if self.use_eleven:
            print("VOICE: Initializing ElevenLabs Bridge (v1.0+)...")
            try:
                self.eleven_client = ElevenLabs(api_key=self.api_key)
                self.eleven_voice_id = eleven_voice_id
                threading.Thread(target=self._worker, daemon=True).start()
            except Exception as e:
                print(f"VOICE ERROR: ElevenLabs Init Failed: {e}")
                self.use_eleven = False
                self._fallback_init()
        else:
            self._fallback_init()

    def _fallback_init(self):
        # macOS Fallback
        self.voice = "Samantha" # Default back to string
        self.is_mac = os.uname().sysname == 'Darwin'
        if self.is_mac:
            print(f"VOICE: Using Mac voice '{self.voice}'")
            threading.Thread(target=self._worker, daemon=True).start()
        else:
            print("VOICE: Not running on macOS. Voice output will be simulated in console.")

    def _worker(self):
        """Sequential worker loop."""
        while True:
            text = self.msg_queue.get()
            if text:
                self.is_currently_speaking = True
                if self.use_eleven:
                    self._run_eleven(text)
                elif self.is_mac:
                    self._run_say(text)
                else:
                    self._run_print(text)
                self.is_currently_speaking = False
            self.msg_queue.task_done()

    def silence(self):
        """Immediately stops speaking and clears the queue."""
        self.is_currently_speaking = False
        with self.msg_queue.mutex:
            self.msg_queue.queue.clear()
        # TODO: Implement process killing for 'say' or 'mpg123' if strictly needed
        print("VOICE: Silenced.")

    def speak(self, text: str, async_mode: bool = True):
        """Narrates the provided text."""
        if not text:
            return

        clean_text = text.replace('"', '').replace("'", "").replace("\n", " ").strip()
        if not clean_text:
            return
            
        if async_mode:
            self.msg_queue.put(clean_text)
        else:
            # Synchronous override
            if self.use_eleven:
                self._run_eleven(clean_text)
            elif hasattr(self, 'is_mac') and self.is_mac:
                self._run_say(clean_text)
            else:
                self._run_print(clean_text)

    def _run_eleven(self, text: str):
        """Generates and plays audio via ElevenLabs."""
        try:
            audio_generator = self.eleven_client.text_to_speech.convert(
                text=text,
                voice_id=self.eleven_voice_id,
                model_id="eleven_multilingual_v2"
            )
            
            # Save stream to temporary file and use afplay (macOS native)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                for chunk in audio_generator:
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            if os.uname().sysname == 'Darwin':
                subprocess.run(["afplay", tmp_path], check=True, timeout=10)
            else:
                # Fallback for non-mac if play_audio fails (still likely to fail without ffplay but worth a try)
                # Or just print it.
                play_audio(audio_generator)
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        except Exception as e:
            print(f"ELEVENLABS/AFPLAY ERROR: {e}")
            # Silencing fallback to 'say' as per user request to use "just the good one"
            # self._run_say(text)
            self._run_print(text)

    def _run_print(self, text: str):
        print(f" [AGI VOICE]: {text}")

    def _run_say(self, text: str):
        """Internal runner for the 'say' command."""
        try:
            # Shorten very long texts for voice efficiency if using local tts
            spoken_text = text[:500]
            subprocess.run(["say", "-v", self.voice, spoken_text], check=True, timeout=15)
        except Exception as e:
            print(f"VOICE ERROR: Failed to execute 'say' command: {e}")

if __name__ == "__main__":
    # Test
    vb = VoiceBridge()
    vb.speak("Hello. I am Echo Prime. My cognitive systems are online.", async_mode=False)
