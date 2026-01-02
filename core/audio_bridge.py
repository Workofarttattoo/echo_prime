import os
import time
import threading
import queue
import speech_recognition as sr
import numpy as np

class AudioBridge:
    """
    Multimodal audio perception system for ECH0-PRIME.
    Uses background listening for low latency and high reliability.
    """
    def __init__(self, watch_dir: str = "audio_input"):
        self.watch_dir = watch_dir
        if not os.path.exists(watch_dir):
            os.makedirs(watch_dir)
        
        self.transcription_queue = queue.Queue()
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.is_listening = True
        self.is_talking = False # Controlled by main_orchestrator
        
        # Hardware Microphone is DISABLED by default to prevent macOS lockups.
        # Use the Dashboard 'TALK' button for high-fidelity voice input.
        self.microphone = None
        self.stop_listening = lambda wait_for_stop=False: None
        print("AUDIO: Terminal Mic Listeners DISABLED (Using Dashboard Bridge).")


        # Fallback directory watcher
        threading.Thread(target=self._directory_watcher, daemon=True).start()

    def _on_speech(self, recognizer, audio):
        """Callback for background listener."""
        if not self.is_listening or self.is_talking:
            return
            
        try:
            # Quick check for zero-signal (Permission check)
            raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            if np.abs(raw_data).max() == 0:
                print(" [⚠️ SENSORY WARNING]: Zero signal detected. Possible macOS Permission blockage.")
                return

            text = recognizer.recognize_google(audio)
            if text:
                print(f" [👂 MIC]: {text}")
                self.transcription_queue.put(text)
        except sr.UnknownValueError:
            pass
        except Exception as e:
            # print(f"SPEECH ERROR: {e}")
            pass

    def pause_perception(self): self.is_listening = False
    def resume_perception(self): self.is_listening = True
    def set_talking(self, talking: bool): self.is_talking = talking

    def _directory_watcher(self):
        """Polls the directory for manual text-based input."""
        while True:
            files = [f for f in os.listdir(self.watch_dir) if f.endswith(".txt")]
            if files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(self.watch_dir, x)))
                file_path = os.path.join(self.watch_dir, files[0])
                try:
                    with open(file_path, "r") as f:
                        text = f.read().strip()
                    os.remove(file_path)
                    if text:
                        print(f" [📄 HUB]: {text}")
                        self.transcription_queue.put(text)
                except Exception as e:
                    print(f"AUDIO ERROR: {e}")
            time.sleep(1)

    def get_latest_transcription(self) -> str:
        """Retrieves the next item from the combined audio stream."""
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return ""

if __name__ == "__main__":
    ab = AudioBridge()
    print("Testing Audio Bridge (Speak now or drop a .txt file)...")
    while True:
        text = ab.get_latest_transcription()
        if text:
            print(f"PERCEIVED: {text}")
        time.sleep(0.5)
