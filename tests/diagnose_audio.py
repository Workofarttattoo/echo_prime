import speech_recognition as sr
import time

def diagnose_audio():
    print("--- AUDIO DIAGNOSTIC ---")
    r = sr.Recognizer()
    
    print("Microphones found:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f" {i}: {name}")
    
    try:
        with sr.Microphone(device_index=2) as source:
            print("\nListening for ambient noise for 2 seconds (STAY QUIET)...")
            r.adjust_for_ambient_noise(source, duration=2)
            print(f"Adjusted energy threshold: {r.energy_threshold}")
            
            print("\nListening for speech for 5 seconds (SAY SOMETHING)...")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
            print("Processing...")
            text = r.recognize_google(audio)
            print(f"SUCCESS! Recognized: '{text}'")
            
    except sr.WaitTimeoutError:
        print("ERROR: Timeout waiting for speech.")
    except sr.UnknownValueError:
        print("ERROR: Speech was detected but could not be understood.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    diagnose_audio()
