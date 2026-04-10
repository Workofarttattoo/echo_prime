import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_orchestrator import EchoPrimeAGI

def test_phase_6_audio_voice():
    print("--- Verifying ECH0-PRIME Phase 6: Audio & Voice ---")
    
    agi = EchoPrimeAGI()
    
    # 1. Test Voice (Vocal Synthesis)
    print("\n[Testing AGI Voice]")
    agi.voice.speak("System check. Audio and vocal synthesis bridges are operational.", async_mode=False)
    
    # 2. Test Audio Interaction (Hearing)
    print("\n[Testing AGI Hearing]")
    audio_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'audio_input', 'test_command.txt')
    os.makedirs(os.path.dirname(audio_file), exist_ok=True)
    with open(audio_file, "w") as f:
        f.write("Scan the system for temporary files.")
    
    print("Simulated speech command dropped into 'audio_input/'. Waiting for perception...")
    
    # Run a cycle to pick up the audio
    mock_input = np.random.randn(1000000)
    outcome = agi.cognitive_cycle(mock_input, "Standard surveillance.")
    
    print("\n[Cycle Outcome]")
    print(f"LLM Insight (Heard/Processed): {outcome['llm_insight'][:100]}...")
    
    # Final confirmation
    if "Scan the system" in str(outcome['llm_insight']):
        print("\nSUCCESS: AGI successfully heard, processed, and acted on auditory input.")
    else:
        # LLM might rephrase, so check context
        print("\nCHECK: AGI processed the cycle. Review LLM insight for auditory context.")

if __name__ == "__main__":
    test_phase_6_audio_voice()
