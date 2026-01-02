import pyaudio
import wave

def record_test():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "test_mic.wav"

    p = pyaudio.PyAudio()

    print(f"Opening stream on default device...")
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Saved to {WAVE_OUTPUT_FILENAME}")
        
        # Check if file is silent (mostly zeros)
        import numpy as np
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        max_val = np.abs(audio_data).max()
        print(f"Max amplitude detected: {max_val}")
        if max_val < 100:
            print("WARNING: Audio signal is extremely weak or silent. Check permissions or mute status.")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    record_test()
