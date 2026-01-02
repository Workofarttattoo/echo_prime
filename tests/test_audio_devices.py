import speech_recognition as sr
import pyaudio

print("Searching for working audio devices...")
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    dev_info = p.get_device_info_by_host_api_device_index(0, i)
    if dev_info.get('maxInputChannels') > 0:
        print(f"Index {i}: {dev_info.get('name')}")
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, input_device_index=i, frames_per_buffer=1024)
            print(f"  ✅ SUCCESSFULLY OPENED index {i}")
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"  ❌ FAILED to open index {i}: {e}")

p.terminate()
