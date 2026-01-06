import json
import os
import time
import numpy as np

def repair_metadata():
    npy_path = "memory_data/episodic.npy"
    metadata_path = "memory_data/episodic_metadata.json"
    
    if not os.path.exists(npy_path):
        print(f"❌ Episodic data not found: {npy_path}")
        return

    # Load episodic data to get count
    try:
        storage = np.load(npy_path, allow_pickle=True)
        count = len(storage)
        print(f"📊 Found {count} episodes in .npy file")
    except Exception as e:
        print(f"❌ Error loading .npy: {e}")
        return

    # Load existing metadata
    metadata = []
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"🔍 Found {len(metadata)} metadata entries")
        except Exception as e:
            print(f"⚠️ Error loading metadata: {e}")
            metadata = []

    # Adjust metadata length
    if len(metadata) > count:
        print(f"✂️ Truncating metadata from {len(metadata)} to {count}")
        metadata = metadata[:count]
    elif len(metadata) < count:
        print(f"➕ Padding metadata from {len(metadata)} to {count}")
        current_time = time.time()
        for i in range(len(metadata), count):
            metadata.append({
                "timestamp": current_time - (count - i) * 60, # 1 minute apart
                "importance": 0.5,
                "repaired": True
            })

    # Save repaired metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata repaired and saved to {metadata_path}")

if __name__ == "__main__":
    repair_metadata()

