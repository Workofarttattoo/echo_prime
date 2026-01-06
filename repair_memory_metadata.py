import json
import os
import time

def repair_metadata():
    metadata_path = "memory_data/episodic_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"🔍 Analyzing {len(metadata)} metadata entries...")
    repaired_count = 0
    current_time = time.time()

    for i, entry in enumerate(metadata):
        if not entry or 'timestamp' not in entry:
            # Backfill with current time minus a small decrement to keep some order
            entry['timestamp'] = current_time - (len(metadata) - i) * 0.1
            if 'importance' not in entry:
                entry['importance'] = 0.5
            repaired_count += 1

    if repaired_count > 0:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"✅ Repaired {repaired_count} entries.")
    else:
        print("✨ No repair needed.")

if __name__ == "__main__":
    repair_metadata()

