import os
import sys
import time
import subprocess
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import requests
from main_orchestrator import EchoPrimeAGI

def update_dashboard_mission(agi, insight, action="Scanning Drive..."):
    """Pushes mission progress to the live dashboard via HTTP."""
    try:
        data = {
            "engine": {
                "free_energy": 0.0,
                "surprise": "System prioritized for Mission Execution.",
                "levels": ["Sensory", "Internal", "Prefrontal", "Meta"],
                "mission_goal": "Step 1: Deep Wisdom Ingestion (External Drive)",
                "mission_complete": False,
                "voice_enabled": True,
                "phi": 10.0
            },
            "reasoning": {
                "insight": insight,
                "actions": [action]
            },
            "timestamp": time.time()
        }
        requests.post("http://127.0.0.1:8000/update_state", json=data, timeout=1)
    except:
        pass

def scan_for_external_wisdom():
    """Scans /Volumes for any new drives and crawls them for wisdom (PDF/JSON)."""
    print("[🔍] Scanning for External Wisdom (Drives)...")
    base_volumes = set(['Macintosh HD', '.timemachine', 'com.apple.TimeMachine.localsnapshots'])

    check_count = 0
    while check_count < 12:  # Limit to 12 checks (1 minute)
        try:
            current_volumes = set(os.listdir("/Volumes"))
            new_volumes = current_volumes - base_volumes

            print(f"[🔍] Check {check_count + 1}: Found {len(current_volumes)} volumes, {len(new_volumes)} new")

            if new_volumes:
                for vol in new_volumes:
                    vol_path = os.path.join("/Volumes", vol)
                    print(f"[📂] NEW DRIVE DETECTED: {vol} at {vol_path}")
                    crawl_and_ingest(vol_path)
                break # Found at least one
            else:
                print("[⏳] Waiting for external drive to be mounted...")
                time.sleep(5)
                check_count += 1
        except Exception as e:
            print(f"SCAN ERROR: {e}")
            time.sleep(5)
            check_count += 1

    if check_count >= 12:
        print("[❌] Timeout: No external drives detected within 1 minute")
        print("[💡] Make sure your external drive is mounted and contains wisdom files")

def crawl_and_ingest(path):
    """Recursively finds PDFs and JSONs and moves/copies them to research_drop or processes them."""
    print(f"[🕷️] Crawling {path} for PDFs and JSONs...")
    agi = EchoPrimeAGI(enable_voice=False)
    
    wisdom_found = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.pdf', '.json')):
                file_path = os.path.join(root, file)
                if file.lower().endswith('.pdf'):
                    # Command ECH0 to ingest it
                    print(f"[💎] Found Wisdom: {file}")
                    update_dashboard_mission(agi, f"Ingesting PDF Wisdom: {file}", action=f"Reading {file}...")
                    intent = f"ACTION: {{'tool': 'ingest_pdf', 'args': {{'file_path': '{file_path}'}}}}"
                    agi.cognitive_cycle(np.random.randn(1000000), intent)
                else:
                    # JSON: Just store a note about its existence or summarize
                    update_dashboard_mission(agi, f"Detected JSON Data: {file}", action=f"Parsing metadata: {file}")
                    agi.cognitive_cycle(np.random.randn(1000000), f"Note: Found a technical JSON data file at {file_path}. Ingesting metadata to memory.")
                
                wisdom_found += 1
                if wisdom_found > 50: # Cap for safety in first run
                    print("[⚠️] Wisdom cap reached for this session.")
                    return
    
    print(f"[✅] Finished scanning external drive. Found {wisdom_found} items.")

def run_all_missions():
    # 1. External Scan & Mission 1 (PDF Ingestion)
    print("\n--- STEP 1: EXTERNAL DRIVE WISDOM SCAN ---")
    scan_for_external_wisdom()
    
    # 2. Mission 2: Autonomous Invention
    print("\n--- STEP 2: MISSION 2 - AUTONOMOUS INVENTION ---")
    try:
        subprocess.run([sys.executable, "missions/autonomous_invention.py"], check=True)
    except Exception as e:
        print(f"Mission 2 Error: {e}")
        
    # 3. Mission 3: ARC-AGI Reflection
    print("\n--- STEP 3: MISSION 3 - ARC-AGI REFLECTION ---")
    try:
        subprocess.run([sys.executable, "missions/arc_reflection.py"], check=True)
    except Exception as e:
        print(f"Mission 3 Error: {e}")

    print("\n[🏁] ALL MISSIONS COMPLETE. THE WISDOM HAS BEEN ARCHIVED.")

if __name__ == "__main__":
    run_all_missions()
