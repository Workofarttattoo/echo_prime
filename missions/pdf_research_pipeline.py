import os
import time
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_orchestrator import EchoPrimeAGI

WATCH_DIR = "research_drop"
PROCESSED_LOG = "research_drop/processed.log"

import threading

def run_research_pipeline():
    print(f"🚀 MISSION 1 INITIALIZED: PDF Research Engine watching '{WATCH_DIR}'")
    agi = EchoPrimeAGI(enable_voice=False)
    
    
    # Ensure log exists
    if not os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "w") as f:
            f.write("--- Processed PDFs Log ---\n")

    try:
        while True:
            # 1. Look for new PDFs
            files = [f for f in os.listdir(WATCH_DIR) if f.lower().endswith(".pdf")]
            
            with open(PROCESSED_LOG, "r") as f:
                processed = f.read()

            new_files = [f for f in files if f not in processed]

            if new_files:
                for filename in new_files:
                    path = os.path.abspath(os.path.join(WATCH_DIR, filename))
                    print(f"\n[📥] NEW RESEARCH DETECTED: {filename}")
                    
                    # Command ECH0 to ingest it via tool
                    intent = f"ACTION: {{'tool': 'ingest_pdf', 'args': {{'file_path': '{path}'}}}}"
                    outcome = agi.cognitive_cycle(np.random.randn(1000000), intent)
                    
                    log_entry = f"[{time.ctime()}] Ingested: {filename} -> {outcome.get('llm_insight')[:100]}...\n"
                    print(f"[✅] {log_entry}")
                    
                    # Mark as processed
                    with open(PROCESSED_LOG, "a") as f:
                        f.write(f"{filename}\n")
                    
                    # Brief commentary from ECH0
                    follow_up = f"I have finished ingesting {filename}. Shall I perform a deep analysis of its core premises?"
                    agi.cognitive_cycle(np.random.randn(1000000), follow_up)

            time.sleep(10)
    except KeyboardInterrupt:
        print("\nResearch pipeline deactivated.")

if __name__ == "__main__":
    run_research_pipeline()
