import json
import os

state_file = "bbb_infinite_state.json"
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    existing_names = [c["name"] for c in state["companies"]]
    
    new_companies = [
        {"name": "CHATTERTECH", "status": "AUTOMATED", "cash_flow": "STABLE"},
        {"name": "FLOWSTATUS", "status": "AUTOMATED", "cash_flow": "STABLE"}
    ]
    
    for nc in new_companies:
        if nc["name"] not in existing_names:
            state["companies"].insert(1, nc) # Insert after 'Work of Art'
            print(f"✅ Added {nc['name']} to Infinite Engine.")
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)
    print("🚀 BBB Infinite State updated and synchronized.")
else:
    print("❌ State file not found.")

