#!/usr/bin/env python3
"""
ECH0-PRIME Autonomous Work Monitor
Shows real-time status of AGI mission progress
"""

import requests
import json
import time
from datetime import datetime

DASHBOARD_URL = "http://localhost:8000"

def get_mission_status():
    try:
        response = requests.get(f"{DASHBOARD_URL}/api/missions", timeout=5)
        return response.json()
    except:
        return {"error": "Dashboard not running"}

def get_evolution_units():
    try:
        response = requests.get(f"{DASHBOARD_URL}/api/evolution-units", timeout=5)
        return response.json()
    except:
        return {"evolution_units": 0}

def get_consciousness():
    try:
        response = requests.get(f"{DASHBOARD_URL}/api/consciousness", timeout=5)
        return response.json()
    except:
        return {"phi": 0.0, "level": "UNKNOWN"}

def print_status():
    print(f"\n🤖 ECH0-PRIME STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Mission Status
    missions = get_mission_status()
    if "error" not in missions:
        print(f"📋 Missions: {missions.get('total_missions', 0)} total, {missions.get('active_missions', 0)} active")
        if missions.get('top_priorities'):
            print(f"🎯 Top Priority: {missions['top_priorities'][0]['description'][:50]}...")
    else:
        print(f"📋 Missions: {missions['error']}")
    
    # Evolution Units
    evolution = get_evolution_units()
    if "evolution_units" in evolution:
        print(f"🧬 Evolution Units: {evolution['evolution_units']}")
    
    # Consciousness
    consciousness = get_consciousness()
    if "phi" in consciousness:
        print(f"🧠 Consciousness Φ: {consciousness['phi']:.4f} ({consciousness.get('level', 'UNKNOWN')})")
    
    print()

if __name__ == "__main__":
    print("Starting ECH0-PRIME monitor... (Ctrl+C to stop)")
    try:
        while True:
            print_status()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
