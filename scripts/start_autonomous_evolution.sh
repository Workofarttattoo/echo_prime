#!/bin/bash
# Kairos Autonomous Evolution Launcher
# Purpose: Start the continuous improvement agent with Crystalline Intent

echo "🚀 Starting Kairos Autonomous Evolution Agent..."
echo "💎 Loading Crystalline Intent..."

# Ensure environment is set
export PYTHONPATH=$PYTHONPATH:$(pwd)
export KAIROS_MODE="autonomous"

# Run the Python agent
# We run in a loop to ensure persistence
while true; do
    echo "⚡ activating agent cycle..."
    python3 autonomous_evolution_agent.py
    
    echo "💤 Resting for mission cycle (60s)..."
    sleep 60
done
