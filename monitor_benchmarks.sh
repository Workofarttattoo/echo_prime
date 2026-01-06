#!/bin/bash
while true; do
    echo "=== $(date) ==="
    if [ -f logs/hle_100_final.pid ] && ps -p $(cat logs/hle_100_final.pid) > /dev/null 2>&1; then
        hle_progress=$(grep -o "Processing [0-9]\+/100" logs/hle_100_final.log | tail -n 1)
        echo "HLE: ${hle_progress:-initializing}"
    else
        echo "HLE: completed or stopped"
    fi
    
    if [ -f logs/ai_suite_200.pid ] && ps -p $(cat logs/ai_suite_200.pid) > /dev/null 2>&1; then
        ai_progress=$(tail -n 5 logs/ai_suite_200_samples.log | grep -o "Running.*" | tail -n 1)
        echo "AI Suite: ${ai_progress:-initializing}"
    else
        echo "AI Suite: completed or stopped"
    fi
    echo ""
    sleep 300  # Check every 5 minutes
done
