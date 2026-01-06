#!/bin/bash
echo "=== ECH0-PRIME Autonomous Evolution Status ==="
echo "Timestamp: $(date)"
echo ""

# Check if service is running
if [ -f "autonomous_evolution.pid" ]; then
    PID=$(cat autonomous_evolution.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Service Status: RUNNING (PID: $PID)"
    else
        echo "❌ Service Status: STOPPED (stale PID file)"
        rm -f autonomous_evolution.pid
    fi
else
    echo "❌ Service Status: NOT RUNNING"
fi

echo ""
echo "=== Recent Evolution Activity ==="
if [ -f "autonomous_evolution.log" ]; then
    tail -10 autonomous_evolution.log
else
    echo "No log file found"
fi

echo ""
echo "=== System Resources ==="
echo "CPU Usage: $(ps -A -o %cpu | awk '{s+=$1} END {print s "%"}')"
echo "Memory Usage: $(ps -A -o %mem | awk '{s+=$1} END {print s "%"}')"
echo "Disk Usage: $(df -h . | tail -1 | awk '{print $5}')"

echo ""
echo "=== Evolution Dashboard ==="
if [ -f "evolution_dashboard.html" ]; then
    echo "📊 Dashboard available: evolution_dashboard.html"
    echo "🌐 Open with: open evolution_dashboard.html"
else
    echo "📊 Dashboard not generated yet"
fi
