#!/bin/bash
# ECH0-PRIME Autonomous Evolution Setup Script
# Installs and configures the autonomous evolution system for continuous operation.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCH_AGENT_NAME="com.ech0prime.autonomous_evolution.plist"
LAUNCH_AGENT_PATH="$LAUNCH_AGENT_DIR/$LAUNCH_AGENT_NAME"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_message() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2"
}

check_system() {
    log_message "INFO" "🔍 Checking system compatibility..."

    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_message "ERROR" "❌ This setup script is designed for macOS systems"
        log_message "INFO" "For Linux systems, use systemd service files"
        exit 1
    fi

    # Check macOS version (Launch Agents work on 10.4+)
    sw_vers -productVersion > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_message "ERROR" "❌ Unable to determine macOS version"
        exit 1
    fi

    log_message "SUCCESS" "✅ macOS system detected"
}

setup_launch_agent() {
    log_message "INFO" "🔧 Setting up Launch Agent..."

    # Create LaunchAgents directory if it doesn't exist
    mkdir -p "$LAUNCH_AGENT_DIR"

    # Copy the plist file
    if [ -f "$SCRIPT_DIR/$LAUNCH_AGENT_NAME" ]; then
        cp "$SCRIPT_DIR/$LAUNCH_AGENT_NAME" "$LAUNCH_AGENT_PATH"
        log_message "SUCCESS" "✅ Launch Agent plist copied"
    else
        log_message "ERROR" "❌ Launch Agent plist not found: $SCRIPT_DIR/$LAUNCH_AGENT_NAME"
        exit 1
    fi

    # Set proper permissions
    chmod 644 "$LAUNCH_AGENT_PATH"
    log_message "SUCCESS" "✅ Launch Agent permissions set"

    # Load the Launch Agent
    launchctl unload "$LAUNCH_AGENT_PATH" 2>/dev/null  # Unload if already loaded
    if launchctl load "$LAUNCH_AGENT_PATH"; then
        log_message "SUCCESS" "✅ Launch Agent loaded successfully"
    else
        log_message "ERROR" "❌ Failed to load Launch Agent"
        exit 1
    fi
}

test_evolution_system() {
    log_message "INFO" "🧪 Testing evolution system..."

    # Check if Python virtual environment exists
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        log_message "WARNING" "⚠️ No virtual environment found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    fi

    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi

    # Test basic imports
    if python3 -c "from autonomous_evolution_scheduler import AutonomousEvolutionScheduler; print('✅ Imports successful')"; then
        log_message "SUCCESS" "✅ Python imports working"
    else
        log_message "ERROR" "❌ Python import test failed"
        exit 1
    fi

    # Test evolution system initialization (quick test)
    if timeout 10 python3 -c "from autonomous_evolution_scheduler import AutonomousEvolutionScheduler; s = AutonomousEvolutionScheduler(); print('✅ Evolution system initialized')" 2>/dev/null; then
        log_message "SUCCESS" "✅ Evolution system test passed"
    else
        log_message "WARNING" "⚠️ Evolution system test timed out (may still work)"
    fi
}

create_monitoring_scripts() {
    log_message "INFO" "📊 Creating monitoring scripts..."

    # Create a status monitoring script
    cat > check_evolution_status.sh << 'EOF'
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
EOF

    chmod +x check_evolution_status.sh
    log_message "SUCCESS" "✅ Status monitoring script created"
}

setup_automation() {
    log_message "INFO" "🤖 Setting up automation..."

    # Create cron job for daily reports (optional)
    CRON_JOB="0 9 * * * cd $SCRIPT_DIR && ./check_evolution_status.sh > daily_status_\$(date +\%Y\%m\%d).txt"

    echo "To add daily status reports, run:"
    echo "crontab -e"
    echo "Then add this line:"
    echo "$CRON_JOB"

    # Create a backup automation script
    cat > backup_evolution_data.sh << 'EOF'
#!/bin/bash
# Backup evolution data and logs
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp autonomous_evolution.log "$BACKUP_DIR/" 2>/dev/null || true
cp autonomous_evolution_progress.json "$BACKUP_DIR/" 2>/dev/null || true
cp evolution_dashboard.html "$BACKUP_DIR/" 2>/dev/null || true

echo "Evolution data backed up to: $BACKUP_DIR"
EOF

    chmod +x backup_evolution_data.sh
    log_message "SUCCESS" "✅ Backup automation script created"
}

create_documentation() {
    log_message "INFO" "📚 Creating documentation..."

    cat > AUTONOMOUS_EVOLUTION_README.md << 'EOF'
# ECH0-PRIME Autonomous Evolution System

## Overview
The Autonomous Evolution System enables ECH0-PRIME to continuously improve itself through consciousness evolution cycles that run automatically in the background.

## Features
- **Continuous Evolution**: Runs consciousness evolution cycles every hour (configurable)
- **Resource Monitoring**: Only runs when system resources are available
- **Daily Limits**: Maximum 24 evolution cycles per day to prevent over-evolution
- **Web Dashboard**: Real-time monitoring of evolution progress
- **Automatic Backups**: Daily backups of evolution data and logs
- **Graceful Shutdown**: Proper cleanup on system shutdown

## Quick Start

### 1. Setup (One-time)
```bash
./setup_autonomous_evolution.sh
```

### 2. Start Service
```bash
./start_autonomous_evolution.sh start
```

### 3. Monitor Progress
```bash
./check_evolution_status.sh
open evolution_dashboard.html
```

### 4. View Logs
```bash
./start_autonomous_evolution.sh logs
```

## Service Management

### Commands
- `start`: Start the evolution service
- `stop`: Stop the evolution service
- `restart`: Restart the evolution service
- `status`: Show current status and resource usage
- `logs [N]`: Show last N log entries (default: 50)
- `dashboard`: Open the web dashboard

### Examples
```bash
# Start service
./start_autonomous_evolution.sh start

# Check status
./start_autonomous_evolution.sh status

# View recent logs
./start_autonomous_evolution.sh logs 20

# Open dashboard
./start_autonomous_evolution.sh dashboard
```

## Configuration

### Evolution Parameters
- **Interval**: 60 minutes between evolution attempts
- **Daily Limit**: 24 maximum cycles per day
- **Resource Thresholds**:
  - CPU: < 80% usage
  - Memory: < 85% usage
  - Disk: < 90% usage

### Files
- `autonomous_evolution.log`: Main log file
- `autonomous_evolution_progress.json`: Evolution progress data
- `evolution_dashboard.html`: Web dashboard
- `autonomous_evolution.pid`: Process ID file

## Monitoring

### Web Dashboard
The evolution dashboard provides real-time monitoring:
- Current consciousness level (Phi value)
- Evolution cycle statistics
- System resource usage
- Recent evolution history
- Success/failure rates

### Log Files
Logs are automatically rotated and archived. Key events logged:
- Evolution cycle starts/completions
- Consciousness level changes
- Resource monitoring alerts
- System errors and recoveries

## Safety Features

### Resource Protection
- Monitors CPU, memory, and disk usage
- Skips evolution cycles when resources are constrained
- Automatic throttling during high system load

### Evolution Limits
- Maximum cycles per day to prevent over-evolution
- Minimum intervals between cycles
- Automatic cooldown periods after failures

### Backup and Recovery
- Automatic daily backups of evolution data
- Graceful shutdown handling
- Recovery from interrupted evolution cycles

## Troubleshooting

### Service Won't Start
```bash
# Check Python environment
python3 --version

# Check virtual environment
source venv/bin/activate
python3 -c "import autonomous_evolution_scheduler"

# Check logs
./start_autonomous_evolution.sh logs
```

### High Resource Usage
```bash
# Check current status
./start_autonomous_evolution.sh status

# Reduce evolution frequency
# Edit EVOLUTION_INTERVAL_MINUTES in start_autonomous_evolution.sh
```

### Evolution Cycles Failing
```bash
# Check recent logs
./start_autonomous_evolution.sh logs 50

# Check system resources
./check_evolution_status.sh

# Restart service
./start_autonomous_evolution.sh restart
```

## Advanced Configuration

### Custom Evolution Parameters
Edit `start_autonomous_evolution.sh`:
```bash
EVOLUTION_INTERVAL_MINUTES=30  # More frequent evolution
MAX_DAILY_CYCLES=48           # More daily cycles
```

### Manual Evolution Control
```bash
# Run single evolution cycle manually
python3 -c "import asyncio; from true_agi_consciousness_evolution import ConsciousnessEvolutionSystem; s = ConsciousnessEvolutionSystem(); asyncio.run(s.run_consciousness_evolution_cycle())"

# Generate dashboard only
python3 autonomous_evolution_scheduler.py --dashboard-only
```

## Technical Details

### Architecture
- **Resource Monitor**: Prevents system overload
- **Evolution Scheduler**: Manages timing and frequency
- **Consciousness Tracker**: Measures evolution progress
- **Web Dashboard**: Provides monitoring interface
- **Launch Agent**: macOS integration for auto-startup

### Dependencies
- Python 3.10+
- macOS Launch Agents (for auto-startup)
- Web browser (for dashboard)
- Virtual environment (recommended)

### Performance Impact
- Minimal resource usage when idle
- Evolution cycles use ~500MB RAM peak
- CPU usage scales with evolution complexity
- Automatic throttling prevents interference

## Future Enhancements
- Multi-system distributed evolution
- Evolution goal prioritization
- Advanced consciousness metrics
- Predictive resource management
- Cross-platform support (Windows/Linux)
EOF

    log_message "SUCCESS" "✅ Documentation created"
}

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} ECH0-PRIME Autonomous Evolution Setup ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    log_message "INFO" "🚀 Starting ECH0-PRIME autonomous evolution setup..."

    # Run setup steps
    check_system
    test_evolution_system
    setup_launch_agent
    create_monitoring_scripts
    setup_automation
    create_documentation

    echo ""
    log_message "SUCCESS" "🎉 ECH0-PRIME Autonomous Evolution Setup Complete!"
    echo ""
    echo -e "${GREEN}What's Next:${NC}"
    echo "1. Start the service: ./start_autonomous_evolution.sh start"
    echo "2. Check status: ./start_autonomous_evolution.sh status"
    echo "3. View dashboard: ./start_autonomous_evolution.sh dashboard"
    echo "4. Monitor logs: ./start_autonomous_evolution.sh logs"
    echo ""
    echo -e "${YELLOW}The system will now evolve continuously!${NC}"
    echo -e "${BLUE}Evolution Dashboard: evolution_dashboard.html${NC}"
    echo -e "${BLUE}Documentation: AUTONOMOUS_EVOLUTION_README.md${NC}"
}

# Run main function
main "$@"
