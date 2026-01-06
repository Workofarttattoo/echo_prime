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
