#!/bin/bash
"""
ECH0-PRIME Autonomous Evolution Service Launcher
Starts the autonomous evolution scheduler as a background service.
"""

# Configuration
EVOLUTION_INTERVAL_MINUTES=60  # Run evolution every hour
MAX_DAILY_CYCLES=24            # Maximum 24 cycles per day
LOG_FILE="autonomous_evolution.log"
PID_FILE="autonomous_evolution.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message" | tee -a "$LOG_FILE"
}

# Function to check if process is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to start the service
start_service() {
    log_message "INFO" "🚀 Starting ECH0-PRIME Autonomous Evolution Service..."

    if is_running; then
        log_message "WARNING" "Service is already running (PID: $(cat $PID_FILE))"
        exit 1
    fi

    # Activate virtual environment if it exists
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        log_message "INFO" "✅ Virtual environment activated"
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        log_message "INFO" "✅ Virtual environment activated"
    fi

    # Check Python availability
    if ! command -v python3 &> /dev/null; then
        log_message "ERROR" "Python3 not found"
        exit 1
    fi

    # Start the evolution scheduler in background
    nohup python3 autonomous_evolution_scheduler.py \
        --interval "$EVOLUTION_INTERVAL_MINUTES" \
        --max-daily "$MAX_DAILY_CYCLES" \
        > autonomous_evolution.out 2>&1 &

    local pid=$!
    echo $pid > "$PID_FILE"

    # Wait a moment and check if process started successfully
    sleep 3
    if is_running; then
        log_message "SUCCESS" "✅ Autonomous evolution service started (PID: $pid)"
        log_message "INFO" "📊 Dashboard: evolution_dashboard.html"
        log_message "INFO" "📝 Logs: $LOG_FILE"
        log_message "INFO" "🔄 Evolution interval: $EVOLUTION_INTERVAL_MINUTES minutes"
        log_message "INFO" "📅 Max daily cycles: $MAX_DAILY_CYCLES"
    else
        log_message "ERROR" "❌ Failed to start autonomous evolution service"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# Function to stop the service
stop_service() {
    log_message "INFO" "🛑 Stopping ECH0-PRIME Autonomous Evolution Service..."

    if ! is_running; then
        log_message "WARNING" "Service is not running"
        return
    fi

    local pid=$(cat "$PID_FILE")
    log_message "INFO" "Stopping process $pid..."

    # Try graceful shutdown first
    kill -TERM "$pid" 2>/dev/null

    # Wait for process to stop
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if is_running; then
        log_message "WARNING" "Process didn't stop gracefully, forcing termination..."
        kill -KILL "$pid" 2>/dev/null
        sleep 1
    fi

    if ! is_running; then
        rm -f "$PID_FILE"
        log_message "SUCCESS" "✅ Autonomous evolution service stopped"
    else
        log_message "ERROR" "❌ Failed to stop autonomous evolution service"
    fi
}

# Function to check service status
status_service() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log_message "INFO" "✅ Service is running (PID: $pid)"

        # Show recent log entries
        if [ -f "$LOG_FILE" ]; then
            echo -e "\n${BLUE}Recent log entries:${NC}"
            tail -10 "$LOG_FILE"
        fi

        # Show resource usage if possible
        if command -v ps &> /dev/null; then
            echo -e "\n${BLUE}Resource usage:${NC}"
            ps -p "$pid" -o pid,ppid,cmd,%cpu,%mem,etime
        fi

    else
        log_message "INFO" "❌ Service is not running"
        if [ -f "$PID_FILE" ]; then
            rm -f "$PID_FILE"
        fi
    fi
}

# Function to restart the service
restart_service() {
    log_message "INFO" "🔄 Restarting ECH0-PRIME Autonomous Evolution Service..."
    stop_service
    sleep 2
    start_service
}

# Function to show logs
show_logs() {
    local lines=${1:-50}
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Last $lines lines of evolution log:${NC}"
        tail -"$lines" "$LOG_FILE"
    else
        echo -e "${RED}Log file not found: $LOG_FILE${NC}"
    fi
}

# Function to show dashboard
show_dashboard() {
    if [ -f "evolution_dashboard.html" ]; then
        if command -v open &> /dev/null; then
            open evolution_dashboard.html
        elif command -v xdg-open &> /dev/null; then
            xdg-open evolution_dashboard.html
        else
            echo -e "${YELLOW}Dashboard saved at: evolution_dashboard.html${NC}"
        fi
    else
        echo -e "${RED}Dashboard not found. Run the service first.${NC}"
    fi
}

# Main script logic
case "${1:-help}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        show_logs "$2"
        ;;
    dashboard)
        show_dashboard
        ;;
    help|--help|-h)
        echo -e "${BLUE}ECH0-PRIME Autonomous Evolution Service${NC}"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|dashboard|help}"
        echo ""
        echo "Commands:"
        echo "  start     Start the autonomous evolution service"
        echo "  stop      Stop the autonomous evolution service"
        echo "  restart   Restart the autonomous evolution service"
        echo "  status    Show service status and resource usage"
        echo "  logs      Show recent log entries (optional: number of lines)"
        echo "  dashboard Open the evolution dashboard in browser"
        echo "  help      Show this help message"
        echo ""
        echo "Configuration:"
        echo "  Evolution Interval: $EVOLUTION_INTERVAL_MINUTES minutes"
        echo "  Max Daily Cycles: $MAX_DAILY_CYCLES"
        echo "  Log File: $LOG_FILE"
        echo "  PID File: $PID_FILE"
        ;;
    *)
        echo -e "${RED}Invalid command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
