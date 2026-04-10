#!/bin/bash

# ECH0-PRIME Dashboard Auto-Restart Script
# This script keeps the dashboard server running continuously

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/dashboard_server.py"
LOG_FILE="$SCRIPT_DIR/dashboard_auto_restart.log"
PID_FILE="$SCRIPT_DIR/dashboard.pid"

# Configuration
MAX_RESTARTS=10          # Maximum restarts per hour
RESTART_WINDOW=3600      # 1 hour window
CHECK_INTERVAL=30        # Check every 30 seconds
STARTUP_GRACE=10         # Wait 10 seconds after starting before checking

# Initialize restart tracking
declare -a restart_times
restart_count=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cleanup() {
    log "Shutting down auto-restart script..."
    if [ -f "$PID_FILE" ]; then
        SERVER_PID=$(cat "$PID_FILE")
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Stopping dashboard server (PID: $SERVER_PID)"
            kill "$SERVER_PID"
            sleep 2
            if kill -0 "$SERVER_PID" 2>/dev/null; then
                kill -9 "$SERVER_PID" 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

check_restart_limit() {
    local current_time=$(date +%s)

    # Remove old restart times outside the window
    local new_times=()
    for time in "${restart_times[@]}"; do
        if [ $((current_time - time)) -lt $RESTART_WINDOW ]; then
            new_times+=("$time")
        fi
    done
    restart_times=("${new_times[@]}")

    # Check if we're over the limit
    if [ ${#restart_times[@]} -ge $MAX_RESTARTS ]; then
        log "ERROR: Too many restarts (${#restart_times[@]}) in $RESTART_WINDOW seconds. Stopping auto-restart."
        return 1
    fi

    return 0
}

start_server() {
    log "Starting dashboard server..."

    # Check restart limits
    if ! check_restart_limit; then
        return 1
    fi

    # Start the server in background
    cd "$SCRIPT_DIR"

    # Export environment variables
    export TOGETHER_API_KEY="tgp_v1_RRivmWsKWdYBs6sXl0m5MFTw1_fKB4j-DJyyyOE905g"
    export DASHBOARD_LLM_PROVIDER="together"

    python3 "$SERVER_SCRIPT" > /dev/null 2>&1 &
    local server_pid=$!

    # Save PID
    echo $server_pid > "$PID_FILE"

    # Record restart time
    restart_times+=("$(date +%s)")
    restart_count=$((restart_count + 1))

    log "Dashboard server started (PID: $server_pid, Restart: $restart_count)"

    # Wait for startup grace period
    sleep $STARTUP_GRACE

    return 0
}

check_server() {
    if [ ! -f "$PID_FILE" ]; then
        log "No PID file found"
        return 1
    fi

    local server_pid=$(cat "$PID_FILE")

    # Check if process is running
    if ! kill -0 "$server_pid" 2>/dev/null; then
        log "Dashboard server process $server_pid is not running"
        return 1
    fi

    # Check if port 8000 is responding
    if ! curl -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
        log "Dashboard server health check failed (port 8000)"
        return 1
    fi

    return 0
}

main() {
    log "Starting ECH0-PRIME Dashboard Auto-Restart Service"
    log "Script directory: $SCRIPT_DIR"
    log "Log file: $LOG_FILE"
    log "PID file: $PID_FILE"
    log "Check interval: ${CHECK_INTERVAL}s"
    log "Max restarts per hour: $MAX_RESTARTS"
    log "Press Ctrl+C to stop"

    # Initial start
    if ! start_server; then
        log "Failed to start server initially. Exiting."
        exit 1
    fi

    # Main monitoring loop
    while true; do
        if ! check_server; then
            log "Server is down. Attempting restart..."
            if [ -f "$PID_FILE" ]; then
                # Clean up old PID
                rm -f "$PID_FILE"
            fi

            if ! start_server; then
                log "Failed to restart server. Will try again in $CHECK_INTERVAL seconds..."
            fi
        fi

        sleep $CHECK_INTERVAL
    done
}

# Run main function
main
