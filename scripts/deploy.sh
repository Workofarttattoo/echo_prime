#!/usr/bin/env bash
# ===========================================================================
# ECH0-PRIME — Production Deployment Script
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
# All Rights Reserved. PATENT PENDING.
#
# Usage:
#   ./scripts/deploy.sh              # Deploy with defaults
#   ./scripts/deploy.sh --build      # Force rebuild images before deploy
#   ./scripts/deploy.sh --dry-run    # Show what would be done
# ===========================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE="docker compose"
ENV_FILE="$PROJECT_ROOT/.env"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

FORCE_BUILD=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --build)    FORCE_BUILD=true ;;
        --dry-run)  DRY_RUN=true ;;
        *)          echo -e "${RED}Unknown argument: $arg${NC}"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }

run_or_echo() {
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        eval "$@"
    fi
}

# ── Pre-flight checks ────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   ECH0-PRIME — Production Deployment        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# 1. Docker available?
if ! command -v docker &>/dev/null; then
    err "Docker is not installed or not in PATH."
    exit 1
fi
ok "Docker found: $(docker --version | head -1)"

# 2. .env exists?
if [ ! -f "$ENV_FILE" ]; then
    warn ".env file not found — copying from .env.example"
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        run_or_echo "cp $PROJECT_ROOT/.env.example $ENV_FILE"
        warn "Please edit .env and fill in required values, then re-run."
        exit 1
    else
        err "Neither .env nor .env.example found."
        exit 1
    fi
fi
ok ".env file present"

# 3. Critical env vars set?
source "$ENV_FILE" 2>/dev/null || true
MISSING_VARS=()
for var in ECH0_PHASE; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS+=("$var")
    fi
done
if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    warn "Optional env vars not set: ${MISSING_VARS[*]}"
fi

# ── Build ─────────────────────────────────────────────────────────────────
info "Step 1/4 — Building images …"
if $FORCE_BUILD; then
    run_or_echo "$COMPOSE build --no-cache"
else
    run_or_echo "$COMPOSE build"
fi
ok "Images built"

# ── Stop old containers ───────────────────────────────────────────────────
info "Step 2/4 — Stopping existing services …"
run_or_echo "$COMPOSE down --remove-orphans"
ok "Old services stopped"

# ── Start ─────────────────────────────────────────────────────────────────
info "Step 3/4 — Starting services …"
run_or_echo "$COMPOSE up -d"
ok "Services started"

# ── Verify ────────────────────────────────────────────────────────────────
info "Step 4/4 — Waiting for health checks …"
if ! $DRY_RUN; then
    sleep 5

    DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
    if curl -fs "http://localhost:${DASHBOARD_PORT}/health" >/dev/null 2>&1; then
        ok "Dashboard API is healthy (port $DASHBOARD_PORT)"
    else
        warn "Dashboard API not yet responding — may still be starting."
        warn "Check logs:  docker compose logs -f dashboard"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅  Deployment Complete                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Dashboard:    http://localhost:${DASHBOARD_PORT:-8000}"
echo -e "  Gradio/HF:    http://localhost:${GRADIO_PORT:-7860}"
echo -e "  Redis:        localhost:${REDIS_PORT:-6379}"
echo ""
echo -e "  Logs:         ${BLUE}docker compose logs -f${NC}"
echo -e "  Stop:         ${BLUE}docker compose down${NC}"
echo ""
