#!/usr/bin/env bash
set -euo pipefail

# CourtListener ingest wrapper for The Gavl tooling.
# Requires COURT_LISTENER_API_KEY, SUPABASE_URL, and SUPABASE_SERVICE_ROLE_KEY.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NODE_SCRIPT="${ROOT_DIR}/scripts/courtlistener-ingest.js"

if [[ ! -f "${NODE_SCRIPT}" ]]; then
  echo "[error] Unable to locate courtlistener-ingest.js at ${NODE_SCRIPT}" >&2
  exit 1
fi

function require_env() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "[error] Environment variable ${name} is required." >&2
    exit 1
  fi
}

require_env "COURT_LISTENER_API_KEY"
require_env "SUPABASE_URL"
require_env "SUPABASE_SERVICE_ROLE_KEY"

ENDPOINT="${ENDPOINT:-recap-documents}"
PARAMS="${PARAMS:-court=nvb&date_filed_min=2024-01-01}"
MAX_PAGES="${MAX_PAGES:-2}"
OUTPUT="${OUTPUT:-}"

LOG_DIR="${ROOT_DIR}/logs/courtlistener"
mkdir -p "${LOG_DIR}"

if [[ -z "${OUTPUT}" ]]; then
  TIMESTAMP="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
  OUTPUT="${LOG_DIR}/${ENDPOINT}-${TIMESTAMP}.json"
fi

NODE_ARGS=(
  --endpoint "${ENDPOINT}"
  --params "${PARAMS}"
  --max-pages "${MAX_PAGES}"
  --output "${OUTPUT}"
)

if [[ -n "${SKIP_SUPABASE:-}" ]]; then
  NODE_ARGS+=(--skip-supabase)
fi

if [[ -n "${REQUIRE_SUPABASE:-}" ]]; then
  NODE_ARGS+=(--require-supabase)
fi

echo "[info] Running CourtListener ingestion..."
node "${NODE_SCRIPT}" "${NODE_ARGS[@]}" "$@"

if [[ -f "${OUTPUT}" ]]; then
  echo "[info] Wrote payload to ${OUTPUT}"
  if command -v open >/dev/null 2>&1; then
    echo "[info] Opening ${OUTPUT}"
    open "${OUTPUT}"
  else
    echo "[warn] 'open' command not available; inspect the JSON manually."
  fi
else
  echo "[warn] Expected output file ${OUTPUT} not found."
fi
