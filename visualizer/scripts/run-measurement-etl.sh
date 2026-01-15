#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[error] DATABASE_URL environment variable is required." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}/qulab-infinite"

if ! command -v poetry >/dev/null 2>&1; then
  echo "[error] Poetry is not on PATH. Install it or activate your virtualenv." >&2
  exit 1
fi

poetry run python "${SCRIPT_DIR}/run-ghz-benchmark.py" \
  --postgres-url "${DATABASE_URL}" "$@"
