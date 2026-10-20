#!/usr/bin/env bash
# SaafScan — one-command cold start.
# Validates environment, launches the API + SPA, waits for /health, prints demo tips.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

HOST="${SAAFSCAN_HOST:-127.0.0.1}"
PORT="${SAAFSCAN_PORT:-8000}"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

# --- validation (fail loudly, never silently) --------------------------------
if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv missing at $VENV" >&2
  echo "       run: uv venv --python 3.12 .venv && uv sync --python .venv/bin/python" >&2
  exit 1
fi

if ! "$PY" -c "import saafscan, fastapi, cv2, numpy" 2>/dev/null; then
  echo "ERROR: dependencies missing — run: uv sync --python .venv/bin/python" >&2
  exit 1
fi

# --- reuse a healthy server if one is already running -------------------------
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "SaafScan already running at http://$HOST:$PORT (reusing)"
  exit 0
fi

echo "Starting SaafScan on http://$HOST:$PORT ..."
"$PY" -m uvicorn saafscan.api:app --host "$HOST" --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# --- wait for readiness -------------------------------------------------------
for _ in $(seq 1 60); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo
    echo "SaafScan is READY → http://$HOST:$PORT"
    echo
    echo "Demo tips:"
    echo "  1. Open the URL above (use the laptop camera, or upload a photo)."
    echo "  2. Demo mode buttons (pure / starch / detergent) run the real pipeline"
    echo "     on synthetic strips — Round-1 screenshots + rehearsal only."
    echo "  3. For 24 Oct, use REAL reagent strips — see README"
    echo "     'Physical kit for 24 Oct'."
    echo
    echo "Ctrl+C stops the server."
    wait "$SERVER_PID"
    exit 0
  fi
  sleep 0.5
done

echo "ERROR: server did not become healthy within 30s — check logs above." >&2
exit 1
