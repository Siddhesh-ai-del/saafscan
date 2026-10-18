#!/usr/bin/env bash
# SaafScan — Round-1 screenshot pack (Phase 6).
#
# Uses native headless Chrome (Playwright's bundled build is not installed in
# a compatible version on this machine; the system Chrome binary works).
#
# Usage:  bash scripts/screenshot.sh [baseUrl]
# Expects the server running (bash start_demo.sh).
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/screenshots"
mkdir -p "$OUT"

CHROME=""
for candidate in \
  "$HOME/.cache/ms-playwright/chromium-1247/chrome-linux64/chrome" \
  "$(command -v chromium || true)" \
  "$(command -v google-chrome || true)" \
  "$(command -v chromium-browser || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] && CHROME="$candidate" && break
done
[[ -n "$CHROME" ]] || { echo "ERROR: no Chrome/Chromium binary found" >&2; exit 1; }

shot() { # shot <name> <url>
  "$CHROME" --headless --no-sandbox --disable-gpu --hide-scrollbars \
    --force-prefers-reduced-motion \
    --window-size=1360,1500 --virtual-time-budget=12000 \
    --screenshot="$OUT/$1" "$2" 2>/dev/null
  echo "✓ $OUT/$1"
}

# 1 — landing / capture UI
shot 01-landing-capture.png "$BASE"

# 2 — verdict: starch (positive)
shot 02-verdict-starch.png "$BASE/?demo=starch"

# 3 — verdict: detergent
shot 03-verdict-detergent.png "$BASE/?demo=detergent"

# 4 — verdict: pure (negative control, after the above populated history)
shot 04-verdict-pure.png "$BASE/?demo=none"

# 5 — history populated with all four analyses
shot 05-evidence-history.png "$BASE/"

echo
echo "Screenshots written to $OUT"
ls -la "$OUT"
