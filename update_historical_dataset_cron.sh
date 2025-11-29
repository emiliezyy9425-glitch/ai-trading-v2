#!/usr/bin/env bash
# Cron helper to trigger the daily historical dataset append.
set -euo pipefail

APP_DIR="/app"
SCRIPT="${APP_DIR}/tsla_ai_master_final_ready.py"
LOG_DIR="${APP_DIR}/logs"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -Is)] $*"
}

resolve_python() {
  local candidate resolved
  candidate="${PYTHON:-}"

  if [[ -n "$candidate" ]]; then
    if [[ "$candidate" == /* && -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi

    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -n "$resolved" ]]; then
      echo "$resolved"
      return 0
    fi

    log "⚠️  Provided PYTHON value '$candidate' is not executable. Ignoring."
  fi

  resolved="$(command -v python3 2>/dev/null || true)"
  if [[ -n "$resolved" ]]; then
    echo "$resolved"
    return 0
  fi

  if [[ -x /usr/local/bin/python3 ]]; then
    echo "/usr/local/bin/python3"
    return 0
  fi

  if [[ -x /usr/bin/python3 ]]; then
    echo "/usr/bin/python3"
    return 0
  fi

  echo ""
}

PYTHON="$(resolve_python)"

if [[ -z "$PYTHON" ]]; then
  log "❌ Python interpreter not found. Set the PYTHON environment variable or ensure python3 is installed."
  exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
  log "❌ Trading agent entry point not found at $SCRIPT"
  exit 1
fi

log "▶️  Triggering historical dataset update via cron..."
if "$PYTHON" "$SCRIPT" update-historical >>"${LOG_DIR}/historical_update.log" 2>&1; then
  log "✅ historical_data.csv update completed."
else
  log "❌ historical_data.csv update failed. See ${LOG_DIR}/historical_update.log for details."
  exit 1
fi
