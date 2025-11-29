#!/usr/bin/env bash
# Cron helper to ensure the AI agent runs hourly without duplicate instances.
set -euo pipefail

APP_DIR="/app"
SCRIPT="${APP_DIR}/tsla_ai_master_final_ready.py"
PIDFILE="${APP_DIR}/ai_agent.pid"
LOG_DIR="${APP_DIR}/logs"
AGENT_LOG="${LOG_DIR}/ai_agent_output.log"

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

log "🐍 Using Python interpreter: $PYTHON"

if [[ ! -f "$SCRIPT" ]]; then
  log "❌ AI agent script not found at $SCRIPT"
  exit 1
fi

stop_existing_agent() {
  local pid="$1"
  log "⏹️  Stopping existing AI agent (PID ${pid})..."
  if kill "$pid" 2>/dev/null; then
    for _ in {1..30}; do
      if kill -0 "$pid" 2>/dev/null; then
        sleep 1
      else
        log "✅ Existing AI agent stopped."
        rm -f "$PIDFILE"
        return 0
      fi
    done
    log "⚠️  AI agent did not stop after 30s; sending SIGKILL."
    kill -9 "$pid" 2>/dev/null || true
  else
    log "⚠️  Failed to send SIGTERM to PID ${pid}; removing stale PID file."
  fi
  rm -f "$PIDFILE"
}

if [[ -f "$PIDFILE" ]]; then
  existing_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${existing_pid:-}" && -d "/proc/${existing_pid}" ]]; then
    if tr '\0' ' ' < "/proc/${existing_pid}/cmdline" | grep -q "tsla_ai_master_final_ready.py"; then
      stop_existing_agent "$existing_pid"
    else
      log "⚠️  Stale PID file detected (PID ${existing_pid}). Removing."
      rm -f "$PIDFILE"
    fi
  else
    log "⚠️  PID file found without live process. Removing."
    rm -f "$PIDFILE"
  fi
fi

log "▶️  Starting AI agent via cron..."
nohup "$PYTHON" "$SCRIPT" >> "$AGENT_LOG" 2>&1 &
child=$!
echo "$child" > "$PIDFILE"
sleep 5
if kill -0 "$child" 2>/dev/null; then
  log "✅ AI agent started (PID $child)."
  exit 0
else
  log "❌ Failed to start AI agent (PID $child)."
  rm -f "$PIDFILE"
  exit 1
fi
