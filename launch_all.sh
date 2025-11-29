#!/usr/bin/env bash
# Master bootstrap with PID-safety for agent & dashboard, and portable cron start
set -euo pipefail

APP_DIR="/app"
LOG_DIR="${APP_DIR}/logs"
MODEL_DIR="${APP_DIR}/models"
DATA_DIR="${APP_DIR}/data"
BACKUP_DIR="${APP_DIR}/backups"

PYTHON="${PYTHON:-python3}"

# Ensure bundled helper scripts (like the nvidia-smi stub) are discoverable.
export PATH="${APP_DIR}/scripts:${PATH}"

AGENT_SCRIPT="${APP_DIR}/tsla_ai_master_final_ready.py"
AGENT_LOG="${LOG_DIR}/ai_agent_output.log"
AGENT_PIDFILE="${APP_DIR}/ai_agent.pid"

DASHBOARD_SCRIPT="${APP_DIR}/dashboard.py"
DASHBOARD_LOG="${LOG_DIR}/dashboard_output.log"
DASHBOARD_PIDFILE="${APP_DIR}/dashboard.pid"

LAUNCH_LOG="${LOG_DIR}/launch_all.log"
RETRAIN_LOG="${LOG_DIR}/retrain.log"
TEST_LOG="${LOG_DIR}/test_output.log"

mkdir -p "$LOG_DIR" "$MODEL_DIR" "$DATA_DIR" "$BACKUP_DIR"

log() { echo "$(date -Is) $*" | tee -a "$LAUNCH_LOG" ; }

# ---------- pre-flight: ensure historical_data.csv ----------
HIST_DATA_FILE="${DATA_DIR}/historical_data.csv"
if [[ ! -f "$HIST_DATA_FILE" ]]; then
  if [[ ! -f "${APP_DIR}/scripts/generate_historical_data.py" ]]; then
    log "❌ scripts/generate_historical_data.py not found. Cannot build historical dataset."
    exit 1
  fi
  log "📈 Generating initial historical_data.csv..."
  set +e
  PYTHONPATH="$APP_DIR" "$PYTHON" - "$HIST_DATA_FILE" >> "$RETRAIN_LOG" 2>&1 <<'PY'
import sys

from scripts.generate_historical_data import generate_historical_data

hist_file = sys.argv[1]
success = generate_historical_data(hist_file)
raise SystemExit(0 if success else 1)
PY
  status=$?
  set -e
  if [[ $status -ne 0 || ! -f "$HIST_DATA_FILE" ]]; then
    log "❌ Failed to generate historical_data.csv. See $RETRAIN_LOG for details."
    exit 1
  fi
  log "✅ historical_data.csv generated at $HIST_DATA_FILE"
else
  log "ℹ️ Found existing historical_data.csv at $HIST_DATA_FILE"
fi

# ---------- helpers ----------
is_pid_running_cmdline() {
  # $1 = pid, $2 = substring to find in /proc/<pid>/cmdline
  local pid="$1" needle="$2"
  [[ -d "/proc/$pid" && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "$needle"
}

start_background() {
  # $1 = command (string), $2 = log file, $3 = pidfile, $4 = cmdline needle
  local cmd="$1" logfile="$2" pidfile="$3" needle="$4"

  if [[ -f "$pidfile" ]]; then
    local oldpid
    oldpid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${oldpid:-}" ]] && is_pid_running_cmdline "$oldpid" "$needle"; then
      log "⚠️  Process already running (PID $oldpid) for [$needle]; skipping launch."
      return 0
    else
      log "⚠️  Removing stale PID file ($pidfile)."
      rm -f "$pidfile"
    fi
  fi

  log "▶️  Starting [$needle] ..."
  set +e
  bash -lc "$cmd" >> "$logfile" 2>&1 &
  local child=$!
  set -e
  echo "$child" > "$pidfile"
  # brief health check
  sleep 3
  if ! kill -0 "$child" 2>/dev/null; then
    log "❌ Failed to start [$needle]. See $logfile"
    return 1
  fi
  log "✅ [$needle] started (PID $child). Logs: $logfile"
}

# ---------- STEP 1: ensure trade_log.csv ----------
if [[ ! -f "${LOG_DIR}/trade_log.csv" ]]; then
  if [[ ! -f "${APP_DIR}/generate_trade_log.py" ]]; then
    log "❌ generate_trade_log.py not found. Exiting."
    exit 1
  fi
  log "📊 Generating initial trade_log.csv..."
  set +e
  $PYTHON "${APP_DIR}/generate_trade_log.py" >> "$RETRAIN_LOG" 2>&1 || log "⚠️  generate_trade_log.py exited non-zero."
  set -e
fi

# ---------- STEP 2: unit tests (optional) ----------
log "🧪 Running unit tests (if available)..."
if command -v pytest >/dev/null 2>&1 && [[ -d "${APP_DIR}/tests" ]]; then
  if [[ ! -f "${APP_DIR}/test_predictions.py" || ! -f "${APP_DIR}/test_indicators.py" ]]; then
    log "⚠️ Test scripts missing. Skipping tests."
  else
    set +e
    pytest -v "${APP_DIR}/test_predictions.py" "${APP_DIR}/test_indicators.py" > "$TEST_LOG" 2>&1
    [[ $? -eq 0 ]] && log "✅ Tests passed." || log "⚠️ Some tests failed; continuing. See $TEST_LOG"
    set -e
  fi
else
  log "ℹ️ Pytest or /app/tests not found; skipping tests."
fi

# ---------- STEP 3: start AI agent (PID-safe) ----------
if [[ ! -f "$AGENT_SCRIPT" ]]; then
  log "❌ AI agent script ($AGENT_SCRIPT) not found. Exiting."
  exit 1
fi
start_background "$PYTHON \"$AGENT_SCRIPT\"" "$AGENT_LOG" "$AGENT_PIDFILE" "tsla_ai_master_final_ready.py"

# ---------- STEP 4: start dashboard (PID-safe) ----------
if [[ -f "$DASHBOARD_SCRIPT" ]]; then
  start_background "$PYTHON \"$DASHBOARD_SCRIPT\"" "$DASHBOARD_LOG" "$DASHBOARD_PIDFILE" "dashboard.py"
else
  log "ℹ️ Dashboard script not found; skipping."
fi

# ---------- STEP 5: install cron jobs (no crontab wipe) ----------
log "🕒 Installing cron jobs..."
cron_entries=()

if [[ -f "${APP_DIR}/ai_agent_cron.sh" ]]; then
  cron_entries+=("0 * * * * ${APP_DIR}/ai_agent_cron.sh >> ${LOG_DIR}/cron.log 2>&1")
else
  log "❌ ai_agent_cron.sh not found. Skipping hourly agent cron."
fi

if [[ -f "${APP_DIR}/update_historical_dataset_cron.sh" ]]; then
  cron_entries+=("0 7 * * * ${APP_DIR}/update_historical_dataset_cron.sh >> ${LOG_DIR}/cron.log 2>&1")
else
  log "❌ update_historical_dataset_cron.sh not found. Skipping daily historical dataset cron."
fi

if [[ -f "${APP_DIR}/cron_backup.sh" ]]; then
  cron_entries+=("0 0 * * 0 ${APP_DIR}/cron_backup.sh >> ${LOG_DIR}/cron.log 2>&1")
else
  log "❌ cron_backup.sh not found. Skipping weekly backup cron."
fi

if [[ ${#cron_entries[@]} -gt 0 ]]; then
  tmp_cron="$(mktemp)"
  crontab -l 2>/dev/null > "$tmp_cron" || true
  sed -i "\|${APP_DIR}/ai_agent_cron.sh|d" "$tmp_cron"
  sed -i "\|${APP_DIR}/update_historical_dataset_cron.sh|d" "$tmp_cron"
  sed -i "\|${APP_DIR}/cron_backup.sh|d" "$tmp_cron"
  printf "%s\n" "${cron_entries[@]}" >> "$tmp_cron"
  crontab "$tmp_cron" || { log "❌ Failed to install cron jobs"; rm -f "$tmp_cron"; exit 1; }
  rm -f "$tmp_cron"
else
  log "ℹ️ No cron entries to install."
fi

# ---------- STEP 6: start cron service ----------
log "🕒 Starting cron in foreground..."
sudo cron -f -L "${LOG_DIR}/crond.log" &

# ---------- STEP 7: backtest hint ----------
log "ℹ️ Backtest: $PYTHON ${APP_DIR}/tsla_ai_master_final_ready_backtest.py --start-date 2025-01-01 --end-date 2025-08-18 --timeframe 1H"

# ---------- keep container alive ----------
log "🟢 AI Agent & (optional) Dashboard running. Tailing launch log..."
tail -F "$LAUNCH_LOG"