#!/usr/bin/env bash
# Safe self_learn runner: single-instance via PID file + /proc check
set -euo pipefail

APP_DIR="/app"
PYTHON="${PYTHON:-python3}"
SCRIPT="${APP_DIR}/self_learn.py"
ARGS=(--horizon-days 5 --threshold 0.02 --test-size 0.2)

PIDFILE="${APP_DIR}/self_learn.pid"
LOGDIR="${APP_DIR}/logs"
LOGFILE="${LOGDIR}/self_learn.log"

mkdir -p "$LOGDIR"
cd "$APP_DIR"

is_same_process() {
  # $1 = pid
  [[ -d "/proc/$1" ]] || return 1
  [[ -r "/proc/$1/cmdline" ]] && tr '\0' ' ' < "/proc/$1/cmdline" | grep -q "self_learn.py"
}

cleanup() {
  if [[ -f "$PIDFILE" ]]; then
    cur="$(cat "$PIDFILE" 2>/dev/null || true)"
    [[ "${cur:-}" = "$child" ]] && rm -f "$PIDFILE" || true
  fi
}
trap cleanup EXIT INT TERM

# Handle existing PID file
if [[ -f "$PIDFILE" ]]; then
  oldpid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${oldpid:-}" ]] && is_same_process "$oldpid"; then
    echo "self_learn already running (PID $oldpid); exiting." | tee -a "$LOGFILE"
    exit 0
  else
    echo "Removing stale PID file (${oldpid:-unknown})." | tee -a "$LOGFILE"
    rm -f "$PIDFILE"
  fi
fi

echo "[$(date -Is)] Starting self_learn..." | tee -a "$LOGFILE"
set +e
"$PYTHON" "$SCRIPT" "${ARGS[@]}" >> "$LOGFILE" 2>&1 &
child=$!
set -e
echo "$child" > "$PIDFILE"

# Wait for completion so trap can clean PID file on exit
wait "$child"
rc=$?
echo "[$(date -Is)] self_learn finished with code $rc" | tee -a "$LOGFILE"
exit "$rc"
