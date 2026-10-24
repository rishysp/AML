#!/usr/bin/env bash
# Is the run alive, how far has it got, and how much memory is it using?
#   bash runpod/status.sh          # snapshot
#   bash runpod/status.sh -f       # snapshot, then follow the log live (Ctrl+C to stop watching;
#                                  # the run itself keeps going)
WORKDIR="${WORKDIR:-/workspace/aml_run}"
PID_FILE="$WORKDIR/logs/run.pid"
LOG="$WORKDIR/logs/latest.log"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "RUNNING (PID $(cat "$PID_FILE"))"
else
  echo "NOT RUNNING"
  grep -q "FINISHED" "$LOG" 2>/dev/null && echo "-> finished: $(ls "$WORKDIR"/aml_results_*.zip 2>/dev/null | tr '\n' ' ')"
  if grep -q "^STOPPED" "$LOG" 2>/dev/null; then
    echo "--- last log lines (the error is usually here)"; tr '\r' '\n' < "$LOG" | grep -v '%|' | tail -15
  fi
fi
echo "--- memory"; free -g | head -2
command -v nvidia-smi >/dev/null && { echo "--- GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; }
echo "--- disk"; df -h "$WORKDIR" | tail -1
echo "--- last progress lines"; grep -E "^====|^\[|trial|epoch|STOPPED|ERROR" "$LOG" 2>/dev/null | tail -12
if [ "${1:-}" = "-f" ]; then tail -n 0 -f "$LOG"; fi
exit 0
