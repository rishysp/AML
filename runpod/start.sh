#!/usr/bin/env bash
# Start the full run in the background, so it keeps going if you close the browser or terminal.
#   bash runpod/start.sh                         # HI-Medium
#   QUICK=1 DATASET=HI-Small bash runpod/start.sh  # 10-15 min smoke test first
# Then watch it with:  bash runpod/status.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-/workspace/aml_run}"
mkdir -p "$WORKDIR/logs"
if [ -f "$WORKDIR/logs/run.pid" ] && kill -0 "$(cat "$WORKDIR/logs/run.pid")" 2>/dev/null; then
  echo "A run is already in progress (PID $(cat "$WORKDIR/logs/run.pid")). Watch it with: bash runpod/status.sh"
  exit 1
fi
LOG="$WORKDIR/logs/run_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$REPO/runpod/run_all.sh" > "$LOG" 2>&1 &
echo $! > "$WORKDIR/logs/run.pid"
ln -sf "$LOG" "$WORKDIR/logs/latest.log"
echo "Started (PID $!). Log: $LOG"
echo "Watch progress:  bash runpod/status.sh      (or: tail -f $WORKDIR/logs/latest.log)"
