#!/bin/bash
# Kill a training run whose queue log has not progressed for STALL seconds (GPU hang after sleep / throttling).
# The queue runner then records it as failed and moves on; failed jobs are retried with --retry-failed.
cd "$(dirname "$0")/.." || exit 1
STALL=${STALL:-1200}
idle=0
while true; do
  pid=$(pgrep -f "train/train.py" | head -1)
  if [ -n "$pid" ]; then
    idle=0
    log=$(ls -t runs/queue/*/*.log 2>/dev/null | head -1)
    age=$(( $(date +%s) - $(stat -f %m "$log") ))
    if [ "$age" -gt "$STALL" ]; then
      echo "$(date '+%F %T') watchdog: killing stalled pid $pid (log $log idle ${age}s)" | tee -a runs/queue/watchdog.log
      kill "$pid"
      sleep 30
    fi
  else
    idle=$((idle + 60))
    [ "$idle" -gt 900 ] && ! pgrep -f "run_queue.py" >/dev/null && { echo "$(date '+%F %T') watchdog: no queue activity, exiting" >> runs/queue/watchdog.log; exit 0; }
  fi
  sleep 60
done
