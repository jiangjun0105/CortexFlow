#!/usr/bin/env bash
# Restart the breakfast agent: one process serves both the backend (/ws, /health) and the page (web/).
#   ./restart.sh                 ASR + Jev on, kitchen prompt off (the known-good setup)
#   ASR=0 JEV=0 ./restart.sh     voice agent only
#   KITCHEN=1 ./restart.sh       also the kitchen system prompt
# Log: /tmp/breakfast-server.log
cd "$(dirname "$0")"
LOG=/tmp/breakfast-server.log

pkill -f "python server.py" && echo "stopped old server"
while pgrep -f "python server.py" >/dev/null; do sleep 0.5; done

ASR=${ASR:-1} JEV=${JEV:-1} KITCHEN=${KITCHEN:-0} nohup .venv/bin/python server.py >"$LOG" 2>&1 &
echo "starting (ASR=${ASR:-1} JEV=${JEV:-1} KITCHEN=${KITCHEN:-0}), log: $LOG"

for _ in $(seq 1 100); do  # model load takes ~20 s
  if curl -sf localhost:8000/health >/dev/null; then
    curl -s localhost:8000/health; echo
    echo "ready: http://localhost:8000  (refresh the page)"
    exit 0
  fi
  if ! pgrep -f "python server.py" >/dev/null; then echo "server died:"; tail -20 "$LOG"; exit 1; fi
  sleep 1
done
echo "not ready after 100 s:"; tail -20 "$LOG"; exit 1
