#!/usr/bin/env bash
# Launch (or stop) the AI-PhotoViewer web server (Linux / macOS / Git Bash).
#
# Usage:
#   scripts/run-server.sh [--db PATH] [--model DIR] [--host IP] [--port N]
#   scripts/run-server.sh --stop [--port N]      # stop whatever serves on --port
#
# Default host is 127.0.0.1 (local only). Pass --host 0.0.0.0 to expose on LAN.
# Model dir can also be set via SIGLIP_MODEL. An empty photos.db is created
# automatically on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

DB="$ROOT/photos.db"
MODEL="${SIGLIP_MODEL:-$ROOT/../models/siglip2-so400m}"
HOST="127.0.0.1"
PORT="8000"
STOP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --stop)  STOP=1;     shift ;;
    --db)    DB="$2";    shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host)  HOST="$2";  shift 2 ;;
    --port)  PORT="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# --- Stop mode: kill whatever is listening on $PORT ------------------------
if [ "$STOP" = "1" ]; then
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
    if [ -z "$pids" ]; then echo "No server listening on port $PORT."; exit 0; fi
    echo "Stopping PID(s): $pids"; kill $pids; exit 0
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null && echo "Stopped server on port $PORT." || echo "No server on port $PORT."
    exit 0
  else
    echo "Need lsof or fuser to stop by port (on Windows use run-server.ps1 -Stop)." >&2
    exit 1
  fi
fi

# --- Start mode -----------------------------------------------------------
# venv python: Linux/macOS use bin/, Windows (Git Bash) use Scripts/
if [ -x "$ROOT/.venv/bin/python" ]; then
  VENV="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  VENV="$ROOT/.venv/Scripts/python.exe"
else
  echo "venv python not found under $ROOT/.venv  (create it first: uv venv)" >&2
  exit 1
fi

echo "DB:    $DB"
echo "Model: $MODEL"
echo "URL:   http://$HOST:$PORT"
exec "$VENV" "$ROOT/web_demo/main.py" --db "$DB" --model "$MODEL" --host "$HOST" --port "$PORT"
