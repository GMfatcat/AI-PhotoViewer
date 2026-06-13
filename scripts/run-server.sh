#!/usr/bin/env bash
# Launch the AI-PhotoViewer web server (Linux / macOS / Git Bash).
#
# Usage:
#   scripts/run-server.sh [--db PATH] [--model DIR] [--host IP] [--port N]
#
# Model dir can also be set via SIGLIP_MODEL. An empty photos.db is created
# automatically on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

# venv python: Linux/macOS use bin/, Windows (Git Bash) use Scripts/
if [ -x "$ROOT/.venv/bin/python" ]; then
  VENV="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  VENV="$ROOT/.venv/Scripts/python.exe"
else
  echo "venv python not found under $ROOT/.venv  (create it first: uv venv)" >&2
  exit 1
fi

DB="$ROOT/photos.db"
MODEL="${SIGLIP_MODEL:-$ROOT/../models/siglip2-so400m}"
HOST="127.0.0.1"
PORT="8000"

while [ $# -gt 0 ]; do
  case "$1" in
    --db)    DB="$2";    shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --host)  HOST="$2";  shift 2 ;;
    --port)  PORT="$2";  shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "DB:    $DB"
echo "Model: $MODEL"
echo "URL:   http://$HOST:$PORT"
exec "$VENV" "$ROOT/web_demo/main.py" --db "$DB" --model "$MODEL" --host "$HOST" --port "$PORT"
