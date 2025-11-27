#!/usr/bin/env bash
set -e

# auto-resolve absolute project path
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT"

export OLLAMA_HOST="http://127.0.0.1:11434"
export TOKENIZERS_PARALLELISM="false"

case "$1" in
  start)
    exec python -m uvicorn main:app --host 127.0.0.1 --port 8081 --log-level info
    ;;
  restart)
    pkill -f "uvicorn main:app" || true
    sleep 1
    exec python -m uvicorn main:app --host 127.0.0.1 --port 8081 --log-level info
    ;;
  stop)
    pkill -f "uvicorn main:app" || true
    ;;
  *)
    echo "Usage: $0 {start|restart|stop}" >&2
    exit 1
    ;;
esac