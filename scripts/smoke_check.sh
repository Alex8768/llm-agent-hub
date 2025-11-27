#!/usr/bin/env bash
set -euo pipefail

MODE="post"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
MAX_ATTEMPTS=20
SLEEP_SECONDS=2

usage() {
    echo "Usage: $0 [--pre|--post] [--host HOST] [--port PORT] [--path /health]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pre)
            MODE="pre"
            shift
            ;;
        --post)
            MODE="post"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --path)
            HEALTH_PATH="$2"
            shift 2
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --path=*)
            HEALTH_PATH="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

URL="http://${HOST}:${PORT}${HEALTH_PATH}"

check_ready_once() {
    local tmp status
    tmp=$(mktemp)
    status=$(curl -sS -m 5 -o "$tmp" -w "%{http_code}" "$URL" || true)
    if [[ "$status" != "200" ]]; then
        rm -f "$tmp"
        return 1
    fi

    if python3 - "$tmp" <<'PY'; then
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
except Exception:
    sys.exit(1)

core = data.get("core")
sys.exit(0 if core == "active" else 1)
PY
        rm -f "$tmp"
        return 0
    fi

    rm -f "$tmp"
    return 1
}

if [[ "$MODE" == "pre" ]]; then
    if check_ready_once; then
        echo "[CHECK] (pre) ✅ Ядро уже активно."
        exit 0
    fi
    echo "[CHECK] (pre) ⚠ Ядро не отвечает."
    exit 1
fi

# post mode with retries
for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
    if check_ready_once; then
        echo "[CHECK] ✅ Ядро активно."
        exit 0
    fi
    sleep "$SLEEP_SECONDS"
done

echo "[CHECK] ❌ Ошибка: ядро не отвечает!"
exit 1
