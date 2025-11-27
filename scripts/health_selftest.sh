#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
HEALTH_PATH="${HEALTH_PATH:-/health}"

validate_json_string() {
  local payload="$1"
  python3 - "$payload" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
sys.exit(0 if data.get("core") == "active" else 1)
PY
}

validate_json_file() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, sys, pathlib
content = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
data = json.loads(content)
sys.exit(0 if data.get("core") == "active" else 1)
PY
}

ok=0

check_python() {
  if output=$(python3 main.py --health 2>python_health.err); then
    if validate_json_string "$output"; then
      echo "[SELFTEST] python --health ✅ core=active"
    else
      echo "[SELFTEST] python --health ❌ core != active"
      ok=1
    fi
  else
    echo "[SELFTEST] python --health ❌ команда завершилась с ошибкой"
    ok=1
  fi
  rm -f python_health.err
}

check_curl() {
  local url="http://$HOST:$PORT$HEALTH_PATH"
  local tmp_resp
  tmp_resp=$(mktemp)
  if curl -sS "$url" -o "$tmp_resp"; then
    if validate_json_file "$tmp_resp"; then
      echo "[SELFTEST] curl /health ✅ core=active"
    else
      echo "[SELFTEST] curl /health ❌ core != active"
      ok=1
    fi
  else
    echo "[SELFTEST] curl /health ❌ запрос не удался"
    ok=1
  fi
  rm -f "$tmp_resp"
}

check_python
check_curl

if [[ "$ok" -eq 0 ]]; then
  echo "[SELFTEST] ✅ Проверки пройдены"
else
  echo "[SELFTEST] ❌ Проверки провалены"
fi

exit "$ok"
