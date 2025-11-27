# LLM-Agent Hub — quick ops

Local orchestration hub for LLM + RAG + tool execution, with health checks and a minimal web UI.

## Run / Stop / Restart
- Start: `./run.sh start`
- Stop: `./run.sh stop`
- Restart (with pre/post smoke): `./run.sh restart`

## Smoke-check
- Pre-start check: `./scripts/smoke_check.sh --pre`
- Wait for readiness (up to ~40s): `./scripts/smoke_check.sh --post`
- Defaults: `HOST=127.0.0.1`, `PORT=8081`, `PATH=/health`; override via flags or env vars.

## Health endpoints
- CLI: `python main.py --health`
- HTTP: `curl http://127.0.0.1:8081/health`
- Status: `curl http://127.0.0.1:8081/status`

## Logs
- Main file: `data/logs/bridge.log`
- Tail: `tail -f data/logs/bridge.log`
