from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict


def _resolve_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _post(path: str, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    url = f"{_resolve_host()}{path}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    data = json.dumps(payload).encode("utf-8")
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def dry_ping(model: str) -> bool:
    try:
        out = _post("/api/generate", {"model": model, "prompt": "ping", "stream": False}, timeout=10.0)
        return bool((out.get("response") or "").strip())
    except Exception:
        return False


def generate(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    stream: bool = False,
    options: Dict[str, Any] | None = None,
) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "options": {"temperature": temperature, **(options or {})},
    }

    if not stream:
        out = _post("/api/generate", body, timeout=120.0)
        return (out.get("response") or "").strip()

    url = f"{_resolve_host()}/api/generate"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    data = json.dumps(body).encode("utf-8")
    acc: list[str] = []
    with urllib.request.urlopen(req, data=data, timeout=300.0) as resp:
        for raw in resp:
            try:
                chunk = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            seg = chunk.get("response")
            if seg:
                acc.append(seg)
    return "".join(acc).strip()
