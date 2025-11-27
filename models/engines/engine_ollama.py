from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Dict, Optional, Set

from .base import GenerateRequest, GenerateResponse, LLMEngine
from .registry import register

DEFAULT_QWEN = os.getenv("DEFAULT_OLLAMA_MODEL", "qwen2.5:3b-instruct")
PREFERRED_QWEN = os.getenv("PREFERRED_QWEN", "qwen2.5:3b-instruct-q4_K_M")
PREFERRED_PHI = os.getenv("PREFERRED_PHI", "phi3:mini")

CODE_HINTS = re.compile(
    r"(def |class |import |for |while |try:|except:|Traceback|AssertionError|doctest|pytest|TypeError|ValueError|\bO\(\w+\)|algorithm|pseudocode|function|code)",
    re.I,
)
RU_HINTS = re.compile(r"[\u0400-\u04FF]")


def _list_ollama_models(host: str) -> Set[str]:
    req = urllib.request.Request(f"{host}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {m.get("name") for m in data.get("models", []) if m.get("name")}


def resolve_ollama_model(host: str, payload: Optional[Dict], text: str) -> str:
    payload = payload or {}
    explicit = (payload.get("model") or "").strip()
    if explicit:
        return explicit

    if CODE_HINTS.search(text or ""):
        candidate = PREFERRED_PHI or DEFAULT_QWEN
    elif RU_HINTS.search(text or ""):
        candidate = PREFERRED_QWEN or DEFAULT_QWEN
    else:
        candidate = PREFERRED_QWEN or DEFAULT_QWEN

    try:
        available = _list_ollama_models(host)
        if candidate in available:
            return candidate
        if DEFAULT_QWEN in available:
            return DEFAULT_QWEN
    except Exception:
        pass
    return candidate


class OllamaEngine(LLMEngine):
    name = "ollama"

    def __init__(self, host: str | None = None):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        model_name = req.model or resolve_ollama_model(self.host, req.payload, req.prompt)
        body = {
            "model": model_name,
            "prompt": req.prompt,
            "stream": False,
        }
        if req.options:
            body["options"] = req.options
        http_req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_req, timeout=120.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("response") or "").strip()
        return GenerateResponse(text=text, raw=data)


register(OllamaEngine())
