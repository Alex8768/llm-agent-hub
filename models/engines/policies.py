from __future__ import annotations

import os
from typing import Any, Dict, Tuple


def _local_llm_enabled() -> bool:
    val = os.getenv("ENABLE_LOCAL_LLM", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


def select_engine(payload: Dict[str, Any], pack_meta: Dict[str, Any]) -> Tuple[str, str | None]:
    payload = payload or {}
    meta = pack_meta or {}

    req_engine = (payload.get("engine") or "").strip().lower()
    requested_model = (payload.get("model") or "").strip() or None

    local_enabled = _local_llm_enabled()

    if req_engine == "ollama":
        if not local_enabled:
            raise ValueError("local LLM disabled")
        return "ollama", requested_model
    if req_engine == "api":
        return "api", requested_model

    prefer = (payload.get("prefer") or os.getenv("AUTO_PREF", "")).strip().lower()
    if prefer in ("local", "ollama"):
        if local_enabled:
            return "ollama", requested_model
        # fall through to API if local disabled
    if prefer == "api":
        return "api", requested_model

    tokens = int(meta.get("approx_tokens") or meta.get("estimated_tokens") or 0)
    doc_k = int(meta.get("rag_docs") or meta.get("doc_count") or 0)
    lang = (meta.get("lang") or "").lower()

    tok_gate = int(os.getenv("TOK_AUTO_API", "3500"))
    doc_gate = int(os.getenv("DOC_HEAVY", "6"))

    if tokens < tok_gate and doc_k < doc_gate:
        if local_enabled:
            return "ollama", requested_model
        return "api", requested_model

    return "api", requested_model
