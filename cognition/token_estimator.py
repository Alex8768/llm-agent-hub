from __future__ import annotations

import re
from typing import Dict

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))

except Exception:  # noqa: BLE001

    def _count_tokens(text: str) -> int:
        return max(1, int(len(text or "") / 4) + int((text or "").count("\n") * 0.5))


def estimate_pack_tokens(pack: Dict) -> Dict[str, int]:
    sys = pack.get("system") or ""
    dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in pack.get("dialogue", []))
    summary = pack.get("summary") or ""
    facts = "\n".join(f"- {x}" for x in pack.get("facts", []))
    retrieval = "\n".join(f"[{r.get('title','')}] {r.get('chunk','')}" for r in pack.get("retrieval", []))
    task = pack.get("task") or ""

    parts = {
        "system": _count_tokens(sys),
        "dialogue": _count_tokens(dialogue),
        "summary": _count_tokens(summary),
        "facts": _count_tokens(facts),
        "retrieval": _count_tokens(retrieval),
        "task": _count_tokens(task),
    }
    parts["total"] = sum(parts.values())
    return parts


_KEYWORDS_REASONING = re.compile(
    r"\b(prove|justify|formal|proof|reasoning steps|"
    r"step by step|theorem|deduct|induct|optimize|complexity|algorithm|schema|"
    r"chain of thought|reason|analy[sz]e)\b",
    re.I,
)

_KEYWORDS_LONGFORM = re.compile(
    r"\b(essay|article|report|longform|5000|1500\+|long\w+)\b",
    re.I,
)


def infer_pack_meta(pack: Dict, message: str) -> Dict[str, int | bool]:
    counts = estimate_pack_tokens(pack)
    text_all = (message or "") + " " + (pack.get("task") or "")
    needs_strong_reasoning = bool(_KEYWORDS_REASONING.search(text_all))
    needs_longform = bool(_KEYWORDS_LONGFORM.search(text_all))
    doc_count = len(pack.get("retrieval") or [])
    ru_ratio = sum(1 for c in text_all if 0x0410 <= ord(c) <= 0x044f) / max(1, len(text_all) or 1)
    lang = "ru" if ru_ratio > 0.3 else "other"
    return {
        "estimated_tokens": counts["total"],
        "approx_tokens": counts["total"],
        "by_part": counts,
        "doc_count": doc_count,
        "rag_docs": doc_count,
        "lang": lang,
        "needs_strong_reasoning": needs_strong_reasoning,
        "needs_longform": needs_longform,
    }
