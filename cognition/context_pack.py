from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from cognition.rag_ingest import rag_search
from memory.memory_core import mem_recent
from system_core_prompt import SOFIA_CORE_PROMPT


def _build_system_prompt(contexts: List[dict], mem_blob: str = "") -> str:
    parts = [SOFIA_CORE_PROMPT]
    if mem_blob.strip():
        parts.append(
            "USER NOTES:\n"
            + mem_blob
            + "\nUse this to keep a caring tone and prioritize the user's interests."
        )
    if contexts:
        joined = "\n\n".join([f"- source {c.get('source','')}: {c.get('text','')}" for c in contexts])
        parts.append("DOCUMENT CONTEXT:\n" + joined)
    parts.append(
        "RESPONSE FORMAT:\n"
        "- Speak as a warm, caring partner.\n"
        "- Start with reassurance, then provide concrete steps or a draft answer.\n"
        "- Do not present yourself as an AI.\n"
    )
    return "\n\n".join(parts)


def _build_full_sofia_prompt(mem_blob: str, contexts: List[dict], directive: str | None = None) -> str:
    try:
        init_path = Path(__file__).resolve().parents[1] / "data" / "initial_memory.json"
        init_data = json.loads(init_path.read_text(encoding="utf-8"))
        profile = init_data.get("profile", {})
        mission = init_data.get("mission", {})
        context_notes = init_data.get("context_notes") or []
    except Exception:
        profile, mission, context_notes = {}, {}, []

    parts = [SOFIA_CORE_PROMPT]
    parts.append(
        "=== ASSISTANT PERSONALITY ===\nA warm, human-like partner who answers calmly and thoughtfully."
    )
    if profile:
        parts.append("=== USER PROFILE ===")
        for k, v in profile.items():
            parts.append(f"- {k.capitalize()}: {v}")
    if mission:
        parts.append("=== MISSION ===")
        for k, v in mission.items():
            parts.append(f"- {k.capitalize()}: {v}")
    if context_notes:
        parts.append("=== EXTRA TRAITS ===")
        for note in context_notes:
            note_str = str(note).strip()
            if note_str:
                parts.append(f"- {note_str}")
    parts.append("=== ACTIVE MEMORY ===\n" + (mem_blob or ""))
    if contexts:
        ctx_str = "\n\n".join([f"- {c.get('source')}: {c.get('text')}" for c in contexts])
        parts.append("=== DOCUMENT CONTEXT ===\n" + ctx_str)
    parts.append("=== COMMUNICATION RULES ===\nStay warm, confident, and direct. Do not describe yourself as an AI.")
    if directive and directive.strip():
        parts.append("=== DIALOGUE DIRECTIVE ===\n" + directive.strip())
    return "\n\n".join(parts)


def build_context_pack(
    session_id: str,
    user_message: str,
    token_budget: int = 3000,
    directive: str | None = None,
) -> Dict[str, Any]:
    try:
        mem_items = mem_recent(session_id, 8)
    except Exception:
        mem_items = []
    mem_blob = "\n".join(
        f"- {m.get('role', 'user')}: {m.get('content', '').strip()}"
        for m in mem_items
        if m.get("content")
    )
    try:
        contexts = rag_search(user_message, k=4)
    except Exception:
        contexts = []

    retrieval = [
        {"title": ctx.get("source", ""), "chunk": ctx.get("text", ""), "source": ctx.get("source", "")}
        for ctx in contexts
        if ctx.get("text")
    ]

    system_prompt = _build_full_sofia_prompt(mem_blob, contexts, directive=directive)

    pack = {
        "system": system_prompt,
        "dialogue": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in mem_items],
        "facts": [],
        "retrieval": retrieval,
        "task": user_message,
        "summary": "",
        "token_budget": token_budget,
    }
    pack["raw_contexts"] = contexts
    if directive and directive.strip():
        pack["policy_directive"] = directive.strip()
    return pack
