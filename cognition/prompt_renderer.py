from __future__ import annotations

from typing import Dict, List


def render_prompt_from_pack(pack: Dict) -> str:
    sys = (pack.get("system") or "").strip()
    facts = pack.get("facts") or []
    retrieval = pack.get("retrieval") or []
    dialogue = pack.get("dialogue") or []
    task = (pack.get("task") or "").strip()
    summary = (pack.get("summary") or "").strip()

    facts_block = ""
    if facts:
        facts_block = "### Key facts (memory)\n" + "\n".join(f"- {f}" for f in facts) + "\n"

    retrieval_block = ""
    if retrieval:
        retrieval_block = "### Materials (RAG)\n" + "\n".join(
            f"- [{r.get('title','')}] {r.get('chunk','')}" for r in retrieval
        ) + "\n"

    dialogue_block = ""
    if dialogue:
        lines: List[str] = []
        for msg in dialogue:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content','')}")
        dialogue_block = "### Dialogue context\n" + "\n".join(lines) + "\n"

    summary_block = f"### Summary\n{summary}\n" if summary else ""
    task_block = f"### Task\n{task}\n" if task else ""

    out: List[str] = []
    if sys:
        out.append(f"<SYSTEM>\n{sys}\n</SYSTEM>\n")
    out.append(facts_block)
    out.append(retrieval_block)
    out.append(dialogue_block)
    out.append(summary_block)
    out.append(task_block)
    out.append(
        "### Instructions\n"
        "- Provide a clear, concise answer.\n"
        "- Keep a warm, supportive tone.\n"
        "- If unsure, say that the answer needs verification.\n"
        "- Do not prefix the reply with a name or signature.\n"
    )
    return "\n".join(filter(None, out)).strip()
