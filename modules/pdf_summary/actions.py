# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from executor.planner import _call_llm_for_plan
from .logic import list_pdfs, read_pdf

logger = getLogger(__name__)


async def pdf_list_docs(args: Dict[str, Any]) -> Dict[str, Any]:
    docs = list_pdfs()
    return {"docs": docs}


async def pdf_read(args: Dict[str, Any]) -> Dict[str, Any]:
    path = (args.get("path") or args.get("file_path") or "").strip()
    if not path:
        raise ValueError("path is required")
    text = read_pdf(path)
    if not text.strip():
        raise ValueError(f"PDF is empty or unreadable: {path}")
    return {"path": path, "content": text}


def _summarize_text(text: str) -> str:
    system_prompt = "You are a concise summarizer. Return 5 bullet points capturing key insights."
    user_prompt = f"Summarize the following PDF in 5 concise bullet points:\n\n{text}"
    raw = _call_llm_for_plan(system_prompt, user_prompt)
    return raw.strip()


async def pdf_summarize(args: Dict[str, Any]) -> Dict[str, Any]:
    path = (args.get("path") or args.get("file_path") or "").strip()
    if not path:
        raise ValueError("path is required")
    text = read_pdf(path)
    if not text.strip():
        raise ValueError(f"PDF is empty or unreadable: {path}")
    logger.info("Summarizing PDF: %s", path)
    summary = _summarize_text(text[:8000])
    if not summary.strip():
        raise ValueError(f"Summary generation failed for: {path}")
    return {"path": path, "summary": summary}
