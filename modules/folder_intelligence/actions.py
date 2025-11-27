# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from executor.planner import _call_llm_for_summary
from .logic import list_items, compute_stats

logger = getLogger(__name__)


async def folder_list(args: Dict[str, Any]) -> Dict[str, Any]:
    path = (args.get("path") or "").strip() or None
    items = list_items(path)
    return {"items": items}


async def folder_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    path = (args.get("path") or "").strip() or None
    items = list_items(path)
    stats = compute_stats(items)
    return {"path": path, "stats": stats}


async def folder_analyze(args: Dict[str, Any]) -> Dict[str, Any]:
    path = (args.get("path") or "").strip() or None
    items = list_items(path)
    stats = compute_stats(items)

    text = (
        f"Folder: {path or 'DATA_ROOT'}\n"
        f"Total items: {stats['total_items']}\n"
        f"Files: {stats['files']}\n"
        f"Folders: {stats['folders']}\n"
    )

    if stats["largest"]:
        text += f"Largest file: {stats['largest']['name']} ({stats['largest']['size']} bytes)\n"

    prompt = (
        "Analyze the following folder metadata and produce:\n"
        "- key observations\n"
        "- potential anomalies\n"
        "- recommended actions\n"
        "- short business-relevant summary\n\n"
        f"{text}"
    )

    summary = await _call_llm_for_summary(prompt)

    if not summary or not summary.strip():
        raise ValueError(f"Summary generation failed for folder: {path}")

    logger.info("Folder analysis completed for: %s", path or "DATA_ROOT")
    return {
        "path": path,
        "stats": stats,
        "summary": summary.strip(),
    }
