# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from executor.planner import _call_llm_for_summary
from .logic import list_excels, read_excel

logger = getLogger(__name__)


async def excel_list(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"files": list_excels()}


async def excel_read(args: Dict[str, Any]) -> Dict[str, Any]:
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError("path is required")
    rows = read_excel(path)
    return {
        "path": path,
        "rows": rows[:50],
    }


async def excel_analyze(args: Dict[str, Any]) -> Dict[str, Any]:
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError("path is required")

    rows = read_excel(path)
    if not rows:
        raise ValueError(f"Excel file is empty or unreadable: {path}")

    text = "\n".join([", ".join(str(c) for c in r) for r in rows[:50]])
    prompt = (
        "Analyze the following spreadsheet data and extract key insights, trends, anomalies, "
        "and 3-5 business-relevant conclusions:\n\n"
        f"{text}"
    )

    summary = await _call_llm_for_summary(prompt)

    if not summary or not summary.strip():
        raise ValueError(f"Summary generation failed for: {path}")

    logger.info("Excel analysis completed for: %s", path)
    return {
        "path": path,
        "summary": summary.strip(),
    }
