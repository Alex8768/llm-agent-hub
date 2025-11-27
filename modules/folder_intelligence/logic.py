# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from cognition.rag_ingest import DATA_DIR


def list_items(path: str | None = None) -> List[Dict[str, Any]]:
    base = Path(DATA_DIR)
    target = base if not path else (base / path)
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Folder not found: {path or DATA_DIR}")

    items: List[Dict[str, Any]] = []
    for p in target.iterdir():
        items.append(
            {
                "name": p.name,
                "path": str(p.relative_to(base)),
                "is_dir": p.is_dir(),
                "size": p.stat().st_size if p.is_file() else 0,
            }
        )

    return items


def compute_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    files = sum(1 for i in items if not i["is_dir"])
    dirs = sum(1 for i in items if i["is_dir"])
    largest = max(items, key=lambda x: x["size"], default=None)
    return {
        "total_items": total,
        "files": files,
        "folders": dirs,
        "largest": largest,
    }
