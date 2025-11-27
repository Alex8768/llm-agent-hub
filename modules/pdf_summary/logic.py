# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from cognition.rag_ingest import DATA_DIR, extract_text


def list_pdfs() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not DATA_DIR.exists():
        return entries
    for file in DATA_DIR.glob("*.pdf"):
        if not file.is_file():
            continue
        stat = file.stat()
        entries.append(
            {
                "name": file.name,
                "path": str(file.relative_to(DATA_DIR)),
                "size": stat.st_size,
            }
        )
    return entries


def read_pdf(path: str) -> str:
    base = Path(DATA_DIR)
    raw = Path(path)
    if not raw.is_absolute():
        candidate = base / raw
        if candidate.is_file():
            return extract_text(candidate) or ""
        matches = list(base.rglob(raw.name))
        if len(matches) == 1 and matches[0].is_file():
            return extract_text(matches[0]) or ""
        if len(matches) > 1:
            raise FileNotFoundError(f"Multiple PDFs found matching name: {raw.name}")
        raise FileNotFoundError(f"PDF not found: {path}")
    if raw.is_file():
        return extract_text(raw) or ""
    raise FileNotFoundError(f"PDF not found: {path}")
