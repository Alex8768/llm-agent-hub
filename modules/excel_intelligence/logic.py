# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from cognition.rag_ingest import DATA_DIR


def list_excels() -> List[Dict[str, str]]:
    base = Path(DATA_DIR)
    files: List[Dict[str, str]] = []
    if not base.exists():
        return files
    for p in base.iterdir():
        if p.suffix.lower() in [".xlsx", ".xlsm"]:
            files.append(
                {
                    "name": p.name,
                    "path": str(p.relative_to(base)),
                    "size": p.stat().st_size,
                }
            )
    return files


def read_excel(path: str):
    base = Path(DATA_DIR)
    raw = Path(path)
    target = raw if raw.is_absolute() else base / raw
    if not target.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    try:
        import openpyxl  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError("openpyxl is required to read Excel files") from exc

    wb = openpyxl.load_workbook(target, data_only=True)
    sheet = wb.active

    rows = []
    for row in sheet.iter_rows(values_only=True):
        rows.append(list(row))
    return rows
