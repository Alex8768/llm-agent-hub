#!/usr/bin/env python3
"""Bulk-import dialog archives for the assistant runtime."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from html import unescape
from pathlib import Path
import re

from cognition.rag_ingest import ingest_all, rag_count

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMPORT_DIR = DATA_DIR / "memory_import"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import dialogs into RAG memory.")
    parser.add_argument("source", help="Path to folder/archive containing dialog files")
    return parser.parse_args()


def materialize_source(src: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_dir():
        return src
    tmp_dir = Path(tempfile.mkdtemp(prefix="assistant_import_"))
    shutil.unpack_archive(str(src), str(tmp_dir))
    return tmp_dir


def clean_text(raw: str) -> str:
    text = unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def copy_and_normalize(src_dir: Path, dst_dir: Path) -> None:
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="ignore")
        normalized = clean_text(text)
        target.write_text(normalized, encoding="utf-8")


def main():
    args = parse_args()
    src_path = Path(args.source).expanduser()
    origin = materialize_source(src_path)

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    copy_and_normalize(origin, IMPORT_DIR)

    before = rag_count()
    ingest_all()
    after = rag_count()
    print(f"[IMPORT] RAG docs: {before} -> {after} (+{after - before})")


if __name__ == "__main__":
    main()
