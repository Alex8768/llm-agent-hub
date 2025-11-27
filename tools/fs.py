from __future__ import annotations

from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _safe_path(path: str) -> Path:
    """Resolve path under project root and prevent escaping with .."""
    p = (PROJECT_ROOT / path).expanduser().resolve()
    if PROJECT_ROOT not in p.parents and p != PROJECT_ROOT:
        raise ValueError(f"Path {p} is outside of project root")
    return p


def read_text_file(path: str, encoding: str = "utf-8", max_bytes: int = 512 * 1024) -> Dict[str, str]:
    """
    Read a text file under the project root.

    Returns a dict with 'path', 'content', 'truncated' flags and 'error' if any.
    """
    try:
        p = _safe_path(path)
        if not p.is_file():
            return {"path": str(p), "content": "", "truncated": "false", "error": "Not a file"}
        data = p.read_bytes()
        truncated = False
        if len(data) > max_bytes:
            data = data[:max_bytes]
            truncated = True
        try:
            text = data.decode(encoding, errors="replace")
        except Exception as e:  # noqa: BLE001
            return {"path": str(p), "content": "", "truncated": "false", "error": f"Decode error: {e}"}
        return {"path": str(p), "content": text, "truncated": "true" if truncated else "false", "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"path": path, "content": "", "truncated": "false", "error": f"Exception: {e}"}


def list_dir(path: str = ".") -> Dict[str, List[Dict[str, str]]]:
    """
    List files in a directory under the project root.

    Returns a list of entries with 'name', 'is_dir', and 'size'.
    """
    try:
        p = _safe_path(path)
        if not p.exists() or not p.is_dir():
            return {"path": str(p), "entries": [], "error": "Not a directory"}
        entries: List[Dict[str, str]] = []
        for child in sorted(p.iterdir()):
            info = {
                "name": child.name,
                "is_dir": "true" if child.is_dir() else "false",
                "size": str(child.stat().st_size if child.is_file() else 0),
            }
            entries.append(info)
        return {"path": str(p), "entries": entries, "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"path": path, "entries": [], "error": f"Exception: {e}"}
