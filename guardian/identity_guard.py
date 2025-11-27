"""Digital identity guard for the assistant runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IDENTITY_PATH = DATA_DIR / "identity.json"
WATCH_PATHS = [
    ROOT_DIR / "main.py",
    ROOT_DIR / "system_core_prompt.py",
    ROOT_DIR / "cognition",
    ROOT_DIR / "cognition" / "resonant_thoughts.json",
    ROOT_DIR / "guardian",
]

logger = logging.getLogger("identity_guard")


def _sha256_of_path(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        h = hashlib.sha256()
        for p in sorted(path.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(path).as_posix().encode("utf-8"))
                h.update(hashlib.sha256(p.read_bytes()).digest())
        return h.hexdigest()
    return ""


def _collect_signature() -> Dict[str, str]:
    return {p.name: _sha256_of_path(p) for p in WATCH_PATHS}


def _platform_snapshot() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def get_identity() -> Dict[str, str]:
    if not IDENTITY_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "uuid": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": _platform_snapshot(),
            "core_signature": _collect_signature(),
        }
        IDENTITY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[IDENTITY] New digital identity created.")
        return data

    with open(IDENTITY_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def assert_identity_integrity() -> None:
    identity = get_identity()
    expected_sig = identity.get("core_signature", {})
    current_sig = _collect_signature()

    if expected_sig != current_sig:
        logger.info("[IDENTITY] Core changes detected: %s", _diff_signatures(expected_sig, current_sig))
    else:
        logger.info("[IDENTITY] Core signature matches.")


def _diff_signatures(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    diff = {}
    keys = set(old) | set(new)
    for key in keys:
        if old.get(key) != new.get(key):
            diff[key] = f"{old.get(key)} -> {new.get(key)}"
    return diff
