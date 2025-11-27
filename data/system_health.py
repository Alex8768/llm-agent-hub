"""Core health reporting utilities."""

from __future__ import annotations

import os
from datetime import datetime, timezone


def check_core_status() -> dict:
    """Return basic runtime health metadata."""
    return {
        "core": "active",
        "time": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
