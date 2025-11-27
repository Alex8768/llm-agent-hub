# EXPERIMENTAL / OPENAI-SPECIFIC MODULE
# This module calls the OpenAI API directly for auxiliary tasks
# (token accounting, local model helpers, etc.).
# The core assistant runtime works through the unified engines/ layer
# and does not depend on this module.
"""Thin wrapper around cognition.usage_meter for simple chat calls."""

from __future__ import annotations

import os
from typing import List, Dict

from cognition.usage_meter import call_openai

API_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def openai_chat(message: str, temperature: float = 0.2, max_tokens: int = 256) -> str:
    """Send a single-turn prompt to OpenAI using the configured chat model."""
    prompt = (message or "").strip()
    if not prompt:
        return ""
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": prompt},
    ]
    return call_openai(
        messages=messages,
        model=API_MODEL,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
