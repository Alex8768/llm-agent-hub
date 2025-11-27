# LEGACY MODULE — archived persona/experimental logic
# Not used by the LLM-Agent Hub runtime.
"""Simple affective layer for Sofia replies."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict

PRIMARY_MAP_PATH = Path(__file__).with_name("affect_map.json")
LEGACY_MAP_PATH = Path(__file__).with_name("emotion_map.json")
DEFAULT_STATE = "calm"
KNOWN_STATES = {"calm", "focus", "tired", "stressed", "energized"}


@lru_cache(maxsize=1)
def _load_map() -> Dict[str, Dict[str, str]]:
    path = PRIMARY_MAP_PATH if PRIMARY_MAP_PATH.exists() else LEGACY_MAP_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def infer_state(text: str, meta: Dict[str, str] | None = None) -> str:
    """
    Derive a coarse affective state from the latest user text.
    Uses quick heuristics based on message length, punctuation, and keywords.
    """
    if not text:
        return DEFAULT_STATE

    meta = meta or {}
    txt = text.strip()
    lower = txt.lower()
    length = len(txt)
    exclam = txt.count("!")
    question = txt.count("?")
    upper_ratio = _uppercase_ratio(txt)

    markers = {
        "tired": ("устал", "нет сил", "опустош", "как-то тяжело"),
        "stressed": ("горит", "срочно", "не понимаю", "ненавижу", "кошмар"),
        "focus": ("план", "структур", "шаги", "алгоритм", "техника"),
        "energized": ("ура", "круто", "готов", "огонь", "полёт", "люблю"),
    }
    for state, keys in markers.items():
        if any(k in lower for k in keys):
            return state

    if length > 900 or "..." in txt and "не понимаю" in lower:
        return "tired"

    if exclam >= 3 or upper_ratio > 0.35 or ("?!" in txt) or exclam + question >= 5:
        return "stressed"

    if "?" in txt and 200 < length < 800:
        return "focus"

    freq = float(meta.get("freq") or 0)
    if freq > 10 and length > 600:
        return "tired"

    if freq > 8 and (exclam or upper_ratio > 0.25):
        return "stressed"

    if length < 200 and exclam and "!" in txt:
        return "energized"

    return DEFAULT_STATE


def adapt_style(state: str, reply: str) -> str:
    """
    Apply lightweight style adaptation for the reply.
    - Prefix with friendly tag + emoji.
    - Clamp length.
    """
    if not reply:
        return reply

    load_high = False
    state_name = state
    if isinstance(state, dict):
        load_high = (state.get("sasha_load") or "").lower() == "high"
        state_name = state.get("affect_state") or state.get("state") or state_name

    state = state_name if state_name in KNOWN_STATES else DEFAULT_STATE
    conf = _load_map().get(state, {})
    prefix = conf.get("prefix")
    emoji = conf.get("emoji", "")
    max_length = int(conf.get("max_length") or 900)

    text = reply.strip()

    if load_high:
        # Бережный режим: при перегрузе сокращаем ответ, чтобы не перегружать пользователя.
        soft_limit = 300
        hard_prefix = "Саша, я отвечу коротко: "
        available = max(0, soft_limit - len(hard_prefix))
        if len(text) > available > 0:
            cut = text[:available]
            boundary = -1
            for sep in (". ", "! ", "? ", " "):
                pos = cut.rfind(sep)
                if pos > boundary:
                    boundary = pos
            if boundary > 0:
                cut = cut[:boundary].rstrip(" .!?,;:-")
            text = cut or text[:available].rstrip()
        text = f"{hard_prefix}{text}"

    if len(text) > max_length > 0:
        text = text[:max_length].rsplit(" ", 1)[0].rstrip() + "…"

    parts = []
    if prefix:
        parts.append(prefix.strip())
    parts.append(text)

    result = " ".join(parts).strip()
    if emoji:
        result = f"{emoji} {result}"
    return result


def _uppercase_ratio(text: str) -> float:
    total = sum(1 for ch in text if ch.isalpha())
    if not total:
        return 0.0
    upper = sum(1 for ch in text if ch.isupper())
    return upper / total
