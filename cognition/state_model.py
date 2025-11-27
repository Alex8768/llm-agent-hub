"""Assistant state tracking and heuristics."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT_DIR / "data" / "assistant_state.json"


def _default_state() -> Dict[str, object]:
    return {
        "current_focus": "assistant_core",
        "user_load": "medium",
        "assistant_mode": "mentor",
        "last_topics": [],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def load_state() -> Dict[str, object]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "sasha_load" in data and "user_load" not in data:
                data["user_load"] = data.pop("sasha_load")
            if "sofia_mode" in data and "assistant_mode" not in data:
                data["assistant_mode"] = data.pop("sofia_mode")
            if data.get("current_focus") == "sofia_bridge_core":
                data["current_focus"] = "assistant_core"
            data.setdefault("current_focus", "assistant_core")
            data.setdefault("user_load", "medium")
            data.setdefault("assistant_mode", "mentor")
            data.setdefault("last_topics", [])
            data.setdefault("last_updated", datetime.now(timezone.utc).isoformat())
            return data
    except Exception:
        pass
    return _default_state()


def save_state(state: Dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


_FOCUS_MAP = [
    ("money", ["money", "income", "revenue", "client"]),
    ("assistant_core", ["assistant", "bridge", "main.py", "codex", "architecture"]),
    ("content", ["video", "channel", "content", "youtube"]),
    ("rest", ["tired", "burned out", "not in the mood", "need rest"]),
]


def infer_focus(user_msg: str, current_focus: str) -> str:
    text = (user_msg or "").lower()
    for focus, keywords in _FOCUS_MAP:
        for kw in keywords:
            if kw in text:
                return focus
    return current_focus


def infer_user_load(user_msg: str, current_load: str) -> str:
    text = (user_msg or "").lower()
    if any(kw in text for kw in ["tired", "burned out", "no energy", "hate this", "mess"]):
        return "high"
    if any(kw in text for kw in ["fine", "calm", "step by step"]):
        return "medium"
    if any(kw in text for kw in ["rested", "full of energy"]):
        return "low"
    return current_load


def infer_assistant_mode(state: Dict[str, object]) -> str:
    focus = (state.get("current_focus") or "").lower()
    load = (state.get("user_load") or "").lower()
    if load == "high":
        return "gentle_friend"
    if focus == "assistant_core":
        return "architect"
    if focus == "money":
        return "mentor"
    if focus == "rest":
        return "gentle_friend"
    return "mentor"


_TOPIC_KEYWORDS = {
    "code": ["code", "main.py", "architecture"],
    "money": ["money", "income", "revenue"],
    "content": ["video", "channel", "content", "youtube"],
    "rest": ["tired", "rest", "burned out"],
}


def _extract_topics(user_msg: str) -> List[str]:
    text = (user_msg or "").lower()
    topics: List[str] = []
    for label, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            topics.append(label)
    return topics


def update_on_message(user_msg: str) -> Dict[str, object]:
    state = load_state()
    state["current_focus"] = infer_focus(user_msg, str(state.get("current_focus", "assistant_core")))
    state["user_load"] = infer_user_load(user_msg, str(state.get("user_load", "medium")))
    topics = deque((state.get("last_topics") or [])[:], maxlen=6)
    for topic in _extract_topics(user_msg):
        if topic in topics:
            topics.remove(topic)
        topics.append(topic)
    state["last_topics"] = list(topics)
    state["assistant_mode"] = infer_assistant_mode(state)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    return state
