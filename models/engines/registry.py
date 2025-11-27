from __future__ import annotations

from typing import Dict, Optional, Tuple

from .base import LLMEngine

_REGISTRY: Dict[str, LLMEngine] = {}


def register(engine: LLMEngine) -> None:
    _REGISTRY[engine.name] = engine


def get(name: str) -> Optional[LLMEngine]:
    return _REGISTRY.get(name)


def all_engines() -> Dict[str, LLMEngine]:
    return dict(_REGISTRY)


def choose(names: Tuple[str, ...]) -> Optional[LLMEngine]:
    for name in names:
        engine = get(name)
        if engine and engine.health():
            return engine
    return None
