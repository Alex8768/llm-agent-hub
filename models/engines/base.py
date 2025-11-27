from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerateRequest:
    prompt: str
    model: str
    options: Optional[Dict[str, Any]] = None
    stream: bool = False
    system: Optional[str] = None
    stop: Optional[list[str]] = None
    payload: Optional[Dict[str, Any]] = None


@dataclass
class GenerateResponse:
    text: str
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


class LLMEngine:
    """Unified interface for all LLM engines."""

    name: str

    def health(self) -> bool:
        """Return True if engine is reachable."""
        raise NotImplementedError

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        """Synchronous text generation."""
        raise NotImplementedError

    def capabilities(self) -> Dict[str, Any]:
        return {}
