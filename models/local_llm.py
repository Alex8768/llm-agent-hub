# EXPERIMENTAL / OPENAI-SPECIFIC MODULE
# This module calls the OpenAI API directly for auxiliary tasks
# (token accounting, local model helpers, etc.).
# The core assistant runtime works through the unified engines/ layer
# and does not depend on this module.

import re
import logging
from typing import Literal, Tuple

from models.ollama_client import generate as ollama_generate, dry_ping as ollama_ok
from services.usage_meter import openai_chat

LOGGER = logging.getLogger("local_llm")
OLLAMA_LOG = logging.getLogger("ollama")

EngineCode = Literal["auto", "api"]
ActualEngine = Literal["api", "ollama-qwen", "ollama-phi"]

QWEN_MODEL = "qwen2.5:3b-instruct"
PHI_MODEL = "phi3:mini"


def _is_codey(text: str) -> bool:
    t = (text or "").lower()
    hot = (
        "```", "def ", "class ", "import ", "doctest", "pytest",
        "python", "sql", "dockerfile", "code",
    )
    if any(k in t for k in hot):
        return True
    return bool(re.search(r"\b(java(script)?|go|rust|c\+\+|c#|bash|zsh|powershell|sql|python)\b", t))


def _choose_ollama_model(prompt: str) -> Tuple[str, int]:
    if _is_codey(prompt):
        return PHI_MODEL, 2048
    return QWEN_MODEL, 4096


class AssistantLocal:
    def __init__(self):
        self.last_engine: ActualEngine = "api"

    def ask(self, message: str, engine: EngineCode) -> Tuple[str, ActualEngine]:
        if engine == "auto":
            if ollama_ok(QWEN_MODEL):
                model, num_ctx = _choose_ollama_model(message)
                try:
                    opts = {"num_ctx": num_ctx}
                    reply = ollama_generate(model=model, prompt=message, temperature=0.0, stream=False, options=opts)
                    self.last_engine = "ollama-phi" if model == PHI_MODEL else "ollama-qwen"
                    OLLAMA_LOG.info("[OLLAMA] model=%s num_ctx=%d ok", model, num_ctx)
                    return reply, self.last_engine
                except Exception as exc:
                    OLLAMA_LOG.warning(f"[FALLBACK] to=api cause=ollama_error err={exc!r}")
            else:
                OLLAMA_LOG.warning("[FALLBACK] to=api cause=ollama_unavailable")
            reply = openai_chat(message)
            self.last_engine = "api"
            return reply, self.last_engine

        reply = openai_chat(message)
        self.last_engine = "api"
        return reply, self.last_engine
