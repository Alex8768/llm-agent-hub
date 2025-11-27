# EXPERIMENTAL / OPENAI-SPECIFIC MODULE
# This module calls the OpenAI API directly for auxiliary tasks
# (token accounting, local model helpers, etc.).
# The core assistant runtime works through the unified engines/ layer
# and does not depend on this module.
"""OpenAI usage metering and resilient call helpers."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

_openai_err = None
try:
    from openai import OpenAI
except Exception as _openai_err:  # noqa: F841
    OpenAI = None  # type: ignore

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOG_PATH = DATA_DIR / "usage_log.jsonl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF_SECONDS = 2

MODEL_PRICING = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4.1": {"prompt": 0.01, "completion": 0.03},
}

_client: OpenAI | None = None
_usage_totals: Dict[str, float] = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost_estimate": 0.0,
}

_logger = logging.getLogger("usage_meter")


def _get_client() -> OpenAI:
    global _client
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed. Install it to use the API."
        ) from _openai_err  # type: ignore
    if _client is None:
        _client = OpenAI()
    return _client


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING.get(model, {"prompt": 0.0, "completion": 0.0})
    pt = (prompt_tokens or 0) / 1000
    ct = (completion_tokens or 0) / 1000
    return pt * rates["prompt"] + ct * rates["completion"]


def _record_usage(entry: Dict[str, Any]) -> None:
    prompt_tokens = entry.get("pt") or entry.get("prompt_tokens") or 0
    completion_tokens = entry.get("ct") or entry.get("completion_tokens") or 0

    if not (prompt_tokens or completion_tokens):
        return

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _usage_totals["calls"] += 1
    _usage_totals["prompt_tokens"] += prompt_tokens or 0
    _usage_totals["completion_tokens"] += completion_tokens or 0
    _usage_totals["cost_estimate"] += entry.get("cost") or entry.get("cost_estimate") or 0.0


def call_openai(
    messages: Sequence[Dict[str, Any]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Call OpenAI chat completion with retries and usage tracking."""
    client = _get_client()
    backoff = BACKOFF_SECONDS
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                timeout=REQUEST_TIMEOUT,
            )
            choice = response.choices[0].message
            reply_text = choice.content or ""
            text = reply_text.strip()

            # --- extract usage safely (supports multiple SDK layouts)
            usage = getattr(response, "usage", None)

            def _u(obj, *names):
                """Return int counter from dataclass/object with possible alternative names."""
                if obj is None:
                    return 0
                for n in names:
                    v = getattr(obj, n, None)
                    if v is not None:
                        try:
                            return int(v)
                        except Exception:
                            try:
                                return int(float(v))
                            except Exception:
                                pass
                return 0

            prompt_tokens = _u(usage, "prompt_tokens", "input_tokens")
            completion_tokens = _u(usage, "completion_tokens", "output_tokens")
            total_tokens = _u(usage, "total_tokens")
            if total_tokens == 0 and (prompt_tokens or completion_tokens):
                total_tokens = prompt_tokens + completion_tokens

            cost = _estimate_cost(model, prompt_tokens, completion_tokens)

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "engine": "api",
                "model": model,
                "pt": prompt_tokens,
                "ct": completion_tokens,
                "cost": round(cost, 8),
            }
            _record_usage(entry)
            return text
        except Exception as exc:
            last_error = exc
            _logger.warning("[USAGE] OpenAI call failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                break
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} attempts: {last_error}")


def get_usage_summary() -> Dict[str, float]:
    """Return aggregated usage stats for the current runtime."""
    return {
        "calls": int(_usage_totals["calls"]),
        "prompt_tokens": int(_usage_totals["prompt_tokens"]),
        "completion_tokens": int(_usage_totals["completion_tokens"]),
        "cost_estimate": round(_usage_totals["cost_estimate"], 6),
        "log_path": str(LOG_PATH),
    }
