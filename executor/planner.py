# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

import json
import os
from logging import getLogger
from typing import Any

from models.engines import registry
from models.engines.base import GenerateRequest
from models.engines.policies import select_engine

from .actions import ActionRegistry
from .models import Plan

logger = getLogger(__name__)


class PlannerError(Exception):
    """Planner-level exception."""


class BasePlanner:
    def make_plan(self, user_goal: str) -> Plan:
        """Produce a Plan for the given user goal."""
        raise NotImplementedError


def _local_llm_enabled() -> bool:
    val = os.environ.get("ENABLE_LOCAL_LLM", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


def _engine_default_model(engine_name: str) -> str:
    if engine_name == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return os.getenv("CHAT_MODEL", "gpt-4o-mini")


def _engine_default_options(engine_name: str) -> dict[str, Any]:
    if engine_name == "ollama":
        ctx = os.getenv("OLLAMA_CTX")
        if ctx:
            return {"num_ctx": int(ctx)}
    return {}


def _get_action_list(registry: ActionRegistry) -> list[str]:
    return sorted(registry.actions.keys())


def _call_llm_for_plan(system_prompt: str, user_prompt: str, payload: dict[str, Any] | None = None) -> str:
    payload_dict = dict(payload or {})
    approx_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
    pack_meta = {
        "approx_tokens": approx_tokens,
        "estimated_tokens": approx_tokens,
        "doc_count": 0,
    }
    try:
        engine_choice, requested_model = select_engine(payload_dict, pack_meta)
    except Exception as exc:  # noqa: BLE001
        raise PlannerError(f"Engine selection failed: {exc}") from exc

    payload_opts = payload_dict.get("options") or {}

    def _make_request(target_engine: str) -> tuple[str, GenerateRequest]:
        model = requested_model or _engine_default_model(target_engine)
        opts = {}
        opts.update(_engine_default_options(target_engine))
        opts.update(payload_opts)
        opts.setdefault("temperature", 0.1)
        if target_engine == "api":
            prompt_text = user_prompt
            req_system = system_prompt
        else:
            prompt_text = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()
            req_system = None
        return model, GenerateRequest(
            prompt=prompt_text,
            model=model or "",
            options=opts,
            stream=False,
            payload=payload_dict,
            system=req_system,
        )

    def _try_generate(engine_obj, req_obj):
        if not engine_obj:
            raise RuntimeError("engine unavailable")
        resp_obj = engine_obj.generate(req_obj)
        text_val = (resp_obj.text or "").strip()
        if not text_val:
            raise RuntimeError(f"{engine_obj.name} returned empty text")
        return text_val

    primary_name = engine_choice
    fallback_name: str | None
    if primary_name == "ollama":
        if not _local_llm_enabled():
            raise PlannerError("Local model is disabled by configuration")
        fallback_name = "api"
    else:
        fallback_name = "ollama" if _local_llm_enabled() else None
    primary_engine = registry.get(primary_name)
    fallback_engine = registry.get(fallback_name) if fallback_name else None
    allow_fallback = (payload_dict.get("engine") or "auto").strip().lower() == "auto"

    if not primary_engine or not primary_engine.health():
        if allow_fallback and fallback_engine and fallback_engine.health():
            primary_name, primary_engine = fallback_name, fallback_engine
            fallback_engine = None
            fallback_name = None
        else:
            raise PlannerError("No healthy engines available")

    model_name, req_primary = _make_request(primary_name)
    try:
        return _try_generate(primary_engine, req_primary)
    except Exception as primary_err:  # noqa: BLE001
        primary_error_msg = str(primary_err)
        if allow_fallback and fallback_engine and fallback_engine.health():
            fallback_model, req_fallback = _make_request(fallback_name)
            try:
                return _try_generate(fallback_engine, req_fallback)
            except Exception as fallback_err:  # noqa: BLE001
                detail = {
                    "error": str(fallback_err),
                    "primary_error": primary_error_msg,
                    "attempted": [primary_name, fallback_name],
                }
                raise PlannerError(detail) from fallback_err
        detail = {
            "error": primary_error_msg,
            "attempted": [primary_name],
            "fallback_tried": bool(fallback_engine),
        }
        raise PlannerError(detail) from primary_err


async def _call_llm_for_summary(user_prompt: str, system_prompt: str | None = None) -> str:
    sys_prompt = system_prompt or "You are a concise summarizer."
    return _call_llm_for_plan(sys_prompt, user_prompt)


class LLMPlanner(BasePlanner):
    def __init__(self, registry: ActionRegistry):
        self.registry = registry

    def _build_system_prompt(self, actions: list[str]) -> str:
        example_action = actions[0] if actions else "list_dir"
        return (
            "You are a strict JSON planner for LLM-Agent Hub.\n"
            "Output ONLY valid JSON and nothing else. No prose, no markdown.\n"
            "Schema:\n"
            "{\n"
            '  "steps": [\n'
            '    {"id": 1, "action": "action_name", "args": {}}\n'
            "  ]\n"
            "}\n"
            "Available actions and their arguments:\n"
            "- pdf_list_docs: no arguments (lists available PDFs under data/).\n"
            "- pdf_read: requires \"path\": \"<relative path or filename>\".\n"
            "- pdf_summarize: requires \"path\": \"<relative path or filename>\".\n"
            "- excel_list: no arguments (lists available Excel files under data/).\n"
            "- excel_read: requires \"path\": \"<relative path or filename>\".\n"
            "- excel_analyze: requires \"path\": \"<relative path or filename>\".\n"
            "- folder_list: optional \"path\" to list files/folders under data/.\n"
            "- folder_stats: optional \"path\" for basic counts and largest file.\n"
            "- folder_analyze: use folder_list + folder_stats + produce summary for the folder.\n"
            f"Allowed actions: {', '.join(actions) if actions else 'none provided'}.\n"
            "Each step must include:\n"
            "- id: integer step id (1-based, increasing)\n"
            "- action: one of the allowed actions\n"
            "- args: object with arguments for that action\n"
            "Return ONLY JSON, no explanations.\n"
            "Example:\n"
            '{\"steps\": [{\"id\": 1, \"action\": \"%s\", \"args\": {\"path\": \".\"}}]}' % example_action
        )

    def _parse_plan(self, raw_text: str) -> Plan:
        try:
            data = json.loads(raw_text)
            plan = Plan(**data)
            return plan
        except Exception as exc:  # noqa: BLE001
            raise PlannerError(f"Invalid plan JSON: {exc}") from exc

    def make_plan(self, user_goal: str) -> Plan:
        actions = _get_action_list(self.registry)
        system_prompt = self._build_system_prompt(actions)
        user_prompt = f"User goal:\n{user_goal}\n\nReturn the JSON plan now."

        logger.debug("Planner sees %d actions: %s", len(self.registry.actions), sorted(self.registry.actions.keys()))
        raw_text = _call_llm_for_plan(system_prompt, user_prompt)
        logger.info("[PLANNER] raw plan text: %s", raw_text)
        try:
            plan = self._parse_plan(raw_text)
        except PlannerError:
            repair_prompt = (
                "The following JSON is invalid. Fix it to match the Plan schema. "
                "Output ONLY the corrected JSON:\n"
                f"{raw_text}"
            )
            logger.warning("[PLANNER] repairing invalid JSON plan")
            repaired_text = _call_llm_for_plan(system_prompt, repair_prompt)
            logger.info("[PLANNER] repaired plan text: %s", repaired_text)
            plan = self._parse_plan(repaired_text)

        return plan
