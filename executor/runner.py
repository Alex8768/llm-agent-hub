# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from logging import getLogger

from .actions import ActionRegistry, ModuleLoader
from .engine import ExecutorEngine
from .models import ExecutionResult
from .planner import LLMPlanner, PlannerError
from .recovery import RecoveryStrategy

logger = getLogger(__name__)


async def run_task(user_goal: str) -> ExecutionResult:
    """
    High-level entry point:
    - build a shared ActionRegistry
    - load all modules (built-ins + external)
    - ask the LLMPlanner for a Plan
    - execute it with ExecutorEngine
    """
    registry = ActionRegistry()

    # Load all module actions into this registry
    loader = ModuleLoader(registry)
    loader.load_all()
    logger.warning("REGISTRY ACTIONS: %s", sorted(registry.actions.keys()))
    logger.info("ActionRegistry initialized with %d actions", len(registry.actions))

    planner = LLMPlanner(registry)

    try:
        plan = planner.make_plan(user_goal)
    except PlannerError as e:
        logger.error("Planner failed to build plan: %s", e)
        return ExecutionResult(
            success=False,
            steps=[],
            summary=f"Planner failed: {e}",
        )

    recovery = RecoveryStrategy()
    engine = ExecutorEngine(registry, recovery=recovery)

    return await engine.execute(plan)
