# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from logging import getLogger

from .actions import ActionRegistry
from .models import ExecutionResult, Plan
from .recovery import RecoveryStrategy

logger = getLogger(__name__)


class ExecutorEngine:
    def __init__(self, registry: ActionRegistry, recovery: RecoveryStrategy | None = None):
        self.registry = registry
        self.recovery = recovery or RecoveryStrategy()

    async def execute(self, plan: Plan) -> ExecutionResult:
        for step in plan.steps:
            logger.info("Executing step %s: %s", step.id, step.action)
            step.status = "running"
            try:
                result = await self.registry.call(step.action, step.args)
                step.result = result
                step.status = "success"
                logger.info("Step %s succeeded", step.id)
            except Exception as exc:  # noqa: BLE001
                step.error = str(exc)
                step.status = "error"
                logger.warning("Step %s failed: %s", step.id, exc)
                decision = self.recovery.on_step_error(step, exc)
                if decision.action == "retry":
                    logger.info("Retrying step %s once due to error: %s", step.id, exc)
                    try:
                        result = await self.registry.call(step.action, step.args)
                        step.result = result
                        step.status = "success"
                        logger.info("Step %s succeeded after retry", step.id)
                        continue
                    except Exception as exc_final:  # noqa: BLE001
                        step.error = str(exc_final)
                        step.status = "error"
                        logger.error("Step %s failed after retry: %s", step.id, exc_final)
                        break
                elif decision.action == "skip":
                    logger.info("Skipping step %s after failure: %s", step.id, exc)
                    continue
                else:  # fail
                    logger.error("Stopping execution at step %s due to failure.", step.id)
                    break

        success = all(s.status == "success" for s in plan.steps)
        if success:
            summary = "All steps executed successfully."
        else:
            summary = self.recovery.describe_failure(plan.steps)
        logger.info("Execution summary: %s", summary)
        return ExecutionResult(success=success, steps=plan.steps, summary=summary)
