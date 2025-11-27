# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from typing import Literal, Optional
from typing import Dict

from pydantic import BaseModel

from .models import Step


class RecoveryDecision(BaseModel):
    action: Literal["retry", "skip", "fail"]
    reason: Optional[str] = None


class RecoveryStrategy:
    def __init__(self):
        self._retries: Dict[int, int] = {}

    def on_step_error(self, step: Step, error: Exception) -> RecoveryDecision:
        """
        Decide what to do when a step fails.
        Minimal implementation:
        - retry exactly once per step id
        - on second failure, fail
        """
        count = self._retries.get(step.id, 0)
        if count == 0:
            self._retries[step.id] = 1
            return RecoveryDecision(action="retry", reason=str(error))
        return RecoveryDecision(action="fail", reason=str(error))

    def describe_failure(self, steps: list[Step]) -> str:
        """Return a readable failure summary."""
        for step in steps:
            if step.status == "error":
                return f"Execution failed at step {step.id} ({step.action}): {step.error}"
        return "Execution failed."
