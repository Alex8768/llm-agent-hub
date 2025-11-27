# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

from typing import Any, Optional
from typing import Literal

from pydantic import BaseModel


class Step(BaseModel):
    id: int
    action: str
    args: dict
    status: Literal["pending", "running", "success", "error"] = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


class Plan(BaseModel):
    steps: list[Step]


class ExecutionResult(BaseModel):
    success: bool
    steps: list[Step]
    summary: str
