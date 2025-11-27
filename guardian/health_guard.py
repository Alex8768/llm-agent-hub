"""Background health guard that periodically snapshots core status."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from data.system_health import check_core_status

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEALTH_FILE = DATA_DIR / "health_status.json"


class HealthGuard:
    """Async task that persists health snapshots for external monitors."""

    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger("guardian.health")

    def start(self) -> Optional[asyncio.Task]:
        """Launch the guard loop if it is not already running."""
        if self._task and not self._task.done():
            return self._task
        self._logger.info("[GUARD] Health guard starting (interval=%ss).", self.interval_seconds)
        task = asyncio.create_task(self._run(), name="health_guard")
        task.add_done_callback(self._on_task_done)
        self._task = task
        return task

    async def stop(self) -> None:
        """Cancel the running guard loop and wait for a graceful stop."""
        task = self._task
        if not task:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._logger.info("[GUARD] Health guard stopped.")

    async def _run(self) -> None:
        """Append health snapshots to disk on a fixed cadence."""
        try:
            while True:
                try:
                    status = check_core_status()
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    with HEALTH_FILE.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(status, ensure_ascii=False) + "\n")
                    self._logger.info("[GUARD] Health snapshot written: %s", status)
                except Exception:
                    self._logger.exception("[GUARD] Failed to persist health snapshot.")

                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            self._logger.info("[GUARD] Health guard loop cancelled.")
            raise

    def _on_task_done(self, task: asyncio.Task) -> None:
        if self._task is task:
            self._task = None
