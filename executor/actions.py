# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

import importlib
import inspect
from logging import getLogger
from typing import Any, Awaitable, Callable
import importlib.util
import json
from pathlib import Path

from tools.fs import read_text_file, list_dir
from tools.shell import run_shell_command

logger = getLogger(__name__)

# Built-in action implementations ------------------------------------------------


async def action_read_file(args: dict) -> dict[str, Any]:
    path = args.get("path")
    if not path:
        raise ValueError("path is required for read_file")
    encoding = args.get("encoding") or "utf-8"
    max_bytes = args.get("max_bytes") or 524288
    return read_text_file(path, encoding=encoding, max_bytes=int(max_bytes))


async def action_list_dir(args: dict) -> dict[str, Any]:
    path = args.get("path") or "."
    return list_dir(path)


async def action_shell(args: dict) -> dict[str, Any]:
    command = args.get("command")
    if not command:
        raise ValueError("command is required for shell")
    timeout = int(args.get("timeout") or 30)
    return run_shell_command(command, timeout=timeout)


class ActionRegistry:
    def __init__(self):
        self.actions: dict[str, Callable[..., Any]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register("read_file", action_read_file)
        self.register("list_dir", action_list_dir)
        self.register("shell", action_shell)

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self.actions[name] = func

    def get(self, name: str) -> Callable[..., Any] | None:
        return self.actions.get(name)

    async def call(self, name: str, args: dict) -> Any:
        func = self.get(name)
        if not func:
            raise ValueError(f"Action not found: {name}")
        if not callable(func):
            raise TypeError(f"Action {name} is not callable")
        result = func(args)
        if inspect.iscoroutine(result) or isinstance(result, Awaitable):
            return await result  # type: ignore[arg-type]
        return result

    def load_module_actions(self, module_path: str) -> None:
        """Dynamically import a module and let it register its actions."""
        module = importlib.import_module(module_path)
        register_fn = getattr(module, "register_actions", None)
        if callable(register_fn):
            register_fn(self)
        else:
            logger.warning("Module %s has no register_actions; nothing registered.", module_path)


class ModuleLoader:
    def __init__(self, registry: ActionRegistry):
        self.registry = registry

    def load_all(self, modules_dir: str = "modules"):
        root = Path(modules_dir)
        logger.warning("MODULE LOADER scanning directory: %s", modules_dir)
        if not root.exists():
            logger.info("No modules directory found: %s", modules_dir)
            return

        for module_dir in root.iterdir():
            if not module_dir.is_dir():
                continue

            manifest_path = module_dir / "manifest.json"
            actions_path = module_dir / "actions.py"
            logger.warning("FOUND: %s", module_dir)
            logger.warning("MANIFEST: %s", manifest_path.exists())
            logger.warning("ACTIONS: %s", actions_path.exists())

            if not manifest_path.exists() or not actions_path.exists():
                logger.warning("Skipping module %s: missing manifest or actions.py", module_dir.name)
                continue

            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)

                module_name = manifest.get("name", module_dir.name)
                action_names = manifest.get("actions", [])

                spec = importlib.util.spec_from_file_location(
                    f"modules.{module_name}.actions",
                    actions_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]

                logger.info("Loaded module: %s", module_name)

                for action_name in action_names:
                    func = getattr(mod, action_name, None)
                    if callable(func):
                        self.registry.register(action_name, func)
                        logger.info("Registered action: %s.%s", module_name, action_name)
                        logger.warning("REGISTERED ACTION: %s", action_name)
                    else:
                        logger.warning("Missing callable for action: %s in module %s", action_name, module_name)

            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to load module %s: %s", module_dir.name, e)
