from typing import Dict
import subprocess
import shlex


def run_shell_command(command: str, timeout: int = 30) -> Dict[str, str]:
    """
    Run a shell command and return a dict with stdout, stderr and returncode.
    This is a low-level helper used by API tool endpoints.
    """
    if not isinstance(command, str) or not command.strip():
        return {
            "stdout": "",
            "stderr": "Empty command",
            "returncode": "1",
        }

    try:
        args = shlex.split(command)
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "returncode": str(proc.returncode),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": e.stdout or "",
            "stderr": f"TimeoutExpired: {e}",
            "returncode": "124",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "stdout": "",
            "stderr": f"Exception: {e}",
            "returncode": "1",
        }
