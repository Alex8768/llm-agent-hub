# guardian/guardian_core.py
import os
import logging
from fastapi import HTTPException
from .signature_engine import verify_core_signature

FORBIDDEN = (
    "rm -rf", "sudo ", "killall", "mkfs", "dd if=", ":(){:|:&};:",
    "shutdown", "reboot", "diskutil erase", "scutil --set",
    "networksetup", "launchctl remove", "chmod -R /", "chown -R /",
    "curl | sh", "wget | sh"
)

logger = logging.getLogger(__name__)

def allow_action(action: str) -> bool:
    a = (action or "").lower()
    return not any(bad in a for bad in FORBIDDEN)

def guard_or_raise(action: str):
    if not allow_action(action):
        raise HTTPException(status_code=403, detail="Action blocked by ethical guard.")

def startup_guard_check():
    ok, info = verify_core_signature()
    if not ok:
        strict = os.getenv("HUB_GUARD_STRICT", "0").strip().lower()
        is_strict = strict in ("1", "true", "yes")
        logger.error("[GUARD] Core signature mismatch: %s", info)
        if is_strict:
            raise RuntimeError(f"[GUARD] Core signature mismatch: {info}")
        logger.warning(
            "[GUARD] Soft mode: core mismatch ignored (HUB_GUARD_STRICT=0). "
            "Continuing startup at owner's risk."
        )
