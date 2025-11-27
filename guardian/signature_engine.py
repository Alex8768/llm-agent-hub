# guardian/signature_engine.py
from pathlib import Path
import hashlib, json

CORE_FILES = [
    "guardian/meta_rules.json",
    "main.py",
    "system_core_prompt.py",
    "memory/memory_core.py",
]

SIG_FILE = Path(__file__).parent / "core_signature.json"

def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def current_signature():
    base = Path(__file__).resolve().parents[1]
    data = {}
    for rel in CORE_FILES:
        p = base / rel
        if p.exists():
            data[rel] = _sha256(p)
    return data

def verify_core_signature():
    cur = current_signature()
    if not SIG_FILE.exists():
        SIG_FILE.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        return True, "signature initialized"
    saved = json.loads(SIG_FILE.read_text(encoding="utf-8"))
    if saved == cur:
        return True, "signature ok"
    return False, {"saved": saved, "current": cur}

def reseal_signature():
    SIG_FILE.write_text(json.dumps(current_signature(), ensure_ascii=False, indent=2), encoding="utf-8")
    return True
