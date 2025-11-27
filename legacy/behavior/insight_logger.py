# LEGACY MODULE — archived persona/experimental logic
# Not used by the LLM-Agent Hub runtime.
# behavior/insight_logger.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, os

# Опционально: используем шифрование, если есть ключ
try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None  # шифрование будет отключено

LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "insight_log.json"


def _load_log() -> list[dict]:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_log(items: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_reflection(summary: str = "", meta: dict | None = None) -> None:
    """
    Сохраняет запись ночной рефлексии:
      - время в ISO-строке
      - summary (в открытую) или summary_enc (если есть SOFIA_ENCRYPT_KEY)
    ВАЖНО: никаких bytes в JSON — всё строками.
    """
    items = _load_log()

    rec: dict = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "summary": (summary or "").strip(),
        "meta": meta or {},
    }

    key = (os.getenv("SOFIA_ENCRYPT_KEY") or "").strip()
    if key and Fernet:
        try:
            f = Fernet(key.encode("utf-8"))
            token: bytes = f.encrypt(rec["summary"].encode("utf-8"))
            rec["summary_enc"] = token.decode("utf-8")  # ← было bytes, теперь str
            rec["summary"] = ""  # можно убирать «в открытую»
        except Exception as e:
            # не роняем процесс — пишем как есть без шифрования
            rec["meta"]["encrypt_error"] = str(e)

    items.append(rec)
    _save_log(items)
