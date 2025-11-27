# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

# memory/memory_core.py
# Unified memory module for LLM-Agent Hub: short-term (messages) + long-term (longterm_facts)
# Compatible with calls from main.py:
# mem_save, mem_recent, build_context_snippet, maybe_remember, update_memory,
# remember_fact, get_longterm_facts, get_memory_summary, update_memory_fact,
# DB_PATH, mem_count_all, init_memory

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

from pathlib import Path
import os, sqlite3
from typing import List, Dict, Any, Tuple

from cryptography.fernet import Fernet, InvalidToken  # ← add if needed

ROOT = Path(__file__).resolve().parent
DB_NAME = "hub_memory.db"  # new neutral DB file name
DB_PATH = ROOT / DB_NAME

logger = logging.getLogger(__name__)

raw_key = os.getenv("HUB_ENCRYPT_KEY", "").strip()
if raw_key:
    try:
        FERNET = Fernet(raw_key.encode("utf-8"))
        ENCRYPTION_ENABLED = True
    except Exception as e:
        raise RuntimeError("Invalid HUB_ENCRYPT_KEY") from e
else:
    FERNET = None
    ENCRYPTION_ENABLED = False
    logger.warning("HUB_ENCRYPT_KEY not set: using plaintext mode.")

def _to_bytes(x) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("utf-8", errors="ignore")
    return str(x).encode("utf-8", errors="ignore")

def _enc(s: str) -> bytes:
    data = s.encode("utf-8")
    if ENCRYPTION_ENABLED:
        return FERNET.encrypt(data)
    return data

def _dec_maybe(b: bytes) -> Tuple[str, bool]:
    """
    Return (text, was_plaintext).
    If the record was unencrypted (legacy format), return the text and flag migration as needed.
    """
    data = _to_bytes(b)
    if ENCRYPTION_ENABLED:
        try:
            return FERNET.decrypt(data).decode("utf-8"), False
        except (InvalidToken, ValueError, TypeError):
            # Likely plaintext → return as is
            try:
                return data.decode("utf-8"), True
            except Exception:
                return str(data), True
    try:
        return data.decode("utf-8"), False
    except Exception:
        return str(data), False

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR   = PROJECT_ROOT / "memory"
MEMORY_DIR.mkdir(exist_ok=True)
DB_PATH      = MEMORY_DIR / "sofia_memory.db"   # single canonical path

# --- Schema -------------------------------------------------------------------
CREATE_MESSAGES_SQL = """
CREATE TABLE IF NOT EXISTS messages(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  role       TEXT NOT NULL,        -- 'user' | 'assistant' | 'system'
  content    TEXT NOT NULL,
  created_at TEXT NOT NULL         -- ISO timestamp
);
"""

CREATE_LONGTERM_SQL = """
CREATE TABLE IF NOT EXISTS longterm_facts(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  key        TEXT,
  value      TEXT,
  created_at TEXT NOT NULL,
  source     TEXT
);
"""

def _migrate_if_needed(cx: sqlite3.Connection) -> None:
    """Gentle migrations from legacy tables, if they still exist."""
    cur = cx.cursor()

    # mem_events → messages (if the old schema was ever used)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='mem_events'")
    if cur.fetchone():
        # Create messages if missing
        cur.execute(CREATE_MESSAGES_SQL)
        # Attempt to copy
        try:
            cur.execute("""
              INSERT INTO messages(session_id, role, content, created_at)
              SELECT COALESCE(session_id,'sasha_dev'),
                     COALESCE(role,'user'),
                     COALESCE(value, content),
                     COALESCE(created_at, datetime('now'))
              FROM mem_events
            """)
        except Exception:
            pass
        # cur.execute("DROP TABLE mem_events")  # optional

    # memory → longterm_facts (if another long-term table existed)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory'")
    if cur.fetchone():
        cur.execute(CREATE_LONGTERM_SQL)
        try:
            cur.execute("""
              INSERT INTO longterm_facts(key, value, created_at)
              SELECT key, value, COALESCE(created_at, datetime('now'))
              FROM memory
            """)
        except Exception:
            pass
        # cur.execute("DROP TABLE memory")  # optional

def _ensure_schema(cx: sqlite3.Connection):
    cx.execute("""
      CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY,
        session_id TEXT,
        role TEXT,
        content BLOB NOT NULL,
        created_at TEXT NOT NULL
      )
    """)
    cx.execute(CREATE_LONGTERM_SQL)
    _migrate_if_needed(cx)

def _open() -> sqlite3.Connection:
    cx = sqlite3.connect(DB_PATH)
    cx.row_factory = sqlite3.Row

    # Fast SQLite pragmas
    cx.execute("PRAGMA journal_mode=WAL;")
    cx.execute("PRAGMA synchronous=NORMAL;")
    cx.execute("PRAGMA temp_store=MEMORY;")
    cx.execute("PRAGMA mmap_size=268435456;")  # 256 MB

    _ensure_schema(cx)

    # Indexes for faster lookups/selections
    cx.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id DESC);")
    cx.execute("CREATE INDEX IF NOT EXISTS idx_longterm_created ON longterm_facts(created_at DESC);")

    return cx

# --- Short-term memory (chat) ----------------------------------------------

def mem_save(session_id: str, role: str, content: str) -> None:
    with sqlite3.connect(DB_PATH) as cx:
        _ensure_schema(cx)
        cx.execute(
            "INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,datetime('now'))",
            (session_id, role, _enc(content)),
        )
        cx.commit()

def mem_recent(session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as cx:
        cx.row_factory = sqlite3.Row
        _ensure_schema(cx)
        rows = cx.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()

        to_update: List[Tuple[bytes, int]] = []
        out: List[Dict[str, Any]] = []

        for r in rows:
            text, was_plain = _dec_maybe(r["content"])
            if was_plain:
                to_update.append((_enc(text), r["id"]))
            out.append({"role": r["role"], "content": text, "created_at": r["created_at"]})

        if to_update:
            for enc_blob, _id in to_update:
                cx.execute("UPDATE messages SET content=? WHERE id=?", (enc_blob, _id))
            cx.commit()

        return out

def mem_count_all() -> int:
    """Total chat log entries across all sessions."""
    with _open() as cx:
        return int(cx.execute("SELECT COUNT(*) FROM messages").fetchone()[0])

# --- Long-term memory (facts) -------------------------------------------

def add_longterm_fact(key: str, value: str, source: Optional[str] = None) -> None:
    """Insert a long-term fact into storage."""
    k = (key or "").strip()
    v = (value or "").strip()
    if not k or not v:
        return
    with _open() as cx:
        _ensure_schema(cx)
        cx.execute(
            "INSERT INTO longterm_facts(key, value, created_at, source) "
            "VALUES(?,?,datetime('now'), ?)",
            (k, v, source),
        )
        cx.commit()


def remember_fact(key: str, value: str) -> str:
    """Add a new fact to long-term memory."""
    k = (key or "").strip()
    v = (value or "").strip()
    if not k or not v:
        return "Nothing to remember."
    add_longterm_fact(k, v, source="manual")
    return f"Recorded: {k} → {v}"

def update_memory_fact(key: str, value: str) -> str:
    """Simple versioning policy: add a new record as the latest."""
    return remember_fact(key, value)

def update_memory(key: str, value: str) -> str:
    """Compatibility with main.py: alias for update_memory_fact."""
    return update_memory_fact(key, value)

def get_longterm_facts(limit: int = 200) -> List[Dict[str, Any]]:
    """Return long-term facts (latest first)."""
    with _open() as cx:
        cur = cx.execute(
            "SELECT key, value, created_at AS 'when' "
            "FROM longterm_facts ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

def get_memory_summary() -> str:
    """Short text summary of facts for user display."""
    facts = get_longterm_facts(limit=50)
    if not facts:
        return "Long-term memory is currently empty — only the base mission and initial profile are stored."
    lines = [f"- ({f['when']}) {f['key']}: {f['value']}" for f in facts[:20]]
    return "Short summary of your facts:\n" + "\n".join(lines)

# --- Utilities for the system prompt ------------------------------------------

def build_context_snippet(messages: List[Dict[str, str]], max_len: int = 8) -> str:
    """
    Build a compact snippet of the latest turns for the model.
    Expects a list of dicts with keys 'role' and 'content'.
    """
    if not messages:
        return ""
    items = messages[:max_len]
    lines = []
    for m in items:
        role = (m.get("role") or "user").strip()
        text = (m.get("content") or "").strip()
        if not text:
            continue
        lines.append(f"- {role}: {text}")
    return "\n".join(lines)

def maybe_remember(text: str) -> Optional[str]:
    """
    Simple auto-memory helper: reacts to 'remember ' or 'record '.
    Format: 'remember key: value'
    """
    if not text:
        return None
    t = text.strip()
    lower = t.lower()
    if lower.startswith("remember ") or lower.startswith("record ") or \
       lower.startswith("запомни ") or lower.startswith("зафиксируй "):
        body = t.split(" ", 1)[1].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            return remember_fact(k, v)
    return None

# --- Initial memory bootstrap ------------------------------------------

def init_memory() -> None:
    """
    Load data/initial_memory.json (if present) into long-term memory,
    marking profile/mission fields as facts. Safe to run multiple times.
    """
    data_file = PROJECT_ROOT / "data" / "initial_memory.json"
    if not data_file.exists():
        logger.warning("initial_memory.json not found; skipping longterm_facts bootstrap")
        return
    try:
        data = json.loads(data_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error loading initial_memory.json: %s", exc)
        return

    profile = data.get("profile") or {}
    mission = data.get("mission") or {}

    try:
        with _open() as cx:
            _ensure_schema(cx)
            cur = cx.cursor()
            # light idempotency: avoid duplicating identical records
            cur.execute("SELECT key, value FROM longterm_facts")
            existing = set((row["key"], row["value"]) for row in cur.fetchall())

            now = datetime.now().isoformat(timespec="seconds")
            rows = []
            for k, v in profile.items():
                kv = (f"profile.{k}", str(v))
                if kv not in existing:
                    rows.append((kv[0], kv[1], now))
            for k, v in mission.items():
                kv = (f"mission.{k}", str(v))
                if kv not in existing:
                    rows.append((kv[0], kv[1], now))

            if rows:
                cur.executemany(
                    "INSERT INTO longterm_facts(key, value, created_at) VALUES(?,?,?)",
                    rows,
                )
                cx.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Error bootstrapping longterm_facts from initial_memory.json: %s",
            exc,
        )
