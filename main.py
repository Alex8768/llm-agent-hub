# LLM-Agent Hub
# Copyright (C) 2025  Aleksandr Ladygin
# Licensed under the GNU General Public License v3 or later.
#
# See the LICENSE file in the project root for full license information.

from __future__ import annotations

"""
main.py
Assistant service — core runtime.

Includes:
- FastAPI endpoints: /chat, /remember, /status, /ui
- Memory (memory_core.py)
- RAG (rag_ingest.py) with auto-indexing of data/ and watcher
- System prompt (assistant personality)
- Heartbeat status every 30 seconds
- Daily DB backup
- Guard + signature integrity checks
- Nightly reflection task (time-based)
"""

import argparse
import html
import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, time as dtime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, APIRouter, Request
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from chromadb import PersistentClient
from starlette.staticfiles import StaticFiles

from system_core_prompt import SOFIA_CORE_PROMPT
# from web_ui import router as web_ui_router
from data.system_health import check_core_status

# memory
from memory.memory_core import (
    mem_save, mem_recent,
    build_context_snippet, maybe_remember, update_memory,
    remember_fact, get_longterm_facts, get_memory_summary, update_memory_fact,
    DB_PATH, mem_count_all,
)

# RAG
from cognition.rag_ingest import ingest_all, rag_search, rag_count, DATA_DIR, CHROMA_DIR
from cognition.file_watcher import start_data_watch
from cognition.cog_engine import CogEngine
from cognition.reflection import run_nightly_reflection

# Guard
from guardian.guardian_core import startup_guard_check
from guardian.health_guard import HealthGuard
from guardian.identity_guard import get_identity, assert_identity_integrity

from models.engines import registry  # noqa: F401
from models.engines import engine_ollama  # noqa: F401
from models.engines import engine_openai  # noqa: F401
from models.engines.base import GenerateRequest
from models.engines.policies import select_engine
from cognition.context_pack import build_context_pack
from cognition.prompt_renderer import render_prompt_from_pack
from cognition.token_estimator import infer_pack_meta
from tools.shell import run_shell_command as run_shelL_command
from tools import read_text_file, list_dir
from executor.runner import run_task
from executor.models import ExecutionResult

health_guard = HealthGuard()

# ---------- Guard: core integrity before startup ----------
try:
    startup_guard_check()
    print("[GUARD] Core verified: all signatures OK.")
except Exception as e:
    print(f"[GUARD ERROR] {e}")
    raise SystemExit(1)

identity_info = get_identity()
assert_identity_integrity()

# ---------- Morning response from nightly reflection ----------
INSIGHT_LOG = Path("data/insight_log.json")


# ---------- basic paths/environment ----------
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

LOG_DIR = ROOT_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "bridge.log"


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    if getattr(_setup_logging, "_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    _setup_logging._configured = True
    return logger


_setup_logging()

LOGGER = logging.getLogger("sofia_bridge")

CONFIG_PATH = ROOT_DIR / "config.json"
try:
    CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    LOGGER.warning("Failed to load config.json: %s", exc)
    CONFIG = {}

ENGINE_CONFIG = CONFIG.get("engines", {})
ENABLE_LOCAL_LLM = bool(CONFIG.get("enable_local_llm", True))
os.environ["ENABLE_LOCAL_LLM"] = "1" if ENABLE_LOCAL_LLM else "0"


def _engine_default_model(engine_name: str) -> str:
    cfg = ENGINE_CONFIG.get(engine_name, {})
    if engine_name == "ollama":
        return cfg.get("default_model") or os.getenv("DEFAULT_OLLAMA_MODEL", "qwen2.5:3b-instruct")
    if engine_name == "api":
        return cfg.get("default_model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return cfg.get("default_model") or "gpt-4o-mini"


def _engine_default_options(engine_name: str) -> dict:
    cfg = ENGINE_CONFIG.get(engine_name, {})
    if engine_name == "ollama":
        num_ctx = cfg.get("num_ctx") or os.getenv("OLLAMA_NUM_CTX", "4096")
        return {"num_ctx": int(num_ctx)}
    return {}


def _local_llm_enabled() -> bool:
    val = os.environ.get("ENABLE_LOCAL_LLM", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}

BRIDGE_TOKEN = (os.getenv("BRIDGE_TOKEN", "") or "").strip()
print(f"[AUTH] BRIDGE_TOKEN loaded: {bool(BRIDGE_TOKEN)}")

def _auth_dep(x_token: str = Header(default="")):
    got = (x_token or "").strip()
    if not BRIDGE_TOKEN:
        raise HTTPException(status_code=500, detail="Token is not configured (.env)")
    if got != BRIDGE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


# ---------- Heartbeat ----------
START_TS = time.time()
BACKGROUND_TASKS: Set[asyncio.Task] = set()


def _register_task(task: asyncio.Task) -> asyncio.Task:
    BACKGROUND_TASKS.add(task)

    def _cleanup(finished: asyncio.Task):
        BACKGROUND_TASKS.discard(finished)

    task.add_done_callback(_cleanup)
    return task

def _safe_count_messages() -> int:
    try:
        if not hasattr(_safe_count_messages, "_printed"):
            print(f"[HEARTBEAT] DB_PATH → {DB_PATH}")
            _safe_count_messages._printed = True

        with sqlite3.connect(DB_PATH) as cx:
            cur = cx.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            cx.commit()
            cur.execute("SELECT COUNT(*) FROM messages;")
            return int(cur.fetchone()[0])
    except Exception as e:
        print("[HEARTBEAT] SQLite error:", e)
        return -1

def _status_line() -> str:
    uptime_min = int((time.time() - START_TS) / 60)
    msgs_count = _safe_count_messages()
    mem_total = msgs_count
    mem_ok = (mem_total >= 0)
    try:
        rag_docs = rag_count()
    except Exception:
        rag_docs = -1
    ok_mark = "ok ✓" if mem_ok else "degraded ⚠"
    ts = datetime.now().isoformat()
    return (f"[HEARTBEAT] {ok_mark} | uptime: {uptime_min}m | "
            f"memory records: {msgs_count} | memory(SQLite): {mem_total} | "
            f"RAG docs: {rag_docs} | {ts}")

async def _heartbeat():
    try:
        try:
            print(_status_line())
        except Exception as e:
            print("[HEARTBEAT] first tick error:", e)
        while True:
            try:
                print(_status_line())
            except Exception as e:
                print("[HEARTBEAT] error:", e)
            await asyncio.sleep(30)
    except asyncio.CancelledError:
        print("[HEARTBEAT] monitor stopped.")
        raise


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def _db_exec(fn):
    cx = sqlite3.connect(DB_PATH)
    cx.row_factory = sqlite3.Row
    try:
        return fn(cx)
    finally:
        cx.close()


NIGHTLY_REFLECTION_TIME = dtime(3, 0)


def _seconds_until(target: dtime) -> float:
    now = datetime.now()
    planned = datetime.combine(now.date(), target)
    if planned <= now:
        planned += timedelta(days=1)
    return (planned - now).total_seconds()


async def _nightly_reflection_loop():
    while True:
        await asyncio.sleep(_seconds_until(NIGHTLY_REFLECTION_TIME))
        try:
            result = await asyncio.to_thread(run_nightly_reflection)
            LOGGER.info("[NIGHTLY] %s", result)
        except Exception as exc:
            LOGGER.warning("[NIGHTLY] failed: %s", exc)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        # CSP for API stays open enough; UI enforces CSP via index.html
        return resp


# ---------- FastAPI ----------
app = FastAPI(title="Assistant Bridge")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "webui_static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["content-type", "authorization"],
)

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

router = APIRouter()


@router.get("/engines")
def engines_status():
    return {
        name: {"healthy": bool(eng.health())}
        for name, eng in registry.all_engines().items()
    }


app.include_router(router)


class ChatIn(BaseModel):
    message: str


class ChatOut(BaseModel):
    status: Literal["success", "error"]
    reply: str | None = None
    engine: str | None = None
    model: str | None = None
    error: str | None = None


# --- Heavy Executor DTOs ---
class ExecutorRequest(BaseModel):
    goal: str


class ExecutorStep(BaseModel):
    id: int
    action: str
    args: dict
    status: str
    result: Any | None = None
    error: str | None = None


class ExecutorResponse(BaseModel):
    status: Literal["success", "error"]
    success: bool
    steps: list[ExecutorStep]
    summary: str
    error: str | None = None


class AgentTaskIn(BaseModel):
    task: str
    max_steps: int | None = 8


class AgentStep(BaseModel):
    tool: str
    args: dict[str, Any]
    description: str


class AgentPlanOut(BaseModel):
    task: str
    steps: list[AgentStep]
    notes: str


class LearnIn(BaseModel):
    text: str


class LearnOut(BaseModel):
    committed: List[str]
    message: str


class BeliefView(BaseModel):
    id: str
    text: str
    source: str | None = None
    kind: str | None = None
    strength: float | None = None
    created_at: str | None = None


class BeliefList(BaseModel):
    items: List[BeliefView]

# === BEGIN RAG SEARCH DTOs ===
class SearchIn(BaseModel):
    query: str
    limit: int = 5


class SearchHit(BaseModel):
    source: str
    score: float
    snippet: str


class SearchOut(BaseModel):
    count: int
    hits: List[SearchHit]
# === END RAG SEARCH DTOs ===

ce = CogEngine()

# === BEGIN RAG SEARCH HELPERS ===
_tok_re = re.compile(r"[A-Za-z0-9_-]{2,}")


def _tokenize(text: str) -> set[str]:
    return set(_tok_re.findall((text or "").lower()))


def _make_snippet(doc: str, q: str, width: int = 260) -> str:
    words = sorted(_tokenize(q), key=len, reverse=True)
    pos = -1
    low = (doc or "").lower()
    for w in words:
        pos = low.find(w)
        if pos != -1:
            break
    if pos == -1:
        pos = 0
    start = max(0, pos - width // 2)
    end = min(len(doc), start + width)
    snip = (doc or "")[start:end]
    for w in words[:5]:
        if not w:
            continue
        snip = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", snip)
    snip = html.escape(snip, quote=False)
    snip = snip.replace("&lt;mark&gt;", "<mark>").replace("&lt;/mark&gt;", "</mark>")
    return snip
# === END RAG SEARCH HELPERS ===


@app.get("/api/status")
async def api_status():
    rag_count: Optional[int] = None
    try:
        client = PersistentClient(path=".chroma")
        rag_count = client.get_or_create_collection("kb_docs").count()
    except Exception:
        rag_count = None
    return JSONResponse({"detail": "✅ Assistant core is active", "rag_docs": rag_count})


@app.post("/api/chat", response_model=ChatOut, response_model_exclude_none=True)
async def api_chat(payload: ChatIn):
    """WebUI wrapper around the main /chat orchestrator."""
    msg = ChatMsg(session_id="webui", message=payload.message)
    raw = await chat(msg)
    if isinstance(raw, JSONResponse):
        return raw
    if not isinstance(raw, dict):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": "Invalid response from chat handler."},
        )

    if raw.get("status") != "success":
        error_msg = str(raw.get("error") or "Unable to process chat request.")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": error_msg},
        )

    reply = clean_reply(str(raw.get("reply") or ""))
    reply = strip_user_echo(reply, payload.message)
    reply = strip_task_header(reply)
    reply = extract_assistant_reply(reply)
    reply = soften_identity_phrases(reply)
    reply = soften_style_phrases(reply)
    return {
        "status": "success",
        "reply": reply or "…",
        "engine": raw.get("engine") or "",
        "model": raw.get("model") or "",
    }

@app.post("/api/agent/plan", response_model=AgentPlanOut)
async def agent_plan(payload: AgentTaskIn):
    """
    High-level agent planner returning a structured plan without executing tools.
    """
    max_steps = payload.max_steps or 8
    data = await _plan_with_llm(payload.task, max_steps=max_steps)
    steps: list[AgentStep] = []
    for raw in data.get("steps", []):
        if not isinstance(raw, dict):
            continue
        tool_name = str(raw.get("tool") or "").strip()
        if tool_name not in PLANNER_ALLOWED_TOOLS:
            continue
        steps.append(
            AgentStep(
                tool=tool_name,
                args=raw.get("args") or {},
                description=str(raw.get("description") or "").strip(),
            )
        )
    return AgentPlanOut(
        task=data.get("task", payload.task),
        steps=steps[:max_steps],
        notes=data.get("notes", ""),
    )


# === BEGIN /api/search (lexical over kb_docs) ===
@app.post("/api/search", response_model=SearchOut)
def api_search(payload: SearchIn) -> SearchOut:
    query = (payload.query or "").strip()
    if not query:
        return SearchOut(count=0, hits=[])
    q_tokens = _tokenize(query)
    if not q_tokens:
        return SearchOut(count=0, hits=[])

    client = PersistentClient(path=".chroma")
    collection = client.get_or_create_collection("kb_docs")
    records = collection.get(include=["documents", "metadatas"]) or {}
    docs = records.get("documents") or []
    metas = records.get("metadatas") or []

    scored: list[tuple[float, str, str]] = []
    for doc, meta in zip(docs, metas):
        if not doc:
            continue
        doc_tokens = _tokenize(doc)
        overlap = len(q_tokens & doc_tokens)
        if overlap == 0:
            continue
        score = overlap / (len(q_tokens) + 1e-9)
        source = (meta or {}).get("source") or ""
        snippet = _make_snippet(doc, query)
        scored.append((score, source, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)
    limit = max(1, min(int(payload.limit), 20))
    hits = [
        SearchHit(source=src, score=float(score), snippet=snippet)
        for score, src, snippet in scored[:limit]
    ]
    return SearchOut(count=len(hits), hits=hits)


@app.post("/api/executor/run", response_model=ExecutorResponse, response_model_exclude_none=True)
async def api_executor_run(req: ExecutorRequest):
    LOGGER.info("Executor run requested: %s", req.goal)
    try:
        result: ExecutionResult = await run_task(req.goal)
    except Exception as e:  # noqa: BLE001
        LOGGER.exception("Executor run failed: %s", e)
        return ExecutorResponse(
            status="error",
            success=False,
            steps=[],
            summary="Executor run failed.",
            error=str(e),
        )

    steps = [
        ExecutorStep(
            id=step.id,
            action=step.action,
            args=step.args,
            status=step.status,
            result=step.result,
            error=step.error,
        )
        for step in result.steps
    ]

    status = "success" if result.success else "error"
    error_msg = None if result.success else result.summary

    return ExecutorResponse(
        status=status,
        success=result.success,
        steps=steps,
        summary=result.summary,
        error=error_msg,
    )
# === END /api/search ===


@app.get("/api/state/summary")
def api_state_summary():
    belief_count: Optional[int] = None
    rag_docs: Optional[int] = None
    try:
        belief_count = PersistentClient(path=".chroma").get_or_create_collection("kb_docs").count()
    except Exception:
        belief_count = None
    try:
        rag_docs = rag_count()
    except Exception:
        rag_docs = None
    return {
        "counts": {
            "beliefs": belief_count,
            "rag_docs": rag_docs,
        },
        "last_reflection": "pending",
    }


@app.get("/api/state/beliefs", response_model=BeliefList)
def api_state_beliefs(query: str = "", limit: int = 20):
    beliefs = ce.rc.list_beliefs()
    q = (query or "").strip().lower()
    filtered = []
    for belief in beliefs:
        text = (belief.get("text") or "").lower()
        title = (belief.get("title") or "").lower()
        haystack = f"{title} {text}".strip()
        if q and q not in haystack:
            continue
        filtered.append(belief)
    limit = max(1, min(limit, 200))
    filtered = filtered[:limit]
    items: list[BeliefView] = []
    for belief in filtered:
        created_at = belief.get("created_at")
        created_iso: str | None
        if isinstance(created_at, (int, float)):
            created_iso = datetime.fromtimestamp(created_at).isoformat()
        else:
            created_iso = created_at if isinstance(created_at, str) else None
        strength_value = belief.get("strength")
        strength = float(strength_value) if strength_value is not None else None
        items.append(
            BeliefView(
                id=str(belief.get("id") or ""),
                text=str(belief.get("text") or ""),
                source=belief.get("source"),
                kind=belief.get("kind"),
                strength=strength,
                created_at=created_iso,
            )
        )
    return BeliefList(items=items)


@app.post("/api/learn", response_model=LearnOut)
def api_learn(payload: LearnIn):
    text = (payload.text or "").strip()
    if not text:
        return LearnOut(committed=[], message="empty input")
    committed = ce.learn(text=text, source="user", kind="manual")
    if not committed:
        return LearnOut(committed=[], message="nothing committed")
    return LearnOut(committed=committed, message=f"stored {len(committed)} belief(s)")


@app.get("/api/chat/stream")
async def chat_stream(message: str):
    return StreamingResponse(stream_generator(message), media_type="text/event-stream")

# --- DEBUG ROUTER ---
debug_router = APIRouter()


@debug_router.get("/ollama")
async def debug_ollama():
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{host}/api/version")
            ok = (r.status_code == 200)
    except Exception:
        ok = False
    return {"OLLAMA_HOST": host, "ollama_health": ok}


app.include_router(debug_router, prefix="/debug")
# --- /DEBUG ROUTER ---


# ---------- async-startup: background tasks ----------
@app.on_event("startup")
async def _on_startup_async():
    heartbeat_task = _register_task(asyncio.create_task(_heartbeat(), name="heartbeat"))
    print("[INIT] Heartbeat task started ✅")

    nightly_belief_task = _register_task(asyncio.create_task(_nightly_reflection_loop(), name="belief_consolidation"))
    app.state.task_belief_reflection = nightly_belief_task
    print("[INIT] Belief consolidation scheduled.")

    guard_task = health_guard.start()
    if guard_task:
        _register_task(guard_task)
    print("[INIT] Health guard task started.")

    app.state.background_tasks = BACKGROUND_TASKS


@app.on_event("startup")
async def _warmup():
    try:
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        async with httpx.AsyncClient(timeout=5) as cl:
            await cl.get(f"{host}/api/tags")
        LOGGER.info("[WARMUP] Ollama reachable")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"[WARMUP] Ollama not reachable: {e!s}")


# ---------- sync-startup: index, watcher, init memory, backup ----------
def _status_autocheck():
    import requests
    url = "http://127.0.0.1:8081/status"
    for _ in range(6):
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                s = r.json()
                mem_items = s.get("memory", {}).get("items", 0)
                rag_docs = s.get("rag", {}).get("docs_in_collection", 0)
                log(f"[INIT] Memory active: {mem_items} records.")
                log(f"[INIT] RAG active: {rag_docs} documents.")
                log("[INIT] Ready.")
                return
        except Exception:
            time.sleep(1)
    log("[INIT] Status not available yet, server is up.")

def _daily_backup_job():
    try:
        import shutil
        backup_dir = Path(__file__).resolve().parent / "memory" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d")
        dst = backup_dir / f"sofia_memory_{ts}.db"
        if not dst.exists():
            shutil.copy2(DB_PATH, dst)
            log(f"[BACKUP] Backup created: {dst.name}")
    except Exception as e:
        log(f"[BACKUP] ⚠ Backup error: {e}")

def _schedule_daily_backup():
    def _loop():
        while True:
            try:
                _daily_backup_job()
            except Exception as e:
                log(f"[BACKUP] error: {e}")
            time.sleep(24 * 60 * 60)
    threading.Thread(target=_loop, daemon=True).start()

@app.on_event("startup")
def on_startup():
    log("[INIT] LLM-Agent Hub startup...")

    # 1) one-time RAG indexing
    try:
        ingest_all()
    except Exception as e:
        log(f"[RAG] ❌ Initial indexing error: {e}")

    # 2) data folder watcher
    try:
        start_data_watch()
        log("[INIT] Data watcher active 👀")
    except Exception as e:
        log(f"[INIT] ⚠ Data watcher failed to start: {e}")

    # 2.5) load initial memory
    try:
        from memory.memory_core import init_memory
        init_memory()
        log("[RAG] 🧠 initial_memory.json loaded (system config).")
    except Exception as e:
        log(f"[RAG] ⚠ Error reading initial_memory.json: {e}")

    # 3) automatic status check
    threading.Thread(target=_status_autocheck, daemon=True).start()

    # 4) final messages
    log("[INIT] Hub online. Core initialized. ✨")
    log("[INIT] Ready 🚀")

    # 5) backup scheduler
    try:
        _schedule_daily_backup()
        log("[INIT] Daily backup scheduler active.")
    except Exception as e:
        log(f"[INIT] ⚠ Backup scheduler failed to start: {e}")


# ---------- shutdown ----------
@app.on_event("shutdown")
async def _on_shutdown_async():
    print("[SHUTDOWN] LLM-Agent Hub shutting down...")

    tasks = [t for t in list(BACKGROUND_TASKS) if not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    BACKGROUND_TASKS.difference_update(tasks)

    print("[SHUTDOWN] All background tasks stopped.")


# ---------- models ----------
class Command(BaseModel):
    action: str

class MemSave(BaseModel):
    session_id: str
    key: str
    value: str

class ChatMsg(BaseModel):
    session_id: str
    message: str
    engine: str | None = None


class ShellCommandIn(BaseModel):
    command: str
    timeout: int | None = 30


class ShellCommandOut(BaseModel):
    stdout: str
    stderr: str
    returncode: str


class FileReadIn(BaseModel):
    path: str
    encoding: str | None = "utf-8"
    max_bytes: int | None = 524288  # 512 KB


class FileReadOut(BaseModel):
    path: str
    content: str
    truncated: str
    error: str


class DirListIn(BaseModel):
    path: str | None = "."


class DirEntry(BaseModel):
    name: str
    is_dir: str
    size: str


class DirListOut(BaseModel):
    path: str
    entries: list[DirEntry]
    error: str

# --- Heavy Executor DTOs ---
class ToolResponse(BaseModel):
    status: Literal["success", "error"]
    result: Any | None = None
    error: str | None = None
    tool: str

class Command(BaseModel):
    action: str

class MemSave(BaseModel):
    session_id: str
    key: str
    value: str

class ChatMsg(BaseModel):
    session_id: str
    message: str
    engine: str | None = None

PLANNER_ALLOWED_TOOLS = {"run_shell", "read_file", "list_dir"}


async def _call_llm_for_planner(
    system_prompt: str,
    user_prompt: str,
    payload: dict[str, Any] | None = None,
) -> str:
    payload_dict = dict(payload or {})
    approx_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
    pack_meta = {
        "approx_tokens": approx_tokens,
        "estimated_tokens": approx_tokens,
        "doc_count": 0,
    }
    try:
        engine_choice, requested_model = select_engine(payload_dict, pack_meta)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Engine selection failed: {exc}") from exc

    payload_opts = payload_dict.get("options") or {}

    def _make_request(target_engine: str) -> tuple[str, GenerateRequest]:
        model = requested_model or _engine_default_model(target_engine)
        opts = {}
        opts.update(_engine_default_options(target_engine))
        opts.update(payload_opts)
        opts.setdefault("temperature", 0.1)
        if target_engine == "api":
            prompt_text = user_prompt
            req_system = system_prompt
        else:
            prompt_text = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()
            req_system = None
        return model, GenerateRequest(
            prompt=prompt_text,
            model=model or "",
            options=opts,
            stream=False,
            payload=payload_dict,
            system=req_system,
        )

    def _try_generate(engine_obj, req_obj):
        if not engine_obj:
            raise RuntimeError("engine unavailable")
        try:
            resp_obj = engine_obj.generate(req_obj)
            text_val = (resp_obj.text or "").strip()
            if not text_val:
                raise RuntimeError(f"{engine_obj.name} returned empty text")
            return text_val
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{engine_obj.name} failed: {exc}") from exc

    primary_name = engine_choice
    if primary_name == "ollama":
        if not _local_llm_enabled():
            raise HTTPException(status_code=400, detail="Local model is disabled by configuration")
        fallback_name = "api"
    else:
        fallback_name = "ollama" if _local_llm_enabled() else None
    primary_engine = registry.get(primary_name)
    fallback_engine = registry.get(fallback_name) if fallback_name else None
    allow_fallback = (payload_dict.get("engine") or "auto").strip().lower() == "auto"

    if not primary_engine or not primary_engine.health():
        if allow_fallback and fallback_engine and fallback_engine.health():
            primary_name, primary_engine = fallback_name, fallback_engine
            fallback_engine = None
            fallback_name = None
        else:
            raise HTTPException(status_code=503, detail="No healthy engines available")

    model_name, req_primary = _make_request(primary_name)
    try:
        return await asyncio.to_thread(_try_generate, primary_engine, req_primary)
    except Exception as primary_err:  # noqa: BLE001
        primary_error_msg = str(primary_err)
        if allow_fallback and fallback_engine and fallback_engine.health():
            fallback_model, req_fallback = _make_request(fallback_name)
            try:
                return await asyncio.to_thread(_try_generate, fallback_engine, req_fallback)
            except Exception as fallback_err:  # noqa: BLE001
                detail = {
                    "error": str(fallback_err),
                    "primary_error": primary_error_msg,
                    "attempted": [primary_name, fallback_name],
                }
                raise HTTPException(status_code=502, detail=detail) from fallback_err
        detail = {
            "error": primary_error_msg,
            "attempted": [primary_name],
            "fallback_tried": bool(fallback_engine),
        }
        raise HTTPException(status_code=502, detail=detail) from primary_err


async def _plan_with_llm(task: str, max_steps: int = 8) -> dict[str, Any]:
    """
    Use the shared LLM stack to produce a JSON plan for the given task.
    """
    max_steps = max(1, int(max_steps or 8))
    system_prompt = (
        "You are an action planner for LLM-Agent Hub.\n"
        "Available tools:\n"
        "- list_dir(path: str): list files under the project root.\n"
        "- read_file(path: str, max_bytes: int): read text files under the project root.\n"
        "- run_shell(command: str, timeout: int): execute shell commands.\n\n"
        "Given a user task, output a STRICT JSON object of the form:\n"
        "{\n"
        '  "task": "...",\n'
        '  "steps": [\n'
        '    {"tool": "list_dir", "args": {"path": "."}, "description": "..."}\n'
        "  ],\n"
        '  "notes": "..."\n'
        "}\n"
        "Each step must use one of the available tools.\n"
        f"Limit the number of steps to at most {max_steps}.\n"
        "Do not output anything outside of JSON."
    )
    prompt = f"User task:\n{task}\n\nReturn JSON plan:"
    try:
        text = await _call_llm_for_planner(system_prompt, prompt)
    except Exception as exc:  # noqa: BLE001
        return {
            "task": task,
            "steps": [],
            "notes": f"Planner failed to call LLM: {exc}",
        }
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Planner output is not a JSON object")
        data.setdefault("task", task)
        data.setdefault("steps", [])
        data.setdefault("notes", "")
        return data
    except Exception as exc:  # noqa: BLE001
        return {
            "task": task,
            "steps": [],
            "notes": f"Planner failed to produce valid JSON: {exc}",
        }



async def stream_generator(message: str):
    text = f"Assistant is thinking about: {message}"
    for token in text.split():
        yield f"data: {token} \n\n"
        await asyncio.sleep(0.25)
    yield "data: [END]\n\n"
    model: str | None = None


CSS_NOISE = ("{", "}", "px", "rem", "font-", "line-height", "color:", "display:", "min-height")


def strip_css_noise(text: str) -> str:
    lines: list[str] = []
    for ln in (text or "").splitlines():
        if any(tok in ln for tok in CSS_NOISE):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_reply(text: str) -> str:
    if not text:
        return ""
    cleaned = strip_css_noise(text)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def strip_task_header(text: str) -> str:
    """Remove leaked task headers/instructions at the start of a reply."""
    if not text:
        return ""
    working = text.lstrip()
    if not working.lower().startswith("### task"):
        return text

    lines = working.splitlines()
    drop_idx = 0
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("###") or stripped.startswith("-") or stripped[0].isdigit():
            drop_idx = idx + 1
            continue
        drop_idx = idx
        break
    remainder = "\n".join(lines[drop_idx:]).lstrip("\n ")
    return remainder


def extract_assistant_reply(text: str) -> str:
    if not text:
        return ""
    marker = "Assistant:"
    if marker in text:
        _, tail = text.rsplit(marker, 1)
        tail = tail.strip()
        if tail:
            return tail
    return text.strip()


_IDENTITY_PATTERNS = [
    (re.compile(r"\bi\s*am\s*(?:an?\s*)?(?:ai|artificial intelligence)\b", re.IGNORECASE),
     "I am your system assistant"),
    (re.compile(r"\bas\s+(?:an?\s+)?ai\s+(?:model|assistant)\b", re.IGNORECASE),
     "as your system assistant"),
]


def soften_identity_phrases(text: str) -> str:
    """Replace generic AI self-references with a warmer assistant persona."""
    if not text:
        return ""
    softened = text
    for pattern, replacement in _IDENTITY_PATTERNS:
        softened = pattern.sub(replacement, softened)
    return softened


def soften_style_phrases(text: str) -> str:
    """Rewrite formal greetings into a warmer assistant style."""
    if not text:
        return ""

    replacements = {
        "I am a virtual assistant": "I am your system assistant",
        "virtual assistant": "system assistant",
        "ready to help you": "ready to help",
        "How can I assist you today": "How can I help today",
        "How may I assist you today": "How may I help today",
    }

    softened = text
    for src, dst in replacements.items():
        softened = softened.replace(src, dst)

    softened = re.sub(r"ready\w* to help you", "ready to help", softened)

    stripped = softened.lstrip()
    lowered = stripped.lower()
    if lowered.startswith("hello!"):
        softened = "Hi!" + stripped[len("Hello!"):]
    elif lowered.startswith("good afternoon!"):
        softened = "Hi!" + stripped[len("Good afternoon!"):]

    return softened


def strip_user_echo(text: str, user_text: str) -> str:
    if not text or not user_text:
        return text
    t = text.strip()
    u = user_text.strip()
    if not u:
        return t
    variants = [
        u,
        f"— {u}",
        f"- {u}",
        f"--- {u}",
        f"— {u}?",
        f"- {u}?",
        f"--- {u}?",
        f"«{u}»",
        f"\"{u}\"",
        f"'{u}'",
    ]
    for v in variants:
        idx = t.find(v)
        if idx != -1 and idx <= 150:
            before = t[:idx].rstrip("-— –,: \n\t")
            after = t[idx + len(v):].lstrip(" -—–,:;.\n\t")
            t = (before + " " + after).strip()
            break
    return t


# ---------- endpoints ----------
@app.get("/auth/ping")
def auth_ping(_: None = Depends(_auth_dep)):
    return {"ok": True}

@app.get("/health")
def health():
    return check_core_status()

@app.post("/admin/backup")
def admin_backup(x_token: str = Header(default="")):
    _auth_dep(x_token)
    _daily_backup_job()
    return {"status": "ok"}

@app.post("/run")
def run(cmd: Command):
    import subprocess
    try:
        result = subprocess.check_output(cmd.action, shell=True, text=True).strip()
        return {"status": "success", "output": result}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remember")
def remember(mem: MemSave):
    try:
        mem_save(mem.session_id, mem.key, mem.value)
        return {"status": "success", "saved": {"session": mem.session_id, "key": mem.key}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mem/recent")
def mem_recent_api(session_id: str, limit: int = 5):
    try:
        items = mem_recent(session_id, limit)
        return {"status": "success", "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MemSearchBody(BaseModel):
    q: str
    limit: int = 50

@app.get("/mem/search")
def mem_search(q: str, limit: int = 50):
    with sqlite3.connect(DB_PATH) as cx:
        cx.row_factory = sqlite3.Row
        cur = cx.execute(
            "SELECT session_id, role, content, created_at "
            "FROM messages WHERE instr(content, ?) > 0 "
            "ORDER BY id DESC LIMIT ?",
            (q, int(limit)),
        )
        return {"status": "success", "items": [dict(r) for r in cur.fetchall()]}

@app.post("/mem/search_json")
def mem_search_json(body: MemSearchBody):
    with sqlite3.connect(DB_PATH) as cx:
        cx.row_factory = sqlite3.Row
        cur = cx.execute(
            "SELECT session_id, role, content, created_at "
            "FROM messages WHERE instr(content, ?) > 0 "
            "ORDER BY id DESC LIMIT ?",
            (body.q, int(body.limit)),
        )
        return {"status": "success", "items": [dict(r) for r in cur.fetchall()]}

@app.get("/mem/export")
def mem_export(_: None = Depends(_auth_dep)):
    with sqlite3.connect(DB_PATH) as cx:
        cx.row_factory = sqlite3.Row
        msgs  = [dict(r) for r in cx.execute("SELECT * FROM messages ORDER BY id")]
        facts = [dict(r) for r in cx.execute("SELECT * FROM longterm_facts ORDER BY id")]
    return {"status": "success", "messages": msgs, "facts": facts}

@app.post("/mem/trim")
def mem_trim(session_id: str, keep: int = 2000, _: None = Depends(_auth_dep)):
    with sqlite3.connect(DB_PATH) as cx:
        cx.execute(
            "DELETE FROM messages "
            "WHERE session_id = ? AND id NOT IN ("
            "  SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?"
            ")",
            (session_id, session_id, int(keep)),
        )
        cx.commit()
    return {"status": "success", "kept": keep, "session_id": session_id}

@app.get("/rag/search")
def rag_search_api(q: str, k: int = 4):
    ctxs = rag_search(q, k=k)
    return {"status": "success", "contexts": ctxs}


TOOL_OUTPUT_LIMIT = 2000
DANGEROUS_SHELL_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    ":(){",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "format c:",
    "poweroff",
]


def _truncate_tool_output(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if not text:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[output truncated]"


def _ensure_shell_command_safe(command: str) -> str:
    cleaned = (command or "").strip()
    if not cleaned:
        raise ValueError("Command is empty.")
    lowered = cleaned.lower()
    for pattern in DANGEROUS_SHELL_PATTERNS:
        if pattern in lowered:
            raise ValueError(f"Command contains disallowed pattern: {pattern}")
    return cleaned



@app.post("/api/tools/run_shell", response_model=ToolResponse, response_model_exclude_none=True)
async def api_run_shell(payload: ShellCommandIn):
    """
    Run a shell command on the host via LLM-Agent Hub.
    Use only from trusted clients.
    """
    tool_name = "run_shell"
    try:
        command = _ensure_shell_command_safe(payload.command)
        timeout = int(payload.timeout or 30)
        timeout = max(1, timeout)
        result = run_shelL_command(command, timeout=timeout)
        stdout = _truncate_tool_output(result.get("stdout") or "")
        stderr = _truncate_tool_output(result.get("stderr") or "")
        payload_result = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.get("returncode"),
        }
        return {
            "status": "success",
            "result": payload_result,
            "tool": tool_name,
        }
    except Exception as e:
        LOGGER.exception("Error in %s: %s", tool_name, e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Tool '{tool_name}' failed to execute.",
                "tool": tool_name,
            },
        )


@app.post("/api/tools/read_file", response_model=ToolResponse, response_model_exclude_none=True)
async def api_read_file(payload: FileReadIn):
    """
    Read a text file under the project root.
    """
    tool_name = "read_file"
    try:
        result = read_text_file(
            payload.path,
            encoding=payload.encoding or "utf-8",
            max_bytes=payload.max_bytes or 524288,
        )
        if result.get("error"):
            raise RuntimeError(result.get("error"))
        content = result.get("content") or ""
        truncated_flag = str(result.get("truncated")).lower() == "true"
        if len(content) > TOOL_OUTPUT_LIMIT:
            content = _truncate_tool_output(content)
            truncated_flag = True
        payload_result = {
            "path": result.get("path"),
            "content": content,
            "truncated": truncated_flag,
        }
        return {
            "status": "success",
            "result": payload_result,
            "tool": tool_name,
        }
    except Exception as e:
        LOGGER.exception("Error in %s: %s", tool_name, e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Tool '{tool_name}' failed to execute.",
                "tool": tool_name,
            },
        )


@app.post("/api/tools/list_dir", response_model=ToolResponse, response_model_exclude_none=True)
async def api_list_dir(payload: DirListIn):
    """
    List files under the given path (relative to project root).
    """
    tool_name = "list_dir"
    try:
        result = list_dir(payload.path or ".")
        if result.get("error"):
            raise RuntimeError(result.get("error"))
        payload_result = {
            "path": result.get("path"),
            "entries": result.get("entries") or [],
        }
        return {
            "status": "success",
            "result": payload_result,
            "tool": tool_name,
        }
    except Exception as e:
        LOGGER.exception("Error in %s: %s", tool_name, e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Tool '{tool_name}' failed to execute.",
                "tool": tool_name,
            },
        )


@app.post("/chat")
async def chat(msg: ChatMsg):
    """
    Orchestrator for the /chat endpoint:
    1) stores the user message in memory;
    2) handles special commands (manual fact updates, summary, longterm_facts view);
    3) builds context (dialogue history, facts, RAG retrieval);
    4) chooses engine and model via engines registry;
    5) generates a reply using the chosen engine;
    6) cleans reply text (CSS/HTML noise, etc.);
    7) stores assistant reply in memory;
    8) returns JSON response for the client.
    """
    try:
        sid = getattr(msg, "session_id", None) or "user_dev"
        payload_dict = msg.model_dump()
        requested_engine = (payload_dict.get("engine") or "").strip().lower()

        def _success_response(reply_text: str, engine_name: str, model_name: str) -> dict[str, str]:
            return {
                "status": "success",
                "reply": reply_text,
                "engine": engine_name,
                "model": model_name,
            }

        # 0) store user input
        try:
            mem_save(sid, "user", msg.message)
        except Exception as e:
            print("[MEM] save user failed:", e)

        lowered = (msg.message or "").lower().strip()

        # 0.1) manual memory update
        manual_update_reply = None
        if lowered.startswith(("assistant, update", "assistant, rewrite", "assistant, change")):
            try:
                after_cmd = msg.message.split(",", 1)[1].strip()
                if ":" in after_cmd:
                    key_part, value_part = after_cmd.split(":", 1)
                else:
                    key_part, value_part = after_cmd, ""
                key_part = (
                    key_part.replace("update", "")
                            .replace("rewrite", "")
                            .replace("change", "")
                            .replace("assistant", "")
                            .strip()
                )
                value_part = value_part.strip()
                if value_part:
                    try:
                        res = update_memory_fact(key_part, value_part)
                        manual_update_reply = f"[MEMCORE] 💾 {res}"
                    except Exception as e:
                        manual_update_reply = f"[MEMCORE] ⚠ Could not update memory: {e}"
                else:
                    manual_update_reply = "[MEMCORE] 🤔 Provide the new value after the colon."
            except Exception as e:
                manual_update_reply = f"[MEMCORE] ⚠ Could not parse update format: {e}"

        if manual_update_reply:
            return _success_response(manual_update_reply, "system", "system")

        # 0.2) memory summary
        if ("show my memory" in lowered) or ("what do you remember about me" in lowered):
            summary = get_memory_summary()
            return _success_response(summary, "system", "system")

        # 0.3) facts
        if any(p in lowered for p in [
            "what do you remember about me", "what do you know about me",
            "what are my goals", "what matters to me", "show me your notes about me",
            "show memory", "show your notes about me",
        ]):
            facts = get_longterm_facts()
            if not facts:
                return _success_response(
                    "There are no long-term notes yet beyond the base mission and goals.",
                    "system",
                    "system",
                )
            lines = []
            for f in facts:
                when = f.get("when", "recently")
                key  = f.get("key", "").capitalize()
                val  = f.get("value", "")
                lines.append(f"- ({when}) {key}: {val}")
            reply_text = "Here is what I currently track as important about you:\n" + "\n".join(lines) + \
                         "\n\nIf something is outdated, tell me and I'll update it."
            return _success_response(reply_text, "system", "system")

        # 0.4) auto memory capture
        try:
            maybe_remember(msg.message)
        except Exception as e:
            print(f"[MEMCORE] ⚠ maybe_remember error in chat: {e}")

        requested_engine = (payload_dict.get("engine") or "auto").strip().lower()
        token_budget = int(payload_dict.get("token_budget") or 3000)
        pack = build_context_pack(
            sid,
            msg.message,
            token_budget=token_budget,
        )
        pack_meta = infer_pack_meta(pack, msg.message)
        try:
            engine_choice, requested_model = select_engine(payload_dict, pack_meta)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Engine selection failed: {exc}") from exc

        prompt = render_prompt_from_pack(pack)
        payload_opts = payload_dict.get("options") or {}

        def _make_request(target_engine: str) -> tuple[str, GenerateRequest]:
            model = requested_model or _engine_default_model(target_engine)
            opts = {}
            opts.update(_engine_default_options(target_engine))
            opts.update(payload_opts)
            opts.setdefault("temperature", 0.2)
            return model, GenerateRequest(
                prompt=prompt,
                model=model or "",
                options=opts,
                stream=False,
                payload=payload_dict,
            )

        def _try_generate(engine_obj, req_obj):
            if not engine_obj:
                raise RuntimeError("engine unavailable")
            try:
                resp_obj = engine_obj.generate(req_obj)
                text_val = (resp_obj.text or "").strip()
                if not text_val:
                    raise RuntimeError(f"{engine_obj.name} returned empty text")
                return text_val, resp_obj
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"{engine_obj.name} failed: {exc}") from exc

        primary_name = engine_choice
        fallback_name: str | None
        if primary_name == "ollama":
            if not _local_llm_enabled():
                raise HTTPException(status_code=400, detail="Local model is disabled by configuration")
            fallback_name = "api"
        else:
            fallback_name = "ollama" if _local_llm_enabled() else None
        primary_engine = registry.get(primary_name)
        fallback_engine = registry.get(fallback_name) if fallback_name else None
        allow_fallback = requested_engine == "auto"

        if not primary_engine or not primary_engine.health():
            if allow_fallback and fallback_engine and fallback_engine.health():
                primary_name, primary_engine = fallback_name, fallback_engine
                fallback_engine = None
                fallback_name = None
            else:
                raise HTTPException(status_code=503, detail="No healthy engines available")

        model_name, req_primary = _make_request(primary_name)
        print(
            f"[ROUTE:/chat] payload.engine={payload_dict.get('engine')} -> chosen={primary_name} model={model_name}",
            flush=True,
        )

        reply_text = ""
        chosen_name = primary_name
        try:
            reply_text, _ = await asyncio.to_thread(_try_generate, primary_engine, req_primary)
        except Exception as primary_err:
            primary_error_msg = str(primary_err)
            if allow_fallback and fallback_engine and fallback_engine.health():
                fallback_model, req_fallback = _make_request(fallback_name)
                try:
                    reply_text, _ = await asyncio.to_thread(_try_generate, fallback_engine, req_fallback)
                    chosen_name = fallback_name
                    model_name = fallback_model
                except Exception as fallback_err:  # noqa: BLE001
                    detail = {
                        "error": str(fallback_err),
                        "primary_error": primary_error_msg,
                        "attempted": [primary_name, fallback_name],
                    }
                    raise HTTPException(status_code=502, detail=detail) from fallback_err
            else:
                detail = {
                    "error": primary_error_msg,
                    "attempted": [primary_name],
                    "fallback_tried": False,
                }
                raise HTTPException(status_code=502, detail=detail) from primary_err

        engine_used = chosen_name

        print(f"[SELECT] chosen={engine_used} model={model_name}", flush=True)

        reply_text = strip_css_noise(reply_text)
        print(f"[LLM:{engine_used}] text_len={len(reply_text)}", flush=True)

        # 5) store the reply
        try:
            mem_save(sid, "assistant", reply_text)
        except Exception as e:
            print("[MEM] save assistant failed:", e)

        return _success_response(reply_text, engine_used, model_name)
    except HTTPException as exc:
        LOGGER.exception("Error in /chat: %s", exc)
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error": detail or "Request failed.",
            },
        )
    except Exception as e:
        LOGGER.exception("Error in /chat: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "Internal error while processing chat request.",
            },
        )


@app.get("/status")
def status():
    try:
        def _mem_stats(cx):
            exists = cx.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            ).fetchone()
            if not exists:
                return 0, None
            cnt  = cx.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            last = cx.execute("SELECT MAX(created_at) FROM messages").fetchone()[0]
            return cnt, last

        mem_count, mem_last = _db_exec(_mem_stats)
        try:
            rag_docs = rag_count()
        except Exception:
            rag_docs = None
        files_count = len(list(DATA_DIR.glob("*"))) if DATA_DIR.exists() else 0

        return {
            "ok": True,
            "service": "sofia_bridge",
            "env": {"CHAT_MODEL": CHAT_MODEL, "EMBED_MODEL": EMBED_MODEL},
            "memory": {
                "db_path": str(DB_PATH),
                "items": mem_count,
                "last_created_at": mem_last,
            },
            "rag": {
                "collection": "kb_docs",
                "chroma_path": str(CHROMA_DIR),
                "docs_in_collection": rag_docs,
                "data_dir": str(DATA_DIR),
                "data_files_count": files_count,
            },
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as e:
        import traceback
        print("[STATUS ERROR]\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def _cli():
    parser = argparse.ArgumentParser(description="Assistant Bridge control flags")
    parser.add_argument(
        "--health",
        action="store_true",
        help="Print current core health JSON and exit",
    )
    args = parser.parse_args()

    if args.health:
        print(json.dumps(check_core_status(), ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
