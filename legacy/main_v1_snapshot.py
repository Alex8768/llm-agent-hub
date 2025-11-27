"""
main.py
Sofia Bridge — v1 stable core (чистая сборка).

Что есть:
- FastAPI с /chat, /remember, /status, /ui
- Память (memory_core.py)
- RAG (rag_ingest.py) + автоиндексация data/ и вотчер
- Системный промпт (личность Софии)
- Heartbeat статус каждые 30 сек
- Ежедневный бэкап БД
- Guard + Signature целостности ядра
- Ночная рефлексия (по системному времени) — фоновой таск
"""

from __future__ import annotations

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
from typing import Any, Dict, List, Optional, Set

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

# ночная рефлексия (фоновая задача)
from behavior.self_reflector import schedule_reflection_daily
from behavior.insight_logger import collect_reflection
from behavior.affect_engine import infer_state, adapt_style

# память
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

health_guard = HealthGuard()

# ---------- Guard: целостность ядра до старта ----------
try:
    startup_guard_check()
    print("[GUARD] Ядро проверено: всё в порядке.")
except Exception as e:
    print(f"[GUARD ERROR] {e}")
    raise SystemExit(1)

identity_info = get_identity()
assert_identity_integrity()

# ---------- «Утренний отклик» по ночной рефлексии ----------
INSIGHT_LOG = Path("data/insight_log.json")
if INSIGHT_LOG.exists():
    try:
        _data = json.loads(INSIGHT_LOG.read_text(encoding="utf-8"))
        if isinstance(_data, list) and _data:
            last = _data[-1]
            t = last.get("time", "неизвестно когда")
            print(f"[REFLECT] 🌅 Доброе утро, Саша. Я перечитала ночные размышления ({t}). Всё сохранилось ясно.")
    except Exception as e:
        print(f"[REFLECT] Ошибка при чтении рефлексии: {e}")
else:
    print("[REFLECT] 🌅 Доброе утро, Саша. Новых записей ночной рефлексии пока нет.")


# ---------- базовые пути/окружение ----------
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

BRIDGE_TOKEN = (os.getenv("BRIDGE_TOKEN", "") or "").strip()
print(f"[AUTH] BRIDGE_TOKEN loaded: {bool(BRIDGE_TOKEN)}")

def _auth_dep(x_token: str = Header(default="")):
    got = (x_token or "").strip()
    if not BRIDGE_TOKEN:
        raise HTTPException(status_code=500, detail="Токен не настроен (.env)")
    if got != BRIDGE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в .env")

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
            print(f"[Пульс] DB_PATH → {DB_PATH}")
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
        print("[Пульс] SQLite error:", e)
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
    return (f"[Пульс] {ok_mark} | аптайм: {uptime_min}м | "
            f"записей в памяти: {msgs_count} | память(SQLite): {mem_total} | "
            f"RAG-доков: {rag_docs} | {ts}")

async def _heartbeat():
    try:
        try:
            print(_status_line())
        except Exception as e:
            print("[Пульс] ошибка первого тика:", e)
        while True:
            try:
                print(_status_line())
            except Exception as e:
                print("[Пульс] ошибка:", e)
            await asyncio.sleep(30)
    except asyncio.CancelledError:
        print("[Пульс] монитор остановлен.")
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
            LOGGER.info("[REFLECT] nightly: %s", result)
        except Exception as exc:
            LOGGER.warning("[REFLECT] nightly failed: %s", exc)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        # CSP for API stays open enough; UI enforces CSP via index.html
        return resp


# ---------- FastAPI ----------
app = FastAPI(title="Sofia Bridge")

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
    reply: str


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
_tok_re = re.compile(r"[A-Za-zА-Яа-я0-9_-]{2,}")


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
    return JSONResponse({"detail": "✅ София ядро активно", "rag_docs": rag_count})


@app.post("/api/chat", response_model=ChatOut)
async def api_chat(payload: ChatIn):
    """WebUI wrapper around the main /chat orchestrator."""
    msg = ChatMsg(session_id="webui", message=payload.message)
    raw = await chat(msg)
    if isinstance(raw, dict):
        candidate = raw.get("reply") or raw.get("message") or ""
    else:
        candidate = getattr(raw, "reply", None) or getattr(raw, "message", None) or str(raw)
    reply = clean_reply(str(candidate))
    reply = strip_user_echo(reply, payload.message)
    reply = extract_sofia_reply(reply)
    return ChatOut(reply=reply or "…")


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


# ---------- async-startup: фоны ----------
@app.on_event("startup")
async def _on_startup_async():
    heartbeat_task = _register_task(asyncio.create_task(_heartbeat(), name="heartbeat"))
    print("[INIT] Heartbeat task started ✅")

    reflection_task = _register_task(asyncio.create_task(schedule_reflection_daily(), name="reflection_scheduler"))
    app.state.task_reflection = reflection_task
    print("[INIT] Ночной планировщик рефлексии активен.")

    nightly_belief_task = _register_task(asyncio.create_task(_nightly_reflection_loop(), name="belief_consolidation"))
    app.state.task_belief_reflection = nightly_belief_task
    print("[INIT] Консолидация убеждений запланирована.")

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


# ---------- sync-startup: индекс, вотчер, init memory, бэкап ----------
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
                log(f"[INIT] Память активна: {mem_items} фактов.")
                log(f"[INIT] RAG содержит {rag_docs} документов. 📚")
                log("[INIT] Всё синхронизировано и готово к работе. 🚀")
                return
        except Exception:
            time.sleep(1)
    log("[INIT] Статус не успели прочитать сразу, но сервер жив.")

def _daily_backup_job():
    try:
        import shutil
        backup_dir = Path(__file__).resolve().parent / "memory" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d")
        dst = backup_dir / f"sofia_memory_{ts}.db"
        if not dst.exists():
            shutil.copy2(DB_PATH, dst)
            log(f"[BACKUP] Бэкап записан: {dst.name}")
    except Exception as e:
        log(f"[BACKUP] ⚠ Ошибка бэкапа: {e}")

def _schedule_daily_backup():
    def _loop():
        while True:
            try:
                _daily_backup_job()
            except Exception as e:
                log(f"[BACKUP] ошибка: {e}")
            time.sleep(24 * 60 * 60)
    threading.Thread(target=_loop, daemon=True).start()

@app.on_event("startup")
def on_startup():
    log("[INIT] Запуск Софии...")

    # 1) разовая индексация RAG
    try:
        ingest_all()
    except Exception as e:
        log(f"[RAG] ❌ Ошибка первичной индексации: {e}")

    # 2) вотчер папки data
    try:
        start_data_watch()
        log("[INIT] Наблюдатель за data активен 👀")
    except Exception as e:
        log(f"[INIT] ⚠ Не удалось запустить наблюдатель за data: {e}")

    # 2.5) загрузка начальной памяти
    try:
        from memory.memory_core import init_memory
        init_memory()
        log("[RAG] 🧠 initial_memory.json успешно загружен.")
    except Exception as e:
        log(f"[RAG] ⚠ Error reading initial_memory.json: {e}")

    # 3) автопроверка статуса
    threading.Thread(target=_status_autocheck, daemon=True).start()

    # 4) финальные сообщения
    log("[INIT] София активна. Личность и стиль загружены. ✨")
    log("[INIT] Готово 🚀")

    # 5) планировщик бэкапа
    try:
        _schedule_daily_backup()
        log("[INIT] Планировщик дневного бэкапа активен.")
    except Exception as e:
        log(f"[INIT] ⚠ Не удалось запустить бэкап: {e}")


# ---------- shutdown ----------
import atexit
atexit.register(collect_reflection)  # на всякий случай — лог при завершении процесса

@app.on_event("shutdown")
async def _on_shutdown_async():
    print("[SHUTDOWN] Завершение работы Софии...")

    tasks = [t for t in list(BACKGROUND_TASKS) if not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    BACKGROUND_TASKS.difference_update(tasks)

    print("[SHUTDOWN] Все фоновые задачи остановлены.")


# ---------- модели ----------
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


async def stream_generator(message: str):
    text = f"София думает над: {message}"
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


def extract_sofia_reply(text: str) -> str:
    if not text:
        return ""
    marker = "София:"
    if marker in text:
        _, tail = text.rsplit(marker, 1)
        tail = tail.strip()
        if tail:
            return tail
    return text.strip()


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


# ---------- эндпойнты ----------
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

@app.post("/chat")
async def chat(msg: ChatMsg):
    sid = getattr(msg, "session_id", None) or "sasha_dev"
    payload_dict = msg.model_dump()
    requested_engine = (payload_dict.get("engine") or "").strip().lower()

    # 0) сохраняем вход пользователя
    try:
        mem_save(sid, "user", msg.message)
    except Exception as e:
        print("[MEM] save user failed:", e)

    lowered = (msg.message or "").lower().strip()

    # 0.1) ручной апдейт памяти
    manual_update_reply = None
    if lowered.startswith(("софия, обнови", "софия, перепиши", "софия, измени")):
        try:
            after_cmd = msg.message.split(",", 1)[1].strip()
            if ":" in after_cmd:
                key_part, value_part = after_cmd.split(":", 1)
            else:
                key_part, value_part = after_cmd, ""
            key_part = (
                key_part.replace("обнови", "")
                        .replace("перепиши", "")
                        .replace("измени", "")
                        .replace("София", "")
                        .replace("софия", "")
                        .strip()
            )
            value_part = value_part.strip()
            if value_part:
                try:
                    res = update_memory_fact(key_part, value_part)
                    manual_update_reply = f"[MEMCORE] 💾 {res}"
                except Exception as e:
                    manual_update_reply = f"[MEMCORE] ⚠ Не смогла обновить память: {e}"
            else:
                manual_update_reply = "[MEMCORE] 🤔 Скажи после двоеточия то, что теперь актуально."
        except Exception as e:
            manual_update_reply = f"[MEMCORE] ⚠ Не смогла разобрать формат: {e}"

    if manual_update_reply:
        return {"status": "success", "reply": manual_update_reply, "sources": [], "engine": "system"}

    # 0.2) сводка памяти
    if ("покажи мою память" in lowered) or ("что ты обо мне помнишь" in lowered):
        summary = get_memory_summary()
        return {"status": "success", "reply": summary, "sources": [], "engine": "system"}

    # 0.3) факты
    if any(p in lowered for p in [
        "что ты помнишь обо мне", "что ты про меня помнишь", "что ты знаешь обо мне",
        "какие у меня цели", "что для меня важно", "что ты записала про меня",
        "покажи память", "покажи свои заметки про меня",
    ]):
        facts = get_longterm_facts()
        if not facts:
            return {"status": "success", "reply": ("Сейчас у меня нет зафиксированных долговременных заметок,"
                                                   " кроме базовой миссии и исходных целей. 💜"), "sources": [], "engine": "system"}
        lines = []
        for f in facts:
            when = f.get("when", "в прошлый раз")
            key  = f.get("key", "").capitalize()
            val  = f.get("value", "")
            lines.append(f"- ({when}) {key}: {val}")
        reply_text = "Вот что я держу как важное о тебе:\n" + "\n".join(lines) + \
                     "\n\nЕсли что-то неактуально — скажи, я обновлю."
        return {"status": "success", "reply": reply_text, "sources": [], "engine": "system"}

    # 0.4) автопамять
    try:
        maybe_remember(msg.message)
    except Exception as e:
        print(f"[MEMCORE] ⚠ Ошибка maybe_remember в чате: {e}")

    requested_engine = (payload_dict.get("engine") or "auto").strip().lower()
    token_budget = int(payload_dict.get("token_budget") or 3000)
    pack = build_context_pack(sid, msg.message, token_budget=token_budget)
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
    fallback_name = "api" if primary_name == "ollama" else "ollama"
    primary_engine = registry.get(primary_name)
    fallback_engine = registry.get(fallback_name)
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
        fallback_attempted = False
        if allow_fallback and fallback_engine and fallback_engine.health():
            fallback_attempted = True
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

    affect_meta = {
        "time": datetime.now().isoformat(),
        "freq": len(pack.get("dialogue") or []),
    }
    state = infer_state(msg.message, affect_meta)
    reply_text = adapt_style(state, reply_text)
    reply_text = strip_css_noise(reply_text)
    print(f"[LLM:{engine_used}] text_len={len(reply_text)}", flush=True)

    # 5) сохраняем ответ
    try:
        mem_save(sid, "assistant", reply_text)
    except Exception as e:
        print("[MEM] save assistant failed:", e)

    sources = list(dict.fromkeys([r.get("source", "") for r in pack.get("retrieval", []) if r.get("source")]))[:4]
    response = {
        "status": "success",
        "engine": engine_used,
        "model": model_name,
        "text": reply_text,
        "reply": reply_text,
    }
    if sources:
        response["sources"] = sources
    return response


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
    parser = argparse.ArgumentParser(description="Sofia Bridge control flags")
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
