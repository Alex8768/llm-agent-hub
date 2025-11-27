from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from typing import List

from chromadb import PersistentClient

from .local_embedder import LocalEmbedder


def _canon(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


class ResonanceCore:
    def __init__(self, path: str = ".chroma"):
        self.client = PersistentClient(path=path)
        self.col = self.client.get_or_create_collection("kb_docs")
        self.emb = LocalEmbedder()
        self.links_path = os.path.join("memory", "belief_links.db")
        self._init_links_db()

    def _init_links_db(self) -> None:
        os.makedirs("memory", exist_ok=True)
        with sqlite3.connect(self.links_path) as cx:
            cx.execute(
                """
                CREATE TABLE IF NOT EXISTS links (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (src, dst, kind)
                )
                """
            )
            cx.commit()

    def compute_signature(self, text: str) -> dict:
        ct = _canon(text)
        vec = self.emb.embed(ct)
        sig_source = "|".join([ct[:128], str(sum(vec))])
        h = hashlib.sha256(sig_source.encode()).hexdigest()[:16]
        return {"hash": h, "len": len(ct), "sum": float(sum(vec))}

    def match(self, text: str, k: int = 5) -> List[str]:
        sig = self.compute_signature(text)
        res = self.col.get(where={"hash": sig["hash"]}, include=["metadatas"])
        if not res:
            return []
        ids = res.get("ids") or []
        return ids[:k]

    def upsert_belief(
        self,
        title: str,
        text: str,
        *,
        source: str | None = None,
        kind: str | None = "belief",
        strength_delta: float = 1.0,
    ) -> str:
        sig = self.compute_signature(text)
        can_text = _canon(text)
        now = int(time.time())
        ids = self.match(text, k=3)
        meta = {
            "kind": kind or "belief",
            "title": title,
            "hash": sig["hash"],
            "strength": strength_delta,
            "updated_at": now,
            "source": source,
        }
        if ids:
            bid = ids[0]
            existing = self.col.get(ids=[bid], include=["metadatas"])
            old_meta = {}
            if existing and existing.get("metadatas"):
                old_meta = existing["metadatas"][0] or {}
                old_strength = float(old_meta.get("strength", 1.0))
            else:
                old_strength = 1.0
            meta["strength"] = old_strength + strength_delta
            meta["created_at"] = old_meta.get("created_at", now)
            self.col.update(ids=[bid], metadatas=[meta], documents=[can_text])
            return bid

        bid = f"belief_{sig['hash']}"
        meta["created_at"] = now
        self.col.add(ids=[bid], documents=[can_text], metadatas=[meta])
        return bid

    def link(self, src_id: str, dst_id: str, kind: str = "related", weight: float = 1.0) -> None:
        if not src_id or not dst_id:
            return
        with sqlite3.connect(self.links_path) as cx:
            cx.execute(
                """
                INSERT INTO links (src, dst, kind, weight)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(src, dst, kind)
                DO UPDATE SET weight=excluded.weight
                """,
                (src_id, dst_id, kind, weight),
            )
            cx.commit()

    def neighbors(self, belief_id: str, limit: int = 20) -> List[dict]:
        if not belief_id:
            return []
        with sqlite3.connect(self.links_path) as cx:
            rows = cx.execute(
                """
                SELECT dst, kind, weight FROM links WHERE src=?
                UNION ALL
                SELECT src, kind, weight FROM links WHERE dst=?
                LIMIT ?
                """,
                (belief_id, belief_id, limit),
            ).fetchall()
        return [{"id": row[0], "kind": row[1], "weight": row[2]} for row in rows]

    def list_beliefs(self) -> List[dict]:
        try:
            batch = self.col.get(include=["documents", "metadatas"])
        except Exception:
            return []
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        beliefs = []
        for bid, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            if meta.get("kind") != "belief":
                continue
            entry = {
                "id": bid,
                "text": doc or "",
                "title": meta.get("title") or "",
                "source": meta.get("source"),
                "kind": meta.get("kind"),
                "strength": meta.get("strength"),
                "created_at": meta.get("created_at"),
                "hash": meta.get("hash"),
            }
            beliefs.append(entry)
        return beliefs
