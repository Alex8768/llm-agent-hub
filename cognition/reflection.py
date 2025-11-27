from __future__ import annotations

import time
from typing import Any, Dict

from chromadb import PersistentClient

DECAY_PER_DAY = 0.02
MIN_STRENGTH = 0.1


def run_nightly_reflection() -> Dict[str, Any]:
    client = PersistentClient(path=".chroma")
    col = client.get_collection("kb_docs")
    batch = col.get(include=["metadatas", "documents"])
    ids = batch.get("ids", []) or []
    metas = batch.get("metadatas", []) or []
    docs = batch.get("documents", []) or []
    now = int(time.time())

    decayed = 0
    removed = 0
    merged = 0

    by_hash: dict[str, list[tuple[str, dict, str]]] = {}
    for idx, mid in enumerate(ids):
        meta = metas[idx] or {}
        if meta.get("kind") != "belief":
            continue
        h = meta.get("hash")
        if not h:
            continue
        by_hash.setdefault(h, []).append((mid, meta, docs[idx]))

    for items in by_hash.values():
        for mid, meta, doc in items:
            strength = float(meta.get("strength", 1.0))
            created = int(meta.get("created_at", now))
            age_days = max(0.0, (now - created) / 86400.0)
            decay = max(0.0, 1.0 - DECAY_PER_DAY * age_days)
            new_strength = strength * decay
            if new_strength < MIN_STRENGTH:
                col.delete(ids=[mid])
                removed += 1
            else:
                if abs(new_strength - strength) > 1e-6:
                    meta["strength"] = new_strength
                    meta["updated_at"] = now
                    col.update(ids=[mid], metadatas=[meta], documents=[doc])
                    decayed += 1

    for h, items in by_hash.items():
        if len(items) <= 1:
            continue
        alive = []
        for mid, _, _ in items:
            try:
                res = col.get(ids=[mid], include=["metadatas", "documents"])
                if res.get("ids"):
                    alive.append((mid, res["metadatas"][0], res["documents"][0]))
            except Exception:
                continue
        if len(alive) <= 1:
            continue
        alive.sort(key=lambda x: float((x[1] or {}).get("strength", 1.0)), reverse=True)
        keeper_id, keeper_meta, keeper_doc = alive[0]
        to_merge = alive[1:]
        for mid, meta, _ in to_merge:
            keeper_meta["strength"] = float(keeper_meta.get("strength", 1.0)) + float(meta.get("strength", 1.0))
            keeper_meta["updated_at"] = now
            col.delete(ids=[mid])
            merged += 1
        col.update(ids=[keeper_id], metadatas=[keeper_meta], documents=[keeper_doc])

    return {"decayed": decayed, "removed": removed, "merged": merged, "ts": now}
