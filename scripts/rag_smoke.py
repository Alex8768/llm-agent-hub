#!/usr/bin/env python3
"""Minimal RAG smoke test utility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognition.local_embedder import LocalEmbedder  # noqa: E402

CANDIDATE_COLLECTION_HINTS = ("kb", "sofia", "rag", "docs", "memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick RAG smoke test.")
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors to return")
    return parser.parse_args()


def find_chroma_dir() -> Path:
    for path in [ROOT / "data" / ".chroma", ROOT / ".chroma"]:
        if path.exists():
            return path
    raise FileNotFoundError("Chroma store not found (checked data/.chroma and .chroma)")


def pick_collection(client) -> chromadb.api.models.Collection.Collection:
    collections = client.list_collections()
    if not collections:
        raise RuntimeError("Chroma has no collections")
    by_name = {c.name: c for c in collections}
    for hint in CANDIDATE_COLLECTION_HINTS:
        for name, coll in by_name.items():
            if hint in name.lower():
                return coll
    # fallback: first collection
    return collections[0]


def run_smoke(query: str, top_k: int) -> None:
    chroma_path = find_chroma_dir()
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = pick_collection(client)
    print(f"[RAG] Using collection: {coll.name}")

    try:
        preview = coll.get(where_document={"$contains": query}, limit=min(3, top_k))
        print(f"[RAG] Current filter results: {len(preview.get('ids', []))}")
    except Exception as exc:
        print(f"[RAG] get() preview skipped: {exc}")

    embedder = LocalEmbedder()
    query_embedding = embedder.embed_query(query)
    results = coll.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        print("[RAG] No semantic matches found.")
        return

    print(f"[RAG] Top {len(docs)} matches:")
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        source = (meta or {}).get("source") if isinstance(meta, dict) else None
        snippet = (doc or "").strip().replace("\n", " ")[:200]
        print(f"{idx:02d}. dist={dist:.4f} source={source or '-'}")
        print(f"    {snippet}")


def main():
    args = parse_args()
    run_smoke(args.q, max(1, args.k))


if __name__ == "__main__":
    main()
