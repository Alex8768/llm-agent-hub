from __future__ import annotations

from typing import Dict

from chromadb import PersistentClient


class TruthEvaluator:
    def __init__(self):
        self.client = PersistentClient(path=".chroma")
        try:
            self.col = self.client.get_collection("kb_docs")
        except Exception:
            self.col = None

    def score(self, claim: Dict, ctx: Dict) -> float:
        if claim.get("kind") == "manual":
            return 0.8
        consistency = self._consistency_with_beliefs(claim.get("text", ""))
        corroboration = 0.4 if claim.get("source") != "user" else 0.2
        recency = 0.5
        source_trust = 0.5 if claim.get("source") in {"white_book", "memory"} else 0.3
        contradiction_penalty = 0.0
        s = (
            0.35 * consistency
            + 0.25 * corroboration
            + 0.15 * recency
            + 0.15 * source_trust
            - 0.20 * contradiction_penalty
        )
        return max(0.0, min(1.0, s))

    def decide(self, score: float) -> str:
        if score >= 0.65:
            return "commit"
        if score >= 0.45:
            return "hypothesis"
        return "discard"

    def _consistency_with_beliefs(self, text: str) -> float:
        if not text.strip():
            return 0.5
        if not self.col:
            return 0.5
        try:
            result = self.col.query(
                query_texts=[text],
                n_results=5,
                include=["metadatas", "documents"],
            )
        except Exception:
            return 0.5

        metas = (result.get("metadatas") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        claim_lower = text.lower()

        alignment = False
        contradiction = False

        for meta, doc in zip(metas, docs):
            if not meta or meta.get("kind") != "belief":
                continue
            strength = float(meta.get("strength", 1.0))
            doc_lower = (doc or "").lower()
            if meta.get("conflicting"):
                if strength >= 1.0:
                    contradiction = True
                    break
            else:
                if strength >= 1.0 and (claim_lower in doc_lower or doc_lower in claim_lower):
                    alignment = True

        if contradiction:
            return 0.25
        if alignment:
            return 0.85
        return 0.5
