from __future__ import annotations

from typing import Dict, List

from .memory_policy import MemoryPolicy
from .resonance_core import ResonanceCore
from .truth_eval import TruthEvaluator


class CogEngine:
    def __init__(self):
        self.rc = ResonanceCore()
        self.te = TruthEvaluator()
        self.policy = MemoryPolicy()

    def observe(self, text: str) -> List[Dict]:
        parts = [p.strip() for p in (text or "").split("\n") if p.strip()]
        return [{"title": p[:80], "text": p} for p in parts]

    def learn(self, text: str, source: str = "user", kind: str = "manual") -> List[str]:
        facts = self.observe(text)
        committed: List[str] = []
        for fact in facts:
            fact.setdefault("source", source)
            fact.setdefault("kind", kind)
            claim = {"text": fact["text"], "source": fact.get("source"), "kind": fact.get("kind")}
            score = self.te.score(claim, ctx={})
            decision = self.te.decide(score)
            if decision == "commit":
                policy_factor = self.policy.factor(fact.get("source"), fact.get("kind"))
                bid = self.rc.upsert_belief(
                    fact["title"],
                    fact["text"],
                    source=fact.get("source"),
                    kind=fact.get("kind"),
                    strength_delta=policy_factor,
                )
                committed.append(bid)
        return committed

    def reflect(self) -> Dict:
        return {"status": "ok", "consolidated": 0}
