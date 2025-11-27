from __future__ import annotations


PRIORITY_BY_SOURCE = {
    "philosophy/assistant_white_book.md": 1.5,
}

PRIORITY_BY_KIND = {
    "belief": 1.0,
    "white_book": 1.4,
    "chat": 1.0,
    "web": 0.8,
}


class MemoryPolicy:
    def factor(self, source: str | None, kind: str | None) -> float:
        base = 1.0
        if source and source in PRIORITY_BY_SOURCE:
            base *= PRIORITY_BY_SOURCE[source]
        if kind and kind in PRIORITY_BY_KIND:
            base *= PRIORITY_BY_KIND[kind]
        return base

    def adjusted_strength(self, raw_strength: float, source: str | None, kind: str | None) -> float:
        return raw_strength * self.factor(source, kind)
