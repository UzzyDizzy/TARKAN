"""Knowledge-graph backend over English ConceptNet 5.7 + SenticNet 7 (paper §3.4).

Two backends behind one API:
  - in-memory (dict): fast, used by unit tests and small KGs.
  - sqlite: built by scripts/build_kg.py from the filtered parquet dumps; scales to
    the full English ConceptNet without loading 10 GB into RAM.

Public API used by kg_retrieval.py:
    kg.normalize(term) -> str
    kg.neighbors(term, top, sources) -> list[Triple]
    kg.polarity(term) -> float | None
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from config import CONFIG


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    weight: float
    source: str  # 'conceptnet' | 'senticnet'


_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(term: str) -> str:
    """Canonicalize a surface term to a ConceptNet-style key (lowercase, _-joined)."""
    term = term.strip().lower()
    term = _NORM_RE.sub("_", term).strip("_")
    return term


class KnowledgeGraph:
    def __init__(
        self,
        triples: Optional[Iterable[Tuple[str, str, str, float, str]]] = None,
        polarities: Optional[Dict[str, float]] = None,
        sqlite_path: Optional[str] = None,
    ):
        self._mem: Dict[str, List[Triple]] = defaultdict(list)
        self._pol: Dict[str, float] = {}
        self._conn: Optional[sqlite3.Connection] = None

        if sqlite_path is not None:
            self._conn = sqlite3.connect(str(sqlite_path))
        if triples is not None:
            self.add_triples(triples)
        if polarities:
            for k, v in polarities.items():
                self._pol[normalize(k)] = float(v)

    # ---- build (in-memory) ----
    def add_triples(self, triples: Iterable[Tuple[str, str, str, float, str]]) -> None:
        for h, r, t, w, src in triples:
            hk = normalize(h)
            tr = Triple(head=hk, relation=str(r), tail=str(t), weight=float(w), source=str(src))
            self._mem[hk].append(tr)
            # index by tail too (undirected lookup for commonsense neighbours)
            self._mem[normalize(t)].append(
                Triple(head=normalize(t), relation=str(r), tail=str(h), weight=float(w), source=str(src))
            )

    def set_polarity(self, concept: str, value: float) -> None:
        self._pol[normalize(concept)] = float(value)

    # ---- query ----
    def neighbors(self, term: str, top: int = None, sources=None) -> List[Triple]:
        top = top or CONFIG.top_m_triples
        key = normalize(term)
        sources = tuple(sources) if sources else None
        rows: List[Triple]
        if self._conn is not None:
            rows = self._sqlite_neighbors(key, sources)
        else:
            rows = list(self._mem.get(key, []))
            if sources:
                rows = [r for r in rows if r.source in sources]
        # deterministic ordering: weight desc, then lexical for tie-break
        rows.sort(key=lambda r: (-r.weight, r.relation, r.tail))
        return rows[:top]

    def polarity(self, term: str) -> Optional[float]:
        key = normalize(term)
        if self._conn is not None:
            cur = self._conn.execute("SELECT value FROM polarity WHERE concept=?", (key,))
            row = cur.fetchone()
            return float(row[0]) if row else None
        return self._pol.get(key)

    # ---- sqlite helpers ----
    def _sqlite_neighbors(self, key: str, sources) -> List[Triple]:
        q = "SELECT head, relation, tail, weight, source FROM triples WHERE head=?"
        params: list = [key]
        if sources:
            q += " AND source IN (%s)" % ",".join("?" * len(sources))
            params.extend(sources)
        cur = self._conn.execute(q, params)
        return [Triple(*r) for r in cur.fetchall()]

    @staticmethod
    def init_sqlite(path: str) -> sqlite3.Connection:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS triples(
                head TEXT, relation TEXT, tail TEXT, weight REAL, source TEXT);
            CREATE TABLE IF NOT EXISTS polarity(concept TEXT PRIMARY KEY, value REAL);
            CREATE INDEX IF NOT EXISTS idx_triples_head ON triples(head);
            CREATE INDEX IF NOT EXISTS idx_triples_tail ON triples(tail);
            """
        )
        conn.commit()
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
