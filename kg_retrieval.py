"""Aspect-centered KG retrieval + triple encoding (paper Eqs. 12-14).

    Q_k = {a_k} ∪ O_k ∪ C_k                                              (Eq. 12)
        O_k = sentiment/opinion words near the aspect (spaCy)
        C_k = visual concepts (caption nouns / CLIP zero-shot)
    G_k = top-M retrieved triples (e_p, r, e_q)                          (Eq. 13)
    g_kq = phi([e_p ; r ; e_q])                                          (Eq. 14)

Retrieval/scoring is non-differentiable preprocessing; only TripleEncoder has
parameters. Entity embeddings come from ConceptNet Numberbatch (English, 300-d)
with a deterministic hash fallback for OOV / offline tests.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from config import CONFIG
from kg import KnowledgeGraph, Triple, normalize

# Canonical relation vocabulary (ConceptNet 34 + SenticNet edge types). Unknown -> <unk>.
CONCEPTNET_RELATIONS = [
    "RelatedTo", "FormOf", "IsA", "PartOf", "HasA", "UsedFor", "CapableOf", "AtLocation",
    "Causes", "HasSubevent", "HasFirstSubevent", "HasLastSubevent", "HasPrerequisite",
    "HasProperty", "MotivatedByGoal", "ObstructedBy", "Desires", "CreatedBy", "Synonym",
    "Antonym", "DistinctFrom", "DerivedFrom", "SymbolOf", "DefinedAs", "MannerOf",
    "LocatedNear", "HasContext", "SimilarTo", "EtymologicallyRelatedTo",
    "EtymologicallyDerivedFrom", "CausesDesire", "MadeOf", "ReceivesAction", "ExternalURL",
]
SENTICNET_RELATIONS = ["HasPolarity", "HasMood", "SemanticallyRelated"]
RELATION_VOCAB = ["<unk>"] + CONCEPTNET_RELATIONS + SENTICNET_RELATIONS
REL2ID = {r: i for i, r in enumerate(RELATION_VOCAB)}


@dataclass
class AspectQuery:
    aspect_term: str
    opinion_words: List[str] = field(default_factory=list)   # O_k
    visual_concepts: List[str] = field(default_factory=list)  # C_k

    def terms(self) -> List[str]:
        seen, out = set(), []
        for t in [self.aspect_term, *self.opinion_words, *self.visual_concepts]:
            k = normalize(t)
            if k and k not in seen:
                seen.add(k)
                out.append(t)
        return out


def _triple_key(t: Triple) -> str:
    return f"{t.head}|{t.relation}|{t.tail}"


def retrieve_triples(
    query: AspectQuery,
    kg: KnowledgeGraph,
    top_m: int = None,
    teacher_scores: Optional[Dict[str, float]] = None,
) -> List[Triple]:
    """Gather neighbours of all query terms, score, keep top-M (Eq. 13).

    score = weight + lexical_match + affective_relevance + relation_prior (+ teacher)
    (equal-weight combination — Open-Q #8; deterministic tie-break).
    """
    top_m = top_m or CONFIG.top_m_triples
    cand: Dict[str, Triple] = {}
    aspect_key = normalize(query.aspect_term)
    qkeys = {normalize(t) for t in query.terms()}
    for term in query.terms():
        for tr in kg.neighbors(term, top=top_m * 4, sources=CONFIG.kg_sources):
            cand[_triple_key(tr)] = tr

    def score(tr: Triple) -> float:
        s = float(tr.weight)
        if normalize(tr.tail) in qkeys or tr.head == aspect_key:
            s += 1.0  # lexical match with query context
        pol = kg.polarity(tr.tail)
        if pol is not None:
            s += abs(float(pol))  # affective relevance (SenticNet polarity magnitude)
        if tr.relation in ("HasPolarity", "RelatedTo", "Causes", "HasProperty", "SemanticallyRelated"):
            s += 0.5  # sentiment-bearing relation prior
        if teacher_scores is not None:
            s += float(teacher_scores.get(_triple_key(tr), 0.0))
        return s

    ranked = sorted(cand.values(), key=lambda t: (-score(t), t.relation, t.tail))
    return ranked[:top_m]


class EntityEmbedder:
    """Surface term -> 300-d vector via Numberbatch, with deterministic hash fallback."""

    def __init__(self, numberbatch: Optional[Dict[str, np.ndarray]] = None, dim: int = None):
        self.dim = dim or CONFIG.entity_emb_dim
        self.table = numberbatch or {}

    @classmethod
    def from_txt(cls, path: str, dim: int = None, vocab: Optional[set] = None) -> "EntityEmbedder":
        """Load numberbatch-en-19.08.txt(.gz), STREAMING line-by-line (never reads the
        whole file at once). Pass `vocab` (a set of normalized terms) to keep only the
        embeddings you need — cuts memory from ~600 MB to just the KG vocabulary.
        """
        import gzip

        table: Dict[str, np.ndarray] = {}
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            first = f.readline().split()
            d = int(first[1]) if len(first) == 2 else CONFIG.entity_emb_dim
            if len(first) != 2:  # first line was already a vector
                key = normalize(first[0].split("/")[-1])
                if vocab is None or key in vocab:
                    table[key] = np.asarray(first[1:], dtype=np.float32)
            for line in f:
                parts = line.rstrip().split(" ")
                key = normalize(parts[0].split("/")[-1])  # /c/en/word -> word
                if vocab is not None and key not in vocab:
                    continue
                table[key] = np.asarray(parts[1:], dtype=np.float32)
        return cls(table, dim=d)

    def embed(self, term: str) -> np.ndarray:
        key = normalize(term)
        if key in self.table:
            return self.table[key]
        # deterministic pseudo-embedding so OOV / offline still works (unit-norm).
        h = hashlib.sha256(key.encode("utf-8")).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
        v = rng.randn(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


class TripleEncoder(nn.Module):
    """phi([e_p ; r ; e_q]) -> g_kq in R^d  (Eq. 14)."""

    def __init__(self, d: int = None, entity_dim: int = None, embedder: Optional[EntityEmbedder] = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        entity_dim = entity_dim or CONFIG.entity_emb_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.embedder = embedder or EntityEmbedder(dim=entity_dim)
        self.entity_proj = nn.Linear(entity_dim, d)
        self.relation_emb = nn.Embedding(len(RELATION_VOCAB), d)
        self.phi = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)
        )

    def forward(self, triples: Sequence[Triple]) -> torch.Tensor:
        """Returns g [M, d]; empty [0, d] if no triples."""
        device = self.entity_proj.weight.device
        d = self.entity_proj.out_features
        if not triples:
            return torch.zeros((0, d), device=device)
        head_e, tail_e, rel_ids = [], [], []
        for t in triples:
            head_e.append(self.embedder.embed(t.head))
            tail_e.append(self.embedder.embed(t.tail))
            rel_ids.append(REL2ID.get(t.relation, 0))
        he = torch.from_numpy(np.stack(head_e)).to(device)
        te = torch.from_numpy(np.stack(tail_e)).to(device)
        ep = self.entity_proj(he)                                  # [M, d]
        eq = self.entity_proj(te)                                  # [M, d]
        r = self.relation_emb(torch.tensor(rel_ids, device=device))  # [M, d]
        g = self.phi(torch.cat([ep, r, eq], dim=-1))              # [M, d]
        return g
