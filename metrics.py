"""Evaluation metrics (paper §5.1 / Tables 1, 3) + paired bootstrap significance (§4.3).

Conventions
-----------
A prediction/gold for one instance is a set/list of (start, end, polarity) triples.
  - Joint MABSA (Table 1): micro P/R/F1 over exact (span, polarity) matches.
  - MATE (Table 3): micro P/R/F1 over spans only (polarity ignored).
  - MASC (Table 3): polarity Acc + macro-F1 on gold aspects.
Significance: paired bootstrap, 1000 resamples, two-sided p<0.05 (marked † in tables).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

Pair = Tuple[int, int, str]
Span = Tuple[int, int]


def _prf(tp: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def joint_prf(preds: Sequence[Sequence[Pair]], golds: Sequence[Sequence[Pair]]) -> Dict[str, float]:
    tp = n_pred = n_gold = 0
    for p, g in zip(preds, golds):
        ps, gs = set(p), set(g)
        tp += len(ps & gs)
        n_pred += len(ps)
        n_gold += len(gs)
    p, r, f = _prf(tp, n_pred, n_gold)
    return {"P": 100 * p, "R": 100 * r, "F1": 100 * f}


def mate_prf(preds: Sequence[Sequence[Pair]], golds: Sequence[Sequence[Pair]]) -> Dict[str, float]:
    p2 = [[(s, e) for (s, e, _) in inst] for inst in preds]
    g2 = [[(s, e) for (s, e, _) in inst] for inst in golds]
    tp = n_pred = n_gold = 0
    for p, g in zip(p2, g2):
        ps, gs = set(p), set(g)
        tp += len(ps & gs)
        n_pred += len(ps)
        n_gold += len(gs)
    p, r, f = _prf(tp, n_pred, n_gold)
    return {"P": 100 * p, "R": 100 * r, "F1": 100 * f}


def masc_acc_f1(y_true: Sequence[str], y_pred: Sequence[str], labels=("POS", "NEU", "NEG")) -> Dict[str, float]:
    """Polarity accuracy + macro-F1 on gold aspects (Table 3 MASC)."""
    if not y_true:
        return {"Acc": 0.0, "F1": 0.0}
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = correct / len(y_true)
    f1s = []
    for lab in labels:
        tp = sum(int(t == lab and p == lab) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != lab and p == lab) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == lab and p != lab) for t, p in zip(y_true, y_pred))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    return {"Acc": 100 * acc, "F1": 100 * float(np.mean(f1s))}


def paired_bootstrap(
    preds_a: Sequence[Sequence[Pair]],
    preds_b: Sequence[Sequence[Pair]],
    golds: Sequence[Sequence[Pair]],
    metric_fn: Callable = joint_prf,
    key: str = "F1",
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired bootstrap p-value for (system A - system B) on `key`.

    Returns observed scores, mean delta, and p (fraction of resamples where the
    sign of the delta flips relative to the observed delta).
    """
    rng = np.random.RandomState(seed)
    N = len(golds)
    obs_a = metric_fn(preds_a, golds)[key]
    obs_b = metric_fn(preds_b, golds)[key]
    obs_delta = obs_a - obs_b
    flips = 0
    for _ in range(n_samples):
        idx = rng.randint(0, N, size=N)
        ra = metric_fn([preds_a[i] for i in idx], [golds[i] for i in idx])[key]
        rb = metric_fn([preds_b[i] for i in idx], [golds[i] for i in idx])[key]
        delta = ra - rb
        if (obs_delta >= 0 and delta <= 0) or (obs_delta < 0 and delta >= 0):
            flips += 1
    p = flips / n_samples
    return {f"{key}_A": obs_a, f"{key}_B": obs_b, "delta": obs_delta, "p_value": p, "significant": bool(p < 0.05)}
