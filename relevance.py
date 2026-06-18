"""Aspect representation + teacher-guided aspect-visual relevance (paper Eqs. 6-10).

    t_k  = Pool({h_i | w_i in a_k})                                       (Eq. 6)
    a_kj = softmax_j( t_k^T W_v v_j )                                     (Eq. 7)
    v_bar_k = sum_j a_kj v_j                                              (Eq. 8)
    r_k  = sigma( w_r^T [t_k ; v_bar_k ; t_k ⊙ v_bar_k] + b_r )          (Eq. 9)
    v_tilde_k = r_k * v_bar_k                                             (Eq. 10)

Lrel (Eq. 11) lives in losses.py.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from config import CONFIG


def pool_aspect(text_feats: torch.Tensor, spans: List[Tuple[int, int]], mode: str = "mean") -> torch.Tensor:
    """Pool token reps over each aspect span (Eq. 6).

    Args:
        text_feats: [n, d] token features for ONE instance.
        spans: list of (start, end_exclusive) token index ranges.
        mode: 'mean' | 'max' | 'first'.
    Returns:
        [K, d] aspect-aware textual representations (K = len(spans)).
    """
    d = text_feats.size(-1)
    if not spans:
        return text_feats.new_zeros((0, d))
    out = []
    n = text_feats.size(0)
    for (s, e) in spans:
        s = max(0, min(s, n - 1))
        e = max(s + 1, min(e, n))
        chunk = text_feats[s:e]  # [len, d]
        if mode == "mean":
            out.append(chunk.mean(dim=0))
        elif mode == "max":
            out.append(chunk.max(dim=0).values)
        elif mode == "first":
            out.append(chunk[0])
        else:
            raise ValueError(f"unknown pool mode {mode}")
    return torch.stack(out, dim=0)  # [K, d]


class AspectVisualRelevance(nn.Module):
    """Aspect-conditioned visual attention + relevance estimator (Eqs. 7-10)."""

    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.Wv = nn.Linear(d, d, bias=False)   # Eq. 7 bilinear projection
        self.wr = nn.Linear(3 * d, 1)           # Eq. 9 (w_r, b_r)
        self.drop = nn.Dropout(dropout)

    def forward(self, t_k: torch.Tensor, V: torch.Tensor):
        """Args: t_k [K, d] aspect reps; V [m, d] visual patches (one instance).

        Returns: t_k, v_bar [K,d], r_k [K], v_tilde [K,d], alpha [K,m].
        """
        if t_k.numel() == 0:
            K, d = 0, V.size(-1)
            empty = V.new_zeros((0, d))
            return t_k, empty, V.new_zeros((0,)), empty, V.new_zeros((0, V.size(0)))
        scores = t_k @ self.Wv(V).transpose(-1, -2)        # [K, m]  t_k^T W_v v_j
        alpha = torch.softmax(scores, dim=-1)              # Eq. 7
        v_bar = alpha @ V                                  # [K, d]  Eq. 8
        feat = torch.cat([t_k, v_bar, t_k * v_bar], dim=-1)  # Eq. 9 input
        r_k = torch.sigmoid(self.wr(self.drop(feat))).squeeze(-1)  # [K]
        v_tilde = r_k.unsqueeze(-1) * v_bar                # Eq. 10
        return t_k, v_bar, r_k, v_tilde, alpha
