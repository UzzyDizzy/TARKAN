"""Teacher-guided KG evidence filtering (paper Eqs. 15-17).

    s_kq    = sigma( w_g^T [t_k ; g_kq ; t_k ⊙ g_kq] + b_g )            (Eq. 15)
    g_tilde = ( sum_q s_kq g_kq ) / ( sum_q s_kq + eps )                (Eq. 17)

Lkg (Eq. 16) lives in losses.py. Operates per-aspect (M varies across aspects).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from config import CONFIG


class KGFilter(nn.Module):
    def __init__(self, d: int = None, dropout: float = None, eps: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.eps = CONFIG.kg_eps if eps is None else eps
        self.wg = nn.Linear(3 * d, 1)   # Eq. 15 (w_g, b_g)
        self.drop = nn.Dropout(dropout)
        self.d = d

    def forward(self, t_k: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: t_k [d] one aspect rep; g [M, d] its retrieved triple reps.

        Returns: s [M] usefulness scores, g_tilde [d] filtered KG representation.
        """
        M = g.size(0)
        if M == 0:
            return g.new_zeros((0,)), g.new_zeros((self.d,))
        t_exp = t_k.unsqueeze(0).expand(M, -1)                 # [M, d]
        feat = torch.cat([t_exp, g, t_exp * g], dim=-1)        # Eq. 15 input
        s = torch.sigmoid(self.wg(self.drop(feat))).squeeze(-1)  # [M]
        weighted = (s.unsqueeze(-1) * g).sum(dim=0)            # [d]
        g_tilde = weighted / (s.sum() + self.eps)             # Eq. 17
        return s, g_tilde
