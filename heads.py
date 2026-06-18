"""Prediction heads (paper Eqs. 21, 23).

    p(b_i | T, I) = softmax(W_b h_i + b_b)            token BIO tagging (Eq. 21)
    p(y_k | a_k)  = softmax(W_s z_k + b_s)            span-level ASC (Eq. 23)

Losses (Eqs. 22, 24) live in losses.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import CONFIG, NUM_BIO_TAGS, NUM_POLARITIES


class BIOTaggingHead(nn.Module):
    """Token-level unified BIO sentiment tagger (Eq. 21)."""

    def __init__(self, d: int = None, num_tags: int = NUM_BIO_TAGS, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_tags)

    def forward(self, text_feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.drop(text_feats))  # [B, n, num_tags]


class SpanSentimentHead(nn.Module):
    """Auxiliary span-level polarity classifier over the KAN-fused z_k (Eq. 23)."""

    def __init__(self, d: int = None, num_pol: int = NUM_POLARITIES, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_pol)

    def forward(self, z_k: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.drop(z_k))  # [K, num_pol]
