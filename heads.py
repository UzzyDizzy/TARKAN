"""Prediction head (updated paper Eq. 21).

    p(b_i | T, I) = softmax(W_b h̃_i + b_b)            unified BIO sentiment tagging

`h̃_i` is the KAN-fused multimodal token representation (text + relevance-filtered
visual + filtered KG; see models.TarkanStudent.forward). The unified BIO head performs
BOTH aspect extraction and aspect-level sentiment classification — there is no separate
ASC head in the updated methodology (§3.6). L_tag (Eq. 22) lives in losses.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import CONFIG, NUM_BIO_TAGS


class BIOTaggingHead(nn.Module):
    """Token-level unified BIO sentiment tagger (Eq. 21).

    Consumes the KAN-fused token representation h̃ of shape [..., d] and emits one
    distribution over the 7 unified BIO tags per token.
    """

    def __init__(self, d: int = None, num_tags: int = NUM_BIO_TAGS, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_tags)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.drop(h))  # [..., num_tags]
