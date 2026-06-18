"""Text & visual encoders (paper Eqs. 4-5).

- Text  : BERTweet  -> H_t = [h_1..h_n]  in R^{n x d}      (Eq. 4)
- Visual: CLIP-ViT  -> V   = [v_1..v_m]  in R^{m x d}      (Eq. 5)

Models are loaded lazily from Hugging Face the first time the module is constructed
(nothing is downloaded at import time). A learnable Linear projects CLIP patch
features to the shared hidden dim d (CLIP ViT-B/32 hidden is already 768; the
projection is kept to honor "text/visual share d" and to add capacity).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import CONFIG


class TextEncoder(nn.Module):
    """BERTweet token encoder (Eq. 4)."""

    def __init__(self, model_id: str = None, hidden_dim: int = None, dropout: float = None):
        super().__init__()
        from transformers import AutoModel

        model_id = model_id or CONFIG.text_model_id
        hidden_dim = hidden_dim or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.bert = AutoModel.from_pretrained(model_id, token=CONFIG.hf_token)
        self.out_dim = self.bert.config.hidden_size
        self.proj = nn.Identity() if self.out_dim == hidden_dim else nn.Linear(self.out_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.dropout(self.proj(out))  # [B, n, d]


class VisualEncoder(nn.Module):
    """CLIP-ViT patch encoder (Eq. 5). Returns the m patch tokens (CLS dropped)."""

    def __init__(self, model_id: str = None, hidden_dim: int = None, dropout: float = None):
        super().__init__()
        from transformers import CLIPVisionModel

        model_id = model_id or CONFIG.visual_model_id
        hidden_dim = hidden_dim or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.clip = CLIPVisionModel.from_pretrained(model_id, token=CONFIG.hf_token)
        self.out_dim = self.clip.config.hidden_size
        self.proj = nn.Linear(self.out_dim, hidden_dim)  # always project (alignment)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pixel_values) -> torch.Tensor:
        out = self.clip(pixel_values=pixel_values).last_hidden_state  # [B, 1+m, out_dim]
        patches = out[:, 1:, :]                                       # drop CLS -> [B, m, out_dim]
        return self.dropout(self.proj(patches))                      # [B, m, d]
