"""TARKAN student model — assembles Eqs. 4-24 (Fig. 1).

Per image-text pair the student:
  1. encodes text/image (Eqs. 4-5),
  2. pools aspect reps (Eq. 6),
  3. estimates aspect-visual relevance and filters the image (Eqs. 7-10),
  4. retrieves + encodes aspect-centered KG triples (Eqs. 12-14),
  5. predicts KG usefulness and aggregates filtered KG evidence (Eqs. 15-17),
  6. fuses [t_k ; v_tilde_k ; g_tilde_k] via KAN (Eqs. 18-20),
  7. predicts token BIO tags (Eq. 21) and auxiliary span polarity (Eq. 23).

The offline LLM teacher never enters the forward pass — its signals (r^T, s^T) are
only *targets* consumed by losses.py. Ablation toggles (config) switch streams on/off
to reproduce Table 6.

Batch dict (from data.py collate) — forward consumes:
  input_ids [B,n], attention_mask [B,n], pixel_values [B,3,H,W] (optional if feats given),
  aspect_spans: List[B] of List[(start,end)],
  aspect_queries: List[B] of List[AspectQuery]   (for KG retrieval),
  aspect_triples: Optional List[B] of List[List[Triple]]  (precomputed/cached retrieval).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import CONFIG, NUM_POLARITIES
from heads import BIOTaggingHead, SpanSentimentHead
from kan_fusion import build_fusion
from kg import KnowledgeGraph
from kg_filter import KGFilter
from kg_retrieval import AspectQuery, EntityEmbedder, TripleEncoder, retrieve_triples
from relevance import AspectVisualRelevance, pool_aspect


class TarkanStudent(nn.Module):
    def __init__(
        self,
        config=CONFIG,
        build_encoders: bool = True,
        kg: Optional[KnowledgeGraph] = None,
        entity_embedder: Optional[EntityEmbedder] = None,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.cfg = config
        d = config.hidden_dim
        self.pool_mode = pool_mode
        self.kg = kg

        if build_encoders:
            from encoders import TextEncoder, VisualEncoder

            self.text_encoder = TextEncoder()
            self.visual_encoder = VisualEncoder()
        else:  # tests/feature-precompute: feed text_feats/visual_feats to forward()
            self.text_encoder = None
            self.visual_encoder = None

        dp = config.dropout
        self.relevance = AspectVisualRelevance(d, dropout=dp)
        self.triple_encoder = TripleEncoder(d, embedder=entity_embedder, dropout=dp)
        self.kg_filter = KGFilter(d, dropout=dp)
        self.fusion = build_fusion(config.fusion, d, dropout=dp)
        self.bio_head = BIOTaggingHead(d, dropout=dp)
        self.asc_head = SpanSentimentHead(d, dropout=dp)

    def set_kg(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    # ------------------------------------------------------------------ #
    def _aspect_forward(self, t_k_all, V, queries, triples_cached, want_alpha):
        """Process all aspects of ONE instance. Returns lists across K aspects."""
        cfg = self.cfg
        K = t_k_all.size(0)
        device = t_k_all.device
        d = self.cfg.hidden_dim

        # ---- visual stream (Eqs. 7-10) ----
        if cfg.use_visual_stream and K > 0:
            _, v_bar, r_k, v_tilde, alpha = self.relevance(t_k_all, V)
            if not cfg.use_relevance:
                # keep aspect-conditioned visual but drop the learned gate (Table 6)
                v_tilde = v_bar
                r_k = torch.ones(K, device=device)
        else:
            v_tilde = torch.zeros((K, d), device=device)
            r_k = torch.zeros((K,), device=device)
            alpha = None

        z_list, s_list, tr_list = [], [], []
        for k in range(K):
            t_k = t_k_all[k]
            # ---- KG stream (Eqs. 12-17) ----
            if cfg.use_kg_stream and self.kg is not None:
                triples = triples_cached[k] if triples_cached is not None else retrieve_triples(
                    queries[k], self.kg, cfg.top_m_triples
                )
                g = self.triple_encoder(triples)              # [M, d]
                if cfg.use_kg_filter:
                    s, g_tilde = self.kg_filter(t_k, g)       # Eqs. 15, 17
                else:
                    s = g.new_zeros((g.size(0),))
                    g_tilde = g.mean(dim=0) if g.size(0) > 0 else g.new_zeros((d,))  # unfiltered mean
            else:
                triples, s, g_tilde = [], torch.zeros((0,), device=device), torch.zeros((d,), device=device)

            z_k = self.fusion(
                t_k.unsqueeze(0), v_tilde[k].unsqueeze(0), g_tilde.unsqueeze(0)
            ).squeeze(0)                                       # Eqs. 18-20 -> [d]
            z_list.append(z_k)
            s_list.append(s)
            tr_list.append(triples)

        z = torch.stack(z_list, 0) if z_list else torch.zeros((0, d), device=device)
        return z, r_k, s_list, tr_list, (alpha if want_alpha else None)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: Dict,
        text_feats: Optional[torch.Tensor] = None,
        visual_feats: Optional[torch.Tensor] = None,
        want_alpha: bool = False,
    ) -> Dict:
        if text_feats is None:
            text_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        if visual_feats is None and self.cfg.use_visual_stream:
            visual_feats = self.visual_encoder(batch["pixel_values"])

        B = text_feats.size(0)
        d = self.cfg.hidden_dim
        tag_logits = self.bio_head(text_feats)  # Eq. 21 -> [B, n, 7]

        all_z, all_r, all_s, all_tr, all_alpha, owner = [], [], [], [], [], []
        spans_b = batch["aspect_spans"]
        queries_b = batch.get("aspect_queries", [[] for _ in range(B)])
        triples_b = batch.get("aspect_triples", None)

        for b in range(B):
            spans = spans_b[b]
            t_k_all = pool_aspect(text_feats[b], spans, self.pool_mode)  # [K, d]
            V = visual_feats[b] if (visual_feats is not None) else text_feats.new_zeros((1, d))
            cached = triples_b[b] if triples_b is not None else None
            z, r_k, s_list, tr_list, alpha = self._aspect_forward(
                t_k_all, V, queries_b[b] if queries_b else [], cached, want_alpha
            )
            K = z.size(0)
            if K:
                all_z.append(z)
                all_r.append(r_k)
                all_s.extend(s_list)
                all_tr.extend(tr_list)
                owner.extend([b] * K)
                if alpha is not None:
                    all_alpha.extend(list(alpha))

        if all_z:
            z_cat = torch.cat(all_z, 0)          # [sumK, d]
            r_cat = torch.cat(all_r, 0)          # [sumK]
            asc_logits = self.asc_head(z_cat)    # Eq. 23 -> [sumK, 3]
        else:
            z_cat = text_feats.new_zeros((0, d))
            r_cat = text_feats.new_zeros((0,))
            asc_logits = text_feats.new_zeros((0, NUM_POLARITIES))

        return {
            "tag_logits": tag_logits,
            "asc_logits": asc_logits,
            "relevance": r_cat,
            "kg_scores": all_s,        # list[sumK] of [M_k]
            "kg_triples": all_tr,      # list[sumK] of list[Triple]
            "aspect_batch_idx": torch.tensor(owner, dtype=torch.long, device=text_feats.device),
            "alpha": all_alpha if want_alpha else None,
            "fused_z": z_cat,
        }
