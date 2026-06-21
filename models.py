"""TARKAN student model — assembles the updated methodology (Fig. 1, §3.2-§3.8).

Per image-text pair the student:
  1. encodes text/image (Eqs. 4-5),
  2. pools aspect reps over spans (Eq. 6) — gold spans in training, predicted in inference,
  3. estimates aspect-visual relevance and filters the image (Eqs. 7-10),
  4. retrieves + encodes aspect-centered KG triples (Eqs. 12-14),
  5. predicts KG usefulness and aggregates filtered KG evidence (Eqs. 15-17),
  6. fuses [h^t_i ; v_tilde ; g_tilde] PER TOKEN via KAN -> h̃_i (Eqs. 18-20),
  7. predicts the unified BIO aspect-sentiment tag from h̃_i (Eq. 21).

Updated paper §3.6: the BIO head performs BOTH aspect extraction and sentiment
classification (one 7-class sequence-labeling task) and runs on the KAN-fused
multimodal token representation h̃_i — there is no separate ASC head. Each aspect's
relevance-filtered visual (v_tilde_k) and teacher-filtered KG (g_tilde_k) evidence is
broadcast to that aspect's token positions; tokens outside any aspect get zero evidence.
Setting cfg.use_kan_tag_representation=False feeds the BIO head text-only features
(Table-6 ablation "w/o KAN-enhanced tag representation").

The offline LLM teacher never enters the forward pass — its signals (r^T, s^T) are
only *targets* consumed by losses.py. Ablation toggles (config) switch streams on/off
to reproduce Table 6.

Batch dict (from data.py collate) — forward consumes:
  input_ids [B,n], attention_mask [B,n], pixel_values [B,3,H,W] (optional if feats given),
  aspect_spans: List[B] of List[(start,end)]   (gold in training / predicted in inference;
    empty lists -> stage-1 extraction with zero aspect evidence),
  aspect_queries: List[B] of List[AspectQuery]   (for KG retrieval),
  aspect_triples: Optional List[B] of List[List[Triple]]  (precomputed/cached retrieval).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import CONFIG
from heads import BIOTaggingHead
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
        self.tag_norm = nn.LayerNorm(d)  # stabilizes the residual KAN-enhanced token rep h̃
        self.bio_head = BIOTaggingHead(d, dropout=dp)

    def set_kg(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    # ------------------------------------------------------------------ #
    def _aspect_evidence(self, t_k_all, V, queries, triples_cached, want_alpha):
        """Per-aspect evidence for ONE instance (Eqs. 7-17).

        Returns:
          v_tilde [K, d]   relevance-filtered visual (Eq. 10)
          r_k     [K]      aspect-visual relevance scores (Eq. 9; supervises L_rel)
          g_list  list[K]  filtered KG vector g_tilde_k [d] (Eq. 17)
          s_list  list[K]  per-triple KG usefulness scores (Eq. 15; supervises L_kg)
          tr_list list[K]  retrieved Triple lists
          alpha            attention weights if requested
        """
        cfg = self.cfg
        K = t_k_all.size(0)
        device = t_k_all.device
        d = cfg.hidden_dim

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

        g_list, s_list, tr_list = [], [], []
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
            g_list.append(g_tilde)
            s_list.append(s)
            tr_list.append(triples)

        return v_tilde, r_k, g_list, s_list, tr_list, (alpha if want_alpha else None)

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

        cfg = self.cfg
        B, n, d = text_feats.shape

        # per-token aspect-relevant evidence (zeros for tokens outside any aspect)
        v_tok = text_feats.new_zeros((B, n, d))
        g_tok = text_feats.new_zeros((B, n, d))

        all_r, all_s, all_tr, all_alpha, owner = [], [], [], [], []
        spans_b = batch["aspect_spans"]
        queries_b = batch.get("aspect_queries", [[] for _ in range(B)])
        triples_b = batch.get("aspect_triples", None)

        for b in range(B):
            spans = spans_b[b]
            t_k_all = pool_aspect(text_feats[b], spans, self.pool_mode)  # [K, d]
            V = visual_feats[b] if (visual_feats is not None) else text_feats.new_zeros((1, d))
            cached = triples_b[b] if triples_b is not None else None
            v_tilde, r_k, g_list, s_list, tr_list, alpha = self._aspect_evidence(
                t_k_all, V, queries_b[b] if queries_b else [], cached, want_alpha
            )
            K = t_k_all.size(0)
            # broadcast each aspect's evidence to its token positions (Eq. 21 input h̃_i)
            for k in range(K):
                s_, e_ = spans[k][0], spans[k][1]
                v_tok[b, s_:e_] = v_tilde[k]
                g_tok[b, s_:e_] = g_list[k]
            if K:
                all_r.append(r_k)
                all_s.extend(s_list)
                all_tr.extend(tr_list)
                owner.extend([b] * K)
                if alpha is not None:
                    all_alpha.extend(list(alpha))

        # ---- KAN-fused multimodal token representation h̃ (Eqs. 18-20) ----
        # §3.6: h̃_i "combines its contextual textual representation WITH aspect-relevant
        # visual and KG evidence". We implement this as a RESIDUAL enhancement:
        #   h̃_i = h^t_i + KAN([h^t_i ; v_tilde ; g_tilde])
        # The residual keeps the BERTweet text signal intact so the unified BIO head can
        # still extract spans (text-driven) while the KAN term *enhances* the representation
        # with multimodal evidence (mainly polarity). Without the residual, routing the head
        # entirely through a fresh per-token KAN washes out the text features and the head
        # collapses to all-O. The residual also makes two-stage inference robust: at stage-1
        # (zero evidence) h̃_i ≈ h^t_i + KAN([h^t_i;0;0]) stays text-dominant -> extraction works.
        if cfg.use_kan_tag_representation:
            # evidence dropout (train only): zero a random subset of instances' evidence so the
            # BIO head also learns text-only extraction (matches stage-1 inference; see config).
            if self.training and getattr(cfg, "evidence_dropout", 0.0) > 0:
                keep = (torch.rand(B, 1, 1, device=text_feats.device) >= cfg.evidence_dropout).to(text_feats.dtype)
                v_tok = v_tok * keep
                g_tok = g_tok * keep
            fused = self.fusion(
                text_feats.reshape(B * n, d),
                v_tok.reshape(B * n, d),
                g_tok.reshape(B * n, d),
            ).reshape(B, n, d)
            h_tilde = self.tag_norm(text_feats + fused)  # add & norm (stable scale for the BIO head)
        else:  # Table-6 ablation "w/o KAN-enhanced tag representation": BIO head on text only
            h_tilde = text_feats

        tag_logits = self.bio_head(h_tilde)  # Eq. 21 -> [B, n, 7]

        r_cat = torch.cat(all_r, 0) if all_r else text_feats.new_zeros((0,))

        return {
            "tag_logits": tag_logits,
            "relevance": r_cat,
            "kg_scores": all_s,        # list[sumK] of [M_k]
            "kg_triples": all_tr,      # list[sumK] of list[Triple]
            "aspect_batch_idx": torch.tensor(owner, dtype=torch.long, device=text_feats.device),
            "alpha": all_alpha if want_alpha else None,
        }
