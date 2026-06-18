"""End-to-end student forward (synthetic features, no model downloads) + tiny overfit.

Proves the assembled pipeline is learnable on CPU before any GPU spend (CLAUDE.md gate).
"""
from dataclasses import replace

import torch

from config import CONFIG
from kg import KnowledgeGraph
from kg_retrieval import AspectQuery
from losses import compute_losses
from models import TarkanStudent
from seeding import seed_everything
from teacher import build_targets

D = 16


def _synthetic_batch(B=2, n=6, m=4, d=D):
    text_feats = torch.randn(B, n, d)
    visual_feats = torch.randn(B, m, d)
    batch = {
        "instance_id": [f"x{b}" for b in range(B)],
        "bio_labels": torch.randint(0, 7, (B, n)),
        "aspect_spans": [[(1, 3)] for _ in range(B)],
        "aspect_polarity": [[b % 3] for b in range(B)],
        "aspect_queries": [[AspectQuery(aspect_term="cat", opinion_words=["happy"], visual_concepts=["dog"])] for _ in range(B)],
    }
    return batch, text_feats, visual_feats


def test_forward_shapes_full_path():
    seed_everything(0)
    cfg = replace(CONFIG, hidden_dim=D, kan_hidden=(8,), fusion="kan")
    kg = KnowledgeGraph(triples=[("cat", "RelatedTo", "happy", 1.0, "conceptnet"),
                                 ("happy", "HasPolarity", "positive", 1.0, "senticnet")])
    model = TarkanStudent(cfg, build_encoders=False, kg=kg)
    batch, tf, vf = _synthetic_batch()
    out = model(batch, text_feats=tf, visual_feats=vf)
    assert out["tag_logits"].shape == (2, 6, 7)
    assert out["asc_logits"].shape == (2, 3)
    assert out["relevance"].shape == (2,)
    assert len(out["kg_scores"]) == 2


def test_tiny_overfit():
    seed_everything(0)
    cfg = replace(CONFIG, hidden_dim=D, fusion="concat_mlp", use_kg_stream=False, use_teacher=False, dropout=0.0)
    model = TarkanStudent(cfg, build_encoders=False)
    batch, tf, vf = _synthetic_batch(B=4, n=6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(400):
        out = model(batch, text_feats=tf, visual_feats=vf)
        targets = build_targets(batch, out, cache=None, cfg=cfg)
        loss = compute_losses(out, targets, cfg)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.item())
    assert float(loss.item()) < 0.15, f"did not overfit: {first:.3f} -> {loss.item():.3f}"
