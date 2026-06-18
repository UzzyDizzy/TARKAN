"""Numeric checks of the model math (Eqs. 6-20) on tiny CPU tensors."""
import torch

from kan_fusion import FUSION_REGISTRY, KANFusion, RBFKAN
from kg import KnowledgeGraph
from kg_filter import KGFilter
from kg_retrieval import AspectQuery, TripleEncoder, retrieve_triples
from relevance import AspectVisualRelevance, pool_aspect
from seeding import seed_everything

D = 16


def test_pool_aspect():
    feats = torch.arange(40.0).reshape(10, 4)
    pooled = pool_aspect(feats, [(0, 2), (3, 4)], mode="mean")
    assert pooled.shape == (2, 4)
    assert torch.allclose(pooled[0], feats[0:2].mean(0))


def test_relevance_eqs_7_10():
    seed_everything(0)
    rel = AspectVisualRelevance(d=D)
    t_k = torch.randn(3, D)
    V = torch.randn(5, D)
    t_out, v_bar, r_k, v_tilde, alpha = rel(t_k, V)
    assert alpha.shape == (3, 5)
    assert torch.allclose(alpha.sum(-1), torch.ones(3), atol=1e-5)   # Eq. 7 softmax
    assert (r_k >= 0).all() and (r_k <= 1).all()                     # Eq. 9 sigmoid
    assert torch.allclose(v_tilde, r_k.unsqueeze(-1) * v_bar)        # Eq. 10


def test_kg_inmemory():
    kg = KnowledgeGraph(triples=[("smile", "RelatedTo", "happy", 2.0, "conceptnet")])
    kg.set_polarity("happy", 0.9)
    nbrs = kg.neighbors("smile", top=5)
    assert any(t.tail == "happy" for t in nbrs)
    assert kg.polarity("happy") == 0.9


def test_retrieval_topm_and_encoder():
    seed_everything(0)
    triples = [(f"a{i}", "RelatedTo", f"b{i}", float(i), "conceptnet") for i in range(20)]
    kg = KnowledgeGraph(triples=triples)
    q = AspectQuery(aspect_term="a1", opinion_words=["a2"], visual_concepts=["a3"])
    got = retrieve_triples(q, kg, top_m=10)
    assert len(got) <= 10
    enc = TripleEncoder(d=D, entity_dim=8)
    g = enc(got)
    assert g.shape == (len(got), D)
    assert enc([]).shape == (0, D)


def test_kg_filter_eq_15_17():
    seed_everything(0)
    filt = KGFilter(d=D)
    t_k = torch.randn(D)
    g = torch.randn(4, D)
    s, g_tilde = filt(t_k, g)
    assert s.shape == (4,) and (s >= 0).all() and (s <= 1).all()      # Eq. 15
    manual = (s.unsqueeze(-1) * g).sum(0) / (s.sum() + filt.eps)      # Eq. 17
    assert torch.allclose(g_tilde, manual, atol=1e-6)
    # empty -> zero vector
    s0, g0 = filt(t_k, torch.zeros(0, D))
    assert s0.shape == (0,) and torch.allclose(g0, torch.zeros(D))


def test_all_fusions_shape_and_grad():
    seed_everything(0)
    t, v, g = (torch.randn(3, D, requires_grad=True) for _ in range(3))
    for name, cls in FUSION_REGISTRY.items():
        fusion = cls(d=D)
        z = fusion(t, v, g)
        assert z.shape == (3, D), name
        z.sum().backward(retain_graph=True)


def test_kan_backend_runs():
    # default backend (efficient_kan if installed, else vendored RBF-KAN)
    kan = KANFusion(d=D, hidden=(8,))
    z = kan(torch.randn(2, D), torch.randn(2, D), torch.randn(2, D))
    assert z.shape == (2, D)
    # vendored fallback is a genuine KAN
    net = RBFKAN([3 * D, 8, D])
    assert net(torch.randn(2, 3 * D)).shape == (2, D)
