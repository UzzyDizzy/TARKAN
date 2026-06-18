"""Loss equations (11, 16, 22, 24, 25). lambda3=0 reproduces Eq.25-exact."""
import torch

from config import CONFIG
from dataclasses import replace
from losses import asc_loss, compute_losses, kg_loss, relevance_loss, tag_loss


def test_tag_loss_ignores_minus100():
    logits = torch.zeros(1, 3, 7)
    labels = torch.tensor([[1, -100, 0]])
    loss = tag_loss(logits, labels)
    assert torch.isfinite(loss)


def test_relevance_loss_bce():
    r = torch.tensor([0.9, 0.1])
    tr = torch.tensor([1.0, 0.0])
    loss = relevance_loss(r, tr)
    assert loss.item() < 0.2  # confident-correct -> low BCE


def test_kg_loss_masking():
    scores = [torch.tensor([0.8, 0.2]), torch.tensor([])]
    teacher = [torch.tensor([1.0, 0.0]), torch.tensor([])]
    loss = kg_loss(scores, teacher)
    assert torch.isfinite(loss) and loss.item() < 0.3


def test_total_objective_eq25():
    outputs = {
        "tag_logits": torch.randn(1, 3, 7),
        "asc_logits": torch.randn(2, 3),
        "relevance": torch.tensor([0.6, 0.4]),
        "kg_scores": [torch.tensor([0.5]), torch.tensor([0.5])],
    }
    targets = {
        "bio_labels": torch.tensor([[1, 0, 0]]),
        "aspect_polarity": torch.tensor([0, 1]),
        "teacher_relevance": torch.tensor([1.0, 0.0]),
        "teacher_relevance_mask": torch.tensor([True, True]),
        "teacher_kg": [torch.tensor([1.0]), torch.tensor([0.0])],
        "teacher_kg_mask": [torch.tensor([True]), torch.tensor([True])],
    }
    full = compute_losses(outputs, targets, CONFIG)
    expected = full["l_tag"] + CONFIG.lambda1 * full["l_rel"] + CONFIG.lambda2 * full["l_kg"] + CONFIG.lambda3 * full["l_asc"]
    assert torch.allclose(full["total"], expected)

    # lambda3 = 0 -> Eq.25-exact (no Lasc term)
    cfg0 = replace(CONFIG, lambda3=0.0)
    eq25 = compute_losses(outputs, targets, cfg0)
    exp25 = eq25["l_tag"] + cfg0.lambda1 * eq25["l_rel"] + cfg0.lambda2 * eq25["l_kg"]
    assert torch.allclose(eq25["total"], exp25)
