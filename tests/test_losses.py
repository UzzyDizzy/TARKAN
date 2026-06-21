"""Loss equations (updated §3.7): L = L_tag + λ1 L_rel + λ2 L_kg (no auxiliary ASC loss)."""
import torch

from config import CONFIG
from losses import compute_losses, kg_loss, relevance_loss, tag_loss


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


def test_total_objective():
    outputs = {
        "tag_logits": torch.randn(1, 3, 7),
        "relevance": torch.tensor([0.6, 0.4]),
        "kg_scores": [torch.tensor([0.5]), torch.tensor([0.5])],
    }
    targets = {
        "bio_labels": torch.tensor([[1, 0, 0]]),
        "teacher_relevance": torch.tensor([1.0, 0.0]),
        "teacher_relevance_mask": torch.tensor([True, True]),
        "teacher_kg": [torch.tensor([1.0]), torch.tensor([0.0])],
        "teacher_kg_mask": [torch.tensor([True]), torch.tensor([True])],
    }
    full = compute_losses(outputs, targets, CONFIG)
    expected = full["l_tag"] + CONFIG.lambda1 * full["l_rel"] + CONFIG.lambda2 * full["l_kg"]
    assert torch.allclose(full["total"], expected)
    assert "l_asc" not in full  # auxiliary ASC loss removed in the updated methodology
