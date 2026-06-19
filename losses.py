"""TARKAN training objective (paper Eqs. 11, 16, 22, 24, 25).

    L_tag = - sum_i log p(b*_i | T, I)                                    (Eq. 22)
    L_rel = - sum_k [ r^T_k log r_k + (1-r^T_k) log(1-r_k) ]             (Eq. 11)
    L_kg  = - sum_k sum_q [ s^T_kq log s_kq + (1-s^T_kq) log(1-s_kq) ]   (Eq. 16)
    L_asc = - sum_k log p(y*_k | a_k, T, I)                              (Eq. 24)

    L = L_tag + λ1 L_rel + λ2 L_kg ( + λ3 L_asc )                        (Eq. 25 + aux)

Eq. 25 as written omits L_asc, but §3.7 + Table 6 require it (Open-Q #1):
set cfg.lambda3 = 0.0 to reproduce Eq.25-exact / the "w/o auxiliary ASC loss" ablation.
cfg.use_teacher = False zeroes L_rel and L_kg ("w/o LLM teacher guidance").
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from config import CONFIG


def tag_loss(tag_logits: torch.Tensor, bio_labels: torch.Tensor) -> torch.Tensor:
    """Eq. 22. Token CE over 7 BIO classes; -100 positions ignored (subtoken continuations)."""
    B, n, C = tag_logits.shape
    return F.cross_entropy(
        tag_logits.reshape(B * n, C), bio_labels.reshape(B * n), ignore_index=-100, reduction="mean"
    )


def relevance_loss(r: torch.Tensor, teacher_r: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Eq. 11. BCE over aspects that have a teacher relevance label."""
    if r.numel() == 0:
        return r.new_zeros(())
    if mask is not None:
        if mask.sum() == 0:
            return r.new_zeros(())
        r, teacher_r = r[mask], teacher_r[mask]
    r = r.clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(r, teacher_r.float(), reduction="mean")


def kg_loss(
    kg_scores: List[torch.Tensor],
    teacher_kg: List[torch.Tensor],
    teacher_kg_mask: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Eq. 16. BCE over retrieved triples that have a teacher usefulness label."""
    preds, tgts = [], []
    for i, s in enumerate(kg_scores):
        if s.numel() == 0:
            continue
        t = teacher_kg[i]
        if t.numel() != s.numel():
            continue
        if teacher_kg_mask is not None:
            m = teacher_kg_mask[i]
            if m.sum() == 0:
                continue
            s, t = s[m], t[m]
        preds.append(s)
        tgts.append(t.float())
    if not preds:
        # keep graph connected to wg params if everything is unlabeled
        device = kg_scores[0].device if kg_scores else torch.device("cpu")
        return torch.zeros((), device=device)
    p = torch.cat(preds).clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(p, torch.cat(tgts), reduction="mean")


def asc_loss(asc_logits: torch.Tensor, gold_polarity: torch.Tensor) -> torch.Tensor:
    """Eq. 24. Span-level polarity CE over the KAN-fused representation."""
    if asc_logits.numel() == 0 or gold_polarity.numel() == 0:
        return asc_logits.new_zeros(())
    return F.cross_entropy(asc_logits, gold_polarity, reduction="mean")


def compute_losses(outputs: Dict, targets: Dict, cfg=CONFIG) -> Dict[str, torch.Tensor]:
    """Returns dict with l_tag, l_rel, l_kg, l_asc, total (Eq. 25 + aux)."""
    l_tag = tag_loss(outputs["tag_logits"], targets["bio_labels"])

    if cfg.use_teacher and cfg.use_relevance and cfg.use_visual_stream and "teacher_relevance" in targets:
        l_rel = relevance_loss(
            outputs["relevance"], targets["teacher_relevance"], targets.get("teacher_relevance_mask")
        )
    else:
        l_rel = l_tag.new_zeros(())

    if cfg.use_teacher and cfg.use_kg_stream and cfg.use_kg_filter and "teacher_kg" in targets:
        l_kg = kg_loss(outputs["kg_scores"], targets["teacher_kg"], targets.get("teacher_kg_mask"))
    else:
        l_kg = l_tag.new_zeros(())

    if cfg.lambda3 > 0 and "aspect_polarity" in targets:
        l_asc = asc_loss(outputs["asc_logits"], targets["aspect_polarity"])
    else:
        l_asc = l_tag.new_zeros(())

    total = l_tag + cfg.lambda1 * l_rel + cfg.lambda2 * l_kg + cfg.lambda3 * l_asc
    return {"l_tag": l_tag, "l_rel": l_rel, "l_kg": l_kg, "l_asc": l_asc, "total": total}
