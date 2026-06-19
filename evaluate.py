"""Inference + evaluation (paper Algorithm 2, §5.1).

Joint MABSA output = BIO-decoded (span, polarity) pairs (paper §9 default). The
auxiliary ASC head is used for the MASC subtask (polarity on gold aspects).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from config import CONFIG, ID2TAG, ID2POL
from metrics import joint_prf, mate_prf, masc_acc_f1
from utils import bio_to_spans


def _to_device(batch: Dict, device: str) -> Dict:
    out = dict(batch)
    for k in ("input_ids", "attention_mask", "bio_labels", "pixel_values"):
        if k in batch and torch.is_tensor(batch[k]):
            out[k] = batch[k].to(device)
    return out


def decode_word_tags(tag_logits_b: torch.Tensor, word_ids: List[int], n_words: int) -> List[str]:
    """Map subtoken tag predictions -> word-level BIO (tag at each word's first subtoken)."""
    sub = tag_logits_b.argmax(-1).tolist()
    word_tag = ["O"] * n_words
    seen = set()
    for i, wid in enumerate(word_ids):
        if wid == -1 or wid >= n_words or wid in seen:
            continue
        seen.add(wid)
        word_tag[wid] = ID2TAG[sub[i]]
    return word_tag


def _pred_subspans_for(word_ids, span):
    """Map a predicted WORD span (s,e_excl) -> subtoken index range (mirrors data._subtoken_spans)."""
    s, e, *_ = span
    idxs = [i for i, wid in enumerate(word_ids) if isinstance(wid, int) and wid >= 0 and s <= wid < e]
    return (idxs[0], idxs[-1] + 1) if idxs else (0, 1)


@torch.no_grad()
def predict_joint(model, loader, device: str = None) -> Tuple[List, List]:
    """Joint MABSA (span, polarity) per instance.

    Spans always come from the BIO tagging head. The final polarity source is
    cfg.joint_polarity_source (queries.md A8):
      'bio' -> polarity from the BIO tags (default).
      'asc' -> polarity from the KAN-fused ASC head (Eq. 23) re-run on the PREDICTED
               spans (paper §3.7), so visual/KG/KAN influence the joint metric.
    """
    device = device or CONFIG.device
    model.eval()
    mode = getattr(model.cfg, "joint_polarity_source", "bio")
    preds, golds = [], []

    id2inst, captions = {}, {}
    if mode == "asc":
        from data import opinion_words, visual_concepts  # lazy: rebuild KG queries for predicted spans
        from kg_retrieval import AspectQuery
        ds = getattr(loader, "dataset", None)
        for inst in (getattr(ds, "instances", None) or []):
            id2inst[inst.id] = inst
        captions = getattr(ds, "captions", None) or {}

    for batch in loader:
        batch = _to_device(batch, device)
        text_feats = model.text_encoder(batch["input_ids"], batch["attention_mask"])
        tag_logits = model.bio_head(text_feats)  # [B, n, 7]
        B = tag_logits.size(0)

        batch_spans = []
        for b in range(B):
            wt = decode_word_tags(tag_logits[b], batch["word_ids"][b], batch["n_words"][b])
            batch_spans.append(bio_to_spans(wt))
            golds.append([(s, e, pol) for (s, e, pol) in batch["gold_aspects"][b]])

        if mode != "asc":
            for spans in batch_spans:
                preds.append([(s, e, pol) for (s, e, pol) in spans])
            continue

        # --- 'asc': re-decode polarity from the fused ASC head over the predicted spans ---
        visual_feats = None
        if model.cfg.use_visual_stream and model.visual_encoder is not None and "pixel_values" in batch:
            visual_feats = model.visual_encoder(batch["pixel_values"])

        sub_spans, queries = [], []
        for b in range(B):
            inst = id2inst.get(batch["instance_id"][b])
            tokens = inst.tokens if inst is not None else None
            vc = visual_concepts(captions.get(inst.image_id, "")) if inst is not None else []
            subs, qs = [], []
            for sp in batch_spans[b]:
                subs.append(_pred_subspans_for(batch["word_ids"][b], sp))
                s, e = sp[0], sp[1]
                if tokens is not None:
                    qs.append(AspectQuery(aspect_term=" ".join(tokens[s:e]),
                                          opinion_words=opinion_words(tokens, (s, e)), visual_concepts=vc))
                else:
                    qs.append(AspectQuery(aspect_term=""))
            sub_spans.append(subs)
            queries.append(qs)

        mod = dict(batch)
        mod["aspect_spans"] = sub_spans
        mod["aspect_queries"] = queries
        mod.pop("aspect_triples", None)
        out = model(mod, text_feats=text_feats, visual_feats=visual_feats)
        asc = out["asc_logits"]  # rows flattened over (b, k) in instance order

        a = 0
        for b in range(B):
            inst_preds = []
            for (s, e, pol_bio) in batch_spans[b]:
                pol = ID2POL[int(asc[a].argmax().item())] if a < asc.size(0) else pol_bio
                a += 1
                inst_preds.append((s, e, pol))
            preds.append(inst_preds)
    return preds, golds


@torch.no_grad()
def predict_masc(model, loader, device: str = None) -> Tuple[List[str], List[str]]:
    """Polarity on gold aspects via the ASC head (Table 3 MASC)."""
    device = device or CONFIG.device
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        asc = out["asc_logits"]
        order = [(b, k) for b in range(len(batch["aspect_spans"])) for k in range(len(batch["aspect_spans"][b]))]
        # asc rows must line up 1:1 with `order` (the flattened aspect list). If a future
        # change drops aspects inside forward(), fail loudly rather than silently skew MASC.
        assert asc.size(0) == len(order), f"asc rows {asc.size(0)} != aspects {len(order)}"
        for a, (b, k) in enumerate(order):
            y_true.append(ID2POL[batch["aspect_polarity"][b][k]])
            y_pred.append(ID2POL[int(asc[a].argmax().item())])
    return y_true, y_pred


def evaluate_all(model, loader, device: str = None) -> Dict[str, Dict]:
    preds, golds = predict_joint(model, loader, device)
    yt, yp = predict_masc(model, loader, device)
    return {
        "joint": joint_prf(preds, golds),
        "mate": mate_prf(preds, golds),
        "masc": masc_acc_f1(yt, yp),
    }


if __name__ == "__main__":
    import argparse

    from data import TarkanDataset, collate_fn, load_split
    from models import TarkanStudent
    from utils import load_checkpoint, get_logger

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    log = get_logger("evaluate")

    from torch.utils.data import DataLoader

    data_dir = CONFIG.paths.data / args.dataset
    images = CONFIG.paths.data / "images" / args.dataset
    insts = load_split(data_dir, args.split)
    ds = TarkanDataset(insts, CONFIG, images_dir=images)
    loader = DataLoader(ds, batch_size=CONFIG.batch_size, collate_fn=collate_fn)
    model = TarkanStudent(CONFIG).to(CONFIG.device)
    load_checkpoint(model, args.checkpoint, map_location=CONFIG.device)
    log.info(evaluate_all(model, loader))
