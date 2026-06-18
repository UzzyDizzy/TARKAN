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


@torch.no_grad()
def predict_joint(model, loader, device: str = None) -> Tuple[List, List]:
    device = device or CONFIG.device
    model.eval()
    preds, golds = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        text_feats = model.text_encoder(batch["input_ids"], batch["attention_mask"])
        tag_logits = model.bio_head(text_feats)  # [B, n, 7]
        for b in range(tag_logits.size(0)):
            wt = decode_word_tags(tag_logits[b], batch["word_ids"][b], batch["n_words"][b])
            preds.append([(s, e, pol) for (s, e, pol) in bio_to_spans(wt)])
            golds.append([(s, e, pol) for (s, e, pol) in batch["gold_aspects"][b]])
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
        for a, (b, k) in enumerate(order):
            if a >= asc.size(0):
                break
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
