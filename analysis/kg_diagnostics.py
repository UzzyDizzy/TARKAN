"""Table 8 — KG retrieval & filtering diagnostics.

Reports: aspect KG match rate, avg retrieved vs retained triples/aspect, and the
SenticNet vs ConceptNet contribution split among retained triples. Needs a trained
checkpoint (for the usefulness scores s_kq) and the built KG index.
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from train import make_loader, build_kg  # noqa: E402
from models import TarkanStudent  # noqa: E402
from evaluate import _to_device  # noqa: E402
from utils import load_checkpoint, get_logger  # noqa: E402

log = get_logger("kg_diagnostics")


@torch.no_grad()
def diagnose(model, loader, thresh=0.5, device=None):
    device = device or CONFIG.device
    n_aspects = matched = retrieved = retained = 0
    src = {"conceptnet": 0, "senticnet": 0}
    for batch in loader:
        out = model(_to_device(batch, device))
        for a, triples in enumerate(out["kg_triples"]):
            n_aspects += 1
            retrieved += len(triples)
            if triples:
                matched += 1
            s = out["kg_scores"][a] if a < len(out["kg_scores"]) else torch.zeros(0)
            keep = (s > thresh)
            retained += int(keep.sum().item())
            for i, tr in enumerate(triples):
                if i < keep.numel() and bool(keep[i]):
                    src[tr.source] = src.get(tr.source, 0) + 1
    tot_src = max(1, src["conceptnet"] + src["senticnet"])
    return {
        "aspect_KG_match_rate": round(100 * matched / max(1, n_aspects), 1),
        "avg_retrieved_per_aspect": round(retrieved / max(1, n_aspects), 1),
        "avg_retained_per_aspect": round(retained / max(1, n_aspects), 1),
        "SenticNet_contribution": round(100 * src["senticnet"] / tot_src, 1),
        "ConceptNet_contribution": round(100 * src["conceptnet"] / tot_src, 1),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    ap.add_argument("--device", default=CONFIG.device)
    args = ap.parse_args()

    kg = build_kg()
    if kg is None:
        log.warning(f"NO KG index at {CONFIG.paths.kg_index / 'kg.sqlite'}; all KG stats will be 0. "
                    f"Build it: python data_setup.py  (step 4)")
    rows = []
    for ds in args.datasets:
        cfg = replace(CONFIG, device=args.device)
        ck = cfg.paths.checkpoints / f"{ds}_best.pt"
        if not ck.exists():
            log.warning(f"NO checkpoint at {ck}; skipping {ds} (usefulness scores would be random). "
                        f"Run: python train.py --dataset {ds} --device {args.device}")
            continue
        model = TarkanStudent(cfg, kg=kg).to(cfg.device)
        load_checkpoint(model, ck, map_location=cfg.device)
        model.eval()
        stats = diagnose(model, make_loader(ds, "test", cfg, shuffle=False), device=cfg.device)
        rows.append({"dataset": ds, **stats})
        print(ds, stats)
    if not rows:
        log.warning("no datasets diagnosed (all skipped); nothing written.")
        return
    out = ROOT / "results" / "tables" / "kg_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
