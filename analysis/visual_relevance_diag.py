"""Table 9 — performance under different visual-relevance conditions.

Buckets test instances into: image-useful, image-irrelevant, weak image-text
correspondence, and multiple-aspect; reports joint F1 per bucket. Uses cached teacher
relevance labels (useful/irrelevant) and captions (weak correspondence) when available.
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from captioner import load_captions  # noqa: E402
from data import load_split  # noqa: E402
from evaluate import predict_joint  # noqa: E402
from metrics import joint_prf  # noqa: E402
from models import TarkanStudent  # noqa: E402
from train import build_kg, make_loader  # noqa: E402
from teacher import TeacherCache  # noqa: E402
from utils import load_checkpoint  # noqa: E402


def _bucket(inst, cache, captions):
    if len(inst.aspects) > 1:
        yield "multiple-aspect"
    rels = [cache.rel.get((inst.id, k)) for k in range(len(inst.aspects))]
    if any(r == 1 for r in rels if r is not None):
        yield "image-useful"
    if rels and all(r == 0 for r in rels if r is not None) and any(r is not None for r in rels):
        yield "image-irrelevant"
    cap = captions.get(inst.image_id, "")
    if cap:
        overlap = len(set(cap.lower().split()) & set(" ".join(inst.tokens).lower().split()))
        if overlap <= 1:
            yield "weak image-text correspondence"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    ap.add_argument("--device", default=CONFIG.device)
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        cfg = replace(CONFIG, device=args.device)
        model = TarkanStudent(cfg, kg=build_kg()).to(cfg.device)
        ck = cfg.paths.checkpoints / f"{ds}_best.pt"
        if ck.exists():
            load_checkpoint(model, ck, map_location=cfg.device)
        loader = make_loader(ds, "test", cfg, shuffle=False)
        preds, golds = predict_joint(model, loader, cfg.device)
        insts = load_split(cfg.paths.data / ds, "test")
        cache = TeacherCache.load(ds)
        captions = load_captions(ds)

        buckets = {}
        for i, inst in enumerate(insts):
            for b in _bucket(inst, cache, captions):
                buckets.setdefault(b, ([], []))
                buckets[b][0].append(preds[i])
                buckets[b][1].append(golds[i])
        for b, (p, g) in buckets.items():
            rows.append({"dataset": ds, "condition": b, "n": len(p), "TARKAN_F1": round(joint_prf(p, g)["F1"], 1)})
            print(ds, b, len(p), round(joint_prf(p, g)["F1"], 1))

    if rows:
        out = ROOT / "results" / "tables" / "visual_relevance_conditions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
