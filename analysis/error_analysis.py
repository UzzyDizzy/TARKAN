"""Table 5 — error-type distribution over test errors.

Heuristic auto-categorizer (a starting point for manual inspection, which the paper
used). Categories: ambiguous/sarcastic text, aspect boundary error, misleading/irrelevant
image, incorrect KG evidence, implicit sentiment without clear cue.
"""
import csv
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from data import load_split  # noqa: E402
from evaluate import predict_joint  # noqa: E402
from models import TarkanStudent  # noqa: E402
from train import build_kg, make_loader  # noqa: E402
from teacher import TeacherCache  # noqa: E402
from utils import load_checkpoint  # noqa: E402

SARCASM = {"lol", "yeah", "sure", "great", "wow", "totally", "#sarcasm", "/s"}


def categorize(inst, pred_set, gold_set, cache):
    gold_spans = {(s, e) for (s, e, _) in gold_set}
    pred_spans = {(s, e) for (s, e, _) in pred_set}
    if any(any(ps != gs and (ps[0] < gs[1] and gs[0] < ps[1]) for gs in gold_spans) for ps in pred_spans):
        return "aspect boundary error"
    text = " ".join(inst.tokens).lower()
    if any(w in text for w in SARCASM):
        return "ambiguous / sarcastic text"
    rels = [cache.rel.get((inst.id, k)) for k in range(len(inst.aspects))]
    if rels and any(r == 0 for r in rels if r is not None):
        return "misleading or irrelevant image"
    # span right, polarity wrong -> KG/implicit
    span_correct = pred_spans & gold_spans
    if span_correct:
        return "incorrect KG evidence"
    return "implicit sentiment without clear cue"


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
        counts = Counter()
        for i, inst in enumerate(insts):
            if set(preds[i]) != set(golds[i]):
                counts[categorize(inst, set(preds[i]), set(golds[i]), cache)] += 1
        for cat, n in counts.most_common():
            rows.append({"dataset": ds, "error_type": cat, "count": n})
        print(ds, dict(counts))

    if rows:
        out = ROOT / "results" / "tables" / "error_distribution.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out} (NOTE: heuristic; paper used manual inspection)")


if __name__ == "__main__":
    main()
