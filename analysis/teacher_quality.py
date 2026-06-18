"""Table 7 — quality of teacher-generated supervision.

Compares cached teacher labels (r^T, s^T) against a HUMAN-VERIFIED subset and reports
agreement/accuracy + Cohen's kappa. Provide the human labels as:
  data/teacher_labels/<dataset>_relevance_human.parquet  [instance_id, aspect_idx, label]
  data/teacher_labels/<dataset>_kg_human.parquet          [instance_id, aspect_idx, triple_key, label]
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from teacher import TeacherCache  # noqa: E402


def _agreement(teacher: dict, human_rows, keycols):
    from sklearn.metrics import cohen_kappa_score

    yt, yh = [], []
    for r in human_rows:
        key = tuple(r[c] for c in keycols)
        if key in teacher:
            yt.append(int(round(teacher[key])))
            yh.append(int(round(float(r["label"]))))
    if not yt:
        return None
    acc = 100 * sum(int(a == b) for a, b in zip(yt, yh)) / len(yt)
    kappa = cohen_kappa_score(yt, yh) if len(set(yh)) > 1 else 1.0
    return {"n": len(yt), "accuracy": round(acc, 1), "cohen_kappa": round(kappa, 2)}


def main():
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    args = ap.parse_args()
    base = CONFIG.paths.teacher_labels

    rows = []
    for ds in args.datasets:
        cache = TeacherCache.load(ds)
        rel_h = base / f"{ds}_relevance_human.parquet"
        kg_h = base / f"{ds}_kg_human.parquet"
        if rel_h.exists():
            r = _agreement(cache.rel, pd.read_parquet(rel_h).to_dict("records"), ["instance_id", "aspect_idx"])
            if r:
                rows.append({"dataset": ds, "signal": "aspect-visual relevance", **r})
        if kg_h.exists():
            r = _agreement(cache.kg, pd.read_parquet(kg_h).to_dict("records"), ["instance_id", "aspect_idx", "triple_key"])
            if r:
                rows.append({"dataset": ds, "signal": "KG evidence usefulness", **r})

    if not rows:
        print("No human-verified subset found. Place *_relevance_human.parquet / *_kg_human.parquet in",
              base, "to compute Table 7.")
        return
    out = ROOT / "results" / "tables" / "teacher_quality.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
