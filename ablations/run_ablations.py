"""Table 6 — component ablations. Each variant toggles exactly one mechanism.

Variants (paper Table 6):
  TARKAN | w/o LLM teacher | w/o aspect-visual relevance | w/o KG evidence filtering
  | w/o KG stream | w/o KAN fusion (MLP) | w/o visual stream | w/o auxiliary ASC loss
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from train import train, make_loader, build_kg  # noqa: E402
from evaluate import evaluate_all  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("run_ablations")

VARIANTS = {
    "TARKAN": {},
    "w/o LLM teacher": {"use_teacher": False},
    "w/o aspect-visual relevance": {"use_relevance": False},
    "w/o KG evidence filtering": {"use_kg_filter": False},
    "w/o KG stream": {"use_kg_stream": False},
    "w/o KAN fusion (MLP)": {"fusion": "concat_mlp"},
    "w/o visual stream": {"use_visual_stream": False},
    "w/o auxiliary ASC loss": {"lambda3": 0.0},
}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    if build_kg() is None:
        log.warning("NO KG index built -> 'w/o KG stream' and 'w/o KG evidence filtering' will be "
                    "no-ops identical to TARKAN (the KG branch is inert without data/kg_index/kg.sqlite). "
                    "Run data_setup.py (step 4) before reproducing Table 6.")

    rows = []
    for name, overrides in VARIANTS.items():
        row = {"variant": name}
        for ds in args.datasets:
            cfg = replace(CONFIG, device=args.device, **overrides)
            res = train(cfg, dataset=ds, max_epochs=args.epochs)
            m = evaluate_all(res["model"], make_loader(ds, "test", cfg, shuffle=False), cfg.device)
            row[f"{ds}_F1"] = round(m["joint"]["F1"], 2)
        rows.append(row)
        print(row)

    out = ROOT / "results" / "tables" / "ablation_components.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
