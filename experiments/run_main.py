"""Table 1 — main joint MABSA results for TARKAN on Twitter-2015 & 2017.

Trains the full model per dataset, evaluates on test, writes results/tables/main_results.csv.
Baseline rows are cited from the paper (Table 1); we report the TARKAN (Ours) row.
"""
from dataclasses import replace

from _common import CONFIG, ROOT, flatten_metrics, write_table
from train import train, make_loader
from evaluate import evaluate_all


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        cfg = replace(CONFIG, device=args.device)
        res = train(cfg, dataset=ds, max_epochs=args.epochs)
        model = res["model"]
        test_loader = make_loader(ds, "test", cfg, shuffle=False)
        metrics = evaluate_all(model, test_loader, cfg.device)
        rows.append({"model": "TARKAN", "dataset": ds, "best_dev_F1": round(res["best_dev_f1"], 2),
                     **{k: round(v, 2) for k, v in flatten_metrics(metrics).items()}})
        print(ds, metrics)
    write_table(rows, ROOT / "results" / "tables" / "main_results.csv")


if __name__ == "__main__":
    main()
