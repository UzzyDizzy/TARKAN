"""Table 3 — subtask results (MATE span F1 + MASC polarity Acc/F1) from a checkpoint."""
from dataclasses import replace

from _common import CONFIG, ROOT, write_table
from train import make_loader
from evaluate import evaluate_all
from models import TarkanStudent
from utils import load_checkpoint, get_logger

log = get_logger("run_subtasks")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    ap.add_argument("--device", default=CONFIG.device)
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        cfg = replace(CONFIG, device=args.device)
        ckpt = cfg.paths.checkpoints / f"{ds}_best.pt"
        if not ckpt.exists():
            log.warning(f"NO checkpoint at {ckpt}; skipping {ds} (an untrained model would write garbage). "
                        f"Run: python train.py --dataset {ds} --device {args.device}")
            continue
        model = TarkanStudent(cfg).to(cfg.device)
        load_checkpoint(model, ckpt, map_location=cfg.device)
        loader = make_loader(ds, "test", cfg, shuffle=False)
        m = evaluate_all(model, loader, cfg.device)
        rows.append({"dataset": ds, "MATE_F1": round(m["mate"]["F1"], 2),
                     "MASC_Acc": round(m["masc"]["Acc"], 2), "MASC_F1": round(m["masc"]["F1"], 2)})
    write_table(rows, ROOT / "results" / "tables" / "subtasks.csv")


if __name__ == "__main__":
    main()
