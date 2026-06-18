"""Shared helpers for experiment/ablation runners."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402


def flatten_metrics(m: dict) -> dict:
    return {
        "joint_P": m["joint"]["P"], "joint_R": m["joint"]["R"], "joint_F1": m["joint"]["F1"],
        "mate_F1": m["mate"]["F1"], "masc_Acc": m["masc"]["Acc"], "masc_F1": m["masc"]["F1"],
    }


def write_table(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
