"""Histogram of predicted aspect-visual relevance scores r_k over the test set."""
from dataclasses import replace

import torch

from _viz import ROOT, plt, save  # noqa: E402

from config import CONFIG  # noqa: E402
from models import TarkanStudent  # noqa: E402
from train import build_kg, make_loader  # noqa: E402
from utils import load_checkpoint  # noqa: E402


@torch.no_grad()
def collect_relevance(model, loader):
    vals = []
    for batch in loader:
        out = model(batch)
        vals.extend(out["relevance"].detach().cpu().tolist())
    return vals


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--device", default=CONFIG.device)
    args = ap.parse_args()
    cfg = replace(CONFIG, device=args.device)
    model = TarkanStudent(cfg, kg=build_kg()).to(cfg.device)
    ck = cfg.paths.checkpoints / f"{args.dataset}_best.pt"
    if ck.exists():
        load_checkpoint(model, ck, map_location=cfg.device)
    model.eval()
    vals = collect_relevance(model, make_loader(args.dataset, "test", cfg, shuffle=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=20, range=(0, 1), color="steelblue")
    ax.set_xlabel("aspect-visual relevance r_k")
    ax.set_ylabel("# aspects")
    ax.set_title(f"Relevance score distribution ({args.dataset})")
    save(fig, f"relevance_hist_{args.dataset}.png")


if __name__ == "__main__":
    main()
