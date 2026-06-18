"""Overlay aspect->patch attention (alpha_kj, Eq. 7) on the image for one instance."""
from dataclasses import replace

import numpy as np
import torch

from _viz import plt, save
from config import CONFIG
from data import TarkanDataset, collate_fn, load_split
from models import TarkanStudent
from train import build_kg
from utils import load_checkpoint


@torch.no_grad()
def main():
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--aspect", type=int, default=0)
    ap.add_argument("--device", default=CONFIG.device)
    args = ap.parse_args()

    cfg = replace(CONFIG, device=args.device)
    insts = load_split(cfg.paths.data / args.dataset, "test")
    images = cfg.paths.data / "images" / args.dataset
    ds = TarkanDataset([insts[args.index]], cfg, images_dir=images)
    batch = collate_fn([ds[0]])

    model = TarkanStudent(cfg, kg=build_kg()).to(cfg.device)
    ck = cfg.paths.checkpoints / f"{args.dataset}_best.pt"
    if ck.exists():
        load_checkpoint(model, ck, map_location=cfg.device)
    model.eval()
    out = model(batch, want_alpha=True)
    alphas = out["alpha"]
    if not alphas:
        print("no aspects/alpha for this instance")
        return
    alpha = alphas[args.aspect].cpu().numpy()        # [m]
    g = int(round(len(alpha) ** 0.5))                # 49 -> 7x7
    heat = alpha[: g * g].reshape(g, g)

    img = Image.open(images / insts[args.index].image_id).convert("RGB").resize((224, 224))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.array(img))
    ax.imshow(np.kron(heat, np.ones((224 // g, 224 // g))), cmap="jet", alpha=0.45)
    ax.set_title(f"aspect '{insts[args.index].aspect_terms[args.aspect]}' attention")
    ax.axis("off")
    save(fig, f"attention_{args.dataset}_{args.index}_{args.aspect}.png")


if __name__ == "__main__":
    main()
