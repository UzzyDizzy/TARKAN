"""Training (paper Algorithm 1, Eq. 25).

Loads cached teacher labels (no LLM at train time), runs the student forward, computes
L = L_tag + λ1 L_rel + λ2 L_kg + λ3 L_asc, early-stops on dev joint-F1.

CPU-runnable for smoke tests; the real runs go on the T4 server (set CONFIG.device='cuda').
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import CONFIG
from data import TarkanDataset, collate_fn, load_split
from evaluate import evaluate_all
from losses import compute_losses
from models import TarkanStudent
from seeding import seed_everything, worker_init_fn
from teacher import TeacherCache, build_targets
from utils import get_logger, save_checkpoint

log = get_logger("train")


def build_kg():
    """Load the built KG index if present; else return None (KG stream becomes inert)."""
    from kg import KnowledgeGraph

    sqlite = CONFIG.paths.kg_index / "kg.sqlite"
    if sqlite.exists():
        return KnowledgeGraph(sqlite_path=str(sqlite))
    return None


def make_loader(dataset: str, split: str, cfg, shuffle: bool, captions=None) -> DataLoader:
    data_dir = cfg.paths.data / dataset
    images = cfg.paths.data / "images" / dataset
    insts = load_split(data_dir, split)
    ds = TarkanDataset(insts, cfg, captions=captions, images_dir=images)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_fn, worker_init_fn=worker_init_fn
    )


def train(cfg=CONFIG, dataset: str = "twitter2015", max_epochs: Optional[int] = None) -> dict:
    seed_everything(cfg.seed)
    device = cfg.device
    max_epochs = max_epochs or cfg.max_epochs

    kg = build_kg()
    entity_embedder = None
    nb = cfg.paths.conceptnet / "numberbatch-en.txt"
    if nb.exists():
        from kg_retrieval import EntityEmbedder

        entity_embedder = EntityEmbedder.from_txt(str(nb))

    model = TarkanStudent(cfg, kg=kg, entity_embedder=entity_embedder).to(device)
    cache = TeacherCache.load(dataset)

    train_loader = make_loader(dataset, "train", cfg, shuffle=True)
    dev_loader = make_loader(dataset, "dev", cfg, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max_epochs * max(1, len(train_loader))
    warmup = int(cfg.warmup_ratio * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, s / max(1, warmup)) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )

    best_f1, best_state, patience = -1.0, None, 0
    cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(batch_dev)
            targets = build_targets(batch_dev, outputs, cache, cfg)
            losses = compute_losses(outputs, targets, cfg)
            optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            running += float(losses["total"].item())

        metrics = evaluate_all(model, dev_loader, device)
        dev_f1 = metrics["joint"]["F1"]
        log.info(f"epoch {epoch}: train_loss={running/max(1,len(train_loader)):.4f} dev_joint_F1={dev_f1:.2f} {metrics}")

        if dev_f1 > best_f1:
            best_f1, patience = dev_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, cfg.paths.checkpoints / f"{dataset}_best.pt", {"dev_f1": dev_f1, "epoch": epoch})
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                log.info(f"early stop at epoch {epoch} (best dev F1 {best_f1:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_dev_f1": best_f1, "model": model}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda1", type=float, default=CONFIG.lambda1)
    ap.add_argument("--lambda2", type=float, default=CONFIG.lambda2)
    ap.add_argument("--fusion", default=CONFIG.fusion)
    args = ap.parse_args()
    cfg = replace(CONFIG, device=args.device, lambda1=args.lambda1, lambda2=args.lambda2, fusion=args.fusion)
    res = train(cfg, dataset=args.dataset, max_epochs=args.epochs)
    log.info(f"done. best dev joint-F1 = {res['best_dev_f1']:.2f}")
