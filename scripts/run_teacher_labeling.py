"""One-time offline teacher-labeling pass (paper Algorithm 1 lines 7-11, Table 4).

Runs the captioner + LLM teacher over a split to produce r^T_k and s^T_kq, cached to
data/teacher_labels/. RUN THIS ON THE T4 SERVER (heavy). Training then reads the cache
and never loads the LLM. Resumable: already-labeled (instance, aspect[, triple]) skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG  # noqa: E402
from captioner import Captioner  # noqa: E402
from data import build_queries, load_split  # noqa: E402
from kg import KnowledgeGraph  # noqa: E402
from kg_retrieval import _triple_key as triple_key, retrieve_triples  # noqa: E402
from teacher import LLMTeacher, TeacherCache  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("teacher_labeling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--device", default=CONFIG.device)
    ap.add_argument("--limit", type=int, default=None, help="cap instances (debug)")
    args = ap.parse_args()

    CONFIG.device = args.device
    images_dir = CONFIG.paths.data / "images" / args.dataset
    sqlite = CONFIG.paths.kg_index / "kg.sqlite"
    kg = KnowledgeGraph(sqlite_path=str(sqlite)) if sqlite.exists() else None
    captioner = Captioner(device=args.device)
    teacher = LLMTeacher(device=args.device)

    prev = TeacherCache.load(args.dataset)
    rel_rows, kg_rows = [], []

    for split in args.splits:
        insts = load_split(CONFIG.paths.data / args.dataset, split)
        if args.limit:
            insts = insts[: args.limit]
        log.info(f"{split}: {len(insts)} instances")
        for n, inst in enumerate(insts):
            tweet = " ".join(inst.tokens)
            try:
                caption = captioner.caption_image(images_dir / inst.image_id)
            except Exception:
                caption = ""
            queries = build_queries(inst, {inst.image_id: caption})
            for k, ((s, e, pol), term) in enumerate(zip(inst.aspects, inst.aspect_terms)):
                if (inst.id, k) not in prev.rel:
                    r = teacher.relevance_label(tweet, term, caption)
                    rel_rows.append({"instance_id": inst.id, "aspect_idx": k, "label": r})
                if kg is not None:
                    for tr in retrieve_triples(queries[k], kg, CONFIG.top_m_triples):
                        tk = triple_key(tr)
                        if (inst.id, k, tk) in prev.kg:
                            continue
                        lbl = teacher.kg_label(tweet, term, tr)
                        kg_rows.append({"instance_id": inst.id, "aspect_idx": k, "triple_key": tk, "label": lbl})
            if (n + 1) % 100 == 0:
                log.info(f"  labeled {n+1}/{len(insts)}")

    # merge with previous and save
    import pandas as pd  # noqa: F401

    all_rel = [{"instance_id": i, "aspect_idx": k, "label": v} for (i, k), v in prev.rel.items()] + rel_rows
    all_kg = [{"instance_id": i, "aspect_idx": k, "triple_key": t, "label": v} for (i, k, t), v in prev.kg.items()] + kg_rows
    TeacherCache.save(args.dataset, all_rel, all_kg)
    log.info(f"saved {len(all_rel)} relevance + {len(all_kg)} kg labels for {args.dataset}")


if __name__ == "__main__":
    main()
