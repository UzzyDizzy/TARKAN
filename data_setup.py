"""ONE-TIME heavy setup — RUN THIS ON THE GPU SERVER FIRST.

Downloads/builds everything that must NOT be git-pushed (all lands under data/, which
is git-ignored):
  1. Twitter-2015/2017 data + images       (scripts/prepare_data.py)
  2. ConceptNet 5.7 English + Numberbatch   (scripts/download_conceptnet.py)
  3. SenticNet 7 English                     (scripts/download_senticnet.py)
  4. unified KG sqlite index                 (scripts/build_kg.py)
  5. (optional) teacher labels r^T, s^T      (scripts/run_teacher_labeling.py)  [needs GPU]

Model weights (BERTweet, CLIP, Qwen teacher, BLIP) are NOT downloaded here — they
auto-download on first use by the training/teacher code, cached by HuggingFace.

    python data_setup.py                         # steps 1-4 (CPU-friendly)
    python data_setup.py --senticnet-rdf senticnet7.rdf
    python data_setup.py --teacher --device cuda # also run teacher labeling on GPU
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils import get_logger  # noqa: E402

log = get_logger("data_setup")
PY = sys.executable
S = ROOT / "scripts"


def run(args: list, label: str) -> None:
    log.info(f"=== {label} ===")
    subprocess.run([PY, *[str(a) for a in args]], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-conceptnet", action="store_true")
    ap.add_argument("--skip-senticnet", action="store_true")
    ap.add_argument("--skip-kg", action="store_true")
    ap.add_argument("--senticnet-rdf", default=None, help="path to official SenticNet 7 RDF/XML")
    ap.add_argument("--senticnet-py", default=None, help="path to official senticnet.py (default: auto-find data/senticnet/senticnet.py)")
    ap.add_argument("--senticnet-git", default=None, help="git URL to fetch senticnet.py from if not present locally")
    ap.add_argument("--teacher", action="store_true", help="also run teacher labeling (GPU)")
    ap.add_argument("--datasets", nargs="+", default=["twitter2015", "twitter2017"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not args.skip_data:
        run([S / "prepare_data.py"], "1/5 datasets + images")
    if not args.skip_conceptnet:
        run([S / "download_conceptnet.py", "--numberbatch"], "2/5 ConceptNet 5.7 (EN) + Numberbatch")
    if not args.skip_senticnet:
        sn = [S / "download_senticnet.py"]
        if args.senticnet_rdf:
            sn += ["--rdf", args.senticnet_rdf]
        elif args.senticnet_py:
            sn += ["--py", args.senticnet_py]
        elif args.senticnet_git:
            sn += ["--git", args.senticnet_git]
        # else: download_senticnet auto-finds data/senticnet/senticnet.py
        run(sn, "3/5 SenticNet 7 (EN)")
    if not args.skip_kg:
        run([S / "build_kg.py"], "4/5 build unified KG index")
    if args.teacher:
        for ds in args.datasets:
            run([S / "run_teacher_labeling.py", "--dataset", ds, "--device", args.device],
                f"5/5 teacher labeling [{ds}]")
    log.info("data_setup complete. Next: python train.py --dataset twitter2015 --device cuda")


if __name__ == "__main__":
    main()
