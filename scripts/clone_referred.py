"""Clone the clone_and_run baseline repos into referred_clones/, strip each .git so the
source commits with OUR repo, and write referred_clones/FIXES.md (provenance + per-repo
compatibility fixes). Heavy artifacts inside clones are git-ignored (see .gitignore).

Run: python scripts/clone_referred.py            (clone all)
     python scripts/clone_referred.py VLP-MABSA  (clone a subset by name)
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _force_rmtree(path: Path) -> None:
    """Remove a tree even with read-only files (Windows .git pack files)."""
    def onerror(func, p, exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import get_logger  # noqa: E402

log = get_logger("clone_referred")
DEST = ROOT / "referred_clones"

# name -> (url, original stack, key compatibility fixes for Py3.10+/torch2.x + CPU/T4)
REPOS = {
    # Priority A — authors re-ran (no * in Table 1)
    "VLP-MABSA": ("https://github.com/NUSTM/VLP-MABSA", "torch1.6, transformers3.4, BART, Faster-RCNN",
        ["migrate transformers 3.4->4.4x (TokenizerFast attrs)", "replace fastnlp", "pin h5py>=3",
         "canonical data/feature source: download feats from its Drive/Baidu (code d0tn)"]),
    "JML": ("https://github.com/MANLP-suda/JML", "py3.6, torch1.1, BERT+Mask/Faster-RCNN",
        ["torch 1.1->2.x rewrite", "transformers API migration", "pin numpy<2",
         "17GB supplementary on Baidu (code 53ej); RCNN feats may need detectron2 backport"]),
    "AoM": ("https://github.com/SilyRab/AoM", "builds on VLP-MABSA, transformers3.4",
        ["same VLP-MABSA dep chain", "reuses VLP-MABSA 36-region 2048-d feats", "expect deprecated-API rewrites"]),
    "M2DF": ("https://github.com/grandchicken/M2DF", "py3.7.13, torch1.12, transformers3.4",
        ["torch1.12 OK on T4", "pin numpy 1.24, fastnlp 0.6/0.7", "h5py wheels", "download feats from Drive"]),
    "CMMT": ("https://github.com/yangli-hub/CMMT-Code", "py3.7, torch1.0, RoBERTa+ResNet152+CRF",
        ["torch 1.0->1.13+", "replace pytorch-crf 0.7.2 -> torchcrf", "transformers 3.4 pin", "CoNLL data + ResNet-152 weights"]),
    "MultiPoint": ("https://github.com/YangXiaocui1215/MultiPoint", "py3.8+, torch1.9+, roberta-large+NF-ResNet50",
        ["most modern of the set", "align sentence-transformers/timm to torch", "Drive data"]),
    "DQPSA": ("https://github.com/pengts/DQPSA", "torch1.13, accelerate+deepspeed, spaCy3.5",
        ["pin transformers<=4.26", "upgrade accelerate/deepspeed or keep torch1.13", "data+ckpts on Baidu (code 2024)", "CPU: strip deepspeed"]),
    "TCMT": ("https://github.com/ZouWang-spider/TCMT", "torch~1.13; YOLOv5+ViT-GPT2+Tesseract+FITE",
        ["install Tesseract (system) + pytesseract", "YOLOv5 pin torch<=1.13", "FITE not public (may block full repro)"]),
    "VLHA": ("https://github.com/ZouWang-spider/VLHA", "Scene-Graph-Benchmark.pytorch + BiAffine",
        ["SGB.pytorch unmaintained -> pin torch<=1.13", "BiAffine needs Cython build", "requirements.txt 404: reverse-engineer deps"]),
    # Priority B — data source / cited multimodal
    "TomBERT": ("https://github.com/jefferyYu/TomBERT", "py3.7, torch1.0, BERT+ResNet-152",
        ["torch 1.0->2.x rewrite", "MASC data source (absa_data, 49 region feats)", "transformers API migration"]),
    "UMT": ("https://github.com/jefferyYu/UMT", "py3.7, torch1.0, BERT+ResNet-152+CRF",
        ["as TomBERT + pytorch-crf pin", "MNER collapsed baselines"]),
    "Atlantis": ("https://github.com/Xillv/Atlantis", "py3.9, torch1.12.1, transformers4.32, FLAN-T5",
        ["nearly modern: bump transformers->4.35, CUDA 11.3->11.8", "see sibling Chimera repo for full env"]),
    "CORSA": ("https://github.com/Liuxj-Anya/CORSA", "py3.8+, torch1.9+, YOLO/Faster-RCNN",
        ["builds C-MABSA from Twitter", "torchvision 0.15+ for RCNN", "CPU bbox dynamic shapes"]),
    "MCPL": ("https://github.com/qujiaqi-babu/MCPL", "BERT; minimal README",
        ["uses CopotronicRifat/TwitterDataMABSA", "check requirements.txt in clone", "bottom-up RCNN feats"]),
    "RpBERT": ("https://github.com/Multimodal-NER/RpBERT", "BERT+ResNet-101+BiLSTM-CRF",
        ["torch>=1.9, transformers>=4", "ResNet-101 via torchvision", "MNER baseline"]),
}


def clone_one(name: str) -> str:
    url, stack, fixes = REPOS[name]
    target = DEST / name
    if target.exists():
        log.info(f"exists, skip: {target}")
        return "(already present)"
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
        sha = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        _force_rmtree(target / ".git")  # strip git tracking (robust on Windows read-only files)
        log.info(f"cloned {name} @ {sha[:10]} (.git stripped)")
        return sha
    except Exception as e:
        log.error(f"failed to clone {name}: {e}")
        return "(clone failed)"


def write_fixes(shas: dict) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    lines = ["# referred_clones — provenance & compatibility fixes\n",
             "Each repo was cloned `--depth 1`, its `.git/` stripped, and heavy artifacts are git-ignored.",
             "Each runnable baseline needs its **own isolated venv/conda env** — do NOT mix with the main `myenv`.\n"]
    for name, (url, stack, fixes) in REPOS.items():
        lines.append(f"## {name}")
        lines.append(f"- repo: {url}")
        lines.append(f"- cloned commit: `{shas.get(name, 'n/a')}`")
        lines.append(f"- original stack: {stack}")
        lines.append("- fixes needed:")
        lines.extend([f"  - [ ] {fx}" for fx in fixes])
        lines.append("")
    lines += ["## Cite-only (no public code found — use paper numbers)",
              "- SGBIS (KBS 2025/26), RNG (ICME 2024), Vanessa (EMNLP-F 2024), DSEM (ICMR 2025)",
              "## Text-only (SemEval, cited)", "- SPAN (huminghao16/SpanABSA), D-GCN (cuhksz-nlp/DGSA)\n"]
    (DEST / "FIXES.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {DEST / 'FIXES.md'}")


def main():
    names = sys.argv[1:] or list(REPOS.keys())
    shas = {}
    for name in names:
        if name not in REPOS:
            log.error(f"unknown repo {name}; options: {list(REPOS)}")
            continue
        shas[name] = clone_one(name)
    write_fixes(shas)


if __name__ == "__main__":
    main()
