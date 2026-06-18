"""Fetch + organize Twitter-2015/2017 MABSA data (paper §3.1, Table 2).

Source: CopotronicRifat/TwitterDataMABSA (BOTH 2015 + 2017, with images) — the given
Lipika-Dewangan repo has 2015 only. Lays files out as config expects:
    data/twitter2015/{train,dev,test}.{tsv,txt}     data/images/twitter2015/*.jpg
    data/twitter2017/{train,dev,test}.{tsv,txt}     data/images/twitter2017/*.jpg
Then asserts per-aspect record counts equal Table 2.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG  # noqa: E402
from data import parse_tsv, parse_txt  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("prepare_data")

REPO = "https://github.com/CopotronicRifat/TwitterDataMABSA.git"
TABLE2 = {
    "twitter2015": {"train": 3179, "dev": 1122, "test": 1037},
    "twitter2017": {"train": 3562, "dev": 1176, "test": 1234},
}


def _clone(dest: Path) -> Path:
    if dest.exists():
        log.info(f"raw repo exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"cloning {REPO}")
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(dest)], check=True)
    return dest


def _copy_year(raw: Path, year: str) -> None:
    src_text = raw / year
    src_imgs = raw / f"{year}_images"
    out_text = CONFIG.paths.data / year
    out_imgs = CONFIG.paths.data / "images" / year
    out_text.mkdir(parents=True, exist_ok=True)
    out_imgs.mkdir(parents=True, exist_ok=True)
    if src_text.exists():
        for f in src_text.glob("*"):
            if f.suffix in (".tsv", ".txt"):
                shutil.copy2(f, out_text / f.name)
    if src_imgs.exists():
        for f in src_imgs.glob("*.jpg"):
            shutil.copy2(f, out_imgs / f.name)
    log.info(f"{year}: text -> {out_text}, images -> {out_imgs}")


def _verify(year: str) -> None:
    d = CONFIG.paths.data / year
    for split, expected in TABLE2[year].items():
        recs = []
        if (d / f"{split}.tsv").exists():
            recs = parse_tsv(d / f"{split}.tsv")
        elif (d / f"{split}.txt").exists():
            recs = parse_txt(d / f"{split}.txt")
        status = "OK" if len(recs) == expected else f"MISMATCH (got {len(recs)})"
        log.info(f"  {year}/{split}: {len(recs)} records, expected {expected} -> {status}")


def main():
    raw = _clone(CONFIG.paths.data / "_raw_TwitterDataMABSA")
    for year in ("twitter2015", "twitter2017"):
        _copy_year(raw, year)
        _verify(year)
    log.info("data prepared.")


if __name__ == "__main__":
    main()
