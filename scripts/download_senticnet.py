"""Build the SenticNet 7 (English) affective table (paper §3.3) from ONE canonical source.

Output: data/senticnet/senticnet_en.parquet
        [concept, polarity_value, polarity_label, pleasantness, attention,
         sensitivity, aptitude, primary_mood, secondary_mood, semantics(list)]

Single source of truth = the official `senticnet.py` data module (SenticNet 7), kept at
data/senticnet/senticnet.py (git-ignored, heavy). It is the richest self-contained dump
(4 sentic dimensions + 2 emotions + polarity label/value + 5 semantics per concept).

Resolution order for the source (first hit wins):
  1. --py <path>                       explicit
  2. data/senticnet/senticnet.py       canonical location
  3. ./senticnet/senticnet.py          repo-root drop-in
  4. --git <url> [--git-file name]     clone a repo and pull the file out (opt-in)
  5. --rdf <path>                      parse the official RDF/XML instead (heavy, exact)

The pip `senticnet` package path was removed: it ships SenticNet-5-era data (Open-Q #10)
and isn't installed here. We standardize on the one provided .py file.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG  # noqa: E402
from kg import normalize  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("download_senticnet")

# senticnet['concept'] = [introspection, temper, attitude, sensitivity,
#                         primary_emotion, secondary_emotion, polarity_label,
#                         polarity_value, semantics1 .. semantics5]
_SEP = "] = "


def _fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _row_from(concept: str, v: list) -> dict:
    def at(i):
        return v[i] if i < len(v) else None

    sem, seen = [], set()
    for s in v[8:13]:
        if isinstance(s, str) and s and s not in seen:
            seen.add(s)
            sem.append(s)
    return {
        "concept": concept,
        "polarity_value": _fnum(at(7)),
        "polarity_label": at(6) if isinstance(at(6), str) else "",
        "pleasantness": _fnum(at(0)),   # introspection
        "attention": _fnum(at(1)),      # temper
        "sensitivity": _fnum(at(3)),    # sensitivity
        "aptitude": _fnum(at(2)),       # attitude
        "primary_mood": (at(4) or "").lstrip("#") if isinstance(at(4), str) else "",
        "secondary_mood": (at(5) or "").lstrip("#") if isinstance(at(5), str) else "",
        "semantics": sem,
    }


def from_py(py_path: str) -> list[dict]:
    """Tolerant line parser for the official senticnet.py.

    A handful of emoticon keys in the official dump are not valid Python (unescaped
    quotes), so we don't exec/import the module: we split each `senticnet[k] = [...]`
    line, literal_eval the (clean) value list, and best-effort recover the key. Keys
    that normalize to empty (emoticons) are skipped — they can't index the KG anyway.
    """
    rows, skipped, bad = [], 0, 0
    with open(py_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("senticnet[") or _SEP not in line:
                continue
            key_part, val_part = line[len("senticnet["):].split(_SEP, 1)
            try:
                concept = ast.literal_eval(key_part)
            except Exception:
                concept = key_part.strip().strip("'\"")
            try:
                v = ast.literal_eval(val_part)
            except Exception:
                bad += 1
                continue
            if not isinstance(concept, str) or not isinstance(v, list):
                continue
            if not normalize(concept):
                skipped += 1
                continue
            rows.append(_row_from(concept, v))
    log.info(f"parsed {len(rows)} concepts from {py_path} (skipped {skipped} empty-normalize, {bad} unparseable values)")
    return rows


def from_rdf(rdf_path: str) -> list[dict]:
    """Parse the official SenticNet RDF/XML dump into rows (streaming, memory-bounded)."""
    import xml.etree.ElementTree as ET

    rows: dict = {}
    context = ET.iterparse(rdf_path, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event != "end":
            continue
        tag = elem.tag.split("}")[-1]
        about = elem.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
        if about and tag == "Description":
            concept = about.rsplit("/", 1)[-1]
            row = rows.setdefault(concept, {"concept": concept, "semantics": []})
            for child in elem:
                ctag = child.tag.split("}")[-1].lower()
                val = (child.text or child.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")).strip()
                if ctag in ("pleasantness", "attention", "sensitivity", "aptitude", "polarity_intensity", "polarity_value"):
                    key = "polarity_value" if ctag.startswith("polarity") else ctag
                    try:
                        row[key] = float(val)
                    except Exception:
                        pass
                elif ctag in ("polarity", "polarity_label"):
                    row["polarity_label"] = val.rsplit("/", 1)[-1]
                elif ctag.startswith("semantics"):
                    row["semantics"].append(val.rsplit("/", 1)[-1])
                elif "mood" in ctag:
                    row.setdefault("primary_mood", val.rsplit("/", 1)[-1])
        elem.clear()
        root.clear()
    return list(rows.values())


def _clone_into(url: str, git_file: str, dest_dir: Path) -> Path:
    """Shallow-clone `url` and copy the first matching `git_file` into dest_dir/senticnet.py."""
    import shutil
    import subprocess
    import tempfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        log.info(f"git clone --depth 1 {url}")
        subprocess.run(["git", "clone", "--depth", "1", url, tmp], check=True)
        hits = list(Path(tmp).rglob(git_file))
        if not hits:
            raise FileNotFoundError(f"{git_file} not found in {url}")
        out = dest_dir / "senticnet.py"
        shutil.copy2(hits[0], out)
        log.info(f"fetched {hits[0].name} -> {out}")
        return out


def _resolve_py(args) -> Path | None:
    if args.py:
        return Path(args.py)
    for cand in (CONFIG.paths.senticnet / "senticnet.py", CONFIG.paths.root / "senticnet" / "senticnet.py"):
        if cand.exists():
            return cand
    if args.git:
        return _clone_into(args.git, args.git_file, CONFIG.paths.senticnet)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default=None, help="path to the official senticnet.py (default: auto-find)")
    ap.add_argument("--rdf", default=None, help="parse the official SenticNet RDF/XML instead")
    ap.add_argument("--git", default=None, help="git repo URL to clone the senticnet.py from (opt-in)")
    ap.add_argument("--git-file", default="senticnet.py", help="filename to extract from the cloned repo")
    args = ap.parse_args()
    import pandas as pd

    if args.rdf:
        rows = from_rdf(args.rdf)
        log.info(f"parsed {len(rows)} concepts from RDF {args.rdf}")
    else:
        src = _resolve_py(args)
        if src is None or not Path(src).exists():
            log.error("No senticnet.py found. Put the official file at data/senticnet/senticnet.py "
                      "(or pass --py PATH / --git URL / --rdf RDF). The pip `senticnet` package is not used.")
            sys.exit(1)
        rows = from_py(str(src))

    out = CONFIG.paths.senticnet / "senticnet_en.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    log.info(f"wrote {len(rows)} concepts -> {out}")


if __name__ == "__main__":
    main()
