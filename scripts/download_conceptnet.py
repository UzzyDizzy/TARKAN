"""Download ConceptNet 5.7 assertions and filter to ENGLISH-only triples (paper §3.2).

Output: data/conceptnet/conceptnet_en.parquet  [head, relation, tail, weight, surface_text]
Optional: data/conceptnet/numberbatch-en.txt   (English Numberbatch for entity embeddings, Eq. 14)

Streams the 498 MB .gz line-by-line (never loads the ~10 GB uncompressed file into RAM)
and keeps a row only if BOTH start AND end nodes are /c/en/. License: CC BY-SA 4.0.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("download_conceptnet")

ASSERTIONS_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
NUMBERBATCH_EN_URL = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"


def _download(url: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info(f"exists, skip: {dest}")
        return dest
    log.info(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _node_word(node: str) -> str:
    # /c/en/smile/n -> smile
    parts = node.split("/")
    return parts[3] if len(parts) > 3 else node


def filter_english(gz_path: Path, out_parquet: Path, chunk: int = 200_000) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [("head", pa.string()), ("relation", pa.string()), ("tail", pa.string()),
         ("weight", pa.float32()), ("surface_text", pa.string())]
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_parquet, schema)
    buf = {k: [] for k in ("head", "relation", "tail", "weight", "surface_text")}
    kept = 0

    def flush():
        if not buf["head"]:
            return
        writer.write_table(pa.table(buf, schema=schema))
        for k in buf:
            buf[k].clear()

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            _, rel, start, end, meta = cols[:5]
            if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
                continue
            try:
                m = json.loads(meta)
            except Exception:
                m = {}
            buf["head"].append(_node_word(start))
            buf["relation"].append(rel.split("/")[2] if rel.startswith("/r/") else rel)
            buf["tail"].append(_node_word(end))
            buf["weight"].append(float(m.get("weight", 1.0)))
            buf["surface_text"].append(m.get("surfaceText") or "")
            kept += 1
            if len(buf["head"]) >= chunk:
                flush()
    flush()
    writer.close()
    log.info(f"kept {kept} English edges -> {out_parquet}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numberbatch", action="store_true", help="also download English Numberbatch")
    ap.add_argument("--keep-gz", action="store_true", help="keep the raw .gz after filtering")
    args = ap.parse_args()

    cn = CONFIG.paths.conceptnet
    gz = _download(ASSERTIONS_URL, cn / "conceptnet-assertions-5.7.0.csv.gz")
    filter_english(gz, cn / "conceptnet_en.parquet")
    if not args.keep_gz:
        gz.unlink(missing_ok=True)

    if args.numberbatch:
        nb_gz = _download(NUMBERBATCH_EN_URL, cn / "numberbatch-en-19.08.txt.gz")
        out = cn / "numberbatch-en.txt"
        if not out.exists():
            import shutil

            with gzip.open(nb_gz, "rt", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
                shutil.copyfileobj(fin, fout)
        log.info(f"numberbatch -> {out}")


if __name__ == "__main__":
    main()
