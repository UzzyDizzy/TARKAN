"""Build the unified KG sqlite index from filtered ConceptNet + SenticNet (paper §3.4).

STREAMING build: parquet is read in row-group batches and inserted into sqlite per
batch (never the whole ~millions-row ConceptNet table in RAM). At runtime the KG is
queried straight from this on-disk index (kg.KnowledgeGraph(sqlite_path=...)), so no
full graph is ever loaded into memory.

Output: data/kg_index/kg.sqlite  (tables: triples, polarity)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG  # noqa: E402
from kg import KnowledgeGraph, normalize  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("build_kg")
BATCH = 100_000


def _stream_conceptnet(conn, path: Path) -> int:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["head", "relation", "tail", "weight"]):
        d = batch.to_pydict()
        rows = [(normalize(h), str(r), str(t), float(w), "conceptnet")
                for h, r, t, w in zip(d["head"], d["relation"], d["tail"], d["weight"])]
        conn.executemany("INSERT INTO triples VALUES (?,?,?,?,?)", rows)
        conn.commit()
        n += len(rows)
        if n % (BATCH * 5) == 0:
            log.info(f"  conceptnet: {n} triples...")
    return n


def _stream_senticnet(conn, path: Path):
    import pyarrow.parquet as pq

    import pandas as pd

    pf = pq.ParquetFile(path)
    n_tr = n_pol = 0
    for batch in pf.iter_batches(batch_size=BATCH):
        df = batch.to_pandas()
        tr, pol = [], []
        for _, r in df.iterrows():
            c = normalize(str(r["concept"]))
            sem = r.get("semantics")
            try:                                  # parquet stores the list as a numpy array
                sem_iter = list(sem) if sem is not None else []
            except TypeError:                     # NaN / scalar
                sem_iter = []
            for s in sem_iter:
                if s is None:
                    continue
                tr.append((c, "SemanticallyRelated", normalize(str(s)), 1.0, "senticnet"))
            lbl = r.get("polarity_label")
            if isinstance(lbl, str) and lbl:
                tr.append((c, "HasPolarity", lbl, 1.0, "senticnet"))
            pv = r.get("polarity_value")
            pv = 0.0 if pv is None or pd.isna(pv) else float(pv)
            pol.append((c, pv))
        conn.executemany("INSERT INTO triples VALUES (?,?,?,?,?)", tr)
        conn.executemany("INSERT OR REPLACE INTO polarity VALUES (?,?)", pol)
        conn.commit()
        n_tr += len(tr)
        n_pol += len(pol)
    return n_tr, n_pol


def main():
    out = CONFIG.paths.kg_index / "kg.sqlite"
    if out.exists():
        out.unlink()
    conn = KnowledgeGraph.init_sqlite(str(out))
    # speed pragmas (durability not needed for a derived index)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")

    n_tr = n_pol = 0
    cn = CONFIG.paths.conceptnet / "conceptnet_en.parquet"
    if cn.exists():
        n_tr += _stream_conceptnet(conn, cn)
        log.info(f"conceptnet: {n_tr} triples")
    else:
        log.warning(f"missing {cn} (run scripts/download_conceptnet.py)")

    sn = CONFIG.paths.senticnet / "senticnet_en.parquet"
    if sn.exists():
        t, p = _stream_senticnet(conn, sn)
        n_tr += t
        n_pol += p
        log.info(f"senticnet: {t} triples, {p} polarities")
    else:
        log.warning(f"missing {sn} (run scripts/download_senticnet.py)")

    conn.commit()
    conn.close()
    log.info(f"built {out}: {n_tr} triples, {n_pol} polarities (streamed)")


if __name__ == "__main__":
    main()
