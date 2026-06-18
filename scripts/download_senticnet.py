"""Build the SenticNet 7 (English) affective table (paper §3.3).

Output: data/senticnet/senticnet_en.parquet
        [concept, polarity_value, polarity_label, pleasantness, attention,
         sensitivity, aptitude, primary_mood, secondary_mood, semantics(list)]

Primary path: the `senticnet` pip package (easy, but ships SenticNet-5-era data —
Open-Q #10). For SenticNet 7 fidelity, download the official RDF/XML from
https://sentic.net/downloads/ and pass --rdf <path>; this script will parse it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("download_senticnet")


def from_pip_package() -> list[dict]:
    """Iterate the `senticnet` package dict into rows."""
    from senticnet.senticnet import SenticNet

    sn = SenticNet()
    rows = []
    for concept in sn.data.keys():
        try:
            info = sn.concept(concept)
            sem = sn.semantics(concept)
        except Exception:
            continue
        rows.append({
            "concept": concept,
            "polarity_value": float(info.get("polarity_value", 0.0) or 0.0),
            "polarity_label": info.get("polarity_label", ""),
            "pleasantness": float(info["sentics"].get("pleasantness", 0.0)) if "sentics" in info else 0.0,
            "attention": float(info["sentics"].get("attention", 0.0)) if "sentics" in info else 0.0,
            "sensitivity": float(info["sentics"].get("sensitivity", 0.0)) if "sentics" in info else 0.0,
            "aptitude": float(info["sentics"].get("aptitude", 0.0)) if "sentics" in info else 0.0,
            "primary_mood": (info.get("moodtags") or ["", ""])[0],
            "secondary_mood": (info.get("moodtags") or ["", ""])[-1],
            "semantics": list(sem),
        })
    return rows


def from_rdf(rdf_path: str) -> list[dict]:
    """Parse the official SenticNet 7 RDF/XML dump into rows (streaming, memory-bounded)."""
    import xml.etree.ElementTree as ET

    rows: dict = {}
    context = ET.iterparse(rdf_path, events=("start", "end"))
    _, root = next(context)  # grab the root element so we can free processed children
    for event, elem in context:
        if event != "end":
            continue
        tag = elem.tag.split("}")[-1]
        # SenticNet RDF uses one Description per concept with sentic/polarity/semantics props.
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
        # free processed XML to keep memory bounded during streaming parse
        elem.clear()
        root.clear()
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdf", default=None, help="path to official SenticNet 7 RDF/XML (preferred)")
    args = ap.parse_args()
    import pandas as pd

    if args.rdf:
        rows = from_rdf(args.rdf)
        log.info(f"parsed {len(rows)} concepts from RDF {args.rdf}")
    else:
        log.info("no --rdf given; using pip `senticnet` package (Open-Q #10 version caveat)")
        rows = from_pip_package()
    out = CONFIG.paths.senticnet / "senticnet_en.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    log.info(f"wrote {len(rows)} concepts -> {out}")


if __name__ == "__main__":
    main()
