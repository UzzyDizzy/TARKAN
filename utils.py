"""Shared utilities: BIO<->span conversion, logging, checkpoint IO.

`bio_to_spans` / `spans_to_bio` are the single shared implementation used by both
training (gold spans) and inference (decoded spans) so the two stay consistent
(paper §3.2, Algorithm 2 step 3).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple, Union

from config import BIO_TAGS, ID2TAG, TAG2ID

Span = Tuple[int, int, str]  # (start, end_exclusive, polarity in {POS,NEU,NEG})


# ----------------------------------------------------------------------------- #
# BIO <-> spans
# ----------------------------------------------------------------------------- #
def _to_tag_str(tag: Union[int, str]) -> str:
    return ID2TAG[tag] if isinstance(tag, int) else tag


def bio_to_spans(tags: List[Union[int, str]]) -> List[Span]:
    """Decode a unified BIO tag sequence into aspect spans with polarity.

    A span opens on B-XXX and extends over following I-XXX of the SAME polarity.
    Malformed sequences (I- without a matching open, polarity switch mid-span) are
    handled defensively: an I- with no open span starts a new span; a polarity
    change closes the current span and opens a new one.
    """
    spans: List[Span] = []
    cur_start, cur_pol = None, None
    for i, tag in enumerate(tags):
        t = _to_tag_str(tag)
        if t == "O":
            if cur_start is not None:
                spans.append((cur_start, i, cur_pol))
                cur_start, cur_pol = None, None
            continue
        prefix, pol = t.split("-", 1)  # 'B'/'I', 'POS'/'NEU'/'NEG'
        if prefix == "B" or cur_start is None or pol != cur_pol:
            if cur_start is not None:
                spans.append((cur_start, i, cur_pol))
            cur_start, cur_pol = i, pol
    if cur_start is not None:
        spans.append((cur_start, len(tags), cur_pol))
    return spans


def spans_to_bio(n_tokens: int, spans: List[Span]) -> List[str]:
    """Encode aspect spans into a unified BIO tag sequence of length n_tokens.

    Overlapping spans: later spans overwrite earlier ones (deterministic).
    """
    tags = ["O"] * n_tokens
    for start, end, pol in spans:
        start = max(0, start)
        end = min(n_tokens, end)
        if end <= start:
            continue
        tags[start] = f"B-{pol}"
        for j in range(start + 1, end):
            tags[j] = f"I-{pol}"
    return tags


def bio_ids_to_tags(ids: List[int]) -> List[str]:
    return [ID2TAG[i] for i in ids]


def tags_to_bio_ids(tags: List[str]) -> List[int]:
    return [TAG2ID[t] for t in tags]


# ----------------------------------------------------------------------------- #
# logging
# ----------------------------------------------------------------------------- #
def get_logger(name: str = "tarkan", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ----------------------------------------------------------------------------- #
# checkpoint / json IO
# ----------------------------------------------------------------------------- #
def save_json(obj, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(model, path: Union[str, Path], extra: dict | None = None) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path: Union[str, Path], map_location="cpu"):
    import torch

    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return payload
