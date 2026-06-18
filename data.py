"""Data pipeline (paper §3.1, §5).

The CopotronicRifat/TwitterDataMABSA files are TomBERT/MASC format — ONE (aspect,
sentiment) per line, with the aspect masked as `$T$`. This module:
  1. parses .tsv (5 cols) or .txt (4-line) records,
  2. reconstructs the JOINT annotation by grouping records per (tweet, image),
     recovering each aspect's word span from the `$T$` position,
  3. synthesizes the unified 7-tag BIO sequence,
  4. exposes a torch Dataset that BERTweet-tokenizes with first-subtoken BIO
     alignment + CLIP image features, and builds per-aspect KG AspectQuery objects.

Pure-python steps (1-3) are import-light and unit-tested; tokenizer/image/spacy
are loaded lazily so parsing works offline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CONFIG, TAG2ID, TSV_LABEL2POL, TXT_LABEL2POL, POL2ID
from kg_retrieval import AspectQuery
from utils import spans_to_bio

PLACEHOLDER = "$T$"


@dataclass
class RawRecord:
    text_masked: str   # tweet with $T$ at the aspect position
    aspect: str
    polarity: str      # POS|NEU|NEG
    image_id: str


@dataclass
class Instance:
    id: str
    tokens: List[str]                       # word-level tokens of the full tweet
    image_id: str
    aspects: List[Tuple[int, int, str]]     # (word_start, word_end_excl, polarity)
    aspect_terms: List[str] = field(default_factory=list)

    @property
    def bio(self) -> List[str]:
        return spans_to_bio(len(self.tokens), self.aspects)


# --------------------------------------------------------------------------- #
# 1. parsing
# --------------------------------------------------------------------------- #
def parse_tsv(path) -> List[RawRecord]:
    out: List[RawRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    start = 1 if rows and not rows[0][0].strip().lstrip("-").isdigit() else 0  # skip header
    for row in rows[start:]:
        if len(row) < 5:
            continue
        _, label, image_id, text_masked, aspect = row[0], row[1], row[2], row[3], row[4]
        try:
            pol = TSV_LABEL2POL[int(label)]
        except (ValueError, KeyError):
            continue
        out.append(RawRecord(text_masked.strip(), aspect.strip(), pol, image_id.strip()))
    return out


def parse_txt(path) -> List[RawRecord]:
    out: List[RawRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for i in range(0, len(lines) - 3, 4):
        text_masked, aspect, label, image_id = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
        try:
            pol = TXT_LABEL2POL[int(label.strip())]
        except (ValueError, KeyError):
            continue
        out.append(RawRecord(text_masked.strip(), aspect.strip(), pol, image_id.strip()))
    return out


# --------------------------------------------------------------------------- #
# 2-3. joint reconstruction + BIO
# --------------------------------------------------------------------------- #
def _full_text(rec: RawRecord) -> str:
    return rec.text_masked.replace(PLACEHOLDER, rec.aspect)


def _aspect_word_span(rec: RawRecord) -> Tuple[int, int]:
    """Word span of the aspect in the reconstructed full tweet (via $T$ position)."""
    before = rec.text_masked.split(PLACEHOLDER)[0]
    start = len(before.split())
    end = start + len(rec.aspect.split())
    return start, end


def reconstruct_joint(records: List[RawRecord]) -> List[Instance]:
    """Group per-aspect records into joint instances (one per tweet+image)."""
    groups: Dict[Tuple[str, str], List[RawRecord]] = {}
    for rec in records:
        if PLACEHOLDER not in rec.text_masked:
            continue
        key = (_full_text(rec), rec.image_id)
        groups.setdefault(key, []).append(rec)

    instances: List[Instance] = []
    for idx, ((full, image_id), recs) in enumerate(groups.items()):
        tokens = full.split()
        aspects, terms = [], []
        seen = set()
        for rec in recs:
            s, e = _aspect_word_span(rec)
            s = max(0, min(s, len(tokens)))
            e = max(s + 1, min(e, len(tokens)))
            if (s, e) in seen:
                continue
            seen.add((s, e))
            aspects.append((s, e, rec.polarity))
            terms.append(rec.aspect)
        aspects_sorted = sorted(zip(aspects, terms), key=lambda x: x[0][0])
        aspects = [a for a, _ in aspects_sorted]
        terms = [t for _, t in aspects_sorted]
        instances.append(Instance(id=f"{image_id}#{idx}", tokens=tokens, image_id=image_id, aspects=aspects, aspect_terms=terms))
    return instances


def load_split(data_dir, split: str) -> List[Instance]:
    """Load a split (train/dev/test), preferring .tsv then .txt."""
    data_dir = Path(data_dir)
    tsv = data_dir / f"{split}.tsv"
    txt = data_dir / f"{split}.txt"
    if tsv.exists():
        return reconstruct_joint(parse_tsv(tsv))
    if txt.exists():
        return reconstruct_joint(parse_txt(txt))
    raise FileNotFoundError(f"no {split}.tsv/.txt in {data_dir}")


# --------------------------------------------------------------------------- #
# query building (O_k opinion words, C_k visual concepts) — paper Eq. 12
# --------------------------------------------------------------------------- #
_NLP = None


def _get_spacy():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return _NLP


def opinion_words(tokens: List[str], span: Tuple[int, int], window: int = 4) -> List[str]:
    """O_k: adjectives/adverbs/verbs near the aspect span (spaCy POS)."""
    try:
        doc = _get_spacy()(" ".join(tokens))
    except Exception:
        return []
    s, e = span
    out = []
    for tok in doc:
        if tok.pos_ in ("ADJ", "ADV", "VERB") and (s - window) <= tok.i <= (e + window):
            out.append(tok.lemma_.lower())
    return list(dict.fromkeys(out))


def visual_concepts(caption: Optional[str]) -> List[str]:
    """C_k: noun keywords from the image caption."""
    if not caption:
        return []
    try:
        doc = _get_spacy()(caption)
        return list(dict.fromkeys([t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]))
    except Exception:
        return [w.lower() for w in caption.split() if len(w) > 3][:8]


def build_queries(inst: Instance, captions: Optional[Dict[str, str]] = None) -> List[AspectQuery]:
    cap = (captions or {}).get(inst.image_id)
    vc = visual_concepts(cap)
    qs = []
    for (s, e, _), term in zip(inst.aspects, inst.aspect_terms):
        qs.append(AspectQuery(aspect_term=term, opinion_words=opinion_words(inst.tokens, (s, e)), visual_concepts=vc))
    return qs


# --------------------------------------------------------------------------- #
# 4. torch Dataset + collate
# --------------------------------------------------------------------------- #
class TarkanDataset:
    """Encodes instances with BERTweet (first-subtoken BIO alignment) + CLIP images.

    Lazily loads the tokenizer/image processor. Requires torch + transformers at
    iteration time (not at import).
    """

    def __init__(self, instances: List[Instance], cfg=CONFIG, captions=None, images_dir=None, with_images=True):
        self.instances = instances
        self.cfg = cfg
        self.captions = captions or {}
        self.images_dir = Path(images_dir) if images_dir else None
        self.with_images = with_images
        self._tok = None
        self._img = None

    def __len__(self):
        return len(self.instances)

    def _tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(
                self.cfg.text_model_id, use_fast=False, normalization=True, token=self.cfg.hf_token
            )
        return self._tok

    def _img_processor(self):
        if self._img is None:
            from transformers import CLIPImageProcessor

            self._img = CLIPImageProcessor.from_pretrained(self.cfg.visual_model_id, token=self.cfg.hf_token)
        return self._img

    def _align(self, tokens: List[str]):
        """Manual subtoken alignment for BERTweet's slow tokenizer.

        Returns input_ids, attention_mask, word_ids (subtoken -> word index or -1).
        """
        tok = self._tokenizer()
        bos, eos = tok.bos_token or "<s>", tok.eos_token or "</s>"
        pieces = [bos]
        word_ids = [-1]
        for wi, w in enumerate(tokens):
            sub = tok.tokenize(w) or [w]
            pieces.extend(sub)
            word_ids.extend([wi] * len(sub))
        pieces.append(eos)
        word_ids.append(-1)
        max_len = self.cfg.max_text_len
        pieces, word_ids = pieces[:max_len], word_ids[:max_len]
        input_ids = tok.convert_tokens_to_ids(pieces)
        attn = [1] * len(input_ids)
        return input_ids, attn, word_ids

    def _bio_subtoken_labels(self, word_ids, bio_words):
        labels = []
        prev = None
        for wid in word_ids:
            if wid == -1:
                labels.append(-100)
            elif wid != prev:  # first subtoken of the word
                labels.append(TAG2ID[bio_words[wid]] if wid < len(bio_words) else -100)
            else:
                labels.append(-100)
            prev = wid
        return labels

    def _subtoken_spans(self, word_ids, aspects):
        """Map word-level aspect spans -> subtoken index ranges (for Eq. 6 pooling)."""
        spans = []
        for (ws, we, _pol) in aspects:
            idxs = [i for i, wid in enumerate(word_ids) if ws <= wid < we]
            if idxs:
                spans.append((idxs[0], idxs[-1] + 1))
            else:
                spans.append((0, 1))
        return spans

    def __getitem__(self, i: int):
        import torch

        inst = self.instances[i]
        input_ids, attn, word_ids = self._align(inst.tokens)
        bio_labels = self._bio_subtoken_labels(word_ids, inst.bio)
        sub_spans = self._subtoken_spans(word_ids, inst.aspects)
        polarities = [POL2ID[p] for (_, _, p) in inst.aspects]
        item = {
            "instance_id": inst.id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "bio_labels": torch.tensor(bio_labels, dtype=torch.long),
            "word_ids": word_ids,             # subtoken -> word index (or -1); for word-level eval
            "n_words": len(inst.tokens),
            "gold_aspects": inst.aspects,     # word-level (s,e,pol) for evaluation
            "aspect_spans": sub_spans,
            "aspect_polarity": polarities,
            "aspect_queries": build_queries(inst, self.captions),
        }
        if self.with_images:
            item["pixel_values"] = self._load_image(inst.image_id)
        return item

    def _load_image(self, image_id: str):
        import torch
        from PIL import Image

        try:
            path = self.images_dir / image_id
            img = Image.open(path).convert("RGB")
            return self._img_processor()(images=img, return_tensors="pt")["pixel_values"][0]
        except Exception:
            return torch.zeros(3, 224, 224)  # missing image -> zeros (model suppresses via r_k)


def collate_fn(batch, pad_id: int = 1):
    """Pad text, stack images, keep ragged aspect lists. pad_id=1 (BERTweet <pad>)."""
    import torch

    n = max(x["input_ids"].size(0) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, n), pad_id, dtype=torch.long)
    attn = torch.zeros((B, n), dtype=torch.long)
    bio = torch.full((B, n), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L] = x["input_ids"]
        attn[i, :L] = x["attention_mask"]
        bio[i, :L] = x["bio_labels"]
    out = {
        "instance_id": [x["instance_id"] for x in batch],
        "input_ids": input_ids,
        "attention_mask": attn,
        "bio_labels": bio,
        "word_ids": [x["word_ids"] for x in batch],
        "n_words": [x["n_words"] for x in batch],
        "gold_aspects": [x["gold_aspects"] for x in batch],
        "aspect_spans": [x["aspect_spans"] for x in batch],
        "aspect_polarity": [x["aspect_polarity"] for x in batch],
        "aspect_queries": [x["aspect_queries"] for x in batch],
    }
    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.stack([x["pixel_values"] for x in batch], 0)
    return out
