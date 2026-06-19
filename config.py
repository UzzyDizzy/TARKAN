"""TARKAN central configuration.

All hyperparameters are the paper values (§4.3). Paths are repo-relative. Secrets
(HF_TOKEN, ...) are read from .env.local via python-dotenv and are NEVER hard-coded.

Usage:
    from config import CONFIG
    CONFIG.text_model_id            # 'vinai/bertweet-base'
    CONFIG.lambda1                  # 0.5

Override any field from a YAML/CLI by constructing TarkanConfig(**overrides) — the
experiment/ablation runners do exactly this.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------- #
# Secrets: load .env.local (preferred) then .env, without overriding real env.
# ----------------------------------------------------------------------------- #
try:
    from dotenv import load_dotenv

    _ROOT = Path(__file__).resolve().parent
    load_dotenv(_ROOT / ".env.local", override=False)
    load_dotenv(_ROOT / ".env", override=False)
except Exception:  # python-dotenv not installed yet (e.g. during scaffolding)
    _ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- #
# Label spaces (paper §3.1, Eq. 3) — single source of truth.
# ----------------------------------------------------------------------------- #
# Unified BIO sentiment tags. Order is fixed; index is the class id used by the
# token-tagging head (Eq. 21) and Ltag (Eq. 22).
BIO_TAGS = ["O", "B-POS", "I-POS", "B-NEU", "I-NEU", "B-NEG", "I-NEG"]
TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}
NUM_BIO_TAGS = len(BIO_TAGS)  # 7

# Sentiment polarities for the auxiliary span classifier (Eq. 23).
POLARITIES = ["POS", "NEU", "NEG"]
POL2ID = {p: i for i, p in enumerate(POLARITIES)}
ID2POL = {i: p for p, i in POL2ID.items()}
NUM_POLARITIES = len(POLARITIES)  # 3

# Raw dataset label encodings -> canonical polarity (see §3.1 / §5.2).
TSV_LABEL2POL = {0: "NEG", 1: "NEU", 2: "POS"}   # CopotronicRifat .tsv
TXT_LABEL2POL = {-1: "NEG", 0: "NEU", 1: "POS"}  # .txt 4-line format


@dataclass
class Paths:
    root: Path = _ROOT
    data: Path = _ROOT / "data"
    twitter2015: Path = _ROOT / "data" / "twitter2015"
    twitter2017: Path = _ROOT / "data" / "twitter2017"
    images2015: Path = _ROOT / "data" / "images" / "twitter2015"
    images2017: Path = _ROOT / "data" / "images" / "twitter2017"
    conceptnet: Path = _ROOT / "data" / "conceptnet"
    senticnet: Path = _ROOT / "data" / "senticnet"
    kg_index: Path = _ROOT / "data" / "kg_index"
    captions: Path = _ROOT / "data" / "captions"
    teacher_labels: Path = _ROOT / "data" / "teacher_labels"
    results: Path = _ROOT / "results"
    checkpoints: Path = _ROOT / "results" / "checkpoints"
    logs: Path = _ROOT / "results" / "logs"
    tables: Path = _ROOT / "results" / "tables"
    plots: Path = _ROOT / "results" / "plots"
    reports: Path = _ROOT / "results" / "reports"


@dataclass
class TarkanConfig:
    # ---- models (paper §4.3) ----
    text_model_id: str = "vinai/bertweet-base"
    visual_model_id: str = "openai/clip-vit-base-patch32"
    teacher_llm_id: str = "Qwen/Qwen2.5-7B-Instruct"        # offline teacher (Open-Q #2)
    captioner_id: str = "Salesforce/blip-image-captioning-large"  # Open-Q #3

    # ---- dimensions ----
    hidden_dim: int = 768          # d
    max_text_len: int = 128        # paper §4.3
    num_visual_tokens: int = 49    # CLIP ViT-B/32 patch tokens (Open-Q #5)

    # ---- optimization (paper §4.3) ----
    batch_size: int = 16
    learning_rate: float = 2e-5
    dropout: float = 0.3
    weight_decay: float = 0.01
    max_epochs: int = 30
    warmup_ratio: float = 0.1      # Open-Q #7 (paper-unspecified)
    grad_clip: float = 1.0         # Open-Q #7
    early_stop_patience: int = 5   # Open-Q #7

    # ---- loss weights (Eq. 25 + auxiliary Lasc, Open-Q #1) ----
    lambda1: float = 0.5           # Lrel weight
    lambda2: float = 0.5           # Lkg weight
    lambda3: float = 1.0           # Lasc weight; set 0.0 -> Eq.25-exact / "w/o ASC" ablation

    # ---- KG retrieval/filter ----
    top_m_triples: int = 10        # paper §4.3 (top-M = 10)
    kg_eps: float = 1e-8           # Eq. 17 epsilon
    entity_emb_dim: int = 300      # ConceptNet Numberbatch
    kg_sources: tuple = ("conceptnet", "senticnet")

    # ---- fusion (Eq. 18-20) ----
    fusion: str = "kan"            # one of FUSION_REGISTRY keys (Table 10)
    kan_backend: str = "efficient_kan"  # efficient_kan | fastkan | rkan (Open-Q #11)
    kan_hidden: tuple = (512,)     # hidden widths between 3*d and d
    kan_grid_size: int = 5
    kan_spline_order: int = 3

    # ---- ablation toggles (Table 6) ----
    use_teacher: bool = True          # --no-teacher
    use_relevance: bool = True        # --no-relevance
    use_kg_filter: bool = True        # --no-kg-filter
    use_kg_stream: bool = True        # --no-kg-stream
    use_visual_stream: bool = True    # --no-visual-stream

    # ---- inference: joint-MABSA polarity source (queries.md A8) ----
    # 'bio'  -> joint (span, polarity) both read from the BIO tagging head (text only) [default].
    # 'asc'  -> span from BIO; FINAL polarity from the KAN-fused ASC head (Eq. 23) re-run on the
    #           predicted spans (paper §3.7 inference). Needed to let visual/KG/KAN move the joint
    #           metric in Tables 6 & 10. Does not affect MATE (spans) or the MASC subtask.
    joint_polarity_source: str = "bio"

    # ---- runtime ----
    seed: int = 42
    device: str = "cpu"            # set 'cuda' on the T4 server
    bootstrap_samples: int = 1000  # paper §4.3 (paired bootstrap)
    bootstrap_alpha: float = 0.05  # p < 0.05

    paths: Paths = field(default_factory=Paths)

    # ---- derived / secrets ----
    @property
    def hf_token(self) -> Optional[str]:
        return os.environ.get("HF_TOKEN")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


CONFIG = TarkanConfig()
