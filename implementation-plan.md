# TARKAN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithfully reproduce **TARKAN** (Teacher-Guided Aspect-Relevant Knowledge Fusion with KAN for Multimodal Aspect-Based Sentiment Analysis) end-to-end — student model, offline LLM teacher supervision, ConceptNet+SenticNet knowledge retrieval/filtering, KAN fusion — plus every experiment, ablation, diagnostic, and visualization reported in the paper, using the *exact* formulas, models, and hyperparameters.

**Architecture:** One-teacher-one-student design. An **offline LLM teacher** (training-time only) produces two binary evidence-supervision signals — aspect–visual relevance labels `rᵀ_k` and KG-triple usefulness labels `sᵀ_kq`. The **student** (`BERTweet` text encoder + `CLIP-ViT-B/32` visual encoder) estimates aspect–visual relevance, retrieves compact aspect-centered triples from SenticNet+ConceptNet, filters them, and fuses `[t_k; ṽ_k; g̃_k]` through a **Kolmogorov–Arnold Network** to jointly predict unified BIO aspect–sentiment tags and an auxiliary span-level polarity. The teacher is removed at inference (student-only).

**Tech Stack:** Python 3.10+, PyTorch 2.x, HuggingFace `transformers`, `vinai/bertweet-base`, `openai/clip-vit-base-patch32`, an open-source instruction LLM teacher (via `HF_TOKEN`), a BLIP captioner, an efficient KAN library, spaCy, ConceptNet 5.7 (English), SenticNet 7 (English).

## Global Constraints

- **Compute:** Develop on **CPU (local)**; full training/teacher-labeling targets a **single T4 (16 GB)** server later. Nothing in the default path may require >16 GB VRAM. Teacher LLM runs in **4-bit** quantization and only during a one-time offline labeling pass (labels cached to disk).
- **Exactness:** Use the **exact** equations (Eqs. 1–25), tag set, and hyperparameters from the paper. **No fake/stub implementations** — every module is runnable and tested. Where the paper is silent or inconsistent, the gap is flagged in `## Open Questions & Paper Ambiguities` and a documented default is used (configurable, ablatable).
- **Models:** **Open-source only.** Text=`vinai/bertweet-base`, Visual=`openai/clip-vit-base-patch32`, hidden dim `d=768`. Closed-weight models (ChatGPT-3.5/4V, VisualGLM) appear only as *cited comparison numbers*, never as dependencies.
- **Hyperparameters (paper §4.3, verbatim):** `d=768`; max text length `128`; batch size `16`; optimizer `AdamW`, lr `2e-5`; dropout `0.3`; top-`M` retrieved KG triples `= 10`; loss weights chosen from `{0.1, 0.3, 0.5, 1.0}` with final `λ1 = 0.5`, `λ2 = 0.5`; early stopping on dev F1; paired bootstrap resampling with `1000` samples, significance at `p < 0.05`.
- **Secrets:** All tokens/keys live in `.env.local` (git-ignored). Only `HF_TOKEN` is required now. Code reads secrets via `python-dotenv`; never hard-code.
- **Git hygiene:** Heavy artifacts (datasets, images, model weights, KG dumps, KG indices, checkpoints, captions cache, teacher-label caches, **and heavy contents of `referred_clones/`**) are git-ignored. `referred_clones/` source code is committed *after stripping each clone's `.git/`* so it ships with our repo.
- **Determinism:** A green deterministic CPU battery (`tests/`) must pass before any T4/GPU or LLM-labeling spend. Global seeding (`random`, `numpy`, `torch`, `PYTHONHASHSEED`), `torch.use_deterministic_algorithms(True)` where feasible.

---

## 0. Deliverable Map (what this plan produces)

| Paper artifact | Reproduced by |
|---|---|
| Student model (Eqs. 4–24, Fig. 1) | `models.py` + `encoders.py`, `relevance.py`, `kg_retrieval.py`, `kg_filter.py`, `kan_fusion.py`, `heads.py` |
| Offline LLM teacher (Table 4 prompts, Eqs. 11 & 16) | `teacher.py`, `captioner.py` |
| KG sources (SenticNet 7 + ConceptNet 5.7, English) | `scripts/download_*.py`, `kg.py` |
| Training (Algorithm 1, Eq. 25) | `train.py`, `losses.py` |
| Inference (Algorithm 2) | `evaluate.py` |
| Datasets (Twitter-2015/2017, Table 2) | `scripts/prepare_data.py`, `data.py` |
| Table 1 (main joint MABSA) | `experiments/run_main.py` |
| Table 3 (MATE / MASC / LLM-MLLM) | `experiments/run_subtasks.py` |
| Table 6 (component ablation) | `ablations/run_ablations.py` |
| Table 10 (fusion-strategy ablation) | `ablations/run_fusion_ablation.py` |
| Table 7 (teacher supervision quality) | `analysis/teacher_quality.py` |
| Table 8 (KG retrieval/filtering stats) | `analysis/kg_diagnostics.py` |
| Table 9 (perf by visual-relevance condition) | `analysis/visual_relevance_diag.py` |
| Table 5 (error distribution) | `analysis/error_analysis.py` |
| All plots | `visualizations/*.py` |
| Baseline numbers (cloned where code exists) | `referred_clones/` + `experiments/run_baselines.py` |

---

## 1. Repository Folder Structure

> **Layout rule (user requirement):** the **main paper-pipeline scripts are flat at repo root** (`./config.py`, `./models.py`, …). Experiments, ablations, analyses, visualizations, baseline clones, and data live in subfolders.

```
TARKAN/
├── .env.local                  # secrets (HF_TOKEN, …)            [GIT-IGNORED]
├── .env.example                # committed template
├── .gitignore
├── README.md
├── requirements.txt
├── implementation-plan.md      # this file
│
│   # ── ROOT-LEVEL MAIN PIPELINE (one focused file each) ──
├── config.py                   # dataclass config: paths, model IDs, hyperparams; loads .env.local
├── seeding.py                  # global determinism helpers
├── data.py                     # parse Twitter-2015/2017, BIO conversion, Dataset/collate, span utils
├── encoders.py                 # BERTweet text encoder + CLIP-ViT visual encoder (Eqs. 4–5)
├── relevance.py                # aspect-conditioned attention + relevance estimator (Eqs. 6–10)
├── kg.py                       # ConceptNet+SenticNet loading, indexing, query API
├── kg_retrieval.py             # aspect-centered query build + triple retrieval + encoding (Eqs. 12–14)
├── kg_filter.py                # teacher-guided KG usefulness scoring + aggregation (Eqs. 15–17)
├── kan_fusion.py               # KAN fusion (Eqs. 18–20) + all Table-10 alternative fusions
├── heads.py                    # BIO tagging head (Eq. 21) + span-level ASC head (Eq. 23)
├── models.py                   # TARKAN student nn.Module assembling all of the above
├── losses.py                   # Ltag, Lrel, Lkg, Lasc, total objective (Eqs. 11,16,22,24,25)
├── captioner.py                # image -> caption (BLIP) for the teacher relevance prompt
├── teacher.py                  # offline LLM teacher: rᵀ_k & sᵀ_kq generation + caching
├── train.py                    # Algorithm 1 training loop, early stopping
├── evaluate.py                 # Algorithm 2 inference + evaluation entrypoint
├── metrics.py                  # joint/MATE/MASC P-R-F1 + paired bootstrap significance
├── utils.py                    # logging, checkpoint IO, span<->BIO helpers, registry
│
├── scripts/                    # one-shot setup utilities
│   ├── download_conceptnet.py  # fetch + English-filter ConceptNet 5.7 -> data/conceptnet/
│   ├── download_senticnet.py   # fetch SenticNet 7 English -> data/senticnet/
│   ├── build_kg.py             # build unified KG index (sqlite/parquet) from both sources
│   ├── prepare_data.py         # pull TwitterDataMABSA + any missing 2017 split; normalize
│   ├── clone_referred.py       # clone baseline repos into referred_clones/, strip .git
│   └── run_teacher_labeling.py # batch teacher labeling over train(+dev) -> cached labels
│
├── experiments/
│   ├── configs/                # YAML per run (seeds, dataset, ablation flags)
│   ├── run_main.py             # Table 1
│   ├── run_subtasks.py         # Table 3
│   └── run_baselines.py        # drive cloned baselines (where runnable)
│
├── ablations/
│   ├── run_ablations.py        # Table 6 (7 component variants)
│   └── run_fusion_ablation.py  # Table 10 (7 fusion strategies)
│
├── analysis/
│   ├── teacher_quality.py      # Table 7 (agreement, accuracy, Cohen's κ)
│   ├── kg_diagnostics.py       # Table 8 (match rate, retrieved/retained, source split)
│   ├── visual_relevance_diag.py# Table 9 (image-useful / irrelevant / weak / multi-aspect)
│   └── error_analysis.py       # Table 5 (error-type distribution)
│
├── visualizations/
│   ├── plot_main_results.py    # grouped bars Table 1/3
│   ├── plot_ablation.py        # Table 6/10 deltas
│   ├── plot_kg_stats.py        # Table 8
│   ├── plot_relevance.py       # Table 9 + relevance-score histograms
│   ├── attention_viz.py        # aspect->patch attention α_kj heatmaps on images
│   └── kan_spline_viz.py       # learned KAN edge functions ψ_ij
│
├── tests/                      # deterministic CPU battery (pytest)
│   ├── test_shapes.py, test_losses.py, test_kan.py, test_relevance.py,
│   ├── test_kg_retrieval.py, test_kg_filter.py, test_data_bio.py,
│   ├── test_metrics.py, test_teacher_cache.py, test_overfit_tiny.py
│
├── data/                                                         [GIT-IGNORED: heavy]
│   ├── twitter2015/  twitter2017/      # text splits (train/dev/test)
│   ├── images/twitter2015/  images/twitter2017/
│   ├── conceptnet/   senticnet/        # raw + English-filtered dumps
│   ├── kg_index/                       # built sqlite/parquet index + entity embeddings
│   ├── captions/                       # cached BLIP captions per image
│   └── teacher_labels/                 # cached rᵀ_k, sᵀ_kq parquet
│
├── results/
│   ├── tables/    # committed: small CSV/markdown result tables
│   ├── plots/     # committed: small PNG/PDF figures
│   ├── reports/   # committed: per-run markdown summaries
│   ├── logs/                                                     [GIT-IGNORED]
│   └── checkpoints/                                              [GIT-IGNORED]
│
├── referred_papers/            # source PDFs                     [GIT-IGNORED: heavy]
└── referred_clones/            # baseline source (committed; .git stripped; heavy parts ignored)
    ├── VLP-MABSA/  JML/  AoM/  M2DF/  CMMT/ ...
    └── FIXES.md                # per-clone compatibility patch notes
```

---

## 2. Environment, Dependencies, Secrets

### 2.1 `.env.example` (committed) → copy to `.env.local` (ignored)
```bash
# Hugging Face (required now): read/accept-gated-model token
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Optional future:
# WANDB_API_KEY=
# OPENAI_API_KEY=        # only if ever comparing against closed APIs (not in default path)
```

### 2.2 `.gitignore` (append to existing)
```gitignore
# data & models
data/
results/checkpoints/
results/logs/
*.pt
*.bin
*.safetensors
*.csv.gz
# secrets & env
.env.local
myenv/
# caches
__pycache__/
*.pyc
.pytest_cache/
# referred material
referred_papers/
# heavy contents inside cloned baselines (keep source, drop weights/data/features)
referred_clones/**/data/
referred_clones/**/datasets/
referred_clones/**/*.pt
referred_clones/**/*.bin
referred_clones/**/*.safetensors
referred_clones/**/*.pth
referred_clones/**/*.h5
referred_clones/**/*.npy
referred_clones/**/*.npz
referred_clones/**/checkpoints/
referred_clones/**/pretrained*/
referred_clones/**/.git/
```

### 2.3 `requirements.txt` (pinned ranges; CPU-first, T4-compatible)
```text
torch>=2.1,<2.4
transformers>=4.40,<4.46
tokenizers>=0.19
huggingface_hub>=0.23
accelerate>=0.30
bitsandbytes>=0.43      # 4-bit teacher on GPU; skipped on CPU
sentencepiece
emoji<2.0               # BERTweet tokenizer normalization dependency
spacy>=3.7
# en_core_web_sm installed via: python -m spacy download en_core_web_sm
pandas>=2.0
pyarrow>=15
numpy>=1.24,<2.0
scikit-learn>=1.3       # Cohen's kappa, metrics helpers
scipy>=1.10             # bootstrap
seqeval>=1.2            # BIO span metric cross-check
Pillow>=10
matplotlib>=3.7
seaborn>=0.13
pyyaml>=6
python-dotenv>=1.0
tqdm>=4.66
fastkan                # KAN backend (RBF). Default B-spline backend via:
#   pip install git+https://github.com/Blealtan/efficient-kan.git   (or vendor its MIT kan.py)
# rkan                 # optional: rational-KAN fusion ablation (paper ref [41])
pytest>=8
```
> **BERTweet note:** `vinai/bertweet-base` requires `emoji` and uses a normalized BPE tokenizer (`AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)`). Pin `emoji<2.0` (TweetNormalizer compatibility).

---

## 3. External Resources — Download & Handling

> Every URL below was **fetched and confirmed to resolve on 2026-06-18** by a 10-agent verification run (`wf_92ed6abe-e86`). Legend: ✓ verified · ⚠ no public code (cite-only).

### 3.1 Benchmark datasets (Twitter-2015 / Twitter-2017)

| Source | URL | Use |
|---|---|---|
| `Lipika-Dewangan/TwitterDataMABSA` (the URL you gave) | https://github.com/Lipika-Dewangan/TwitterDataMABSA | ✓ but **Twitter-2015 only** — incomplete |
| `CopotronicRifat/TwitterDataMABSA` | https://github.com/CopotronicRifat/TwitterDataMABSA | ✓ **PRIMARY** — both 2015 **and** 2017 + image folders |
| `NUSTM/VLP-MABSA` (data + precomputed Faster-RCNN features) | https://github.com/NUSTM/VLP-MABSA | ✓ canonical joint-format annotations; data via Google Drive / Baidu (code `d0tn`) |

> **Action:** `scripts/prepare_data.py` clones `CopotronicRifat/TwitterDataMABSA` (both splits) into `data/`. The original `Lipika-Dewangan` repo is kept only as a 2015 cross-check.

**File formats (verified):**
- **`.tsv` (5 tab-separated cols):** `index ⟶ label(0=neg,1=neu,2=pos) ⟶ ImageID(e.g. 1860693.jpg) ⟶ masked tweet (with `$T$` aspect placeholder) ⟶ aspect/entity term`.
- **`.txt` (4-line records):** `masked tweet (with $T$)` / `aspect term` / `label(-1=neg,0=neu,1=pos)` / `ImageID`.
- **Images:** `twitter2015_images/` (numeric IDs `1234.jpg`), `twitter2017_images/` (date IDs `16_05_01_100.jpg`) — `data.py` must handle **both naming schemes**.

**Critical handling note (drives `data.py`, §5.2):** this is **TomBERT/MASC format — one (aspect, sentiment) record per line**, *not* joint BIO. The paper's joint MABSA (Table 1) needs all aspects of a tweet tagged together, so `prepare_data.py` must **group records by (normalized tweet text + ImageID), recover each aspect's token span from the `$T$` position, and synthesize the unified 7-tag BIO sequence**. Per-aspect record counts in this format equal **Table 2** totals (Twitter-2015 train/dev/test = 3179/1122/1037; Twitter-2017 = 3562/1176/1234) — assert this after preparation. Label encodings differ across `.tsv` (0/1/2) vs `.txt` (-1/0/1): normalize both to `{POS,NEU,NEG}`.

### 3.2 ConceptNet 5.7 (English only)

| Asset | URL | Notes |
|---|---|---|
| Assertions dump | https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz | ✓ ~498 MB gz → **~10 GB** uncompressed, 34,074,917 edges |
| Numberbatch (EN) embeddings | https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz | ✓ ~325 MB, 516,783 EN terms, **300-d** (→ entity embeddings, Eq. 14) |
| Edges/Relations docs | https://github.com/commonsense/conceptnet5/wiki/Edges · https://github.com/commonsense/conceptnet5/wiki/Relations | ✓ 5-field schema, 34 relation types |

- **Row schema (TSV, 5 fields):** `edge_uri ⟶ /r/Relation ⟶ /c/<lang>/start ⟶ /c/<lang>/end ⟶ {JSON}` where JSON has `weight`, `surfaceText` (may be `null`), `sources`, `dataset`, `license`.
- **English filter (`scripts/download_conceptnet.py`):** stream the `.gz` line-by-line; **keep a row only if BOTH start AND end begin with `/c/en/`**. Extract relation name after `/r/`. Surface word = node path segment after `/c/en/` (`/c/en/smile/n` → `smile`). Persist filtered triples to `data/conceptnet/conceptnet_en.parquet` (cols: `head, relation, tail, weight, surface_text`). **Do not hold the 10 GB file in memory** — stream + filter (English subset is a few-hundred-MB parquet).
- **License:** CC BY-SA 4.0 (attribution + share-alike — note in `README.md`).
- Convenience fallback: `pip install conceptnet-lite` (prebuilt SQLite) if streaming the dump is undesirable; default path is the official dump for exactness.

### 3.3 SenticNet 7 (English)

| Asset | URL | Notes |
|---|---|---|
| Official downloads (RDF/XML, AffectiveSpace, OntoSenticNet, PrimeNet) | http://w.sentic.net/downloads/ · https://sentic.net/ | ✓ SenticNet 7 English distribution |
| Python package (fallback) | https://pypi.org/project/senticnet/ (`pip install senticnet`) | ✓ easy API, **⚠ ships SenticNet 5-era data**, not 7 |
| SenticNet 7 paper (field semantics) | https://aclanthology.org/2022.lrec-1.408/ | ✓ |
| Community parser reference | https://github.com/yurimalheiros/senticnetapi · https://github.com/cbroms/senticnet-JSON | ✓ shows the 13-field record structure |

- **Per-concept record (≈200k concepts):** 4 sentics `pleasantness, attention, sensitivity, aptitude` (∈[-1,1]; pleasantness/aptitude bipolar, attention/sensitivity mono-polar), `primary_mood`, `secondary_mood`, `polarity_label`, `polarity_value` (∈[-1,1]), and **5 `semantics` (related concepts)**.
- **`scripts/download_senticnet.py`:** download SenticNet 7 English (RDF/XML), parse to `data/senticnet/senticnet_en.parquet` with cols `concept, polarity_value, polarity_label, pleasantness, attention, sensitivity, aptitude, primary_mood, secondary_mood, semantics(list)`.
- **Triple synthesis for the KG (`kg.py`):** from each concept emit triples (i) `(concept, RelatedTo, semantic_i)` for the 5 `semantics` — the natural commonsense/affective edges used in retrieval; (ii) `(concept, HasPolarity, polarity_label)`; (iii) `(concept, HasMood, primary_mood/secondary_mood)`. `kg.polarity(concept)` returns `polarity_value` (used in the Eq.-12/Top-M affective-relevance score).
- **⚠ Version exactness:** the paper cites SenticNet 7 [20]. Use the **official SenticNet 7 download** for fidelity; the `pip senticnet` package is a convenience fallback only (older data). Flagged in `## Open Questions` (#10).

### 3.4 Pretrained models (HuggingFace) + KAN library

**Student encoders (fixed by paper §4.3):**
| Role | HF id | Notes |
|---|---|---|
| Text encoder | `vinai/bertweet-base` | ✓ 110M, MIT, ungated, ~0.5 GB fp16 |
| Visual encoder | `openai/clip-vit-base-patch32` | ✓ ungated; use vision tower (49 patch tokens) |

**Offline teacher stack (training-time only; not named in paper → chosen, see Open-Q #2/#3):**
| Role | HF id | Notes |
|---|---|---|
| **Teacher LLM (default)** | `Qwen/Qwen2.5-7B-Instruct` | ✓ Apache-2.0, **ungated**, 4-bit ~3.8 GB → fits T4-16GB |
| Teacher LLM (alt, gated) | `meta-llama/Llama-3.1-8B-Instruct` | ✓ **gated** — needs `HF_TOKEN` + license acceptance |
| Teacher LLM (alt, ungated) | `mistralai/Mistral-7B-Instruct-v0.3` | ✓ Apache-2.0 |
| **Captioner (default)** | `Salesforce/blip-image-captioning-large` | ✓ 500M, BSD-3, ungated, ~1.5 GB |
| Captioner (alt) | `Salesforce/blip2-opt-2.7b` | ✓ heavier (~2 GB int4); only if VQA needed |

> Teacher + captioner are loaded **separately** during the one-time offline labeling pass (§7), never together with training → T4 memory is comfortable. Use `bitsandbytes` 4-bit on GPU; on CPU run fp32/fp16 (slow — defer teacher labeling to the T4 server).

**KAN library (Eq. 19):**
| Library | Install | Basis | Verdict |
|---|---|---|---|
| **efficient-kan (default)** | `pip install git+https://github.com/Blealtan/efficient-kan.git` (MIT; or vendor the single `kan.py`) | **B-spline edges** | ✓ most faithful to Eq. 19's learnable univariate edge functions; fast (~4.7 ms/layer) |
| fastkan (alt) | `pip install fastkan` | Gaussian RBF | ✓ ~3.3× faster; needs LayerNorm'd input |
| rkan (ablation) | `pip install rkan` | rational (Padé/Jacobi) | ✓ honors the paper's **rational-KAN** citation [41]; use in fusion ablation |

> **Default `kan_fusion.py` backend = `efficient_kan.KAN(layers_hidden=[2304, 512, 768], grid_size=5, spline_order=3)`** (B-spline edges = closest to Eq. 19; `grid=5,k=3` are KAN defaults). Backend is swappable via a config flag (`efficient_kan|fastkan|rkan`). Avoid `pykan` for training (≈10× slower on GPU).

---

## 4. Referred-Paper Codebases (`referred_clones/`)

**Handling protocol for every clone (`scripts/clone_referred.py`):**
1. `git clone --depth 1 <url> referred_clones/<NAME>`
2. **Strip git tracking:** remove `referred_clones/<NAME>/.git/` so the source commits inside *our* repo.
3. Record exact commit SHA cloned (write to `referred_clones/FIXES.md`) for provenance.
4. Heavy artifacts inside the clone are git-ignored (see §2.2); only source ships.
5. Apply compatibility fixes (see table) and log each as a patch note in `referred_clones/FIXES.md`.

**`referred_papers/` already present (PDFs, git-ignored):** TCMT (ESWA 2025), VLHA (Pattern Recognition 2025), Emotion-aware MABSC (IPM 2026, ref [35]), SGBIS (KBS 2025/26, ref [18]).

### 4.1 Priority A — clone & run (reproduce numbers; **no `*` in paper Table 1** → authors re-ran these)
All URLs ✓ verified to resolve (2026-06-18).

| Method | Repo | Stack (orig) | Key compatibility fixes for Py3.10+/torch2.x + CPU/T4 |
|---|---|---|---|
| **VLP-MABSA** [9] | `NUSTM/VLP-MABSA` | torch 1.6, transformers 3.4, BART, Faster-RCNN feats | migrate transformers 3.4→4.4x (TokenizerFast attrs); replace `fastnlp`; pin `h5py>=3`; **canonical data/feature source** — download feats from its Drive/Baidu |
| **JML** [7] | `MANLP-suda/JML` | py3.6, torch 1.1, BERT+Mask/Faster-RCNN | torch 1.1→2.x rewrite; transformers API migration; pin `numpy<2`; 17 GB supp on Baidu (code `53ej`); RCNN feats may need detectron2 backport |
| **AoM** [10] | `SilyRab/AoM` | builds on VLP-MABSA, transformers 3.4 | same VLP-MABSA chain; reuses VLP-MABSA 36-region 2048-d feats; expect deprecated-API rewrites |
| **M2DF** | `grandchicken/M2DF` | py3.7.13, torch 1.12, transformers 3.4 | torch 1.12 OK on T4; pin `numpy 1.24`, `fastnlp 0.6/0.7`; h5py wheels; download feats from Drive |
| **CMMT** [8] | `yangli-hub/CMMT-Code` | py3.7, torch 1.0, RoBERTa+ResNet-152+CRF | torch 1.0→1.13+; replace `pytorch-crf 0.7.2`→`torchcrf`; transformers 3.4 pin; CoNLL data + ResNet-152 weights |
| **MultiPoint** | `YangXiaocui1215/MultiPoint` | py3.8+, torch 1.9+, RoBERTa-large+NF-ResNet50(timm) | most modern of the set; align `sentence-transformers`/`timm` to torch; Drive data |
| **DQPSA** ("DQPS") [2] | `pengts/DQPSA` | torch 1.13, accelerate+deepspeed, spaCy 3.5 | pin `transformers<=4.26`; upgrade `accelerate`/`deepspeed` or keep torch 1.13; data+ckpts on Baidu (code `2024`); CPU: strip deepspeed |
| **TCMT** [52] | `ZouWang-spider/TCMT` | torch ~1.13; YOLOv5+ViT-GPT2+Tesseract+FITE | heavy external preproc pipeline; install Tesseract (system) + `pytesseract`; YOLOv5 pin torch ≤1.13; FITE not public (may block full repro) |
| **VLHA** [16] | `ZouWang-spider/VLHA` | Scene-Graph-Benchmark.pytorch + BiAffine | SGB.pytorch unmaintained → pin torch ≤1.13; BiAffine needs Cython build; `requirements.txt` 404 — reverse-engineer deps |

### 4.2 Priority B — clone & run (data-source / cited multimodal; mostly `*` in paper)
| Method | Repo | Stack | Notes / fixes |
|---|---|---|---|
| **TomBERT** | `jefferyYu/TomBERT` | py3.7, torch 1.0, BERT+ResNet-152 | torch 1.0→2.x rewrite; **MASC data source** (Twitter `absa_data/`, 49 region feats); transformers API migration |
| **UMT** | `jefferyYu/UMT` | py3.7, torch 1.0, BERT+ResNet-152+CRF | as TomBERT + `pytorch-crf` pin; MNER (collapsed baselines) |
| **Atlantis** [49] | `Xillv/Atlantis` | py3.9, torch 1.12.1, transformers 4.32, FLAN-T5 | nearly modern; bump transformers→4.35, CUDA 11.3→11.8; see sibling `Chimera` repo for full env |
| **CORSA** [15] | `Liuxj-Anya/CORSA` | py3.8+, torch 1.9+, YOLO/Faster-RCNN | builds C-MABSA from Twitter; torchvision 0.15+ for RCNN; CPU bbox dynamic shapes |
| **MCPL-VLP** [3] | `qujiaqi-babu/MCPL` | BERT; minimal README | uses `CopotronicRifat/TwitterDataMABSA`; check `requirements.txt` in clone; bottom-up RCNN feats |
| **RpBERT** | `Multimodal-NER/RpBERT` | BERT+ResNet-101+BiLSTM-CRF | MNER baseline; torch≥1.9, transformers≥4; ResNet-101 via torchvision |

### 4.3 Priority C — text-only baselines (cite numbers; code is SemEval-only, not multimodal)
| Method | Repo | Note |
|---|---|---|
| **SPAN** | `huminghao16/SpanABSA` | ✓ exists but **AllenNLP, text-only, SemEval** — Twitter numbers in paper are `*` (cited). `link_only`. |
| **D-GCN** | `cuhksz-nlp/DGSA` | ✓ exists but **text-only, SemEval** dependency-GCN — `link_only`. |
| **RoBERTa / BART** | — (no special repo) | Reimplement via HuggingFace `roberta-base` / `facebook/bart-base` if re-running; else cite `*`. |

### 4.4 No public code → cite paper numbers only (⚠)
| Method | Status |
|---|---|
| **SGBIS** [18] | ⚠ no repo found (KBS 2025/26). Cite Table-1/3 numbers. (PDF in `referred_papers/`.) |
| **RNG** [50] | ⚠ no repo (IEEE ICME 2024, arXiv 2405.13059). Cite. |
| **Vanessa** [51] | ⚠ no public code (EMNLP-F 2024). Cite. |
| **DSEM** [53] | ⚠ no public code (ICMR 2025). Cite. |
| LLM/MLLM rows (VisualGLM-6B, ChatGPT-3.5/4V, LLaMA+DPCI) | Cited comparison numbers only — **not dependencies** (closed/heavy). |

> `scripts/clone_referred.py` clones the **Priority A + B** repos by default (the `clone_and_run` set), strips each `.git/`, records SHAs and the per-repo fix checklist into `referred_clones/FIXES.md`. Priority C/D are documented as cite-only. Because every clone needs an old, mutually-incompatible stack, **do not** install them into the main `myenv`; each runnable baseline gets its **own venv/conda env** (documented in `FIXES.md`) — only their *result numbers* flow back into our `results/tables/`.

---

## 5. Data Pipeline Specification (`data.py`, `scripts/prepare_data.py`)

### 5.1 Task & label space (paper §3.1)
- Input `X = (T, I)`, `T = {w₁..wₙ}`. Output `Y = {(a_k, y_k)}`, `y_k ∈ {positive, neutral, negative}`.
- **Unified BIO sentiment tag set (Eq. 3):** `B = {B-POS, I-POS, B-NEU, I-NEU, B-NEG, I-NEG, O}` (7 classes). This is the *single source of truth* for the token-tagging head.
- Polarity id map: `POS↔positive`, `NEU↔neutral`, `NEG↔negative`. (Twitter MABSA raw labels are typically `1/0/-1` → `POS/NEU/NEG`.)

### 5.2 Parsing & normalization
- `scripts/prepare_data.py`: obtain splits + images (§3.1), normalize into a canonical JSONL per split:
  ```json
  {"id":"...", "tokens":["..."], "image":"twitter2015/12345.jpg",
   "aspects":[{"span":[start,end], "term":"Klay Thompson", "polarity":"POS"}],
   "bio":["O","B-POS","I-POS","O", ...]}
  ```
- **Both source formats are per-aspect (TomBERT-style), NOT joint** (verified §3.1): the `.txt` 4-line records and the `.tsv` 5-column rows each encode **one (aspect, sentiment) per line**. `data.py` parses either, then **reconstructs the joint annotation**: group rows by `(normalized_tweet, ImageID)`, locate each aspect via its `$T$` position, and emit one unified 7-tag BIO sequence per tweet. Normalize labels (`.tsv` 0/1/2 and `.txt` -1/0/1) → `{POS,NEU,NEG}`.
- **Subtoken alignment:** BERTweet BPE → align each word's gold BIO tag to its **first** subtoken; continuation subtokens get `-100` (ignored by `Ltag`). Store `word_ids` for span pooling (Eq. 6).
- Reproduce **Table 2** counts (per-split POS/NEU/NEG totals, avg length) as a `tests/`-checked invariant after preparation.

### 5.3 Aspect spans (train vs. infer; paper §3.2)
- **Train:** gold spans build `t_k` (Eq. 6) and supervise the heads.
- **Infer:** spans decoded from predicted BIO tags (Algorithm 2 step 3). `utils.bio_to_spans()` is shared so train/infer stay consistent.

### 5.4 `Dataset` / collate outputs (consumed by `models.py`)
Per instance: `input_ids, attention_mask, word_ids, pixel_values, bio_labels, aspect_spans(list[(s,e,polarity)]), image_id, instance_id`. Collate pads text to ≤128, stacks `pixel_values`, keeps `aspect_spans` ragged (per-aspect loop in model).

---

## 6. Model Specification — EXACT formulas → modules

> All shapes use `d = 768`, `n` = text tokens (≤128), `m` = CLIP patches (= 49 for ViT-B/32 at 224px + 1 CLS → use the 49 patch tokens, optionally +CLS; default: 49 patch tokens), `K` = #aspects, `M_k` = retrieved triples (≤10).

### 6.1 `encoders.py` — text & visual (Eqs. 4–5)
- **Text (Eq. 4):** `H_t = Enc_t(T) = [h_1..h_n] ∈ ℝ^{n×768}` from `vinai/bertweet-base` (last_hidden_state). Dropout 0.3 after.
- **Visual (Eq. 5):** `V = Enc_v(I) = [v_1..v_m] ∈ ℝ^{m×768}` from `openai/clip-vit-base-patch32` vision tower **patch** hidden states (`vision_model(...).last_hidden_state[:,1:,:]`), projected `768(clip)→768` by a learnable `Linear` so text/visual share `d`. (CLIP ViT-B/32 hidden=768 already; projection kept for alignment + to honor "share d".)
- **Interface — Produces:** `text_feats[B,n,768]`, `visual_feats[B,m,768]`.

### 6.2 `relevance.py` — aspect repr + relevance estimator (Eqs. 6–10)
- **Aspect repr (Eq. 6):** `t_k = Pool({h_i | w_i ∈ a_k})`. Default `Pool = mean` over the span's first-subtoken positions (configurable: mean/max/attn).
- **Aspect-conditioned visual attention (Eqs. 7–8):**
  `α_kj = softmax_j( t_kᵀ W_v v_j )`, `v̄_k = Σ_j α_kj v_j`, with `W_v ∈ ℝ^{768×768}`.
- **Relevance score (Eq. 9):** `r_k = σ( w_rᵀ [t_k ; v̄_k ; t_k ⊙ v̄_k] + b_r )`, `w_r ∈ ℝ^{3·768}`.
- **Filtered visual (Eq. 10):** `ṽ_k = r_k · v̄_k`.
- **Interface — Produces:** `t_k[K,768]`, `v_bar[K,768]`, `r_k[K]`, `v_tilde[K,768]`, `alpha[K,m]` (alpha exported for `attention_viz.py`).

Reference (exact):
```python
class AspectVisualRelevance(nn.Module):
    def __init__(self, d=768, dropout=0.3):
        super().__init__()
        self.Wv = nn.Linear(d, d, bias=False)          # Eq.7 bilinear
        self.wr = nn.Linear(3 * d, 1)                  # Eq.9  (w_r, b_r)
        self.drop = nn.Dropout(dropout)
    def forward(self, t_k, V):                          # t_k:[K,d], V:[m,d]
        scores = t_k @ self.Wv(V).transpose(-1, -2)     # [K,m]   t_kᵀ W_v v_j
        alpha  = torch.softmax(scores, dim=-1)          # Eq.7
        v_bar  = alpha @ V                              # [K,d]   Eq.8
        feat   = torch.cat([t_k, v_bar, t_k * v_bar], dim=-1)  # Eq.9
        r_k    = torch.sigmoid(self.wr(self.drop(feat))).squeeze(-1)  # [K]
        v_tilde = r_k.unsqueeze(-1) * v_bar             # Eq.10
        return t_k, v_bar, r_k, v_tilde, alpha
```
- **Loss (Eq. 11), in `losses.py`:** `L_rel = − Σ_k [ rᵀ_k·log r_k + (1−rᵀ_k)·log(1−r_k) ]` → `F.binary_cross_entropy(r_k, rᵀ_k, reduction='sum')` over aspects with a teacher label.

### 6.3 `kg.py` — knowledge graph backend
- Build a single queryable index over **English** ConceptNet 5.7 + SenticNet 7 (build details §3.2/§3.3 once verified). Public API:
  - `kg.normalize(term:str) -> key` (lowercase, lemmatize, `/c/en/<word>` form).
  - `kg.neighbors(term, top=K, sources=('conceptnet','senticnet')) -> list[Triple]` where `Triple=(head, relation, tail, weight, source)`.
  - `kg.polarity(term) -> float|None` (SenticNet polarity value).
  - Backend: SQLite (indexed on head/tail) + optional in-memory dict cache; relation vocabulary persisted to `data/kg_index/relations.json`.

### 6.4 `kg_retrieval.py` — aspect-centered retrieval + triple encoding (Eqs. 12–14)
- **Query set (Eq. 12):** `Q_k = {a_k} ∪ O_k ∪ C_k`.
  - `O_k` (opinion/sentiment words): spaCy POS+dependency around the aspect span — adjectives/verbs/adverbs and dependency-linked opinion terms; optional SenticNet-polarity gate.
  - `C_k` (visual concepts): noun keywords from the BLIP caption (`captioner.py`) and/or CLIP zero-shot over a concept vocab.
- **Retrieve (Eq. 13):** for each `q ∈ Q_k`, gather `kg.neighbors(q)`; pool to candidate triples `G_k = {(e_p, r, e_q)}`.
- **Top-M selection:** keep **top-M=10** by combined score `lexical_match + affective_relevance(SenticNet polarity magnitude) + relation_type_prior` (+ teacher usefulness at train time). Deterministic tie-break by (score, source, string).
- **Triple encoding (Eq. 14):** `g_kq = ϕ([e_p ; r ; e_q])`. Entity embeddings `e_p,e_q` from **ConceptNet Numberbatch (English, 300d) → Linear→768** (fallback: BERTweet-encoded surface form, mean-pooled); relation embedding `r` = learnable `nn.Embedding(|R|, 768)`; `ϕ` = 2-layer FFN `(3·768→768→768)` + GELU. Choice flagged in §Open-Questions; configurable.
- **Interface — Produces:** `g[K, M_k, 768]`, `triple_meta[K][M_k]` (for diagnostics/teacher), `match_mask[K]`.

### 6.5 `kg_filter.py` — teacher-guided filtering (Eqs. 15–17)
- **Usefulness (Eq. 15):** `s_kq = σ( w_gᵀ [t_k ; g_kq ; t_k ⊙ g_kq] + b_g )`, `w_g ∈ ℝ^{3·768}`.
- **Filter loss (Eq. 16), `losses.py`:** `L_kg = − Σ_k Σ_q [ sᵀ_kq·log s_kq + (1−sᵀ_kq)·log(1−s_kq) ]` over triples with a teacher label.
- **Aggregation (Eq. 17):** `g̃_k = ( Σ_q s_kq·g_kq ) / ( Σ_q s_kq + ε )`, `ε = 1e-8`. If `M_k = 0`, `g̃_k = 0` vector (no KG match).
- **Interface — Produces:** `s[K, M_k]`, `g_tilde[K, 768]`.

### 6.6 `kan_fusion.py` — KAN fusion (Eqs. 18–20) + Table-10 alternatives
- **Fusion input (Eq. 18):** `u_k = [t_k ; ṽ_k ; g̃_k] ∈ ℝ^{3·768=2304}`.
- **KAN layer (Eq. 19):** `z_{l+1,j} = Σ_i ψ^{(l)}_{ij}(z_{l,i})` — learnable univariate edge functions (spline/rational), no fixed node activations.
- **Fused repr (Eq. 20):** `z_k = KAN(u_k) ∈ ℝ^{d_z}` (default `d_z=768`; KAN width `[2304, 512, 768]`, grid/spline order per §3.4).
- **All Table-10 fusions behind one registry** (`FUSION_REGISTRY`) so `u_k → z_k` is swappable: `concat_linear`, `concat_mlp`, `gated`, `cross_modal_attention`, `bilinear`, `tensor`, `kan`. Each maps `2304→768` (or modality-wise for gated/attention/bilinear/tensor). KAN uses the confirmed library (§3.4); MLP variant = `Linear→GELU→Dropout→Linear` (this is the "w/o KAN, MLP fusion" ablation in Table 6 and "Concatenation+MLP" in Table 10).
- **Interface — Produces:** `z_k[K, 768]`.

### 6.7 `heads.py` — prediction heads (Eqs. 21, 23)
- **Token BIO head (Eq. 21):** `p(b_i|T,I) = softmax(W_b h_i + b_b)`, `W_b ∈ ℝ^{768×7}`. Operates on **all** token reps `H_t`.
- **Span ASC head (Eq. 23):** `p(y_k|a_k,T,I) = softmax(W_s z_k + b_s)`, `W_s ∈ ℝ^{768×3}`, over the KAN-fused `z_k`.
- **Interface — Produces:** `tag_logits[B,n,7]`, `asc_logits[K,3]`.

### 6.8 `models.py` — TARKAN student (assembly)
`forward(batch) ->` dict with `tag_logits, asc_logits, r_k, s_kq, alpha, kg_meta`. Per-aspect loop builds `t_k → (relevance) → (retrieval) → (filter) → (KAN) → asc_logits`; token head runs once over `H_t`. Inference path uses predicted BIO spans (Algorithm 2). All teacher signals are *labels fed to losses*, never inputs to the student forward.

---

## 7. Offline LLM Teacher (`teacher.py`, `captioner.py`) — training-time only

### 7.1 Captioner (`captioner.py`)
- Image → textual **"image description"** required by the relevance prompt. Model = BLIP (final ID in §3.4). Captions cached to `data/captions/<dataset>/<image_id>.txt` (one-time; reused by `C_k` extraction too).

### 7.2 Teacher prompts (Table 4, verbatim)
- **Aspect–Visual Relevance Prompt:** *"Given a tweet, an aspect term, and an image description, decide whether the image provides useful evidence for inferring sentiment toward the aspect. Return 1 if useful and 0 otherwise."*
- **KG Evidence Prompt:** *"Given a tweet, an aspect term, and a candidate KG triple, decide whether the triple is useful for aspect-level sentiment reasoning. Return 1 if useful and 0 otherwise."*

### 7.3 Generation & caching (Eqs. 11 & 16 supervision)
- LLM teacher = open-source instruction model (final ID in §3.4), 4-bit on T4. Deterministic decoding (`do_sample=False`, greedy), strict `{0,1}` parsing (regex; fallback `0`).
- **`rᵀ_k`:** one call per (instance, aspect) using tweet+aspect+caption.
- **`sᵀ_kq`:** one call per (instance, aspect, retrieved triple). To bound cost, label only the **retrieved candidate set** (top-M before filtering) — this matches Algorithm 1 lines 9–11 (retrieve, then teacher-label, then student predicts).
- **Cache** to `data/teacher_labels/{dataset}_relevance.parquet` (keys: instance_id, aspect_span) and `…_kg.parquet` (keys: instance_id, aspect_span, triple_key). Training loads from cache → **teacher LLM not loaded during training**. Cache schema asserted in `tests/test_teacher_cache.py`.
- `scripts/run_teacher_labeling.py` runs the one-time pass (T4 server) with resumable batching.

---

## 8. Training (`train.py`, `losses.py`) — Algorithm 1

### 8.1 Total objective (Eq. 25 + auxiliary, see Open-Questions)
- Paper Eq. 25: `L = L_tag + λ1·L_rel + λ2·L_kg` with `λ1=λ2=0.5`.
- Paper §3.7 + Table 6 ("w/o auxiliary ASC loss") require the auxiliary span loss `L_asc` (Eq. 24). **Eq. 25 omits it — flagged.** Default implemented: `L = L_tag + λ1·L_rel + λ2·L_kg + λ3·L_asc`, `λ3=1.0` (configurable; `λ3=0` reproduces the "w/o auxiliary ASC loss" ablation, and exactly matches Eq. 25 as written).
- `L_tag` (Eq. 22): token CE over 7 BIO classes, ignoring `-100`.
- `L_asc` (Eq. 24): span CE over 3 polarities on `asc_logits`.

### 8.2 Loop (Algorithm 1, exact order)
For each `(T,I)`: encode → for each **gold** aspect: `t_k`, `v̄_k`, load `rᵀ_k`, predict `r_k`→`ṽ_k`, retrieve `G_k`, load `{sᵀ_kq}`, predict `{s_kq}`→`g̃_k`, KAN→`z_k`. Predict BIO tags; compute `L_tag,L_rel,L_kg(,L_asc)`; `AdamW(lr=2e-5)` step.

### 8.3 Schedule & stopping
- Batch 16, dropout 0.3, max len 128. Early stopping on **dev joint-F1** (patience configurable, default 5). Linear warmup (10%) + decay (standard; flagged as paper-unspecified). Grad clip 1.0. Seeds fixed; log per-epoch dev P/R/F1.
- λ1,λ2 sweep over `{0.1,0.3,0.5,1.0}` on dev (reproduces selection of `0.5,0.5`) via `experiments/configs/*`.

---

## 9. Inference (`evaluate.py`) — Algorithm 2
Encode → predict BIO `b̂` → `spans = bio_to_spans(b̂)` → per predicted aspect: `t_k`, `r_k`, `ṽ_k`, retrieve `G_k`, `s_kq`, `g̃_k`, KAN→`z_k` → `ŷ_k = argmax asc_logits` (or read polarity from BIO suffix; **default joint output = BIO-decoded pairs**, ASC head used for MASC setting & auxiliary supervision). **No teacher, no captioner needed if `C_k` falls back to CLIP zero-shot at inference** (captions optional at test). Returns `{(â_k, ŷ_k)}`.

---

## 10. Experiments, Ablations, Diagnostics

### 10.1 Metrics (`metrics.py`)
- **Joint MABSA (Table 1):** micro P/R/F1 over exact `(span, polarity)` matches.
- **MATE (Table 3):** P/R/F1 over aspect **spans** only (polarity ignored).
- **MASC (Table 3):** Acc + macro/micro F1 over gold-aspect polarity (use ASC head).
- **Significance:** paired bootstrap, **1000** resamples, report `p<0.05` (`†`). Cross-check spans with `seqeval`.

### 10.2 Main & subtasks
- `experiments/run_main.py` → **Table 1** (Twitter-2015 & 2017). Our row = TARKAN; baseline rows = cited (`*`) or cloned-and-run (no-`*`; see §4).
- `experiments/run_subtasks.py` → **Table 3** (MATE / MASC / LLM-MLLM comparison rows; LLM/MLLM numbers are cited).

### 10.3 Ablations
- `ablations/run_ablations.py` → **Table 6** flags: `--no-teacher`, `--no-relevance`, `--no-kg-filter`, `--no-kg-stream`, `--mlp-fusion`, `--no-visual-stream`, `--no-asc-loss`. Each flag toggles exactly one mechanism in `config.py`.
- `ablations/run_fusion_ablation.py` → **Table 10** over `FUSION_REGISTRY` (7 strategies), all else fixed.

### 10.4 Diagnostics & analysis
- `analysis/teacher_quality.py` → **Table 7**: agreement/accuracy + Cohen's κ of teacher signals vs. a human-verified subset (`sklearn.metrics.cohen_kappa_score`).
- `analysis/kg_diagnostics.py` → **Table 8**: aspect KG match rate, avg retrieved vs. retained triples, SenticNet vs. ConceptNet contribution split.
- `analysis/visual_relevance_diag.py` → **Table 9**: F1 stratified by image-useful / image-irrelevant / weak-correspondence / multiple-aspect buckets (bucketing rule documented; uses `r_k` + teacher labels).
- `analysis/error_analysis.py` → **Table 5**: error-type tally over inspected test errors.

---

## 11. Visualizations (`visualizations/`)
- `plot_main_results.py`, `plot_ablation.py`, `plot_kg_stats.py`, `plot_relevance.py` — bar/line figures for Tables 1/3/6/8/9/10 → `results/plots/`.
- `attention_viz.py` — overlay `α_kj` aspect→patch attention on the image (qualitative, Fig.-1-style).
- `kan_spline_viz.py` — plot learned KAN edge functions `ψ_ij` (interpretability claim).

---

## 12. Determinism & Verification Battery (`tests/`) — gate before any GPU/LLM spend
1. `test_shapes.py` — forward on a 2-sample CPU batch: all tensor shapes match §6.
2. `test_kan.py` — KAN layer maps `[*,2304]→[*,768]`, gradients flow, deterministic.
3. `test_relevance.py` — Eqs. 7–10 numeric check vs. hand-computed tiny example; `r_k∈[0,1]`, `ṽ_k=r_k·v̄_k`.
4. `test_kg_retrieval.py` / `test_kg_filter.py` — top-M≤10, Eq. 17 aggregation incl. `M_k=0` → zero vector, ε behavior.
5. `test_losses.py` — Eqs. 11/16/22/24/25 against hand values; `λ3=0` ⇒ Eq.25-exact.
6. `test_data_bio.py` — BIO↔span round-trip; subtoken alignment ignores continuations; Table-2 counts.
7. `test_metrics.py` — joint/MATE/MASC P/R/F1 vs. seqeval on toy data; bootstrap reproducible with seed.
8. `test_teacher_cache.py` — cache schema/keys; training reads cache without loading the LLM.
9. `test_overfit_tiny.py` — model overfits 8 instances to ~0 loss in ≤200 steps (CPU) — proves end-to-end learnability before T4.

---

## 13. Task Breakdown (bite-sized, TDD, frequent commits)

> Order respects dependencies. Each task ends with an independently testable deliverable + commit. Steps marked write→test-fail→implement→test-pass→commit per `superpowers:test-driven-development`.

### Phase A — Scaffold & resources
- **Task A1:** Create `.env.example`, `.env.local`, `.gitignore` updates, `requirements.txt`, `config.py` (dataclass; loads dotenv; all §Global-Constraints values). Test: `tests/test_config.py` asserts hyperparams equal paper values. Commit.
- **Task A2:** `seeding.py` + `utils.py` (logging, checkpoint IO, `bio_to_spans`/`spans_to_bio`). Test round-trip. Commit.
- **Task A3:** `scripts/prepare_data.py` + `data.py` parsing → canonical JSONL; reproduce **Table 2** counts. Test `test_data_bio.py`. Commit.
- **Task A4:** `scripts/download_conceptnet.py`, `scripts/download_senticnet.py`, `scripts/build_kg.py`, `kg.py` query API. Test on a tiny fixture KG. Commit.
- **Task A5:** `scripts/clone_referred.py` (clone + strip `.git` + record SHA) + write `referred_clones/FIXES.md` skeleton. Commit (source only; heavy ignored).

### Phase B — Student model (CPU, deterministic)
- **Task B1:** `encoders.py` (Eqs. 4–5) + `test_shapes.py`. Commit.
- **Task B2:** `relevance.py` (Eqs. 6–10) + `test_relevance.py`. Commit.
- **Task B3:** `kg_retrieval.py` (Eqs. 12–14, Numberbatch entity embeddings + ϕ) + `test_kg_retrieval.py`. Commit.
- **Task B4:** `kg_filter.py` (Eqs. 15–17) + `test_kg_filter.py`. Commit.
- **Task B5:** `kan_fusion.py` (Eqs. 18–20 + `FUSION_REGISTRY`) + `test_kan.py`. Commit.
- **Task B6:** `heads.py` (Eqs. 21,23) + shape test. Commit.
- **Task B7:** `models.py` assembly + `losses.py` (Eqs. 11,16,22,24,25) + `test_losses.py` + `test_overfit_tiny.py`. Commit.

### Phase C — Teacher (offline)
- **Task C1:** `captioner.py` (BLIP) + caption cache + test (mock model on CPU). Commit.
- **Task C2:** `teacher.py` + `scripts/run_teacher_labeling.py` (Table-4 prompts, `{0,1}` parse, parquet cache) + `test_teacher_cache.py`. Commit.

### Phase D — Train / eval / metrics
- **Task D1:** `metrics.py` (joint/MATE/MASC + bootstrap) + `test_metrics.py`. Commit.
- **Task D2:** `train.py` (Algorithm 1, early stop, λ-sweep hooks). Smoke-train 1 epoch on 16 samples (CPU). Commit.
- **Task D3:** `evaluate.py` (Algorithm 2). Smoke-eval. Commit.

### Phase E — Experiments, ablations, analysis, viz
- **Task E1:** `experiments/run_main.py` + configs → Table 1 CSV. Commit.
- **Task E2:** `experiments/run_subtasks.py` → Table 3. Commit.
- **Task E3:** `ablations/run_ablations.py` (Table 6) + `ablations/run_fusion_ablation.py` (Table 10). Commit.
- **Task E4:** `analysis/*` (Tables 5,7,8,9). Commit.
- **Task E5:** `visualizations/*` (all plots). Commit.

### Phase F — Baselines (where code exists)
- **Task F1:** Per §4 table, for each `clone_and_run` baseline: apply fixes (log in `FIXES.md`), run on Twitter-2015/2017, record numbers in `results/tables/baselines.csv`. `link_only`/`no_public_code` → cite paper numbers with source. Commit per baseline.

### Phase G — Full runs (T4 server)
- **Task G1:** Run `scripts/run_teacher_labeling.py` (T4). **Task G2:** Full TARKAN training both datasets + λ-selection. **Task G3:** All ablations/diagnostics. **Task G4:** Compile `results/reports/` and compare against paper Tables 1,3,6,7,8,9,10.

---

## 14. Open Questions & Paper Ambiguities (documented defaults)
1. **Eq. 25 vs. `L_asc`:** Eq. 25 lists only 3 terms but §3.7 + Table 6 use the auxiliary ASC loss. → Default `L = L_tag + λ1 L_rel + λ2 L_kg + λ3 L_asc`, `λ3=1.0`; `λ3=0` reproduces Eq.25-exact and the "w/o ASC" ablation.
2. **Teacher LLM identity:** paper says "offline LLM teacher" without a named model. → Open-source instruction LLM chosen in §3.4 (4-bit, T4-feasible). Configurable.
3. **Captioner:** "image description" source unspecified. → BLIP (§3.4). Configurable.
4. **Entity/relation embeddings + ϕ:** Eq. 14 unspecified source. → Numberbatch(EN)→768 entities, learned relation embedding, 2-layer FFN ϕ. Configurable.
5. **`m` (visual tokens):** use CLIP ViT-B/32 49 patch tokens (default). 
6. **`O_k`/`C_k` extraction:** spaCy opinion terms / BLIP+CLIP visual concepts (paper lists options, not a fixed recipe).
7. **LR schedule / warmup / patience:** unspecified → linear warmup 10%, patience 5, grad-clip 1.0.
8. **Top-M scoring function:** paper lists criteria (lexical/affective/relation/teacher) without weights → equal-weight combination, configurable.
9. **Twitter-2017 data source:** ✅ RESOLVED — the given `Lipika-Dewangan` repo has 2015 only; use `CopotronicRifat/TwitterDataMABSA` for both splits (§3.1). Data is per-aspect/MASC format → joint BIO reconstructed in `data.py` (§5.2).
10. **SenticNet version:** `pip senticnet` ships v5-era data; paper cites SenticNet 7. → default = official SenticNet 7 RDF download; pip package is a flagged fallback (§3.3).
11. **KAN backend:** Eq. 19 is the generic edge-function KAN. → default `efficient-kan` (B-spline, most faithful); `fastkan`/`rkan` selectable (§3.4).

---

## 15. Self-Review (completed 2026-06-18)
- [x] **Spec coverage:** all Eqs. 1–25, Tables 1–10, and Algorithms 1–2 map to a module/task — verified mapping: Eqs.1–3→`data.py`/§5.1; 4–5→`encoders.py`; 6–10→`relevance.py`; 11,16,22,24,25→`losses.py`; 12–14→`kg_retrieval.py`; 15,17→`kg_filter.py`; 18–20→`kan_fusion.py`; 21,23→`heads.py`; Alg.1→`train.py`; Alg.2→`evaluate.py`; Tables 1/3→`experiments/`; 2→`prepare_data.py`; 4→`teacher.py`; 5,7,8,9→`analysis/`; 6,10→`ablations/`.
- [x] **Placeholder scan:** no `[[VERIFY-FILL]]`/TBD/TODO remain (grep-verified); every §3/§4 URL fetch-verified 2026-06-18.
- [x] **Type consistency:** `t_k`, `v_bar`, `r_k`, `v_tilde`, `g_tilde`, `z_k`, `tag_logits`, `asc_logits` used identically across §6 module interfaces and §13 tasks.
- [x] **Constraint propagation:** CPU-first + T4 ceiling, open-source-only, exact §4.3 hyperparams, and `.env.local` secrets honored in every task; teacher labeling isolated to a one-time offline pass.
