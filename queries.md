# TARKAN — Open Questions, Ambiguities & Fixes

A consolidated log of (A) **important but vague** things in the paper, (B) things **not
specified / missing**, and (C) **implementation/infra issues** found while building. Each
entry states what the paper says (or doesn't), why it matters, and the concrete fix +
where it lives + whether it's configurable. This is the authoritative companion to
`implementation-plan.md §14`.

Legend: 🔧 = decision implemented · ⚙️ = configurable in `config.py` · 📄 = paper-faithful default.

---

## A. Primary VAGUE / ambiguous things (present in the paper, but underspecified)

### A1. Total loss objective — Eq. 25 vs. the auxiliary ASC loss 🔧⚙️📄
- **Paper:** Eq. 25 writes `L = L_tag + λ1·L_rel + λ2·L_kg` (3 terms), yet §3.7 defines the
  auxiliary span loss `L_asc` (Eq. 24) and Table 6 ablates **"w/o auxiliary ASC loss"** —
  so `L_asc` is clearly used but missing from Eq. 25.
- **Why it matters:** changes what is optimized and the Table-6 ablation.
- **Fix:** `L = L_tag + λ1·L_rel + λ2·L_kg + λ3·L_asc`, default `λ3=1.0`. **`λ3=0` reproduces
  Eq. 25 exactly and the "w/o ASC" ablation.** → `losses.compute_losses`, `config.lambda3`.

### A2. Aspect pooling `Pool(·)` (Eq. 6) 🔧⚙️
- **Paper:** `t_k = Pool({h_i | w_i ∈ a_k})` — the pooling operator is unnamed.
- **Fix:** mean-pool over the span's first-subtoken positions (default); `max`/`first` selectable.
  → `relevance.pool_aspect`, `TarkanStudent(pool_mode=...)`.

### A3. Opinion words `O_k` and visual concepts `C_k` (Eq. 12) 🔧⚙️
- **Paper:** lists *options* — `O_k` from "nearby adjectives, verbs, adverbs, or
  dependency-linked opinion terms"; `C_k` from "CLIP-predicted concepts, object tags, or
  image caption keywords" — but no fixed recipe.
- **Fix:** `O_k` = spaCy POS/dependency (ADJ/ADV/VERB within a window of the aspect);
  `C_k` = noun keywords from the BLIP caption (optional CLIP zero-shot). → `data.opinion_words`,
  `data.visual_concepts`, `data.build_queries`.

### A4. Top-M triple selection score (Eq. 13) 🔧⚙️
- **Paper:** keep top-M "based on lexical match, affective relevance, relation type, or
  teacher usefulness score" — criteria listed, no weights/formula.
- **Fix:** equal-weight sum `weight + lexical_match + |SenticNet polarity| + relation_prior
  (+ teacher score)`, deterministic tie-break by `(score, relation, tail)`. `M=10` (§4.3).
  → `kg_retrieval.retrieve_triples`.

### A5. KAN realization (Eq. 19) 🔧⚙️📄
- **Paper:** generic edge-function form `z_{l+1,j}=Σ_i ψ_ij(z_{l,i})`; cites both the KAN
  survey [41] and rational KANs — does not fix spline vs. rational, grid, or order.
- **Fix:** default **B-spline `efficient-kan`** (closest to Eq. 19) with `grid_size=5,
  spline_order=3` (KAN defaults); backends `fastkan` (RBF) and `rkan` (rational, honors [41])
  selectable; a vendored RBF-KAN is the zero-dependency fallback. → `kan_fusion`, `config.kan_backend`.

### A6. Number of visual tokens `m` (Eq. 5) 🔧📄
- **Paper:** "visual patch or object-level features"; count unstated.
- **Fix:** CLIP ViT-B/32 **49 patch tokens** (CLS dropped). → `encoders.VisualEncoder`, `config.num_visual_tokens`.

### A7. Teacher "image description" (Table 4 relevance prompt) 🔧⚙️
- **Paper:** the relevance prompt consumes an "image description" but never says how it is produced.
- **Fix:** BLIP caption (`Salesforce/blip-image-captioning-large`), cached. → `captioner.py`, `config.captioner_id`.

### A8. Inference: joint output vs. ASC head 🔧⚙️
- **Paper:** has both a BIO suffix polarity (Eq. 21) and an auxiliary ASC head (Eq. 23);
  which yields the *final* joint polarity at inference is ambiguous. §3.7 ("…fused through the
  KAN module to predict the final aspect-level sentiment polarity") leans ASC; but Table 6's
  "w/o ASC loss" = −0.7 leans BIO. The two readings reproduce different tables.
- **Fix:** now a **config toggle** `config.joint_polarity_source ∈ {bio, asc}` (default `bio`).
  - `bio`: span **and** polarity from the BIO head (text only). Visual/KG/KAN affect joint F1 only
    via training-time regularization of the shared encoder → Table-10 joint column ≈ flat.
  - `asc`: span from BIO; **final polarity from the KAN-fused ASC head (Eq. 23) re-run on the
    predicted spans** (paper §3.7) → lets visual/KG/KAN move Tables 6 & 10. MATE & MASC unchanged.
  - Run both on one trained checkpoint and keep whichever reproduces the paper.
  → `evaluate.predict_joint` (two-stage when `asc`), `config.joint_polarity_source`.

### A9. Visual-relevance condition buckets (Table 9) 🔧
- **Paper:** reports F1 for "image-useful / image-irrelevant / weak image–text correspondence /
  multiple-aspect" without defining the buckets.
- **Fix:** useful/irrelevant ← teacher relevance label; weak ← low caption↔tweet token overlap;
  multiple-aspect ← `>1` gold aspect. → `analysis/visual_relevance_diag.py` (documented, tunable).

---

## B. NOT specified / missing in the paper (+ fixes)

### B1. Dataset source mismatch 🔧 (high impact)
- **Missing:** the data repo originally pointed to (`Lipika-Dewangan/TwitterDataMABSA`) contains
  **Twitter-2015 only**; the paper uses 2015 **and** 2017.
- **Fix:** use **`CopotronicRifat/TwitterDataMABSA`** (both splits + images). Data is **per-aspect
  TomBERT/MASC format (`$T$` placeholder), not joint BIO** → reconstruct joint BIO by grouping
  records per (tweet, image) and recovering spans from the `$T$` position. Verified record counts
  match Table 2. → `scripts/prepare_data.py`, `data.reconstruct_joint`.

### B2. Entity/relation embeddings and `ϕ` (Eq. 14) 🔧⚙️
- **Missing:** Eq. 14 `g_kq = ϕ([e_p; r; e_q])` doesn't say where `e_p,e_q,r` come from or what `ϕ` is.
- **Fix:** entities = **ConceptNet Numberbatch-EN (300-d) → Linear→768** (deterministic hash fallback
  for OOV/offline); relation = learned `nn.Embedding` over the ConceptNet-34 + SenticNet relations;
  `ϕ` = 2-layer FFN `(3·768→768→768)` + GELU. → `kg_retrieval.TripleEncoder`, `EntityEmbedder`.

### B3. LR schedule / warmup / patience / grad-clip / weight-decay 🔧⚙️
- **Missing:** §4.3 gives lr=2e-5 (AdamW), dropout 0.3, early stopping on dev F1 — but no schedule,
  warmup, patience, clipping, or weight decay.
- **Fix:** linear warmup 10% + linear decay, patience 5, grad-clip 1.0, weight_decay 0.01. → `train.py`, `config`.

### B4. SenticNet version & distribution 🔧⚙️
- **Missing:** paper cites SenticNet 7 [20]; the easy `pip senticnet` package ships SenticNet-5-era data.
- **Fix:** preferred path parses the **official SenticNet 7 RDF/XML** (`--rdf`); `pip senticnet` is a
  flagged fallback. → `scripts/download_senticnet.py`.

### B5. ConceptNet version 🔧📄
- **Missing:** the citation [21] is ConceptNet 5.5; the paper doesn't pin a download.
- **Fix:** use the latest stable **ConceptNet 5.7 assertions** (superset), **English-only** via the
  `/c/en/` prefix filter. → `scripts/download_conceptnet.py`.

### B6. BIO subtoken alignment 🔧
- **Missing:** BERTweet's slow tokenizer has no `word_ids()`; alignment of word-level BIO to subtokens isn't discussed.
- **Fix:** manual alignment — each word's **first** subtoken carries the BIO label, continuations get `-100`
  (ignored by `L_tag`); word↔subtoken map retained for span pooling and word-level eval. → `data.TarkanDataset`.

### B7. `ε` in the KG aggregation (Eq. 17) 🔧⚙️
- **Missing:** value of the division-by-zero guard.
- **Fix:** `ε = 1e-8`; `M_k=0` ⇒ `g̃_k = 0` vector. → `kg_filter.KGFilter`, `config.kg_eps`.

### B8. KAN width / depth 🔧⚙️
- **Missing:** layer count and hidden widths.
- **Fix:** `[3·768=2304, 512, 768]` (one hidden layer). → `config.kan_hidden`.

### B9. Paired bootstrap procedure 🔧
- **Missing:** §4.3 gives "1000 samples, p<0.05" but not the exact test.
- **Fix:** two-sided **paired bootstrap** over test instances (resample with replacement, recompute both
  systems' F1, count sign flips), seeded/reproducible. → `metrics.paired_bootstrap`.

### B10. Aux preprocessing tools (OCR/object detector/scene graph) 🔧
- **Missing:** none named; TARKAN's student needs only text+image encoders, captions, and KG.
- **Fix:** student uses BERTweet + CLIP only; captioner = BLIP; no OCR/detector required (those appear only
  in some *baselines* — noted in `referred_clones/FIXES.md`).

---

## C. Implementation / infra issues found & fixed (not paper-related)

- **C1.** `config.dropout` wasn't threaded into submodules (they read global `CONFIG`), so a `replace(cfg,
  dropout=…)` was silently ignored. **Fixed** — dropout now flows from `cfg` into every submodule
  (`models.py`, `kg_retrieval`, `kan_fusion`). Caught by the tiny-overfit test.
- **C2.** `fastkan` is **not reliably on PyPI** → install aborted. **Fixed** — default KAN = `efficient-kan`
  (git) with a **vendored RBF-KAN** fallback so fusion always runs; `fastkan`/`rkan` optional.
- **C3.** Windows `.git` strip failed (read-only pack files) → embedded repos would be treated as submodules.
  **Fixed** — `clone_referred.py` uses a chmod-retry `rmtree`; all 15 clones are plain source.
- **C4. Streaming KG (memory safety).** `build_kg.py` now **streams** parquet in 100k-row batches into the
  sqlite index (never materializes the millions-row ConceptNet table); `download_conceptnet.py` streams the
  498 MB `.gz` line-by-line; `download_senticnet.py` (RDF) clears the XML tree per element; the **runtime**
  KG is queried straight from the on-disk sqlite index (`kg.KnowledgeGraph(sqlite_path=…)`), so the full
  graph is **never** loaded into RAM; `EntityEmbedder.from_txt` reads Numberbatch line-by-line and accepts a
  `vocab` filter to load only needed embeddings.
- **C5.** spaCy needed `click` (missing dep) for `en_core_web_sm` — installed; documented in `requirements`.

---

_Status: deterministic CPU battery green (23 tests). All decisions above are configurable via `config.py`
unless marked 📄. See `implementation-plan.md` for full module specs and `walkthrough.md` for run order._
