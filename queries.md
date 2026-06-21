# TARKAN — Open Questions, Ambiguities & Fixes

A consolidated log of (A) **important but vague** things in the paper, (B) things **not
specified / missing**, and (C) **implementation/infra issues** found while building. Each
entry states what the paper says (or doesn't), why it matters, and the concrete fix +
where it lives + whether it's configurable. This is the authoritative companion to
`implementation-plan.md §14`.

Legend: 🔧 = decision implemented · ⚙️ = configurable in `config.py` · 📄 = paper-faithful default.

---

## A. Primary VAGUE / ambiguous things (present in the paper, but underspecified)

### A1. Total loss objective — auxiliary ASC loss REMOVED ✅ (resolved by 2026-06-21 paper update)
- **Updated paper (§3.7):** `L = L_tag + λ1·L_rel + λ2·L_kg` (exactly 3 terms). The earlier
  auxiliary span loss `L_asc` (old Eq. 24), the `λ3` weight, and the "w/o auxiliary ASC loss"
  ablation are **gone**: §3.6 now folds aspect extraction **and** sentiment classification into the
  single unified BIO head, so there is no separate span classifier to supervise.
- **Resolution:** removed `asc_loss`/`l_asc`/`λ3` and `SpanSentimentHead`. → `losses.compute_losses`,
  `config` (no `lambda3`), `heads.py`, `models.py`. See [A8] and [C6].
- **⚠ Paper inconsistency (please fix in the latex master):** §3.1 (≈ line 382) still says "we use an
  auxiliary span-level sentiment classifier over the fused aspect representation," and the §6 ablation
  discussion (≈ line 1382) still says "The auxiliary ASC loss gives a smaller but consistent gain."
  Both are stale remnants — the tables, §3.6, and §3.7 already removed the ASC head/loss. Recommend
  deleting those two sentences. (Edited in the local `TARKAN_latex` copy as well; mirror in your master.)

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

### A8. Inference: joint polarity source — RESOLVED to the unified BIO head ✅ (2026-06-21 update)
- **Updated paper (§3.6 + §3.8):** there is no ASC head. The single unified BIO head runs on the
  **KAN-fused multimodal token representation** h̃_i (Eq. 21) and yields span **and** polarity. The old
  `bio`-vs-`asc` ambiguity is gone, and the `joint_polarity_source` toggle was removed.
- **Resolution — two-stage inference (§3.8):** (1) predict BIO with *empty* aspect spans (zero visual/KG
  evidence) to extract aspect spans; (2) for each predicted span, recompute relevance-filtered visual +
  filtered KG, re-fuse per token via KAN, and re-run the BIO head → final polarity. Span boundaries use a
  boundary-first decode (collapse polarity → segment → majority polarity) to avoid fragmenting multi-word
  spans. MASC reads the BIO polarity on gold spans. → `evaluate.predict_joint` / `predict_masc`, `models.py`.
- **Design note (circularity):** aspect-centered visual/KG need spans, but spans come from BIO. Training
  uses gold spans (§3.2); inference uses the two-stage pass. Stage-1 uses zero evidence (matching how O
  tokens are trained), so it is a text-driven *extraction* pass; stage-2 supplies the multimodal evidence
  that decides the *polarity*. See [C6].

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

### B2. Entity/relation embeddings and `ϕ` (Eq. 14) ✅⚙️ (confirmed by authors 2026-06-21)
- **Missing in paper:** Eq. 14 `g_kq = ϕ([e_p; r; e_q])` doesn't say where `e_p,e_q,r` come from or what `ϕ` is.
- **Confirmed implementation (authors):** `ϕ` = **two-layer feed-forward projection with GELU**. The head
  entity, relation, and tail entity embeddings are each mapped to the same hidden dim, concatenated, and
  projected to a **768-d** triple representation. Entities = ConceptNet Numberbatch-EN (300-d) → Linear→768
  (deterministic hash fallback for OOV/offline); relation = learned `nn.Embedding` over ConceptNet-34 +
  SenticNet relations. → `kg_retrieval.TripleEncoder`, `EntityEmbedder`. (Authors: "current implementation
  is appropriate.")

### B3. LR schedule / warmup / patience / grad-clip / weight-decay 🔧⚙️
- **Missing:** §4.3 gives lr=2e-5 (AdamW), dropout 0.3, early stopping on dev F1 — but no schedule,
  warmup, patience, clipping, or weight decay.
- **Fix:** linear warmup 10% + linear decay, patience 5, grad-clip 1.0, weight_decay 0.01. → `train.py`, `config`.

### B4. SenticNet version & distribution 🔧⚙️
- **Missing:** paper cites SenticNet 7 [20]; the easy `pip senticnet` package ships SenticNet-5-era
  data (and isn't installed here — it raised `ModuleNotFoundError`).
- **Fix:** **single canonical source = the official `senticnet.py` dump at `data/senticnet/senticnet.py`**
  (SenticNet 7; 292,357 EN concepts). `download_senticnet.py` parses it with a tolerant line reader
  (the official file has a few malformed emoticon keys → skipped; they normalize to empty KG keys
  anyway). Auto-found by `data_setup.py`; `--py` / `--git <url>` / `--rdf` are alternatives. The pip
  package path was removed. → `scripts/download_senticnet.py`.

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
- **C6. Unified-BIO methodology refactor (2026-06-21 paper update).** Implemented the updated §3.6/§3.7/§3.8:
  the BIO head (Eq. 21) now runs on the **KAN-fused multimodal token representation** h̃_i — each aspect's
  relevance-filtered visual + filtered KG is broadcast to its token positions, O tokens get zero evidence.
  One head does extraction **and** sentiment; the separate ASC head / `L_asc` / `λ3` are removed (see A1, A8).
  Objective `L = L_tag + λ1 L_rel + λ2 L_kg`. New ablation toggle **`use_kan_tag_representation`** (default
  True; False ⇒ BIO on text-only) reproduces the new Table-6 row **"w/o KAN-enhanced tag representation"**
  (74.1→73.4 / 72.9→72.1). → `models.py`, `heads.py`, `losses.py`, `config.py`, `evaluate.py`, `ablations/run_ablations.py`.
  - **Design decisions — making the under-specified "BIO on KAN-fused token rep" trainable (resolved feasibly;
    the paper's §3.6 + §3.8 are circular: aspect-centered evidence needs spans, spans come from BIO).** Three
    standard, documented mechanisms, each fixing an empirically-observed failure:
    1. **Residual fusion:** §3.6 says h̃_i "combines its contextual textual representation **with** evidence", so
       `h̃_i = h^t_i + KAN([h^t_i; ṽ; g̃])`, NOT h̃_i = KAN(...) replacing text. Replacing text makes the fresh
       per-token KAN wash out BERTweet features → head collapses to all-O (dev F1=0).
    2. **Add & norm:** `h̃_i = LayerNorm(h^t_i + KAN(...))`. Without it the KAN output scale perturbs the head →
       extraction oscillates (dev F1 0↔7 between epochs). With it, polarity learns cleanly (MASC climbs).
    3. **Evidence dropout (`config.evidence_dropout=0.5`, train only):** zero a random subset of instances'
       per-token evidence so the head ALSO learns text-only extraction. Without it the head learns "B/I requires
       evidence" and at stage-1 inference (zero evidence) extracts nothing (recall ~0) even though MASC (gold
       spans, evidence present) reaches ~73%. Matches stage-1's zero-evidence regime. Doesn't touch L_rel/L_kg.
    The ablation (`use_kan_tag_representation=False`) drops the KAN addend entirely → pure text BIO (and the
    LayerNorm/evidence-dropout are bypassed).
- **C7.** Eval-time model build now mirrors training (KG + entity embedder) in `evaluate.py` (`__main__`) and
  `experiments/run_subtasks.py` via `evaluate._build_kg_and_entities` — previously they built the student
  without the KG stream, so a standalone checkpoint eval ran KG-inert. (`experiments/run_main.py` already
  evaluated the in-memory trained model, so it was unaffected.)

---

_Status: deterministic CPU battery green (23 tests, updated for the unified-BIO objective). GPU smoke of the
new pipeline passes (forward+backward + two-stage eval; 5.6 GB peak, fits T4). All decisions above are
configurable via `config.py` unless marked 📄. See `implementation-plan.md` for module specs and
`walkthrough.md` for run order._
