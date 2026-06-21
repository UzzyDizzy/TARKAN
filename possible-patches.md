# TARKAN — Possible Speed-up Patches

Performance patches for the full experiment sweep on a single **T4**. Split into
(1) **pure speed-ups** (result-neutral) and (2) a **speed-up that also fixes a documented
`queries.md` inconsistency**. None are applied yet — this is a menu.

**Baseline (measured, current code):**
- Teacher labeling (one-time, ~99k LLM calls @ ~255 ms): **~7 h**
- Training run: ~154 s/epoch × ~23 epochs ≈ **~1.0 h/run**; ~28 runs (main + Table 6 + Table 10) ≈ **~28–31 h**
- **Total ≈ ~36–39 h** continuous T4.

---

## Paper-faithfulness of each patch

| Patch | Faithfulness | Why |
|---|---|---|
| P1 batch teacher `generate()` | **obeying** | result-neutral (greedy labels identical); paper is silent on batching |
| P2 `num_workers>0` | **obeying** | data-loading infra; results unchanged |
| P3 AMP fp16 | **obeying** | precision only; methodology unchanged (≈neutral, tiny fp16 numerics) |
| P4 cache KG retrieval | **obeying** | same deterministic triples, just cached |
| P5 freeze+cache CLIP | **disobeying** | paper fine-tunes the visual encoder; freezing changes the model |
| P6 Numberbatch vocab filter | **obeying** | identical in-vocab vectors; loads a subset (also fixes C4) |
| A1 class-weighted/focal `L_tag` | **disobeying** | paper's `L_tag` is plain CE (Eq. 22); weighting is a reproduction aid |
| A2 tune `evidence_dropout` | **disobeying** | evidence dropout is a non-paper mechanism (our feasibility fix for §3.6's circularity) |
| A3 layer-wise LR | **disobeying** | §4.3 states a single `lr=2e-5`; per-group LR deviates (extends B3) |
| A4 CRF head | **disobeying** | paper uses a softmax BIO head (Eq. 21), not a CRF |
| A5 label smoothing on `L_tag` | **disobeying** | paper's `L_tag` is plain CE (Eq. 22) |
| A6 infer-time top-M re-rank by learned `s_kq` | **obeying** | §3.4 lists teacher-usefulness as a top-M criterion; `s_kq` is its learned proxy (also fixes A4) |

**Non-disobeying (paper-faithful):** P1, P2, P3, P4, P6, A6. **Disobeying (deviate from the paper):** P5, A1, A2, A3, A4, A5.

---

## 1. Pure speed-ups (do not change results)

### P1. Batch the teacher's `generate()` calls  ★ biggest one-time win
- **Now:** `teacher.py:_ask` runs **one** prompt per `generate()` (greedy). ~99k sequential calls ≈ 7 h.
- **Patch:** batch N prompts per `generate()` with left-padding + attention mask; also collapse the
  **M=10 per-aspect KG-triple prompts into one** prompt that scores all retrieved triples at once.
- **Gain:** GPU batching 8–16× throughput + ~10× fewer KG calls ⇒ labeling **~7 h → ~1–1.5 h**.
- **Result-neutral?** Yes — greedy decoding on padded batches yields identical per-sequence argmax/{0,1}
  labels (padding is masked). → `teacher.py` (`_ask`, `relevance_label`, `kg_label`, `run_teacher_labeling.py`).
- **Effort:** medium. **Risk:** low (verify a few labels match the unbatched output).

### P2. `DataLoader(num_workers>0, pin_memory=True)`
- **Now:** all loaders use `num_workers=0` (`train.py:make_loader`) → image decode + tokenization +
  KG-query build run on the main process, serial with the GPU step.
- **Patch:** `num_workers=4–8`, `pin_memory=True`, `persistent_workers=True` (+ the existing
  `worker_init_fn` already seeds workers).
- **Gain:** ~10–30 % per-epoch (data prep overlaps compute). Across ~28 runs that's hours.
- **Result-neutral?** Yes (seeded workers). → `train.py:make_loader` (+ experiment/ablation loaders).
- **Effort:** trivial. **Risk:** low.

### P3. Mixed-precision training (AMP)
- **Now:** full fp32. **Patch:** `torch.autocast('cuda', dtype=torch.float16)` + `GradScaler` around the
  forward/loss/backward in `train.py`. (T4 = fp16, not bf16.) Keep the KAN/`LayerNorm` in fp32 if any
  instability appears (`autocast` already does this for norms).
- **Gain:** ~1.3–1.7× step time **and** lower memory (room for larger batch → further speed).
- **Result-neutral?** Near-neutral (minor fp16 numerics; F1 within run-to-run noise). → `train.py`.
- **Effort:** low. **Risk:** low–medium (watch KAN spline numerics; fall back to bf16-emulation/fp32 if NaNs).

### P4. Cache aspect-centered KG retrieval (skip sqlite every epoch)
- **Now:** `retrieve_triples()` hits `kg.sqlite` for every aspect on **every** forward (every epoch).
  The model already supports `batch["aspect_triples"]` to bypass this, but training never populates it.
- **Patch:** precompute retrieval once per split (the cached teacher labels in
  `data/teacher_labels/*_kg.parquet` already enumerate the retrieved `triple_key`s), store per-instance
  triples, and feed them via `aspect_triples`.
- **Gain:** removes all per-epoch sqlite I/O (~the dominant CPU cost at `num_workers=0`); compounds with P2.
- **Result-neutral?** Yes (same triples, deterministic retrieval). → `train.py`, `data.py`, `kg_retrieval`.
- **Effort:** medium. **Risk:** low.

### P5. (Conditional) Freeze + cache CLIP visual features
- **Now:** the CLIP visual encoder is **fine-tuned** (no freeze), so its features change each step.
- **Patch:** freeze CLIP, precompute the 49 patch features per image once, cache to disk, and load instead
  of running CLIP each epoch. ~20–40 % per-epoch.
- **Result-neutral?** **No** — freezing CLIP changes results (small, often negligible). Listed for
  completeness; only adopt if the accuracy delta is acceptable. → `encoders.VisualEncoder`, `train.py`.

> **Projected with P1–P4:** labeling ~7 h → ~1.5 h; training ~28–31 h → **~18–22 h**;
> **total ~36–39 h → ~20–24 h** (≈ 1.7× overall, result-neutral).

---

## 2. Speed-up that also fixes a `queries.md` inconsistency

### P6. Use the Numberbatch **vocab filter** when loading entity embeddings
- **Inconsistency fixed: `queries.md` → C4.** C4 states *"`EntityEmbedder.from_txt` reads Numberbatch
  line-by-line and accepts a `vocab` filter to load only needed embeddings"* and the method's own docstring
  promises *"cuts memory from ~600 MB to just the KG vocabulary."* **But both callers ignore it:**
  `train.py` and `evaluate._build_kg_and_entities` call `EntityEmbedder.from_txt(str(nb))` with **no
  `vocab`**, so every run loads **all ~417k** vectors (~600 MB, the ~35 s startup you see). The documented
  capability is never exercised — code vs. C4 are inconsistent.
- **Patch:** build the dataset KG-entity vocabulary once (the head/tail entities of all retrieved triples —
  available from the cached `*_kg.parquet` / P4's precomputed retrieval, normalized the same way as
  `EntityEmbedder`), then pass it: `EntityEmbedder.from_txt(nb, vocab=kg_vocab)`. OOV → existing hash
  fallback (unchanged behavior).
- **Gain:** entity-embedder load **~35 s → ~2–5 s** per run and **~600 MB → tens of MB** RAM. Across ~28
  training runs + every standalone eval that's ~15–25 min wall-clock saved and headroom for a bigger batch
  (compounds with P3). 
- **Result-neutral?** Yes for in-vocab entities (identical vectors); out-of-vocab entities already used the
  hash fallback, so behavior is unchanged for them too **provided** the vocab is built from the same
  retrieval set (so nothing that *would* have matched is dropped).
- **Also resolves C4:** after this, C4's description matches reality (vocab filter actually used). Update the
  C4 note from "accepts a vocab filter" → "uses the dataset KG vocab filter (see P6)".
- **Effort:** low–medium. **Risk:** low (verify in-vocab vectors identical; confirm F1 unchanged on one split).
  → `kg_retrieval.EntityEmbedder.from_txt` (caller side), `train.py`, `evaluate._build_kg_and_entities`.

---

## 3. Accuracy patches — close the gap to the paper

Current test (twitter2015, new methodology): joint **62.4** / MATE **82.9** / MASC Acc **74.4** / macF1 **68.9**
vs paper **74.1 / 89.0 / 82.6 / 82.4**. These are *minimal* patches to narrow that gap. Each notes any
speed side-effect and any `queries.md` inconsistency it also resolves. **Honest expectation:** combined they
plausibly lift joint into the high-60s/low-70s; fully reaching 74.1 may still need the paper's exact (unstated)
recipe — the residual gap is model-quality, not a single bug (see `context-to-be-processed.md`).

### A1. Class-weighted (or focal) `L_tag`  ★ targets the biggest gap (MASC macF1 −13.5)
- **What:** weight the 7 BIO classes by inverse frequency (up-weight the rare **NEG** tags, and B/I vs the
  dominant `O`) in `losses.tag_loss`, or use focal loss. Now that polarity lives in the unified BIO CE, the
  minority-polarity collapse (NEU-majority) shows up here — MASC sits at 68.9 macF1 / 74.4 acc.
- **Accuracy:** macF1 **+5 to +10**, joint **+1–2** (better minority polarity → more correct (span,polarity)).
- **Speed:** none. **Inconsistency:** none. **Effort:** low (`losses.tag_loss`). **Risk:** low (tune weight so
  precision doesn't drop; this is a reproduction aid — the paper presumably used plain CE).

### A2. Tune `evidence_dropout` (0.5 → 0.2–0.3, or anneal)
- **What:** 0.5 zeroes the multimodal evidence for half the training instances, so the head is trained ~half the
  time as text-only — likely why test (62.4) ≈ the old text-only result and visual/KG add little. Lower it (or
  anneal high→low) so the polarity-from-evidence path trains more while keeping stage-1 extraction working.
- **Accuracy:** joint **+1–3** *if* the multimodal signal is real (uncertain — KG was ~inert before).
- **Speed:** none. **Inconsistency:** none. **Effort:** trivial (`config.evidence_dropout`). **Risk:** low
  (too low → stage-1 extraction regresses; keep ≥0.2; sweep {0.2,0.3,0.5}).

### A3. Discriminative / layer-wise learning rate
- **What:** the fresh modules (KAN fusion, relevance gate, KG triple-encoder/filter, BIO head) are likely
  undertrained at the encoders' 2e-5. Use param groups: encoders 2e-5, new modules 1e-4.
- **Accuracy:** joint **+1–3** (lets the multimodal streams actually contribute).
- **Speed:** none. **Inconsistency:** extends **B3** (which fixes a *single* LR/schedule) — a layer-wise LR is a
  more faithful resolution of B3's under-specification. **Effort:** low (param groups in `train.py`). **Risk:** low–medium (tune).

### A4. Linear-chain CRF decoding on the BIO head
- **What:** add a CRF layer so decoding enforces valid BIO transitions and span-internal polarity consistency
  (no `I-` without `B-`, no mid-span flips). Complements the boundary-first decode already in place.
- **Accuracy:** MATE **+1–3**, joint **+1–2** (cleaner spans → both metrics).
- **Speed:** marginally slower (CRF forward/Viterbi; negligible vs the encoders). **Inconsistency:** none.
  **Effort:** medium (`torchcrf` dep + `losses`/`evaluate`). **Risk:** medium (new dep, training change).

### A5. Label smoothing (0.1) on `L_tag`
- **What:** `F.cross_entropy(..., label_smoothing=0.1)` in `tag_loss`.
- **Accuracy:** **+0.5–1** (calibration + minority help). **Speed:** none. **Inconsistency:** none.
  **Effort:** trivial. **Risk:** low. (Cheap to stack with A1.)

### A6. Inference-time top-M re-selection by the learned KG filter — also fixes an inconsistency
- **What:** at inference, select/weight the top-M KG triples by the **learned usefulness `s_kq`** (the student's
  proxy for teacher usefulness), not only the static retrieval score.
- **Accuracy:** small joint gain if KG signal is real (uncertain). **Speed:** none.
- **Inconsistency fixed: `queries.md` A4** — A4's top-M score lists "(+ teacher usefulness score)" but that
  term is only available at *training* (offline labels); using `s_kq` operationalizes that criterion at
  inference and resolves the dangling parenthetical. **Effort:** medium (`kg_retrieval`/`kg_filter`, `evaluate`).
  **Risk:** medium.

> **Combined A1–A5 (skip A4/A6 if avoiding new deps):** optimistic joint ~62 → ~68–70, MASC macF1 ~69 → ~78.
> Verify on twitter2015 first (~1 h) before committing GPU to the full sweep.

---

## Suggested order
**Speed (do first, low-risk):** P2 + P6 → P1 (huge one-time labeling cut) → P4 → P3 → P5 only if freezing CLIP
is acceptable. After P6, edit `queries.md` C4 to reflect the vocab filter is now used.
**Accuracy (one twitter2015 run each to measure):** A1 + A5 together (cheap, target MASC) → A2 sweep → A3 →
A4/A6 only if the cheap ones leave a gap worth a new dependency. After A3, note the layer-wise LR under B3;
after A6, mark A4 in `queries.md` resolved.
