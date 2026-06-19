# TARKAN — GPU-Server Walkthrough

Step-by-step to reproduce every TARKAN result on a single **T4 (16 GB)** (or better).
Local CPU is only for the deterministic test battery; everything below assumes the server.

All heavy outputs land under `data/` and `results/{checkpoints,logs}/` (git-ignored).
Run commands from the repo root.

---

## 0. Environment (once)

```bash
git clone <your-fork>/TARKAN && cd TARKAN
python -m venv venv && source venv/bin/activate        # (Windows: venv\Scripts\activate)

# GPU PyTorch (pick the CUDA build for your driver; cu121 shown):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install bitsandbytes                                # 4-bit teacher (GPU only)
pip install git+https://github.com/Blealtan/efficient-kan.git
python -m spacy download en_core_web_sm

cp .env.example .env.local      # put your HF_TOKEN inside (Qwen teacher is ungated)
```

Sanity check (no GPU needed, no downloads):
```bash
python -m pytest tests/ -q       # expect: 23 passed
```

---

## 1. One-time data + KG setup  (`data_setup.py`)

Downloads datasets/images, ConceptNet 5.7 (EN) + Numberbatch, SenticNet 7 (EN), and
builds the KG sqlite index. **None of this is git-pushed.**

```bash
python data_setup.py                              # steps 1-4 (CPU-OK, ~10 GB transient disk for ConceptNet)
# For SenticNet 7 fidelity, download the RDF from https://sentic.net/downloads/ then:
python data_setup.py --skip-data --skip-conceptnet --senticnet-rdf /path/senticnet7.rdf
```
Produces: `data/twitter2015|twitter2017/*.tsv`, `data/images/...`, `data/conceptnet/conceptnet_en.parquet`,
`data/conceptnet/numberbatch-en.txt`, `data/senticnet/senticnet_en.parquet`, `data/kg_index/kg.sqlite`.
The script prints Table-2 record-count checks (3179/1122/1037 and 3562/1176/1234).

---

## 2. Offline teacher labeling  (GPU; one-time, cached)

Captions images (BLIP) and runs the Qwen teacher (4-bit) to produce `r^T_k`, `s^T_kq`
(Table 4 prompts). Cached to `data/teacher_labels/*.parquet`; training never loads the LLM.

```bash
python scripts/run_teacher_labeling.py --dataset twitter2015 --device cuda
python scripts/run_teacher_labeling.py --dataset twitter2017 --device cuda
# debug a few first:  --limit 20
```
Resumable — re-running skips already-labeled (instance, aspect[, triple]).

---

## 3. Train TARKAN  (Algorithm 1, Eq. 25)

```bash
python train.py --dataset twitter2015 --device cuda
python train.py --dataset twitter2017 --device cuda
```
Early-stops on dev joint-F1; best checkpoint → `results/checkpoints/<dataset>_best.pt`.
λ-sweep (reproduces selecting λ1=λ2=0.5): pass `--lambda1/--lambda2` from `{0.1,0.3,0.5,1.0}`.

Evaluate a checkpoint:
```bash
python evaluate.py --dataset twitter2015 --split test --checkpoint results/checkpoints/twitter2015_best.pt
```

---

## 4. Main results & subtasks  (Tables 1, 3)

```bash
python experiments/run_main.py     --device cuda     # -> results/tables/main_results.csv  (Table 1)
python experiments/run_subtasks.py --device cuda     # -> results/tables/subtasks.csv      (Table 3: MATE/MASC)
```

## 5. Ablations  (Tables 6, 10)

```bash
python ablations/run_ablations.py       --device cuda   # -> ablation_components.csv (8 variants, Table 6)
python ablations/run_fusion_ablation.py --device cuda   # -> ablation_fusion.csv     (7 fusions, Table 10)
```

> **Joint-polarity source (Tables 6 & 10).** The joint metric's polarity comes from
> `config.joint_polarity_source` (default `bio`). With `bio`, the fusion/visual/KG branch only
> affects joint-F1 indirectly, so Table 10's joint column is ~flat across fusion strategies and
> Table 6's visual/KG drops are muted. To make the multimodal components move the **joint** metric
> (paper §3.7 inference), set `joint_polarity_source='asc'` — span from BIO, final polarity from the
> KAN-fused ASC head. Run the ablations under **both** settings and keep whichever matches the paper.
> (Requires the built KG index, else the two KG ablations are no-ops — see §1.)

## 6. Diagnostics  (Tables 5, 7, 8, 9)

```bash
python analysis/kg_diagnostics.py       --device cuda   # Table 8 (match rate, retrieved/retained, source split)
python analysis/visual_relevance_diag.py --device cuda  # Table 9 (F1 by visual-relevance bucket)
python analysis/error_analysis.py       --device cuda   # Table 5 (heuristic; refine by manual inspection)
# Table 7 needs a human-verified subset:
#   data/teacher_labels/<ds>_relevance_human.parquet , <ds>_kg_human.parquet
python analysis/teacher_quality.py
```

## 7. Figures

```bash
python visualizations/plot_main_results.py
python visualizations/plot_ablation.py
python visualizations/plot_kg_stats.py
python visualizations/plot_relevance.py    --dataset twitter2015 --device cuda
python visualizations/attention_viz.py     --dataset twitter2015 --index 0 --aspect 0
python visualizations/kan_spline_viz.py
# -> results/plots/*.png
```

---

## 8. Baselines  (`referred_clones/`)

Each baseline targets an **old, mutually-incompatible** stack — give each its **own**
venv/conda env (do **not** install into the main env). Per-repo provenance + fixes are in
[`referred_clones/FIXES.md`](referred_clones/FIXES.md). Priority reproductions (no `*` in
Table 1): VLP-MABSA, JML, AoM, M2DF, CMMT, MultiPoint, DQPSA, TCMT, VLHA. Record their
numbers in `results/tables/baselines.csv`. Cite-only (no public code): SGBIS, RNG,
Vanessa, DSEM. Text-only (SemEval, cited): SPAN, D-GCN.

---

## T4 memory notes
- Student training (BERTweet + CLIP-ViT-B/32, batch 16, len 128) fits comfortably.
- Teacher (Qwen2.5-7B 4-bit ≈ 3.8 GB) + BLIP (≈ 1.5 GB) run in the **labeling** pass only,
  separate from training — no concurrent load. Use `batch_size 1` for the LLM if memory is tight.
- If you hit OOM: lower `CONFIG.batch_size`, or set `kan_backend='fastkan'` (lighter than B-spline).

## Troubleshooting
- `kg.sqlite` missing → KG stream is inert (runs but `g̃_k=0`); run `data_setup.py` step 4.
- Teacher cache missing → `L_rel`/`L_kg` are 0 (model still trains on `L_tag`+`L_asc`); run step 2.
- Gated teacher (Llama-3.1) → accept its HF license; default Qwen needs none.
