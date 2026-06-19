"""Offline LLM teacher (paper §3.3, §3.5, Table 4) — TRAINING-TIME ONLY.

Produces two binary evidence-supervision signals, cached to parquet so the LLM is
loaded ONCE (offline pass), never during training or inference:
  - r^T_k  aspect-visual relevance   (supervises L_rel, Eq. 11)
  - s^T_kq KG-triple usefulness      (supervises L_kg,  Eq. 16)

Default teacher = Qwen/Qwen2.5-7B-Instruct (4-bit on T4). Greedy decoding, strict
{0,1} parsing. `build_targets` aligns cached labels to a live batch for losses.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from kg import Triple
from kg_retrieval import _triple_key as triple_key

RELEVANCE_PROMPT = (
    "Given a tweet, an aspect term, and an image description, decide whether the image "
    "provides useful evidence for inferring sentiment toward the aspect. "
    "Return 1 if useful and 0 otherwise."
)
KG_PROMPT = (
    "Given a tweet, an aspect term, and a candidate KG triple, decide whether the triple "
    "is useful for aspect-level sentiment reasoning. Return 1 if useful and 0 otherwise."
)
_BIN_RE = re.compile(r"[01]")


def _parse_binary(text: str) -> int:
    m = _BIN_RE.search(text or "")
    return int(m.group(0)) if m else 0


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
class TeacherCache:
    """In-memory lookup over the cached teacher labels for one dataset."""

    def __init__(self, rel: Dict[Tuple[str, int], float] = None, kg: Dict[Tuple[str, int, str], float] = None):
        self.rel = rel or {}
        self.kg = kg or {}

    @classmethod
    def load(cls, dataset: str) -> "TeacherCache":
        import pandas as pd

        base = CONFIG.paths.teacher_labels
        rel, kg = {}, {}
        rp = base / f"{dataset}_relevance.parquet"
        kp = base / f"{dataset}_kg.parquet"
        if rp.exists():
            df = pd.read_parquet(rp)
            for _, r in df.iterrows():
                rel[(str(r["instance_id"]), int(r["aspect_idx"]))] = float(r["label"])
        if kp.exists():
            df = pd.read_parquet(kp)
            for _, r in df.iterrows():
                kg[(str(r["instance_id"]), int(r["aspect_idx"]), str(r["triple_key"]))] = float(r["label"])
        return cls(rel, kg)

    @staticmethod
    def save(dataset: str, rel_rows: List[dict], kg_rows: List[dict]) -> None:
        import pandas as pd

        base = CONFIG.paths.teacher_labels
        base.mkdir(parents=True, exist_ok=True)
        if rel_rows:
            pd.DataFrame(rel_rows).to_parquet(base / f"{dataset}_relevance.parquet", index=False)
        if kg_rows:
            pd.DataFrame(kg_rows).to_parquet(base / f"{dataset}_kg.parquet", index=False)


# --------------------------------------------------------------------------- #
# LLM teacher
# --------------------------------------------------------------------------- #
class LLMTeacher:
    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or CONFIG.teacher_llm_id
        self.device = device or CONFIG.device
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"token": CONFIG.hf_token}
        if self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
                )
                kwargs["device_map"] = "auto"
            except Exception:
                kwargs["torch_dtype"] = torch.float16
        self._tok = AutoTokenizer.from_pretrained(self.model_id, token=CONFIG.hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()
        # We decode greedily (do_sample=False). Clear the model's default sampling params
        # so transformers doesn't warn about temperature/top_p/top_k on every call.
        gc = self._model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
        if gc.pad_token_id is None and self._tok.pad_token_id is not None:
            gc.pad_token_id = self._tok.pad_token_id

    def _ask(self, system: str, user: str) -> int:
        import torch

        self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{system}\n\n{user}\nAnswer:"
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_binary(gen)

    def relevance_label(self, tweet: str, aspect: str, image_description: str) -> int:
        user = f"Tweet: {tweet}\nAspect: {aspect}\nImage description: {image_description}\nAnswer (0 or 1):"
        return self._ask(RELEVANCE_PROMPT, user)

    def kg_label(self, tweet: str, aspect: str, triple: Triple) -> int:
        tr = f"({triple.head}, {triple.relation}, {triple.tail})"
        user = f"Tweet: {tweet}\nAspect: {aspect}\nKG triple: {tr}\nAnswer (0 or 1):"
        return self._ask(KG_PROMPT, user)


# --------------------------------------------------------------------------- #
# align cached labels to a live batch (consumed by losses.compute_losses)
# --------------------------------------------------------------------------- #
def build_targets(batch: Dict, outputs: Dict, cache: Optional[TeacherCache] = None, cfg=CONFIG) -> Dict:
    import torch

    B = len(batch["aspect_spans"])
    order = [(b, k) for b in range(B) for k in range(len(batch["aspect_spans"][b]))]
    device = outputs["tag_logits"].device

    targets: Dict = {"bio_labels": batch["bio_labels"]}

    pol = [batch["aspect_polarity"][b][k] for (b, k) in order]
    targets["aspect_polarity"] = torch.tensor(pol, dtype=torch.long, device=device)

    if cache is not None:
        # relevance
        rvals, rmask = [], []
        for (b, k) in order:
            iid = batch["instance_id"][b]
            v = cache.rel.get((iid, k))
            rvals.append(float(v) if v is not None else 0.0)
            rmask.append(v is not None)
        targets["teacher_relevance"] = torch.tensor(rvals, dtype=torch.float, device=device)
        targets["teacher_relevance_mask"] = torch.tensor(rmask, dtype=torch.bool, device=device)

        # kg (aligned to outputs['kg_triples'] order)
        kg_t, kg_m = [], []
        for a, (b, k) in enumerate(order):
            iid = batch["instance_id"][b]
            triples = outputs["kg_triples"][a] if a < len(outputs["kg_triples"]) else []
            tv, tm = [], []
            for tr in triples:
                v = cache.kg.get((iid, k, triple_key(tr)))
                tv.append(float(v) if v is not None else 0.0)
                tm.append(v is not None)
            kg_t.append(torch.tensor(tv, dtype=torch.float, device=device))
            kg_m.append(torch.tensor(tm, dtype=torch.bool, device=device))
        targets["teacher_kg"] = kg_t
        targets["teacher_kg_mask"] = kg_m

    return targets
