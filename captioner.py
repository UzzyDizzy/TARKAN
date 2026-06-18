"""Image captioner — produces the "image description" the teacher relevance prompt
needs (paper Table 4) and the noun keywords for C_k (Eq. 12).

Default model: Salesforce/blip-image-captioning-large (Open-Q #3). Captions are cached
to data/captions/<dataset>/<image_id>.txt so the (heavy) captioning pass runs once.
Models load lazily — importing this module downloads nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from config import CONFIG


class Captioner:
    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or CONFIG.captioner_id
        self.device = device or CONFIG.device
        self._proc = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self._proc = BlipProcessor.from_pretrained(self.model_id, token=CONFIG.hf_token)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_id, token=CONFIG.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()

    @staticmethod
    def _cache_path(dataset: str, image_id: str) -> Path:
        return CONFIG.paths.captions / dataset / (image_id + ".txt")

    def caption_image(self, image_path, max_new_tokens: int = 30) -> str:
        import torch
        from PIL import Image

        self._load()
        img = Image.open(image_path).convert("RGB")
        inputs = self._proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3)
        return self._proc.decode(out[0], skip_special_tokens=True).strip()

    def caption_dataset(self, dataset: str, image_ids: List[str], images_dir) -> Dict[str, str]:
        """Caption all images for a split, caching to disk. Returns {image_id: caption}."""
        images_dir = Path(images_dir)
        out: Dict[str, str] = {}
        for image_id in image_ids:
            cp = self._cache_path(dataset, image_id)
            if cp.exists():
                out[image_id] = cp.read_text(encoding="utf-8").strip()
                continue
            try:
                cap = self.caption_image(images_dir / image_id)
            except Exception:
                cap = ""
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(cap, encoding="utf-8")
            out[image_id] = cap
        return out


def load_captions(dataset: str) -> Dict[str, str]:
    """Load any cached captions for a dataset split-set into {image_id: caption}."""
    out: Dict[str, str] = {}
    d = CONFIG.paths.captions / dataset
    if d.exists():
        for f in d.glob("*.txt"):
            out[f.stem] = f.read_text(encoding="utf-8").strip()
    return out
