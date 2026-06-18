"""Global determinism helpers. Call seed_everything() at the top of every entrypoint.

A green deterministic run on CPU is a hard gate before any GPU/LLM spend
(CLAUDE.md budget rule).
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and torch; optionally force deterministic torch algos."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cuBLAS determinism for matmuls on GPU; harmless on CPU.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def worker_init_fn(worker_id: int) -> None:
    """Deterministic DataLoader workers."""
    seed = (os.environ.get("PYTHONHASHSEED") or "42")
    base = int(seed) if str(seed).isdigit() else 42
    random.seed(base + worker_id)
    try:
        import numpy as np

        np.random.seed((base + worker_id) % (2**32 - 1))
    except Exception:
        pass
