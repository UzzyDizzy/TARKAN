"""KAN-based multimodal fusion (paper Eqs. 18-20) + all Table-10 fusion strategies.

    u_k = [t_k ; v_tilde_k ; g_tilde_k]                                  (Eq. 18)
    z_{l+1,j} = sum_i psi^(l)_ij(z_{l,i})   (learnable univariate edges)  (Eq. 19)
    z_k = KAN(u_k)                                                        (Eq. 20)

KAN backend is selectable (config.kan_backend):
  - 'efficient_kan' (default): B-spline edges (closest to Eq. 19), pip git install.
  - 'fastkan'      : Gaussian-RBF KAN (pip fastkan), faster.
  - 'rkan'         : rational-KAN activations (paper ref [41]).
A self-contained RBF-KAN (RBFKAN) is vendored as a zero-dependency fallback so the
fusion ALWAYS runs (it is a genuine KAN: learnable univariate edge functions).

Every fusion in FUSION_REGISTRY implements forward(t, v, g) -> z, mapping three
[K, d] modality tensors to a fused [K, d] (Table 10).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


# --------------------------------------------------------------------------- #
# Vendored fallback KAN (Gaussian-RBF edges) — always available.
# --------------------------------------------------------------------------- #
class RBFKANLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, num_grids: int = 8, grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids))
        self.h = (grid_max - grid_min) / max(num_grids - 1, 1)
        self.coeff = nn.Parameter(torch.randn(out_f, in_f, num_grids) * 0.1)
        self.base = nn.Linear(in_f, out_f)
        self.norm = nn.LayerNorm(in_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        rbf = torch.exp(-(((xn.unsqueeze(-1) - self.grid) / self.h) ** 2))  # [*, in, G]
        spline = torch.einsum("...ig,oig->...o", rbf, self.coeff)            # [*, out]
        return spline + self.base(F.silu(xn))


class RBFKAN(nn.Module):
    def __init__(self, layers_hidden: List[int], num_grids: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [RBFKANLayer(layers_hidden[i], layers_hidden[i + 1], num_grids) for i in range(len(layers_hidden) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _build_kan(backend: str, widths: List[int], grid_size: int, spline_order: int) -> nn.Module:
    backend = (backend or "efficient_kan").lower()
    if backend == "efficient_kan":
        try:
            from efficient_kan import KAN  # B-spline KAN

            return KAN(layers_hidden=widths, grid_size=grid_size, spline_order=spline_order)
        except Exception:
            pass  # fall through to RBF fallback
    if backend == "fastkan":
        try:
            from fastkan import FastKAN

            return FastKAN(layers_hidden=widths, num_grids=max(grid_size + 3, 8))
        except Exception:
            pass
    if backend == "rkan":
        try:
            return _RationalKAN(widths)
        except Exception:
            pass
    # universal fallback (genuine RBF-KAN)
    return RBFKAN(widths, num_grids=max(grid_size + 3, 8))


class _RationalKAN(nn.Module):
    """Linear layers with rational-KAN activations (paper ref [41])."""

    def __init__(self, widths: List[int]):
        super().__init__()
        from rkan.torch import JacobiRKAN  # type: ignore

        blocks = []
        for i in range(len(widths) - 1):
            blocks.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                blocks.append(JacobiRKAN(3))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Fusion strategies (Table 10). All: forward(t, v, g) -> z  ([K, d] each).
# --------------------------------------------------------------------------- #
class KANFusion(nn.Module):
    def __init__(self, d: int = None, hidden=None, backend: str = None, grid_size: int = None, spline_order: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        hidden = list(hidden if hidden is not None else CONFIG.kan_hidden)
        widths = [3 * d, *hidden, d]
        self.net = _build_kan(
            backend or CONFIG.kan_backend, widths,
            grid_size or CONFIG.kan_grid_size, spline_order or CONFIG.kan_spline_order,
        )

    def forward(self, t, v, g):
        return self.net(torch.cat([t, v, g], dim=-1))  # Eq. 18 -> Eq. 20


class ConcatLinear(nn.Module):
    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        self.lin = nn.Linear(3 * d, d)

    def forward(self, t, v, g):
        return self.lin(torch.cat([t, v, g], dim=-1))


class ConcatMLP(nn.Module):
    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.net = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, t, v, g):
        return self.net(torch.cat([t, v, g], dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        self.proj = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.gate = nn.Linear(3 * d, 3)

    def forward(self, t, v, g):
        mods = [self.proj[i](x) for i, x in enumerate([t, v, g])]  # each [K, d]
        w = torch.softmax(self.gate(torch.cat([t, v, g], dim=-1)), dim=-1)  # [K, 3]
        stacked = torch.stack(mods, dim=1)                          # [K, 3, d]
        return (w.unsqueeze(-1) * stacked).sum(dim=1)               # [K, d]


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, d: int = None, heads: int = 8, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        dropout = CONFIG.dropout if dropout is None else dropout
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, t, v, g):
        tokens = torch.stack([t, v, g], dim=1)        # [K, 3, d]
        out, _ = self.attn(tokens, tokens, tokens)    # [K, 3, d]
        return self.norm(out.mean(dim=1))             # [K, d]


class BilinearFusion(nn.Module):
    def __init__(self, d: int = None, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        self.bil = nn.Bilinear(d, d, d)
        self.lin_g = nn.Linear(d, d)

    def forward(self, t, v, g):
        return self.bil(t, v) + self.lin_g(g)


class TensorFusion(nn.Module):
    """Low-rank tensor fusion (LMF; Liu et al. 2018) — tractable outer-product fusion."""

    def __init__(self, d: int = None, rank: int = 4, dropout: float = None):
        super().__init__()
        d = d or CONFIG.hidden_dim
        self.rank = rank
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(rank, d + 1, d) * 0.1) for _ in range(3)]
        )

    def forward(self, t, v, g):
        zs = []
        for x, W in zip([t, v, g], self.factors):
            ones = x.new_ones((x.size(0), 1))
            xb = torch.cat([x, ones], dim=-1)               # [K, d+1]
            zs.append(torch.einsum("kd,rdo->kro", xb, W))   # [K, rank, d]
        fused = zs[0] * zs[1] * zs[2]                        # [K, rank, d]
        return fused.sum(dim=1)                              # [K, d]


FUSION_REGISTRY = {
    "kan": KANFusion,
    "concat_linear": ConcatLinear,
    "concat_mlp": ConcatMLP,        # also the "w/o KAN, MLP fusion" ablation (Table 6)
    "gated": GatedFusion,
    "cross_modal_attention": CrossModalAttentionFusion,
    "bilinear": BilinearFusion,
    "tensor": TensorFusion,
}


def build_fusion(name: str = None, d: int = None, dropout: float = None) -> nn.Module:
    name = name or CONFIG.fusion
    if name not in FUSION_REGISTRY:
        raise ValueError(f"unknown fusion '{name}'. options: {list(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name](d=d, dropout=dropout)
