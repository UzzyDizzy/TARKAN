"""Visualize learned KAN edge functions psi_ij (Eq. 19).

Probes the first KAN layer: sweeps a scalar along a single input dimension (others 0)
and plots the layer's response for several output dims — an empirical view of the
learnable univariate edge functions that distinguish KAN from a fixed-activation MLP.
"""
from dataclasses import replace

import torch

from _viz import plt, save
from config import CONFIG
from kan_fusion import KANFusion


@torch.no_grad()
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dims", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dims", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

    d = CONFIG.hidden_dim
    fusion = KANFusion(d=d)
    net = fusion.net
    layer0 = net.layers[0] if hasattr(net, "layers") else net  # efficient_kan/fastkan/RBFKAN
    in_dim = 3 * d
    xs = torch.linspace(-2, 2, 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i in args.in_dims:
        probe = torch.zeros(100, in_dim)
        probe[:, i] = xs
        try:
            y = layer0(probe)
        except Exception:
            y = net(probe)
        for o in args.out_dims:
            if o < y.shape[1]:
                ax.plot(xs.numpy(), y[:, o].numpy(), alpha=0.6, label=f"in{i}->out{o}")
    ax.set_xlabel("input activation")
    ax.set_ylabel("edge response")
    ax.set_title("Learned KAN edge functions (Eq. 19)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "kan_edges.png")


if __name__ == "__main__":
    main()
