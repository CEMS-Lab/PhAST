"""Mixed-precision CG demo on a 1D Laplacian (issue #118).

Builds K = D^T D (forward-difference) of size n=5000, cond ~ O(n^2) ~ 1e7,
then solves K x = b in three precision modes via cg_mixed_precision and
reports wall-time, residual, and error vs the float64 reference.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from phast.mixed_precision_cg import cg_mixed_precision

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX"],
    "mathtext.fontset": "stix",
    "font.size": 9,
})


def build_laplacian(n: int, device: str = "cpu"):
    """Sparse 1D Laplacian K = D^T D with Dirichlet ends; SPD, cond ~ n^2."""
    idx_main = torch.arange(n)
    idx_off = torch.arange(n - 1)
    rows = torch.cat([idx_main, idx_off, idx_off + 1])
    cols = torch.cat([idx_main, idx_off + 1, idx_off])
    vals = torch.cat([2.0 * torch.ones(n), -torch.ones(n - 1), -torch.ones(n - 1)])
    indices = torch.stack([rows, cols])
    K64 = torch.sparse_coo_tensor(indices, vals.double(), (n, n)).coalesce().to(device)
    K32 = torch.sparse_coo_tensor(indices, vals.float(), (n, n)).coalesce().to(device)
    return K64, K32


def make_matvec(K64, K32):
    def matvec(v):
        K = K32 if v.dtype == torch.float32 else K64
        return torch.sparse.mm(K, v.unsqueeze(1)).squeeze(1)
    return matvec


def residual(K64, x, b):
    Kx = torch.sparse.mm(K64, x.double().unsqueeze(1)).squeeze(1)
    return float(torch.linalg.norm(Kx - b.double()).item())


def main():
    torch.manual_seed(0)
    n = 5000
    K64, K32 = build_laplacian(n)
    matvec = make_matvec(K64, K32)
    b = torch.randn(n, dtype=torch.float64)

    # Baseline truth (float64)
    t0 = time.perf_counter()
    x_ref, it_ref, conv_ref = cg_mixed_precision(
        matvec, b, tol=1e-10, max_iter=20000, precision="float64"
    )
    t_ref = time.perf_counter() - t0
    res_ref = residual(K64, x_ref, b)

    # float32 single-precision
    b32 = b.float()
    t0 = time.perf_counter()
    x_f32, it_f32, conv_f32 = cg_mixed_precision(
        matvec, b32, tol=1e-6, max_iter=20000, precision="float32"
    )
    t_f32 = time.perf_counter() - t0
    res_f32 = residual(K64, x_f32, b)
    err_f32 = float(torch.linalg.norm(x_f32.double() - x_ref).item())

    # mixed precision
    t0 = time.perf_counter()
    x_mix, it_mix, conv_mix = cg_mixed_precision(
        matvec, b, tol=1e-10, max_iter=20000, precision="mixed", max_refine=5
    )
    t_mix = time.perf_counter() - t0
    res_mix = residual(K64, x_mix, b)
    err_mix = float(torch.linalg.norm(x_mix - x_ref).item())
    err_ref = 0.0

    rows = [("float64", t_ref, it_ref, res_ref, err_ref, conv_ref),
            ("float32", t_f32, it_f32, res_f32, err_f32, conv_f32),
            ("mixed",   t_mix, it_mix, res_mix, err_mix, conv_mix)]
    print(f"Mixed-precision CG demo  (n={n}, cond~{4*(n+1)**2/np.pi**2:.2e})")
    print("-" * 76)
    print(f"{'precision':<10}{'time [s]':>10}{'iters':>8}{'||Kx-b||':>14}{'||x-x*||':>14}{'converged':>12}")
    print("-" * 76)
    for name, t, it, r, e, c in rows:
        print(f"{name:<10}{t:>10.4f}{it:>8d}{r:>14.3e}{e:>14.3e}{str(c):>12}")
    print("-" * 76)

    # Plot
    out_dir = Path(__file__).parent
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=150)
    names = [r[0] for r in rows]
    times = [r[1] for r in rows]
    colors = ["#2c7fb8", "#7fcdbb", "#edf8b1"]
    bars = ax.bar(names, times, color=colors, edgecolor="black", linewidth=0.6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"Mixed-precision CG, $n={n}$ (CPU)")
    ax.set_ylim(0, max(times) * 1.20)
    fig.tight_layout()
    png_path = out_dir / "mixed_precision_cg_demo.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sz = png_path.stat().st_size
    print(f"plot saved: {png_path}  ({sz/1024:.1f} kB)")


if __name__ == "__main__":
    main()
