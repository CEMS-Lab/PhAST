#!/usr/bin/env python
"""Scalar fracture-toughness inversion for Paper 1.

This driver fixes the Paper 1 differentiability audit point by running the
same scalar ``G_c`` recovery problem with a genuine autograd gradient through
the phase-field forward solve. Central finite differences remain available as
a baseline and as a gradient-parity check, but they are no longer the only
gradient path for the manuscript inversion demo.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from phast.boundary_conditions import BoundaryConditions
from phast.material import create_material
from phast.mesh import FEMMesh
from phast.mesh_generator import rectangular_sent
from phast.staggered_solver import SolverConfig, StaggeredSolver


GC_TRUE = 3.0e-3
L0_TRUE = 0.5
SIGMA_LOAD = 1.0


def _history_jsonable(history: dict[str, list[float | int]]) -> dict:
    out: dict[str, list[float | int]] = {}
    for key, values in history.items():
        out[key] = [int(v) if isinstance(v, int) else float(v) for v in values]
    return out


def build_mesh(device: str, h_crack: float, h_coarse: float) -> FEMMesh:
    """Build the SENT mesh used by the existing Paper 1 scalar inversion."""
    mesh_path = os.path.join(tempfile.gettempdir(), "paper1_gc_inverse.msh")
    rectangular_sent(
        mesh_path,
        W=100.0,
        H=40.0,
        a=50.0,
        l0=L0_TRUE,
        h_crack=h_crack,
        h_coarse=h_coarse,
        branching=True,
    )
    mesh = FEMMesh(mesh_path, device=device, dtype=torch.float64)
    mesh.identify_boundaries()
    return mesh


def run_forward(
    mesh: FEMMesh,
    gc: float | torch.Tensor,
    n_steps: int,
    device: str,
    *,
    differentiable: bool,
    h_update_method: str,
    h_update_scale: float | None,
    softmax_h_beta: float | None,
) -> torch.Tensor:
    """Run one explicit dynamic solve and return the final damage field."""
    bcs = BoundaryConditions(mesh.n_nodes, device=device, dtype=torch.float64)
    bcs.add_neumann(mesh.node_sets["top"], [0.0, 1.0])
    bcs.add_neumann(mesh.node_sets["bottom"], [0.0, -1.0])
    bcs.fix(mesh.node_sets["left"], 0)

    # The material object stores the fixed mechanical constants.  When ``gc``
    # is a gradient-tracked tensor, StaggeredSolver forwards it to the damage
    # solver via ``diff_Gc`` so the scalar implicit-adjoint path is used.
    gc_nominal = float(gc.detach().item()) if isinstance(gc, torch.Tensor) else float(gc)
    mat = create_material("glass_borden", l0=L0_TRUE, Gc=gc_nominal)
    cfg = SolverConfig(
        solver_type="explicit",
        differentiable=differentiable,
        preconditioner="jacobi",
        H_update_method=h_update_method,
        H_update_scale=h_update_scale,
        softmax_H_beta=softmax_h_beta,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)
    solver.f_ext = SIGMA_LOAD * bcs.get_neumann_forces(mesh)
    if isinstance(gc, torch.Tensor):
        solver.diff_Gc = gc

    for _ in range(n_steps):
        solver.step_full()
    return solver.d


def fd_loss_and_grad(
    mesh: FEMMesh,
    d_target: torch.Tensor,
    gc_value: float,
    n_steps: int,
    device: str,
    *,
    rel_delta: float,
    h_update_method: str,
    h_update_scale: float | None,
    softmax_h_beta: float | None,
) -> tuple[float, float, int]:
    """Return ``(loss, dL/dlogGc, forward_count)`` from central FD."""
    delta = rel_delta * gc_value
    losses: list[float] = []
    with torch.no_grad():
        for gc in (gc_value, gc_value + delta, gc_value - delta):
            d_pred = run_forward(
                mesh,
                gc,
                n_steps,
                device,
                differentiable=False,
                h_update_method=h_update_method,
                h_update_scale=h_update_scale,
                softmax_h_beta=softmax_h_beta,
            )
            losses.append(float(((d_pred - d_target) ** 2).mean().item()))
    loss_c, loss_p, loss_m = losses
    dloss_dgc = (loss_p - loss_m) / (2.0 * delta)
    return loss_c, dloss_dgc * gc_value, 3


def autograd_loss(
    mesh: FEMMesh,
    d_target: torch.Tensor,
    log_gc: torch.Tensor,
    n_steps: int,
    device: str,
    *,
    h_update_method: str,
    h_update_scale: float | None,
    softmax_h_beta: float | None,
) -> torch.Tensor:
    gc_tensor = torch.exp(log_gc)
    d_pred = run_forward(
        mesh,
        gc_tensor,
        n_steps,
        device,
        differentiable=True,
        h_update_method=h_update_method,
        h_update_scale=h_update_scale,
        softmax_h_beta=softmax_h_beta,
    )
    return ((d_pred - d_target) ** 2).mean()


def gradient_parity_check(
    mesh: FEMMesh,
    d_target: torch.Tensor,
    gc_value: float,
    n_steps: int,
    device: str,
    *,
    rel_delta: float,
    h_update_method: str,
    h_update_scale: float | None,
    softmax_h_beta: float | None,
) -> dict[str, float]:
    """Compare autograd and central-FD gradients at a single point."""
    log_gc = torch.tensor(
        torch.log(torch.tensor(gc_value, dtype=torch.float64)).item(),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    loss = autograd_loss(
        mesh,
        d_target,
        log_gc,
        n_steps,
        device,
        h_update_method=h_update_method,
        h_update_scale=h_update_scale,
        softmax_h_beta=softmax_h_beta,
    )
    loss.backward()
    grad_ad = float(log_gc.grad.item())
    loss_fd, grad_fd, fd_forwards = fd_loss_and_grad(
        mesh,
        d_target,
        gc_value,
        n_steps,
        device,
        rel_delta=rel_delta,
        h_update_method=h_update_method,
        h_update_scale=h_update_scale,
        softmax_h_beta=softmax_h_beta,
    )
    rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-30)
    return {
        "Gc": float(gc_value),
        "loss_autograd": float(loss.item()),
        "loss_fd_center": float(loss_fd),
        "grad_autograd_dlogGc": grad_ad,
        "grad_fd_dlogGc": float(grad_fd),
        "relative_error": float(rel),
        "autograd_forward_evals": 1,
        "fd_forward_evals": int(fd_forwards),
    }


def optimise(
    mesh: FEMMesh,
    d_target: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    log_gc = torch.tensor(
        torch.log(torch.tensor(args.Gc_init, dtype=torch.float64)).item(),
        dtype=torch.float64,
        device=args.device,
        requires_grad=True,
    )
    if args.optimizer == "lbfgs":
        optimiser = torch.optim.LBFGS(
            [log_gc],
            lr=1.0,
            max_iter=1,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    else:
        optimiser = torch.optim.Adam([log_gc], lr=args.lr)

    history: dict[str, list[float | int]] = {
        "iter": [],
        "Gc": [],
        "loss": [],
        "rel_err": [],
        "forward_evals": [],
    }
    counter = {"n": 0}
    t0 = time.time()

    def closure_autograd() -> torch.Tensor:
        optimiser.zero_grad()
        loss = autograd_loss(
            mesh,
            d_target,
            log_gc,
            args.n_steps,
            args.device,
            h_update_method=args.H_update_method,
            h_update_scale=args.H_update_scale,
            softmax_h_beta=args.softmax_H_beta,
        )
        loss.backward()
        counter["n"] += 1
        return loss

    def closure_fd() -> torch.Tensor:
        optimiser.zero_grad()
        loss_v, grad_v, n_fwds = fd_loss_and_grad(
            mesh,
            d_target,
            float(torch.exp(log_gc).item()),
            args.n_steps,
            args.device,
            rel_delta=args.fd_rel_delta,
            h_update_method=args.H_update_method,
            h_update_scale=args.H_update_scale,
            softmax_h_beta=args.softmax_H_beta,
        )
        log_gc.grad = torch.tensor(grad_v, dtype=torch.float64, device=args.device)
        counter["n"] += n_fwds
        return torch.tensor(loss_v, dtype=torch.float64, device=args.device)

    closure = closure_autograd if args.gradient_mode == "autograd" else closure_fd
    for it in range(args.n_iters):
        if args.optimizer == "lbfgs":
            loss = optimiser.step(closure)
        else:
            loss = closure()
            optimiser.step()

        gc_val = float(torch.exp(log_gc).item())
        rel_err = abs(gc_val - GC_TRUE) / GC_TRUE
        history["iter"].append(it)
        history["Gc"].append(gc_val)
        history["loss"].append(float(loss.item()))
        history["rel_err"].append(rel_err)
        history["forward_evals"].append(counter["n"])
        print(
            f"  iter {it:3d}: Gc={gc_val:.6e}, "
            f"err={rel_err:.3%}, loss={float(loss.item()):.3e}, "
            f"forward_evals={counter['n']}",
            flush=True,
        )
        if rel_err < args.stop_rel_err:
            break

    total_time = time.time() - t0
    final_gc = float(torch.exp(log_gc).item())
    return {
        "optimizer": args.optimizer,
        "gradient_mode": args.gradient_mode,
        "H_update_method": args.H_update_method,
        "H_update_scale": args.H_update_scale,
        "softmax_H_beta": args.softmax_H_beta,
        "Gc_true": float(GC_TRUE),
        "Gc_init": float(args.Gc_init),
        "final_Gc": final_gc,
        "final_rel_err": abs(final_gc - GC_TRUE) / GC_TRUE,
        "total_forward_evals": int(counter["n"]),
        "total_time_s": float(total_time),
        "n_steps_per_sim": int(args.n_steps),
        "history": _history_jsonable(history),
    }


def write_outputs(result: dict, out_dir: Path, optimizer: str, gradient_mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if optimizer == "adam" else f"_{optimizer}"
    mode_suffix = "" if gradient_mode == "autograd" else f"_{gradient_mode}"
    json_path = out_dir / f"demo_inversion{suffix}{mode_suffix}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            from phast.paper_figure_style import apply_style

            apply_style()
        except Exception:
            pass

        hist = result["history"]
        step_iters = [int(i) + 1 for i in hist["iter"]]
        gc_iters = [0, *step_iters]
        gc_values = [result["Gc_init"] * 1e3, *[g * 1e3 for g in hist["Gc"]]]
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0))
        axes[0].plot(gc_iters, gc_values, "b-o", ms=2)
        axes[0].axhline(GC_TRUE * 1e3, color="r", ls="--", lw=1.2)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(r"$G_c$ ($10^{-3}$ N/mm)")
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(step_iters, hist["loss"], "r-o", ms=2)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("MSE loss")
        axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(step_iters, hist["rel_err"], "g-o", ms=2)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel(r"$|G_c-G_c^*|/G_c^*$")
        axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"demo_inversion{suffix}{mode_suffix}.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"  plot skipped: {exc}")

    print(f"  wrote {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--n_iters", type=int, default=80)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--Gc_init", type=float, default=6.0e-3)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument(
        "--gradient_mode",
        choices=["autograd", "fd"],
        default="autograd",
        help="Gradient source used by the optimiser.",
    )
    parser.add_argument("--fd_rel_delta", type=float, default=0.05)
    parser.add_argument("--stop_rel_err", type=float, default=0.0)
    parser.add_argument("--h_crack", type=float, default=1.5)
    parser.add_argument("--h_coarse", type=float, default=8.0)
    parser.add_argument("--output_dir", type=str, default="papers/paper/demo_data")
    parser.add_argument(
        "--skip_gradient_check",
        action="store_true",
        help="Do not run the one-point autograd-vs-FD check before optimisation.",
    )
    parser.add_argument(
        "--H_update_method",
        choices=["hard_max", "softmax", "smooth_max", "log_smooth", "custom_subgrad"],
        default="hard_max",
    )
    parser.add_argument("--H_update_scale", type=float, default=None)
    parser.add_argument("--softmax_H_beta", type=float, default=None)
    args = parser.parse_args()
    if args.optimizer == "lbfgs" and args.n_iters == 80:
        args.n_iters = 10
    return args


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("  Paper 1 scalar Gc inversion")
    print(f"  gradient_mode={args.gradient_mode}, optimizer={args.optimizer}")
    print("=" * 72)

    mesh = build_mesh(args.device, args.h_crack, args.h_coarse)
    print(f"  mesh: {mesh.n_nodes} nodes, {len(mesh.elements)} elements")
    print(f"  n_steps={args.n_steps}, n_iters={args.n_iters}")

    with torch.no_grad():
        d_target = run_forward(
            mesh,
            GC_TRUE,
            args.n_steps,
            args.device,
            differentiable=False,
            h_update_method=args.H_update_method,
            h_update_scale=args.H_update_scale,
            softmax_h_beta=args.softmax_H_beta,
        ).detach()
    print(
        f"  target: max(d)={float(d_target.max().item()):.4f}, "
        f"damaged_nodes(d>0.5)={int((d_target > 0.5).sum().item())}"
    )

    result = optimise(mesh, d_target, args)
    if not args.skip_gradient_check:
        print("  running autograd-vs-FD gradient parity check at initial guess ...")
        result["gradient_check"] = gradient_parity_check(
            mesh,
            d_target,
            args.Gc_init,
            args.n_steps,
            args.device,
            rel_delta=args.fd_rel_delta,
            h_update_method=args.H_update_method,
            h_update_scale=args.H_update_scale,
            softmax_h_beta=args.softmax_H_beta,
        )
        gc = result["gradient_check"]
        print(
            "  gradient_check: "
            f"autograd={gc['grad_autograd_dlogGc']:.6e}, "
            f"fd={gc['grad_fd_dlogGc']:.6e}, "
            f"rel={gc['relative_error']:.3e}"
        )

    write_outputs(result, Path(args.output_dir), args.optimizer, args.gradient_mode)


if __name__ == "__main__":
    main()
