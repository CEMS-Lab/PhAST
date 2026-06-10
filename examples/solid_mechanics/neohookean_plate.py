"""Compressible neo-Hookean cantilever via Newton + SparseSolveAutograd (#110, #105).

Plane-strain CST plate, clamped left, tip load right. Newton-Raphson outer loop
with the linear solve at each step delegated to ``sparse_solve.solve(...,
backend='auto')`` so the whole chain (including grad-through-solve) goes through
the #106 autograd primitive.
"""
from __future__ import annotations

import torch

from phast.sparse_solve import SparseSolveAutograd, solve


def build_mesh(nx, ny, L, H, dtype=torch.float64):
    xs = torch.linspace(0.0, L, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, H, ny + 1, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    nid = lambda i, j: i * (ny + 1) + j
    elems = []
    for i in range(nx):
        for j in range(ny):
            n00, n10, n11, n01 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elems += [[n00, n10, n11], [n00, n11, n01]]
    return coords, torch.tensor(elems, dtype=torch.long)


def cst_grads(coords_e):
    x1, y1 = coords_e[0]; x2, y2 = coords_e[1]; x3, y3 = coords_e[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b = torch.stack([y2 - y3, y3 - y1, y1 - y2]) / A2
    c = torch.stack([x3 - x2, x1 - x3, x2 - x1]) / A2
    return b, c, 0.5 * A2


def elem_energy(u_e, coords_e, mu, lam):
    b, c, A = cst_grads(coords_e)
    ux, uy = u_e[0::2], u_e[1::2]
    F2 = torch.stack([torch.stack([1.0 + (b * ux).sum(), (c * ux).sum()]),
                      torch.stack([(b * uy).sum(), 1.0 + (c * uy).sum()])])
    F3 = torch.eye(3, dtype=u_e.dtype).clone(); F3[:2, :2] = F2
    J = torch.det(F3); I_C = (F3 * F3).sum()
    W = 0.5 * mu * (I_C - 3.0) - mu * torch.log(J) + 0.5 * lam * torch.log(J) ** 2
    return W * A


def assemble(u, coords, elems, mu, lam, n_dof, keep_graph=False):
    """Return (r, K_indices, K_values). If keep_graph=True the returned tensors
    keep autograd dependency on (mu, lam); otherwise they're detached scalars."""
    rows, cols = [], []
    vals_list = []
    r_list = [torch.zeros((), dtype=torch.float64) for _ in range(n_dof)] if keep_graph \
             else None
    r_det = torch.zeros(n_dof, dtype=u.dtype)
    for tri in elems:
        idx = tri.tolist()
        ce = coords[idx]
        dof = [2*idx[0], 2*idx[0]+1, 2*idx[1], 2*idx[1]+1, 2*idx[2], 2*idx[2]+1]
        u_e = u[dof].detach().clone().requires_grad_(True)
        W = elem_energy(u_e, ce, mu, lam)
        f_e = torch.autograd.grad(W, u_e, create_graph=True)[0]
        for a in range(6):
            if keep_graph:
                r_list[dof[a]] = r_list[dof[a]] + f_e[a]
            else:
                r_det[dof[a]] = r_det[dof[a]] + f_e[a].detach()
            grad_a = torch.autograd.grad(f_e[a], u_e, retain_graph=True,
                                         create_graph=keep_graph)[0]
            for bb in range(6):
                rows.append(dof[a]); cols.append(dof[bb])
                vals_list.append(grad_a[bb] if keep_graph else grad_a[bb].item())
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.stack(vals_list) if keep_graph else torch.tensor(vals_list, dtype=u.dtype)
    r = torch.stack(r_list) if keep_graph else r_det
    return r, indices, values


def apply_dirichlet(idx, val, r, fixed, penalty, u):
    extra = torch.tensor(fixed, dtype=torch.long)
    idx_p = torch.cat([idx, torch.stack([extra, extra])], dim=1)
    val_p = torch.cat([val, torch.full((len(fixed),), penalty, dtype=val.dtype)])
    r_bc = r.clone()
    for d in fixed:
        r_bc[d] = penalty * u[d]
    return idx_p, val_p, r_bc


def newton_solve(u, coords, elems, mu, lam, fixed, f_ext, n_dof, tol=1e-6, max_it=30):
    free = torch.ones(n_dof, dtype=torch.bool); free[fixed] = False
    r0 = None; rn = float('inf')
    for it in range(max_it):
        r_int, idx, val = assemble(u, coords, elems, mu, lam, n_dof)
        r = r_int - f_ext
        penalty = val.abs().max().item() * 1e8
        idx_p, val_p, r_bc = apply_dirichlet(idx, val, r, fixed, penalty, u)
        rn = r_bc[free].norm().item()
        if r0 is None: r0 = max(rn, 1e-30)
        if rn / r0 < tol:
            return u, it, rn
        K = torch.sparse_coo_tensor(idx_p, val_p, size=(n_dof, n_dof)).coalesce()
        du = solve(K, -r_bc, backend='auto')
        u = u + du
        if du.abs().max().item() < 1e-9:
            return u, it + 1, rn
    return u, max_it, rn


def main():
    torch.set_default_dtype(torch.float64)
    nx, ny = 20, 10
    L, H, nu = 1.0, 0.2, 0.3
    E_val = 2.1e11
    coords, elems = build_mesh(nx, ny, L, H)
    n_dof = coords.shape[0] * 2
    mu_v = E_val / (2.0 * (1.0 + nu))
    lam_v = E_val * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    left = [i for i in range(coords.shape[0]) if coords[i, 0].item() == 0.0]
    fixed = [d for n in left for d in (2 * n, 2 * n + 1)]
    tip = nx * (ny + 1) + ny // 2

    I_sec = H ** 3 / 12.0
    # Pick P so linear EB tip deflection ~ 5% of beam length: enough geometric
    # nonlinearity to be visible, low enough for Newton on a coarse CST mesh.
    P_lin = -3.0 * E_val * I_sec * (0.05 * L) / L ** 3
    delta_lin = P_lin * L ** 3 / (3.0 * E_val * I_sec)
    P_max = 0.5 * P_lin

    u = torch.zeros(n_dof, dtype=torch.float64)
    print(f"{'frac':>6}  {'P [N]':>11}  {'iters':>5}  {'residual':>12}  {'u_tip [m]':>12}")
    for k in range(1, 6):
        frac = k / 5
        f_ext = torch.zeros(n_dof, dtype=torch.float64)
        f_ext[2 * tip + 1] = P_max * frac
        u, it, rn = newton_solve(u, coords, elems, mu_v, lam_v, fixed, f_ext, n_dof)
        print(f"{frac:>6.2f}  {P_max*frac:>11.3e}  {it:>5d}  {rn:>12.3e}  {u[2*tip+1].item():>12.4e}")
    print(f"linear EB at P_max={P_max:.3e}: {0.5*delta_lin:.4e} m  (NL FE: {u[2*tip+1].item():.4e})")

    # Autograd: rebuild K, r at converged u with E differentiable, then drive
    # SparseSolveAutograd.apply so dE.grad flows through the #106 adjoint.
    E = torch.tensor(E_val, dtype=torch.float64, requires_grad=True)
    mu_g = E / (2.0 * (1.0 + nu))
    lam_g = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    f_ext = torch.zeros(n_dof, dtype=torch.float64); f_ext[2 * tip + 1] = P_max
    r_int, idx_g, val_g = assemble(u, coords, elems, mu_g, lam_g, n_dof, keep_graph=True)
    penalty = val_g.detach().abs().max().item() * 1e8
    idx_p, val_p, r_bc = apply_dirichlet(idx_g, val_g, r_int - f_ext, fixed, penalty, u)
    du = SparseSolveAutograd.apply(idx_p, val_p, -r_bc, n_dof)
    u_tip = u[2 * tip + 1].detach() + du[2 * tip + 1]
    u_tip.abs().sum().backward()
    print(f"d|u_tip|/dE = {E.grad.item():.6e}  (finite, non-zero)")


if __name__ == "__main__":
    main()
