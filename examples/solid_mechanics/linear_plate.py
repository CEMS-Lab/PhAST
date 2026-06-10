"""Linear elastic cantilever via SparseSolveAutograd (#110).

Minimal demo proving the #106 building block works for non-fracture problems:
plane-strain CST mesh, clamped-left + tip-load right, autograd-through-solve.
"""
from __future__ import annotations

import torch

from phast.sparse_solve import SparseSolveAutograd


def build_mesh(nx: int, ny: int, L: float, H: float, dtype=torch.float64):
    xs = torch.linspace(0.0, L, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, H, ny + 1, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    nid = lambda i, j: i * (ny + 1) + j
    elems = []
    for i in range(nx):
        for j in range(ny):
            n00, n10, n11, n01 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            elems.append([n00, n10, n11])
            elems.append([n00, n11, n01])
    return coords, torch.tensor(elems, dtype=torch.long)


def plane_strain_D(E: torch.Tensor, nu: float):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = torch.zeros(3, 3, dtype=E.dtype if E.ndim else torch.float64)
    D[0, 0] = c * (1.0 - nu); D[1, 1] = c * (1.0 - nu)
    D[0, 1] = c * nu;        D[1, 0] = c * nu
    D[2, 2] = c * (0.5 - nu)
    return D


def cst_BA(coords_e: torch.Tensor):
    x1, y1 = coords_e[0]; x2, y2 = coords_e[1]; x3, y3 = coords_e[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * A2
    b = torch.stack([y2 - y3, y3 - y1, y1 - y2]) / A2
    c = torch.stack([x3 - x2, x1 - x3, x2 - x1]) / A2
    B = torch.zeros(3, 6, dtype=coords_e.dtype)
    for k in range(3):
        B[0, 2 * k] = b[k]
        B[1, 2 * k + 1] = c[k]
        B[2, 2 * k] = c[k]; B[2, 2 * k + 1] = b[k]
    return B, A


def assemble(coords, elems, E, nu):
    D = plane_strain_D(E, nu)
    n_dof = coords.shape[0] * 2
    rows, cols, vals = [], [], []
    for tri in elems:
        idx = tri.tolist()
        ce = coords[idx]
        B, A = cst_BA(ce)
        Ke = (B.T @ D @ B) * A
        dof = [2 * idx[0], 2 * idx[0] + 1, 2 * idx[1], 2 * idx[1] + 1, 2 * idx[2], 2 * idx[2] + 1]
        for a in range(6):
            for b in range(6):
                rows.append(dof[a]); cols.append(dof[b]); vals.append(Ke[a, b])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.stack(vals)
    return indices, values, n_dof


def apply_dirichlet_penalty(indices, values, n_dof, fixed_dofs, penalty):
    extra_r, extra_c, extra_v = [], [], []
    for d in fixed_dofs:
        extra_r.append(d); extra_c.append(d); extra_v.append(torch.tensor(penalty, dtype=values.dtype))
    new_indices = torch.cat([indices, torch.tensor([extra_r, extra_c], dtype=torch.long)], dim=1)
    new_values = torch.cat([values, torch.stack(extra_v)])
    return new_indices, new_values


def main():
    torch.set_default_dtype(torch.float64)
    nx, ny = 20, 10
    L, H = 1.0, 0.2
    nu = 0.3
    P = -1.0e3
    E = torch.tensor(2.1e11, requires_grad=True)

    coords, elems = build_mesh(nx, ny, L, H)
    indices, values, n_dof = assemble(coords, elems, E, nu)

    left = [i for i in range(coords.shape[0]) if coords[i, 0] == 0.0]
    fixed = []
    for n in left:
        fixed.extend([2 * n, 2 * n + 1])
    penalty = values.abs().max().item() * 1e8
    indices_p, values_p = apply_dirichlet_penalty(indices, values, n_dof, fixed, penalty)

    tip_node = nx * (ny + 1) + ny // 2
    f = torch.zeros(n_dof, dtype=torch.float64)
    f[2 * tip_node + 1] = P

    u = SparseSolveAutograd.apply(indices_p, values_p, f, n_dof)
    u_tip = u[2 * tip_node + 1]

    I = H ** 3 / 12.0
    delta_eb = P * L ** 3 / (3.0 * E.detach().item() * I)
    err = (u_tip.detach().item() - delta_eb) / delta_eb * 100.0

    print(f"tip displacement (FE)         : {u_tip.detach().item():.6e} m")
    print(f"Euler-Bernoulli analytical    : {delta_eb:.6e} m  (error {err:+.1f}%)")

    loss = u_tip.abs()
    loss.backward()
    print(f"d|u_tip|/dE                    : {E.grad.item():.6e}  (finite, non-zero)")


if __name__ == "__main__":
    main()
