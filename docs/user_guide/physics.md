# Physics, units & references

Governing equations, the unit system, and the reference list backing the
solver's formulation choices.

K_sparse = mesh.assemble_sparse_matrix(K_elem_data)

# Phase 2: Build preconditioner from K
precond = build_amg_or_ilu(K_sparse)

# Phase 3: CG with matrix-free matvec + sparse preconditioner
for cg_iter in range(max_iter):
    Ap = fem.internal_force(p, d)     # matrix-free (fast, no K stored)
    z = precond(r)                     # sparse preconditioner (from K)
    ...
```

This preserves all four advantages (memory for matvec, autograd, GPU coalescing,
simplicity) while closing the preconditioner gap. The sparse K is only used for
preconditioner setup — it can live on CPU if VRAM is tight.

### Summary: Matrix-Free vs Sparse

| Aspect | Matrix-Free (ours) | Sparse K Assembly |
|--------|-------------------|-------------------|
| **Memory** | O(E) — element data only | O(nnz) — matrix + indices |
| **Autograd** | Full backward support | Broken or fragile |
| **GPU coalescing** | Good (scatter patterns) | Poor (irregular SpMV) |
| **Portability** | CUDA + MPS + CPU | Needs cuSPARSE or scipy |
| **Preconditioner** | Jacobi + 2-level GMG (moderate) | AMG, ILU, Cholesky (strong) |
| **Direct solve** | Not possible | Possible (MUMPS, SuperLU) |
| **Implementation** | ~500 lines PyTorch | ~1500 lines + PETSc/scipy |
| **NN coupling** | Trivial (same graph) | Requires C++/Python bridge |

**Bottom line:** For a PyTorch-native solver designed for GPU execution and neural
operator coupling, matrix-free is the right default. The 2-level geometric
multigrid preconditioner (v0.9.0) largely closes the preconditioner gap for
moderate-sized meshes. For very large problems (>100K nodes), selective sparse
assembly for AMG (#67) remains on the roadmap.

---

## Sparse linear solver (autograd-enabled)

`phast.sparse_solve.SparseSolveAutograd` is a
`torch.autograd.Function` wrapping a
sparse-direct solve `K x = b` that also supplies the implicit-derivative
adjoint, so PyTorch can backpropagate through the linear system without ever
materialising a dense inverse. The factorisation is cached across forward and
backward, and the same factor reuses for the adjoint solve `K^T λ = ∂L/∂x`.
Multi-backend dispatch is exposed through `solve(K, b, backend=...)` with
`'auto' | 'scipy' | 'mumps' | 'cudss' | 'cg'`.

```python
import torch
from phast.sparse_solve import solve

# K: 5x5 SPD CSR (torch.sparse_csr_tensor); b: dense RHS
indptr  = torch.tensor([0, 2, 4, 6, 8, 9])
indices = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3, 4])
values  = torch.tensor([2., -1., -1., 2., 2., -1., -1., 2., 1.],
                       dtype=torch.float64, requires_grad=True)
K = torch.sparse_csr_tensor(indptr, indices, values, size=(5, 5))
b = torch.ones(5, dtype=torch.float64)
x = solve(K, b, backend="auto")
x.sum().backward()  # gradients flow back to `values` and `b`
```

### Backend selection (`backend='auto'`)

| Condition | Backend chosen |
|---|---|
| CUDA tensor + functional cuDSS/nvmath smoke test | `cudss` |
| Functional PETSc/MUMPS smoke test | `mumps` |
| `scipy.sparse.linalg` available (default CPU) | `scipy` (SuperLU) |
| No sparse-direct backend available | raise a clear install error |

Phase 1 (#106) ships SciPy SuperLU as the always-available baseline; PETSc/MUMPS
(#107) and cuDSS (#108) are optional runtime-checked backends. The
quasi-static solver's own `backend='auto'` uses sparse direct below the
configured sparse-DOF threshold, including spectral/amor/star-convex splits via
the frozen-state secant assembly, and falls back to matrix-free CG above the
threshold or when direct backends are absent.

### Direct solver names

When this repository says `scipy`, it means SciPy's sparse direct solver path,
which uses SuperLU through `scipy.sparse.linalg`. This is the always-available
CPU sparse-direct baseline when SciPy is installed.

When this repository says `mumps`, it means the PETSc/MUMPS path: `petsc4py`
creates a PETSc `KSP` with `pc_type=lu` and `pc_factor_mat_solver_type=mumps`.
This is the PhaseFieldX-like CPU sparse-direct path used for quasi-static
validation jobs when the runtime smoke test passes.

PARDISO and SPOOLES are other sparse direct solver libraries exposed by COMSOL.
PARDISO is a high-performance sparse direct solver from the Intel MKL/PARDISO
family, commonly used for robust shared-memory sparse factorisation. SPOOLES is
an older sparse direct solver based on multifrontal LU/LDL factorisation. We do
not currently call PARDISO or SPOOLES from `phast`; they are listed in
COMSOL comparisons only to explain the commercial direct-solver menu.

For SENS/TPB on an environment with working PETSc/MUMPS, `backend='auto'`
resolves to `mumps`. If PETSc/MUMPS fails its smoke test on another machine,
the same configuration falls back to `scipy` SuperLU for sparse direct
mechanics; if the problem is above the configured sparse-DOF threshold or no
direct backend is available, it falls back to `cg`.

Timing comparisons must be regenerated with the current YAML deck, backend
versions, hardware, and output settings before citation. See
`docs/performance_reproducibility/index.md` for the public reporting checklist.
See `examples/solid_mechanics/` for the linear elastic plate demo and the
neo-Hookean extension; `tests/test_sparse_solve_autograd.py` exercises the
gradient via finite differences in the development test suite.

#### Worked examples

**Tutorial primitives** (small, self-contained — exercise `SparseSolveAutograd`
end-to-end on a plate geometry):

- `python -m phast run examples/solid_mechanics/linear_plate/config.yaml` — plane-strain CST cantilever;
  clamp left, point load right; writes response CSV, plot, thumbnail, and manifest while verifying autograd through E.
- `python -m phast run examples/solid_mechanics/neohookean_plate/config.yaml` — Newton iteration over
  compressible neo-Hookean; same geometry as the linear demo with 5 load
  increments, with load-response and convergence outputs.

**Validation benchmarks** (paper-quality forward + L-BFGS inverse): tracked as
follow-ups under issue #110. The current tree includes the solid-mechanics
private development benchmark archive; keep the
per-benchmark README and compare script as the source of truth for which
subcases are fully validated.

## Physics

AT2 phase-field fracture model (Bourdin, Francfort, Marigo 2000):

- **Elasticity**: 2D plane-strain, linear triangles (T3)
- **Damage PDE**: `-Gc*l0*nabla^2(d) + (Gc/l0 + 2H)*d = 2H`
- **Degradation**: `g(d) = (1-d)^2 + eta` (eta ~ 0 for full degradation)
- **History variable**: `H = max over time of psi+(epsilon)` (irreversibility)
- **Coupling**: Staggered (alternate minimization) — solve u then d each step
- **Convergence**: Iterate until `||d_new - d_old|| < tol` (typically 1e-6)

## Unit System

All examples use mm-N-MPa-s (consistent with Miehe et al.):

| Quantity | Unit |
|----------|------|
| Length | mm |
| Force | N |
| Stress / E | MPa = N/mm^2 |
| Fracture toughness Gc | N/mm |
| Regularization length l0 | mm |

## References

- Miehe, C., Welschinger, F., & Hofacker, M. (2010). Thermodynamically
  consistent phase-field models of fracture. *Int. J. Numer. Meth. Engng.*
- Bourdin, B., Francfort, G.A., & Marigo, J.-J. (2000). Numerical experiments
  in revisited brittle fracture. *J. Mech. Phys. Solids.*
- Amor, H., Marigo, J.-J., & Maurini, C. (2009). Regularized formulation of
  the variational brittle fracture with unilateral contact. *IJNME.*
- Kumar, A., Francfort, G.A., & Lopez-Pamies, O. (2020). Revisiting nucleation
  in the phase-field approach to brittle fracture. *JMPS*, 142, 104027.
- PhaseFieldX: https://phasefieldx.readthedocs.io/
