"""Quick, deterministic CPU verification for a PhAST source checkout.

The measured kernel excludes Python and PyTorch import time. Cold-start wall
time varies by operating system and hardware, so this script reports timing
rather than promising a universal two-second limit.
"""
from __future__ import annotations

from io import BytesIO
import importlib.util
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import torch

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mesh import FEMMesh


def optional_status(module: str) -> str:
    """Return an availability label without importing an optional backend."""
    return "discoverable, not tested" if importlib.util.find_spec(module) else "not installed (optional)"


def main() -> int:
    started = time.perf_counter()
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu", dtype=torch.float64)
    material = Material(
        E=1000.0, nu=0.30, rho=1.0, Gc=1.0, l0=0.20,
        energy_split="amor", pf_model="AT2", plane_stress=False,
    )
    operators = FEMOperators(mesh, material)
    displacement = torch.tensor(
        [[0.0, 0.0], [1.0e-3, 0.0], [0.0, 0.0]], dtype=torch.float64,
    )
    history = operators.compute_psi_plus(displacement)
    damage_solver = PhaseFieldDamageSolver(
        operators, tol=1.0e-10, max_iter=100,
        bounds_method="projected_cg", use_multigrid=False,
    )
    damage = damage_solver.solve(
        history, d_prev=torch.zeros(mesh.n_nodes, dtype=torch.float64),
    )
    bounded = torch.isfinite(damage).all() and ((damage >= 0) & (damage <= 1)).all()
    if not bool(bounded):
        raise RuntimeError("single-element damage solution is not finite and bounded")

    sparse_solution = spsolve(
        csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]])),
        np.array([1.0, 0.0]),
    )
    if not np.allclose(sparse_solution, [2.0 / 3.0, 1.0 / 3.0]):
        raise RuntimeError("SciPy sparse linear solve returned an unexpected result")

    figure, axis = plt.subplots(figsize=(2.4, 1.8))
    axis.tripcolor(
        nodes[:, 0].numpy(), nodes[:, 1].numpy(),
        elements.numpy(), damage.numpy(), shading="gouraud",
    )
    axis.set_title("single T3 damage")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=80)
    plt.close(figure)
    if buffer.tell() == 0:
        raise RuntimeError("Matplotlib did not produce an in-memory PNG")

    elapsed = time.perf_counter() - started
    print("PhAST sanitizer: PASS")
    print(f"platform: {platform.system()} {platform.machine()}")
    print(f"torch: {torch.__version__}; selected verification route: CPU float64")
    print(f"single T3 AT2 damage: max(d)={damage.max().item():.6e}")
    print(f"core sparse backend: SciPy; kernel-and-plot time: {elapsed:.3f} s")
    print(f"CUDA: {'available' if torch.cuda.is_available() else 'not available'}")
    print(f"MPS: {'available' if torch.backends.mps.is_available() else 'not available'}")
    for label, module in (("PETSc", "petsc4py"), ("PyAMG", "pyamg"), ("PyVista", "pyvista")):
        print(f"{label}: {optional_status(module)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
