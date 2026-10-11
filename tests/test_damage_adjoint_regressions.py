"""Regression tests for the reverse rules of the projected damage solve.

The damage update solves a linear system for the unconstrained damage d_unc
and then projects it, d = clamp(max(d_unc, d_prev), 0, 1). Three earlier
defects broke the reverse pass of that composition:

1. the history (H) pullback evaluated the matrix derivative at the projected
   damage instead of d_unc;
2. the history-only route returned no gradient for d_prev, so the
   irreversibility branch (d = d_prev) cut the chain to earlier steps;
3. with fixed material values the solve was dispatched to a forward-only path,
   detaching the damage from H and d_prev.

Tests 1 and 2 drive the production ``_AdjointDamageSolve`` on a two-triangle
operator assembled independently from linear basis functions and compare
against a dense solve-and-project reference differentiated by autograd and
against central differences. Test 3 uses the full solver on a small mesh.
"""

from types import SimpleNamespace

import pytest
import torch

from phast.solvers.damage_solver import _AdjointDamageSolve


class TwoTriangleDamage:
    """Minimal stand-in exposing the attributes the reverse rule reads."""

    def __init__(self):
        dtype = torch.float64
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=dtype)
        elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
        areas = torch.tensor([.5, .5], dtype=dtype)
        self.fem = SimpleNamespace(mesh=SimpleNamespace(elements=elements, areas=areas))
        self.masses = []
        self.K = torch.zeros(4, 4, dtype=dtype)
        for nodes, area in zip(elements, areas):
            interpolation = torch.cat([torch.ones(3, 1, dtype=dtype), xy[nodes]], dim=1)
            gradients = torch.linalg.inv(interpolation)[1:, :].T
            local_mass = area / 12 * (torch.eye(3, dtype=dtype) + torch.ones(3, 3, dtype=dtype))
            scatter = torch.eye(4, dtype=dtype)[:, nodes]
            self.masses.append(scatter @ local_mass @ scatter.T)
            self.K = self.K + scatter @ (area * gradients @ gradients.T) @ scatter.T
        self._cg_device = torch.device("cpu")
        self._cg_dtype = dtype
        self.max_iter = 50
        # Node 0 carries a phase-field Dirichlet value, as on a pre-cracked edge.
        self.mask = torch.tensor([True, False, False, False])
        self.values = torch.tensor([.8, 0., 0., 0.], dtype=dtype)

    def system(self, H):
        A = .7 * self.K + sum((2 * h + .6) * M for h, M in zip(H, self.masses))
        b = sum(2 * h * M.sum(dim=1) for h, M in zip(H, self.masses))
        return A, b

    def unconstrained(self, H):
        A, b = self.system(H)
        free = ~self.mask
        xf = torch.linalg.solve(A[free][:, free],
                                b[free] - A[free][:, self.mask] @ self.values[self.mask])
        return self.values.clone().masked_scatter(free, xf)

    def projected(self, H, d_prev):
        x = self.unconstrained(H)
        return torch.where(self.mask, self.values, torch.maximum(x, d_prev).clamp(0, 1))

    # Interface used by _AdjointDamageSolve.
    def _solve_dispatch(self, H, d_prev, *ignored):
        return self.projected(H, d_prev)

    def _prepare_cg(self, H, d_prev):
        A, b = self.system(H)
        return (d_prev.clone(), d_prev.clone(), H, b, 1 / A.diag(), d_prev.device, d_prev.dtype, False)

    def _prepare_pf_dirichlet(self, mask, values, *ignored):
        return mask, values

    def _Ax(self, x, H):
        return self.system(H)[0] @ x


def _irreversibility_active_state(model):
    """H, d_prev with node 1 held at d_prev and nodes 2, 3 strictly free."""
    H = torch.tensor([.3, .8], dtype=torch.float64, requires_grad=True)
    x = model.unconstrained(H).detach()
    d_prev = (.5 * x).clone()
    d_prev[1] = .5 * (1 + x[1])
    assert d_prev[1] > x[1] + 1e-3 and bool(torch.all(x[2:] > d_prev[2:] + 1e-3))
    return H, d_prev.requires_grad_(True)


@pytest.mark.parametrize("weights", [[.5, -.7, 1.2, -.4], [-.3, .8, -.6, 1.1], [1., 1., 1., 1.]])
def test_history_pullback_uses_unconstrained_damage(weights):
    model = TwoTriangleDamage()
    H, d_prev = _irreversibility_active_state(model)
    w = torch.tensor(weights, dtype=torch.float64)
    reference = torch.autograd.grad(w @ model.projected(H, d_prev), H)[0]
    out = _AdjointDamageSolve.apply(model, H, d_prev, model.mask, model.values)
    torch.testing.assert_close(out, model.projected(H, d_prev), rtol=0, atol=1e-12)
    grad_H = torch.autograd.grad(w @ out, H)[0]
    torch.testing.assert_close(grad_H, reference, rtol=1e-9, atol=1e-11)
    h = 1e-5  # small enough to keep every node on its projection branch
    for i in range(2):
        e = torch.zeros_like(H)
        e[i] = 1
        fd = w @ (model.projected(H + h * e, d_prev) - model.projected(H - h * e, d_prev)) / (2 * h)
        torch.testing.assert_close(grad_H[i], fd, rtol=1e-8, atol=1e-10)


def test_previous_damage_receives_gradient_where_irreversibility_is_active():
    model = TwoTriangleDamage()
    H, d_prev = _irreversibility_active_state(model)
    w = torch.tensor([.5, -.7, 1.2, -.4], dtype=torch.float64)
    reference = torch.autograd.grad(w @ model.projected(H, d_prev), d_prev)[0]
    out = _AdjointDamageSolve.apply(model, H, d_prev, model.mask, model.values)
    (grad_prev,) = torch.autograd.grad(w @ out, d_prev, allow_unused=True)
    assert grad_prev is not None, "irreversibility branch must pass the gradient to d_prev"
    assert abs(float(reference[1])) > .1
    torch.testing.assert_close(grad_prev, reference, rtol=1e-12, atol=1e-12)


def test_history_gradient_is_not_detached_with_fixed_material(tmp_path):
    pytest.importorskip("gmsh")
    from phast.device import DeviceContext
    from phast.fem_operators import FEMOperators
    from phast.material import Material
    from phast.mesh import FEMMesh
    from phast.mesh_generator import rectangular_sent
    from phast.damage_solver import PhaseFieldDamageSolver

    path = tmp_path / "mesh.msh"
    rectangular_sent(str(path), W=1.0, H=1.0, a=0.5, l0=0.1, h_crack=0.2, h_coarse=0.4, verbose=False)
    mesh = FEMMesh(str(path), device="cpu", dtype=torch.float64)
    mesh.identify_boundaries()
    material = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.1, rho=7.8e-9,
                        energy_split="spectral", pf_model="AT2", eta_residual=1e-7)
    ctx = DeviceContext(device="cpu", profile=False, compile_solvers=False)
    solver = PhaseFieldDamageSolver(FEMOperators(mesh, material, ctx=ctx), tol=1e-12, max_iter=2000, ctx=ctx)

    torch.manual_seed(0)
    n_elem = mesh.elements.shape[0]
    H = (5.0 + 3.0 * torch.rand(n_elem, dtype=torch.float64)).requires_grad_(True)
    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    Gc = torch.tensor(2.7, dtype=torch.float64)  # fixed material value, no gradient requested

    loss = lambda d: (d ** 2).sum()  # noqa: E731
    out = solver.solve(H, d_prev, Gc=Gc)
    assert out.requires_grad, "damage must stay on the graph when H requires a gradient"
    grad_H = torch.autograd.grad(loss(out), H)[0]
    assert float(grad_H.abs().max()) > 0.0

    direction = torch.randn(n_elem, dtype=torch.float64)
    h = 1e-6
    with torch.no_grad():
        plus = loss(solver.solve(H.detach() + h * direction, d_prev, Gc=Gc))
        minus = loss(solver.solve(H.detach() - h * direction, d_prev, Gc=Gc))
    fd = (plus - minus) / (2 * h)
    torch.testing.assert_close(grad_H @ direction, fd, rtol=1e-5, atol=1e-10)
