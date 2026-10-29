"""Interaction integral ``compute_sif`` and the cutoff of ``compute_j_integral`` against Williams fields.

The state is the analytic crack-tip displacement of a given (K_I, K_II) imposed on a structured T3 mesh with a slit
along the negative x axis. The slit nodes are duplicated, so the field is continuous in every cell and the displacement
jump of the crack faces is represented exactly. Everything is CPU and float64.

Tolerances. The mesh has 12 800 cells (h = 0.025 mm on a 2 mm square, l0 = 0.05 mm, domains of 2 to 4, 3 to 6 and 4 to
8 l0). The discretisation error of the one-point T3 integration is 2e-4 of K on the smallest domain and below 1e-5 on
the largest; the tests allow 1 % of K, and 1e-3 of K for the spread between domains. A mismatch of the normalisation
of the auxiliary field (the earlier defect was a factor of 3/32 E' in mode I and 1/32 E' in mode II) or of the sign of
a mode misses these bounds by orders of magnitude.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

from phast.core.fem_operators import FEMOperators
from phast.core.mesh import FEMMesh
from phast.physics import fracture_mechanics as fm
from phast.physics.material import Material

E, NU, L0 = 210000.0, 0.3, 0.05
DOMAINS = [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0)]       # [r_inner, r_outer] in units of l0
K_TOL = 1e-2                                          # of K, against the analytic value (achieved 2e-4)
SPREAD_TOL = 1e-3                                     # of K, between domains (achieved 2e-4)


def slit_mesh(n: int = 80, half: float = 1.0):
    """T3 mesh of [-half, half]^2, tip at the origin, slit on y = 0, x < 0 with duplicated nodes.

    Returns ``nodes (N, 2)``, ``elements (E, 3)`` and ``theta (N,)``, the polar angle of each node in (-pi, pi]
    (+pi on the upper face of the slit, -pi on the lower one).
    """
    xs = np.linspace(-half, half, n + 1)
    m = n // 2
    pts = [(xs[i], xs[j]) for j in range(n + 1) for i in range(n + 1)]
    index = lambda i, j: j * (n + 1) + i                                      # noqa: E731
    lower = {}
    for i in range(m):                                                        # lower copy of the slit row
        lower[i] = len(pts)
        pts.append((xs[i], 0.0))

    def node(i, j, below):
        return lower[i] if (below and j == m and i < m) else index(i, j)

    cells = []
    for j in range(n):
        for i in range(n):
            below = j < m
            a, b, c, d = node(i, j, below), node(i + 1, j, below), node(i + 1, j + 1, below), node(i, j + 1, below)
            cells += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    nodes = np.array(pts)
    theta = np.arctan2(nodes[:, 1], nodes[:, 0])
    theta[(n + 1) ** 2:] = -math.pi
    return torch.tensor(nodes), torch.tensor(cells, dtype=torch.long), torch.tensor(theta)


def kolosov(plane_stress: bool) -> float:
    return (3.0 - NU) / (1.0 + NU) if plane_stress else 3.0 - 4.0 * NU


def williams_displacement(xy_local, theta, k1, k2, plane_stress):
    """Displacement of the Williams field of (K_I, K_II) at crack-frame points ``xy_local`` (tip at the origin)."""
    mu, kappa = E / (2.0 * (1.0 + NU)), kolosov(plane_stress)
    a = torch.sqrt(xy_local.norm(dim=1) / (2.0 * math.pi)) / (2.0 * mu)
    c, s = torch.cos(0.5 * theta), torch.sin(0.5 * theta)
    ux = a * (k1 * c * (kappa - 1.0 + 2.0 * s ** 2) + k2 * s * (kappa + 1.0 + 2.0 * c ** 2))
    uy = a * (k1 * s * (kappa + 1.0 - 2.0 * c ** 2) - k2 * c * (kappa - 1.0 - 2.0 * s ** 2))
    return torch.stack([ux, uy], 1)


@pytest.fixture(scope="module")
def slit():
    return slit_mesh()


def setup(slit, plane_stress, *, angle=0.0, origin=(0.0, 0.0), k=(40.0, 0.0)):
    """Mesh rotated by ``angle`` and moved to ``origin`` (the crack frame follows), the field, damage and operators."""
    nodes, cells, theta = slit
    a = math.radians(angle)
    rot = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=torch.float64)
    tip = torch.tensor(origin, dtype=torch.float64)
    mesh = FEMMesh.from_tensors(nodes @ rot.T + tip, cells, device="cpu")
    mat = Material(E=E, nu=NU, Gc=2.7, l0=L0, energy_split="isotropic", pf_model="AT2", plane_stress=plane_stress)
    fem = FEMOperators(mesh, mat)
    u = williams_displacement(nodes, theta, k[0], k[1], plane_stress) @ rot.T      # crack frame -> global axes
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    return mesh, mat, fem, u, d, tip, a


def e_prime(plane_stress: bool) -> float:
    return E if plane_stress else E / (1.0 - NU ** 2)


@pytest.mark.parametrize("plane_stress", [False, True], ids=["plane_strain", "plane_stress"])
@pytest.mark.parametrize("mode", ["I", "II"])
def test_auxiliary_field_is_compatible_and_obeys_hookes_law(plane_stress, mode):
    """Gradient of the auxiliary displacement against autograd of its formula (1e-12 relative), and Hooke's law of the
    plane condition between that gradient and the Williams stress (1e-12 relative)."""
    mu, kappa = E / (2.0 * (1.0 + NU)), kolosov(plane_stress)
    lam = E * NU / (1.0 - NU ** 2) if plane_stress else E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    th = torch.linspace(-3.0, 3.0, 25, dtype=torch.float64)
    r = torch.full_like(th, 0.37)
    (sxx, syy, sxy), (dxx, dxy, dyx, dyy) = fm._williams_unit_fields(r, th, mu, kappa, mode)

    x = (r * torch.cos(th)).requires_grad_(True)
    y = (r * torch.sin(th)).requires_grad_(True)
    rr, tt = torch.sqrt(x ** 2 + y ** 2), torch.atan2(y, x)
    a, c, s = torch.sqrt(rr / (2.0 * math.pi)) / (2.0 * mu), torch.cos(tt / 2), torch.sin(tt / 2)
    if mode == "I":
        ux, uy = a * c * (kappa - 1.0 + 2.0 * s ** 2), a * s * (kappa + 1.0 - 2.0 * c ** 2)
    else:
        ux, uy = a * s * (kappa + 1.0 + 2.0 * c ** 2), -a * c * (kappa - 1.0 - 2.0 * s ** 2)
    gx = torch.autograd.grad(ux.sum(), (x, y), retain_graph=True)
    gy = torch.autograd.grad(uy.sum(), (x, y))
    scale = float(dxx.abs().max())
    for got, ref in ((dxx, gx[0]), (dxy, gx[1]), (dyx, gy[0]), (dyy, gy[1])):
        assert float((got - ref).abs().max()) < 1e-12 * scale

    tr = dxx + dyy
    scale = float(sxx.abs().max())
    assert float((lam * tr + 2.0 * mu * dxx - sxx).abs().max()) < 1e-12 * scale
    assert float((lam * tr + 2.0 * mu * dyy - syy).abs().max()) < 1e-12 * scale
    assert float((mu * (dxy + dyx) - sxy).abs().max()) < 1e-12 * scale


@pytest.mark.parametrize("plane_stress", [False, True], ids=["plane_strain", "plane_stress"])
@pytest.mark.parametrize("k", [(40.0, 0.0), (0.0, 40.0), (30.0, -20.0), (-25.0, 35.0)],
                         ids=["mode_I", "mode_II", "mixed_minus", "mixed_plus"])
def test_compute_sif_recovers_the_williams_k(slit, plane_stress, k):
    """K_I and K_II (signs included) within 1 % of the applied values for every domain radius, the domains within
    1e-3 of K of each other, and J = (K_I^2 + K_II^2)/E'."""
    mesh, mat, fem, u, d, tip, alpha = setup(slit, plane_stress, k=k)
    kmax = max(abs(k[0]), abs(k[1]))
    found = []
    for r_in, r_out in DOMAINS:
        k1, k2, j = fm.compute_sif(mesh, u, d, mat, fem, tip=tip, alpha=alpha,
                                   r_inner_factor=r_in, r_outer_factor=r_out)
        assert abs(k1 - k[0]) < K_TOL * kmax and abs(k2 - k[1]) < K_TOL * kmax, (r_in, r_out, k1, k2)
        assert abs(j / ((k[0] ** 2 + k[1] ** 2) / e_prime(plane_stress)) - 1.0) < K_TOL
        found.append((k1, k2))
    found = np.array(found)
    assert np.ptp(found, axis=0).max() < SPREAD_TOL * kmax


@pytest.mark.parametrize("plane_stress", [False, True], ids=["plane_strain", "plane_stress"])
def test_mode_ii_sign_convention_is_the_physical_one(slit, plane_stress):
    """K_II > 0 means sigma_xy > 0 ahead of the tip (crack frame) and the upper face sliding to +x against the lower one.

    Checked on the stress that the solver operators compute from the imposed displacement (not through the auxiliary
    formulas): ahead of the tip the mean of sigma_xy sqrt(2 pi r) is K_II within 2 % (single cells within 15 %) and sigma_yy is small; the faces slide as stated.
    """
    mesh, mat, fem, u, d, tip, alpha = setup(slit, plane_stress, k=(0.0, 40.0))
    sxx, syy, sxy = fem.compute_stress(u, d, strain=fem.compute_strain(u))
    c = mesh.nodes[mesh.elements].mean(dim=1)
    ahead = (c[:, 0] > 0.1) & (c[:, 0] < 0.3) & (c[:, 1].abs() < 0.02)
    assert int(ahead.sum()) > 10
    kii = sxy[ahead] * torch.sqrt(2.0 * math.pi * c[ahead].norm(dim=1))
    assert abs(float(kii.mean()) / 40.0 - 1.0) < 0.02                   # achieved 0.2 %; one-point cells scatter by 6 %
    assert float(kii.min()) > 0.85 * 40.0 and float(kii.max()) < 1.15 * 40.0
    assert abs(float(syy[ahead].mean() / sxy[ahead].mean())) < 0.05     # sigma_yy = 0 on theta = 0 in mode II (band symmetric about it)
    nodes, _, theta = slit
    upper, lower = (theta == math.pi) & (nodes[:, 0] < -0.1), (theta == -math.pi) & (nodes[:, 0] < -0.1)
    assert upper.sum() == lower.sum() > 10
    assert float((u[upper, 0] - u[lower, 0]).min()) > 0.0               # upper face moves to +x relative to the lower


@pytest.mark.parametrize("plane_stress", [False, True], ids=["plane_strain", "plane_stress"])
@pytest.mark.parametrize("angle", [30.0, -40.0, 90.0, 160.0])
def test_compute_sif_with_a_rotated_and_shifted_crack(slit, plane_stress, angle):
    """The crack frame follows ``alpha``: a field rotated with the mesh gives the same (K_I, K_II), mixed mode, with the
    tip away from the origin. The discretisation is the same, so the agreement with the unrotated value is roundoff."""
    k = (30.0, -20.0)
    mesh0, mat0, fem0, u0, d0, tip0, _ = setup(slit, plane_stress, k=k)
    ref = fm.compute_sif(mesh0, u0, d0, mat0, fem0, tip=tip0, alpha=0.0, r_inner_factor=4.0, r_outer_factor=8.0)
    mesh, mat, fem, u, d, tip, alpha = setup(slit, plane_stress, angle=angle, origin=(0.3, -0.2), k=k)
    k1, k2, _ = fm.compute_sif(mesh, u, d, mat, fem, tip=tip, alpha=alpha, r_inner_factor=4.0, r_outer_factor=8.0)
    assert abs(k1 - k[0]) < K_TOL * 30.0 and abs(k2 - k[1]) < K_TOL * 30.0
    assert abs(k1 - ref[0]) < 1e-8 * 30.0 and abs(k2 - ref[1]) < 1e-8 * 30.0


def test_compute_sif_does_not_scale_with_the_modulus(slit):
    """The earlier defect made K_I proportional to E'. The same stress intensity gives the same K at E and 10 E."""
    nodes, cells, theta = slit
    mesh = FEMMesh.from_tensors(nodes, cells, device="cpu")
    tip = torch.zeros(2, dtype=torch.float64)
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    for factor in (1.0, 10.0):
        mat = Material(E=E * factor, nu=NU, Gc=2.7, l0=L0, energy_split="isotropic", pf_model="AT2")
        u = williams_displacement(nodes, theta, 40.0, 0.0, False) / factor     # displacement is K/(2 mu): same K
        k1, k2, _ = fm.compute_sif(mesh, u, d, mat, FEMOperators(mesh, mat), tip=tip, alpha=0.0,
                                   r_inner_factor=4.0, r_outer_factor=8.0)
        assert k1 == pytest.approx(40.0, rel=K_TOL) and abs(k2) < K_TOL * 40.0


def test_default_cutoff_is_09_for_both_integrals():
    for fn in (fm.compute_j_integral, fm.compute_sif):
        assert inspect.signature(fn).parameters["d_process_threshold"].default == 0.9


@pytest.mark.parametrize("fn", ["j", "sif"])
def test_cutoff_leaves_out_cells_at_or_above_it_and_keeps_the_damage_halo(slit, fn):
    """A damage halo of 0.3 along the crack faces (cells that still carry 49 % of their stiffness) is inside the
    integral with the default 0.9 and outside it with 0.01, which changes the result by more than 1 %; cells at
    d = 1 are outside with the default and with a cutoff of 1."""
    mesh, mat, fem, u, d, tip, alpha = setup(slit, False, k=(30.0, -20.0))
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    halo = torch.where((x < 0.0) & (y.abs() < 0.12), torch.full_like(d, 0.3), d)

    def run(dd, **kw):
        if fn == "j":
            return np.array([fm.compute_j_integral(mesh, u, dd, mat, fem, tip=tip, alpha=0.0, r_inner_factor=4.0,
                                                   r_outer_factor=8.0, **kw)])
        return np.array(fm.compute_sif(mesh, u, dd, mat, fem, tip=tip, alpha=0.0, r_inner_factor=4.0,
                                       r_outer_factor=8.0, **kw))

    default, explicit, strict = run(halo), run(halo, d_process_threshold=0.9), run(halo, d_process_threshold=0.01)
    assert np.array_equal(default, explicit)
    assert np.abs(strict - default).max() > 1e-2 * np.abs(default).max()
    nothing_left_out = run(halo, d_process_threshold=1.01)
    assert np.allclose(default, nothing_left_out, rtol=1e-12)                    # no cell of the halo reaches 0.9
    dead = torch.where((x < 0.0) & (y.abs() < 0.12), torch.ones_like(d), d)
    assert np.array_equal(run(dead), run(dead, d_process_threshold=1.0))         # d = 1 is at or above every cutoff
