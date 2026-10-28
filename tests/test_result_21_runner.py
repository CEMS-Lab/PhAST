"""The 2D runner's result file: ``stress_zz`` for every energy split and stress degradation, the stored damage
lower bound, and the layout-2.1 items (initial state, wave speeds, velocity condition, crack tips).

The in-plane and out-of-plane stress are checked against an independent 3D
reference: the 2D strain is embedded in a 3x3 tensor with ``eps_zz = 0``, the
split is applied with ``numpy.linalg.eigh`` where it is spectral, and the
``zz`` entry is compared.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from phast import load_result
from phast.config.result_output import plane_strain_stress_zz, resolve_result_settings
from phast.core.fem_operators import FEMOperators
from phast.core.mesh import FEMMesh
from phast.physics.material import Material
from result_v2 import runner_helpers as R

SPLITS = ["isotropic", "amor", "star_convex", "spectral", "spectral_stress"]


# --- independent 3D reference ---------------------------------------------------

def _same_numbers(a, b, what=""):
    """Two separate runs of one deck agree to roundoff. Bit equality is not promised: with several CPU threads the solver
    reduces in a different order from run to run (on Linux with 4 threads two identical runs differ by 1e-16 relative)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, what
    scale = max(float(np.max(np.abs(a))) if a.size else 0.0, np.finfo(np.float64).tiny)
    assert float(np.max(np.abs(a - b))) <= 1e-10 * scale, what


def _reference_stress(split, eps2, g, lam, mu):
    """Full 3x3 stress of the degraded law for one point; ``eps2`` = (exx, eyy, gamma_xy), eps_zz = 0."""
    exx, eyy, gxy = eps2
    eps = np.array([[exx, gxy / 2, 0.0], [gxy / 2, eyy, 0.0], [0.0, 0.0, 0.0]])
    eye = np.eye(3)
    tr = np.trace(eps)
    kappa = lam + 2.0 * mu / 3.0
    pos = lambda x: max(x, 0.0)
    neg = lambda x: min(x, 0.0)
    if split == "isotropic":
        return g * (lam * tr * eye + 2 * mu * eps)
    if split == "amor":
        dev = eps - tr / 3.0 * eye
        return g * (kappa * pos(tr) * eye + 2 * mu * dev) + kappa * neg(tr) * eye
    if split == "star_convex":
        dev = eps - tr / 3.0 * eye
        return g * (lam * tr * eye + 2 * mu * eps) if tr >= 0 else g * 2 * mu * dev + kappa * tr * eye
    if split == "spectral":
        w, v = np.linalg.eigh(eps)
        eps_p = sum(pos(w[i]) * np.outer(v[:, i], v[:, i]) for i in range(3))
        eps_m = eps - eps_p
        return g * (lam * pos(tr) * eye + 2 * mu * eps_p) + lam * neg(tr) * eye + 2 * mu * eps_m
    sigma0 = lam * tr * eye + 2 * mu * eps                       # spectral_stress: split the stress
    w, v = np.linalg.eigh(sigma0)
    s_p = sum(pos(w[i]) * np.outer(v[:, i], v[:, i]) for i in range(3))
    return g * s_p + (sigma0 - s_p)


def _fem(split, element_type="T3", **material):
    if element_type == "T3":
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
        elems = torch.tensor([[0, 1, 2], [0, 2, 3]])
    else:
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
                             dtype=torch.float64)
        elems = torch.tensor([[0, 1, 2, 3], [1, 4, 5, 2]])
    mesh = FEMMesh.from_tensors(nodes, elems, device="cpu", element_type=element_type)
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.1, rho=7.8e-9, energy_split=split, pf_model="AT2",
                   plane_stress=False, **material)
    return mesh, mat, FEMOperators(mesh, mat)


STATES = {                                            # displacement gradients: tension, compression, shear, mixed
    "tension": [[2e-3, 0.0], [1e-3, 3e-3]],
    "compression": [[-2e-3, 5e-4], [0.0, -1.5e-3]],
    "shear": [[0.0, 2e-3], [-1e-3, 0.0]],
    "mixed": [[4e-3, 1e-3], [-2e-3, -3e-3]],
}


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("state", sorted(STATES))
def test_stress_zz_matches_the_3d_split(split, state):
    mesh, mat, fem = _fem(split)
    grad = torch.tensor(STATES[state], dtype=torch.float64)
    u = mesh.nodes @ grad.T
    d = torch.tensor([0.0, 0.3, 0.6, 0.2], dtype=torch.float64)
    strain = fem.compute_strain(u)
    sxx, syy, sxy = (x.detach().numpy() for x in fem.compute_stress(u, d, strain=strain))
    szz = plane_strain_stress_zz(fem, d, strain).detach().numpy()
    g = mat.degradation(d[mesh.elements].mean(1)).detach().numpy()
    lam, mu = float(mat.lam), float(mat.mu)
    scale = max(np.abs(sxx).max(), np.abs(syy).max(), 1.0)
    for e in range(2):
        ref = _reference_stress(split, [float(x[e]) for x in strain], g[e], lam, mu)
        assert szz[e] == pytest.approx(ref[2, 2], rel=1e-6, abs=1e-7 * scale), (e, split, state)
        # the in-plane part of the reference is what the solver computes: the reference is independent
        assert (sxx[e], syy[e], sxy[e]) == pytest.approx((ref[0, 0], ref[1, 1], ref[0, 1]), rel=1e-5, abs=1e-6 * scale)   # the solver regularises degenerate eigenvalues with 1e-12


def test_isotropic_zz_is_poisson_times_the_trace_of_the_in_plane_stress():
    mesh, mat, fem = _fem("isotropic")
    u = mesh.nodes @ torch.tensor(STATES["mixed"], dtype=torch.float64).T
    d = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    strain = fem.compute_strain(u)
    sxx, syy, _ = fem.compute_stress(u, d, strain=strain)
    assert torch.allclose(plane_strain_stress_zz(fem, d, strain), mat.nu * (sxx + syy), rtol=1e-12)


@pytest.mark.parametrize("split", ["amor", "spectral", "star_convex"])
def test_stress_zz_of_q4_has_one_value_per_point(split):
    mesh, mat, fem = _fem(split, "Q4")
    u = mesh.nodes @ torch.tensor(STATES["mixed"], dtype=torch.float64).T
    d = torch.linspace(0.0, 0.5, mesh.n_nodes, dtype=torch.float64)
    strain = fem.compute_strain(u)
    szz = plane_strain_stress_zz(fem, d, strain)
    assert szz.shape == (2, 4)
    g = mat.degradation(torch.einsum("qa,ea->eq", mesh.quad_N, d[mesh.elements])).detach().numpy()
    for e in range(2):
        for q in range(4):
            ref = _reference_stress(split, [float(x[e, q]) for x in strain], g[e, q], float(mat.lam), float(mat.mu))
            assert float(szz[e, q]) == pytest.approx(ref[2, 2], rel=1e-6, abs=1e-6)


def test_stress_zz_is_refused_where_it_is_not_defined():
    _, _, fem = _fem("amor")
    fem.material.plane_stress = True
    with pytest.raises(ValueError, match="plane-stress"):
        plane_strain_stress_zz(fem, torch.zeros(4, dtype=torch.float64), fem.compute_strain(torch.zeros(4, 2, dtype=torch.float64)))
    _, _, fem = _fem("amor")
    fem.material.energy_split = "spectral_plane_stress_condensed"
    with pytest.raises(ValueError, match="no out-of-plane stress"):
        plane_strain_stress_zz(fem, torch.zeros(4, dtype=torch.float64), fem.compute_strain(torch.zeros(4, 2, dtype=torch.float64)))


@pytest.mark.parametrize("state", sorted(STATES))
def test_stress_zz_under_full_stress_degradation_is_the_isotropic_law(state):
    """``stress_degradation: full`` degrades the whole stress by one factor, whatever the history split is."""
    mesh, mat, fem = _fem("spectral", stress_degradation="full")
    u = mesh.nodes @ torch.tensor(STATES[state], dtype=torch.float64).T
    d = torch.tensor([0.0, 0.3, 0.6, 0.2], dtype=torch.float64)
    strain = fem.compute_strain(u)
    szz = plane_strain_stress_zz(fem, d, strain).detach().numpy()
    g = mat.degradation(d[mesh.elements].mean(1)).detach().numpy()
    for e in range(2):
        ref = _reference_stress("isotropic", [float(x[e]) for x in strain], g[e], float(mat.lam), float(mat.mu))
        assert szz[e] == pytest.approx(ref[2, 2], rel=1e-9, abs=1e-12)
    # a degradation law other than the standard one changes g(d) only
    _, rational, fem2 = _fem("isotropic", degradation_type="rational_at2")
    g2 = rational.degradation(d[mesh.elements].mean(1)).detach().numpy()
    szz2 = plane_strain_stress_zz(fem2, d, strain).detach().numpy()
    for e in range(2):
        ref = _reference_stress("isotropic", [float(x[e]) for x in strain], g2[e], float(mat.lam), float(mat.mu))
        assert szz2[e] == pytest.approx(ref[2, 2], rel=1e-9, abs=1e-12)


# --- runner files ------------------------------------------------------------------

def _t3_with(tmp, split, fields=None):
    cfg = R.t3_qs(tmp)
    cfg["material"]["energy_split"] = split
    if fields:
        cfg["output"]["result"] = dict(fields=fields)
    return cfg


@pytest.fixture(scope="module")
def amor_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("amor")
    cfg = _t3_with(tmp, "amor", ["displacement", "damage", "H", "damage_lower_bound", "stress", "strain"])
    return R.run(tmp, cfg)[1]


@pytest.mark.timeout(180)
def test_runner_writes_stress_zz_for_a_non_isotropic_split(amor_run):
    R.validate_file(amor_run / "result.vtkhdf")
    with h5py.File(amor_run / "result.vtkhdf", "r") as f:
        zz = f["VTKHDF/CellData/stress_zz"]
        assert "amor" in zz.attrs["meaning"] and zz.attrs["derived_method"] == "exact"
        n_cells = int(f["VTKHDF/NumberOfCells"][0])
        d = f["VTKHDF/PointData/damage"][-int(f["VTKHDF/NumberOfPoints"][0]):]
        eps = f["VTKHDF/CellData/strain"][-n_cells:].astype(float)
        stored = zz[-n_cells:].astype(float)
        cells = f["VTKHDF/Connectivity"][...].reshape(-1, 3)
        tris = d[cells].mean(axis=1)
    lam = 210000.0 * 0.3 / (1.3 * 0.4)
    mu = 210000.0 / 2.6
    g = (1 - tris) ** 2
    ref = np.array([_reference_stress("amor", e, gi, lam, mu)[2, 2] for e, gi in zip(eps, g)])
    scale = np.abs(ref).max()
    assert scale > 0
    assert np.allclose(stored, ref, rtol=2e-4, atol=1e-4 * scale)      # float32 storage and the residual stiffness


@pytest.mark.timeout(180)
def test_damage_lower_bound_is_the_damage_before_the_increment(amor_run):
    with h5py.File(amor_run / "result.vtkhdf", "r") as f:
        n = int(f["VTKHDF/NumberOfPoints"][0])
        lb = f["VTKHDF/PointData/damage_lower_bound"]
        d = f["VTKHDF/PointData/damage"]
        assert lb.dtype == np.float64 and lb.attrs["kind"] == "state" and "definition" in lb.attrs
        assert isinstance(f.get("state/damage_lower_bound", getlink=True), h5py.SoftLink)
        steps = int(f["VTKHDF/Steps"].attrs["NSteps"])
        assert steps == 4
        for s in range(1, steps):
            assert np.array_equal(lb[s * n:(s + 1) * n], d[(s - 1) * n:s * n])     # irreversibility floor of increment s
        assert (d[...] >= lb[...] - 1e-12).all()                                    # the solution respects it
        assert f.attrs["phast_schema_version"] == "2.0.0"


@pytest.mark.timeout(180)
def test_storing_the_lower_bound_leaves_the_numerics_unchanged(tmp_path):
    base = R.run(tmp_path / "a", R.t3_qs(tmp_path / "a"))[1]
    withlb = R.run(tmp_path / "b", _t3_with(tmp_path / "b", "isotropic",
                                            ["displacement", "damage", "H", "damage_lower_bound", "stress", "strain"]))[1]
    a, b = np.load(base / "final.npz"), np.load(withlb / "final.npz")
    for key in ("displacement", "damage", "history_element"):
        _same_numbers(a[key], b[key], key)


@pytest.mark.timeout(180)
def test_explicit_run_stores_the_lower_bound_per_increment(tmp_path):
    cfg = R.dynamic(tmp_path)
    cfg["output"]["result"] = dict(every=1, fields=["displacement", "damage", "H", "velocity", "acceleration",
                                                    "damage_lower_bound"])
    out = R.run(tmp_path, cfg)[1]
    R.validate_file(out / "result.vtkhdf")
    with h5py.File(out / "result.vtkhdf", "r") as f:
        n = int(f["VTKHDF/NumberOfPoints"][0])
        lb, d = f["VTKHDF/PointData/damage_lower_bound"][...], f["VTKHDF/PointData/damage"][...]
        steps = int(f["VTKHDF/Steps"].attrs["NSteps"])
        for s in range(1, steps):
            assert np.array_equal(lb[s * n:(s + 1) * n], d[(s - 1) * n:s * n])


@pytest.mark.timeout(180)
def test_material_options_are_recorded_and_stress_zz_follows_them(tmp_path):
    cfg = R.t3_qs(tmp_path)
    cfg["material"].update(energy_split="spectral", stress_degradation="full", degradation_type="rational_at2")
    out = R.run(tmp_path, cfg)[1]
    R.validate_file(out / "result.vtkhdf")
    with h5py.File(out / "result.vtkhdf", "r") as f:
        phase = f["materials/phases/0"]
        assert phase.attrs["energy_split"] == "spectral" and phase.attrs["stress_degradation"] == "full"
        assert phase.attrs["degradation_type"] == "rational_at2"
        assert f["history/damage_solved"][...].tolist() == [1] * 4      # the projected Newton solve of rational_at2 counts too
        assert "isotropic" in f["VTKHDF/CellData/stress_zz"].attrs["meaning"]
        n_cells = int(f["VTKHDF/NumberOfCells"][0])
        sigma = f["VTKHDF/CellData/stress"][-n_cells:].astype(float)
        zz = f["VTKHDF/CellData/stress_zz"][-n_cells:].astype(float)
    assert np.allclose(zz, 0.3 * (sigma[:, 0] + sigma[:, 1]), rtol=1e-5, atol=1e-6 * np.abs(zz).max())


# --- layout 2.1 in the runner ------------------------------------------------------------------------

def _layout21(cfg, **more):
    cfg["output"]["result"] = dict(layout="2.1", **more)
    return cfg


def test_layout_settings_resolution():
    s = resolve_result_settings(True, explicit=False, h5_every=1)
    assert s["layout"] == "2.0" and s["crack_tips"] is None
    assert resolve_result_settings({"layout": 2.1}, explicit=False, h5_every=1)["layout"] == "2.1"
    s = resolve_result_settings({"crack_tips": {"threshold": 0.9}}, explicit=False, h5_every=1)
    assert s["layout"] == "2.1" and s["crack_tips"] == dict(threshold=0.9, direction="x")     # a 2.1 item implies 2.1
    assert resolve_result_settings({"crack_tips": True}, explicit=False, h5_every=1)["crack_tips"]["threshold"] == 0.5
    for bad in ({"layout": "3.0"}, {"layout": 2}, {"crack_tips": {"treshold": 0.5}}, {"crack_tips": {"threshold": 0.0}},
                {"crack_tips": {"direction": "z"}}, {"crack_tips": 3}):
        with pytest.raises(ValueError):
            resolve_result_settings(bad, explicit=False, h5_every=1)
    assert json.dumps(resolve_result_settings({"crack_tips": True}, explicit=False, h5_every=1))


@pytest.fixture(scope="module")
def qs21(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("qs21")
    cfg = _layout21(R.t3_qs(tmp), crack_tips=dict(threshold=0.5))
    cfg["loading"] = dict(protocol="simple", num_steps=12, dt=0.001)
    return R.run(tmp, cfg)[1]


@pytest.mark.timeout(180)
def test_quasi_static_run_writes_initial_state_and_tips(qs21):
    path = qs21 / "result.vtkhdf"
    R.validate_file(path)
    with h5py.File(path, "r") as f:
        assert f.attrs["phast_schema_version"] == "2.1.0"
        assert "crack_tips" in set(f["history"]) and {"damage", "H", "displacement"} <= set(f["initial_conditions"])
        assert "substeps" not in set(f["history"])                    # the damage solver reports no per-solve record
        tips = f["history/crack_tips"]
        assert tips["increment"][0] > 0 and tips["position"].attrs["units"] == "mm"
        assert tips["speed"].attrs["units"] == "mm"                  # a pseudo-time of unit 1
        assert "find_crack_tip" in tips["position"].attrs["method"]
        assert np.allclose(f["initial_conditions/damage"][...], 0.0)
        assert "specification" not in f["initial_conditions/damage"].attrs
    # the tip is where the isoline of the damage field is: on the notch line y = 0.5, advancing in +x
    ct = load_result(path).crack_tips()
    assert np.all(np.abs(ct["position"][1:, 1] - 0.5) < 0.2) and ct["position"][-1, 0] > ct["position"][0, 0]


@pytest.mark.timeout(240)
def test_layout_21_changes_no_numbers(tmp_path):
    plain = R.run(tmp_path / "a", R.t3_qs(tmp_path / "a"))[1]
    rich = R.run(tmp_path / "b", _layout21(R.t3_qs(tmp_path / "b"), crack_tips=True))[1]
    a, b = np.load(plain / "final.npz"), np.load(rich / "final.npz")
    for key in ("displacement", "damage", "history_element"):
        _same_numbers(a[key], b[key], key)
    with h5py.File(plain / "result.vtkhdf", "r") as fa, h5py.File(rich / "result.vtkhdf", "r") as fb:
        assert fa.attrs["phast_schema_version"] == "2.0.0" and fb.attrs["phast_schema_version"] == "2.1.0"
        for name in ("displacement", "damage", "H"):
            _same_numbers(fa[f"VTKHDF/PointData/{name}" if name != "H" else "VTKHDF/CellData/H"][...],
                          fb[f"VTKHDF/PointData/{name}" if name != "H" else "VTKHDF/CellData/H"][...], name)
        _same_numbers(fa["history/energy/elastic"][...], fb["history/energy/elastic"][...], "elastic energy")


@pytest.mark.timeout(240)
def test_velocity_impact_run_records_velocity_initial_state_and_wave_speeds(tmp_path):
    cfg = _layout21(R.kalthoff(tmp_path))
    cfg["initial_conditions"] = dict(preseed_notch_nodesets=["notch1", "notch2"])
    out = R.run(tmp_path, cfg)[1]
    R.validate_file(out / "result.vtkhdf")
    r = load_result(out / "result.vtkhdf")
    bc = r.boundary_conditions()["bc"]
    vel = next(a for a in bc.values() if a["set"] == "left_impact")
    assert vel["kind"] == "velocity" and vel["value"] == pytest.approx(16.5 * 1e3) and vel["value_units"] == "mm/s"
    assert vel["ramp_type"] == "velocity_impact" and vel["t_ramp"] == pytest.approx(1.0e-6)
    phase = r.materials()["phases"][0]
    E, nu, rho = 190000.0, 0.3, 8.0e-9
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    assert phase["c_p"] == pytest.approx(((lam + 2 * mu) / rho) ** 0.5, rel=1e-12) and phase["units"]["c_p"] == "mm/s"
    assert phase["c_s"] == pytest.approx((mu / rho) ** 0.5, rel=1e-12)
    ic = r.initial_conditions()
    assert set(ic) == {"H", "acceleration", "damage", "displacement", "velocity"} and not ic["velocity"].any()
    assert ic["damage"].max() > 0.9 and ic["H"].max() > 1.0e3       # the pre-seeded notch: seeded history, damage solved once
    with h5py.File(out / "result.vtkhdf", "r") as f:
        spec = json.loads(f["initial_conditions/damage"].attrs["specification"])
    assert spec["preseed_notch_nodesets"] or spec["preseed_damage"]
    plain_cfg = R.kalthoff(tmp_path / "plain")
    plain_cfg["initial_conditions"] = dict(preseed_notch_nodesets=["notch1", "notch2"])
    plain = R.run(tmp_path / "plain", plain_cfg)[1]
    _same_numbers(np.load(plain / "final.npz")["damage"], np.load(out / "final.npz")["damage"], "damage")
    _same_numbers(np.load(plain / "final.npz")["velocity"], np.load(out / "final.npz")["velocity"], "velocity")


@pytest.mark.timeout(180)
def test_crack_tips_are_refused_on_q4(tmp_path):
    cfg = R.q4_qs(tmp_path, result=dict(crack_tips=True))
    proc, out = R.run(tmp_path, cfg, check=False)
    assert proc.returncode != 0 and "Q4 cells" in proc.stderr


def test_an_unknown_boundary_condition_type_is_refused_in_layout_21_only():
    from types import SimpleNamespace
    from phast.config.result_output import RunResultRecorder
    entry = SimpleNamespace(type="mystery", nodes="top", component=1, value=1.0)
    cfg = SimpleNamespace(boundary_conditions=[entry], loading=SimpleNamespace(ramp_type="constant"))
    solver = SimpleNamespace(bcs=SimpleNamespace(neumann_bcs=[]))
    stub = lambda layout21: SimpleNamespace(explicit=False, layout21=layout21, solver=solver)
    bcs, _, _ = RunResultRecorder._loading(stub(False), cfg, None)           # 2.0 keeps the raw type as the kind
    assert next(iter(bcs.values()))["kind"] == "mystery"
    with pytest.raises(ValueError, match="no layout-2.1 kind"):
        RunResultRecorder._loading(stub(True), cfg, None)
