import pytest
import torch
from types import SimpleNamespace

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from examples.quasistatic._run_utils import (
    restore_solver_state,
    snapshot_solver_state,
)


class _OneTriangleMesh:
    device = "cpu"
    dtype = torch.float64
    n_nodes = 3
    n_elems = 1
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    areas = torch.tensor([0.5], dtype=torch.float64)
    grad_phi = torch.zeros(1, 3, 2, dtype=torch.float64)
    M_scalar = torch.ones(3, dtype=torch.float64)
    _elem_flat = elements.flatten()
    h_min = 1.0


def _solver():
    mat = Material(
        E=210.0,
        nu=0.3,
        Gc=2.7,
        l0=0.1,
        pf_model="AT2",
        energy_split="isotropic",
    )
    return PhaseFieldDamageSolver(
        FEMOperators(_OneTriangleMesh(), mat),
        use_multigrid=False,
        preconditioner="jacobi",
    )


def test_damage_viscosity_adds_mass_term_and_lagged_rhs():
    H = torch.tensor([0.3], dtype=torch.float64)
    d_prev = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)
    baseline = _solver()
    _, _, reaction0, b0, *_ = baseline._prepare_cg(H, d_prev)

    viscous = _solver()
    viscous.damage_viscosity = 0.2
    viscous.damage_dt = 0.1
    _, _, reaction1, b1, *_ = viscous._prepare_cg(H, d_prev)

    coeff = viscous.damage_viscosity / viscous.damage_dt
    area = 0.5
    d_sum = float(d_prev.sum())
    expected_reaction_add = coeff * area / 12.0
    expected_rhs_add = torch.tensor(
        [expected_reaction_add * (float(d) + d_sum) for d in d_prev],
        dtype=torch.float64,
    )

    assert torch.allclose(
        reaction1 - reaction0,
        torch.tensor([expected_reaction_add], dtype=torch.float64),
    )
    assert torch.allclose(b1 - b0, expected_rhs_add)


def test_damage_viscosity_uses_fixed_step_reference_for_rhs():
    H = torch.tensor([0.3], dtype=torch.float64)
    current_iterate = torch.tensor([0.5, 0.6, 0.8], dtype=torch.float64)
    accepted_step_ref = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)

    baseline = _solver()
    _, _, _, b0, *_ = baseline._prepare_cg(H, current_iterate)

    viscous = _solver()
    viscous.damage_viscosity = 0.2
    viscous.damage_dt = 0.1
    viscous.damage_viscosity_reference = accepted_step_ref
    _, _, _, b1, *_ = viscous._prepare_cg(H, current_iterate)

    coeff = viscous.damage_viscosity / viscous.damage_dt
    area = 0.5
    d_sum = float(accepted_step_ref.sum())
    expected_rhs_add = torch.tensor(
        [coeff * area / 12.0 * (float(d) + d_sum)
         for d in accepted_step_ref],
        dtype=torch.float64,
    )

    assert torch.allclose(b1 - b0, expected_rhs_add)
    assert torch.allclose(viscous._last_viscous_d_prev, accepted_step_ref)


def test_damage_cg_can_warm_start_without_changing_lower_bound():
    solver = _solver()
    d_prev_step = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)
    current_iterate = torch.tensor([0.5, 0.6, 0.8], dtype=torch.float64)
    solver._cg_initial_guess = current_iterate

    d_init, d_prev_cg, *_ = solver._prepare_cg(
        torch.tensor([0.3], dtype=torch.float64), d_prev_step)

    assert torch.allclose(d_init, current_iterate)
    assert torch.allclose(d_prev_cg, d_prev_step)


def test_damage_viscosity_requires_positive_dt():
    solver = _solver()
    solver.damage_viscosity = 1.0
    solver.damage_dt = None

    with pytest.raises(RuntimeError, match="damage_dt"):
        solver._prepare_cg(
            torch.tensor([0.3], dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )


def test_damage_viscosity_rejects_direct_damage_path():
    solver = _solver()
    solver.damage_viscosity = 1.0
    solver.damage_dt = 1.0

    with pytest.raises(NotImplementedError, match="CG/projected-CG"):
        solver._solve_direct(
            torch.tensor([0.3], dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )


def test_qs_snapshot_restores_damage_viscosity_bookkeeping():
    solver = SimpleNamespace(
        u=torch.tensor([1.0], dtype=torch.float64),
        v=torch.tensor([2.0], dtype=torch.float64),
        a=torch.tensor([3.0], dtype=torch.float64),
        d=torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64),
        H_elem=torch.tensor([4.0], dtype=torch.float64),
        H_nodal=torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64),
        f_ext=torch.tensor([8.0], dtype=torch.float64),
        _step_count=11,
        _last_stagger_iter=3,
        _last_residual=1e-7,
        _last_damage_load_factor=0.25,
        damage_solver=SimpleNamespace(
            damage_dt=0.05,
            damage_viscosity_reference=torch.tensor(
                [0.1, 0.2, 0.4], dtype=torch.float64),
            _last_viscous_d_prev=torch.tensor(
                [0.1, 0.2, 0.4], dtype=torch.float64),
        ),
    )
    state = snapshot_solver_state(solver)

    solver.d = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float64)
    solver._last_damage_load_factor = 0.75
    solver.damage_solver.damage_dt = 0.15
    solver.damage_solver.damage_viscosity_reference = torch.tensor(
        [0.9, 0.9, 0.9], dtype=torch.float64)
    solver.damage_solver._last_viscous_d_prev = torch.tensor(
        [0.8, 0.8, 0.8], dtype=torch.float64)

    restore_solver_state(solver, state)

    assert torch.allclose(
        solver.d, torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64))
    assert solver._last_damage_load_factor == 0.25
    assert solver.damage_solver.damage_dt == 0.05
    assert torch.allclose(
        solver.damage_solver.damage_viscosity_reference,
        torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64))
    assert torch.allclose(
        solver.damage_solver._last_viscous_d_prev,
        torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64))
