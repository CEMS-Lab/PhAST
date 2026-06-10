"""eta_residual default-consistency test.

Audit T1.1 (W4 SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07.md): the
``Material`` dataclass defaulted ``eta_residual = 1e-7`` (the physics
source of truth, in `material.py`) while ``Config.solver.eta_residual``
defaulted to ``1e-6`` (in `config.py`). The 10x discrepancy meant
direct-instantiated solvers and YAML-driven solvers used different
residual-stiffness floors.

Fix: ``Config.solver.eta_residual`` default is now ``1e-7`` to match
``Material``. This test pins both defaults and the YAML-propagation path.
"""

import os
import tempfile

import pytest


def test_material_default_eta_residual_is_1e_minus_7():
    """Material's default eta_residual stays at 1e-7 (physics convention)."""
    from phast.material import Material
    mat = Material()
    assert mat.eta_residual == pytest.approx(1e-7)


def test_config_solver_default_eta_residual_matches_material():
    """Config.solver default must agree with Material default — no 10x split."""
    from phast.config import SolverSettings
    from phast.material import Material
    s = SolverSettings()
    m = Material()
    assert s.eta_residual == pytest.approx(m.eta_residual)
    assert s.eta_residual == pytest.approx(1e-7)


def test_yaml_explicit_eta_residual_propagates_to_material_and_solver():
    """An explicit ``eta_residual`` in YAML must reach both Material and SolverSettings."""
    from phast.config import load_config
    from phast.material import create_material

    yaml_text = """
name: eta-residual-propagation
material:
  preset: glass_borden
solver:
  eta_residual: 5.0e-8
"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)

    # SolverSettings carries the explicit value verbatim.
    assert cfg.solver.eta_residual == pytest.approx(5.0e-8)

    # Resolution pipeline forwards solver.eta_residual to Material when
    # the material block omits it (mirrors config.resolve_config:805-807).
    mat_overrides = dict(cfg.material.overrides)
    inline_names = (
        'E', 'nu', 'Gc', 'l0', 'rho', 'eta_residual',
        'energy_split', 'pf_model', 'kinematics', 'plane_stress',
    )
    for name in inline_names:
        v = getattr(cfg.material, name, None)
        if v is not None:
            mat_overrides[name] = v
    if 'eta_residual' not in mat_overrides:
        mat_overrides['eta_residual'] = cfg.solver.eta_residual
    mat = create_material(preset=cfg.material.preset, **mat_overrides)
    assert mat.eta_residual == pytest.approx(5.0e-8)


def test_yaml_default_eta_residual_propagates_to_material_as_1e_minus_7():
    """When YAML omits eta_residual entirely, both Material and SolverSettings
    end up at 1e-7 (the unified default)."""
    from phast.config import load_config
    from phast.material import create_material

    yaml_text = """
name: eta-residual-default
material:
  preset: glass_borden
"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)

    assert cfg.solver.eta_residual == pytest.approx(1e-7)

    mat_overrides = dict(cfg.material.overrides)
    inline_names = (
        'E', 'nu', 'Gc', 'l0', 'rho', 'eta_residual',
        'energy_split', 'pf_model', 'kinematics', 'plane_stress',
    )
    for name in inline_names:
        v = getattr(cfg.material, name, None)
        if v is not None:
            mat_overrides[name] = v
    if 'eta_residual' not in mat_overrides:
        mat_overrides['eta_residual'] = cfg.solver.eta_residual
    mat = create_material(preset=cfg.material.preset, **mat_overrides)
    # The glass_borden preset itself ships eta_residual=1e-7 (material.py:341),
    # so this assertion is also a sanity-check on preset consistency.
    assert mat.eta_residual == pytest.approx(1e-7)
