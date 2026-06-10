"""Tests for the config/Problem builder pipeline."""

import math
import os
import tempfile
import pytest


class TestLoadingTypes:
    """Test compute_load_factor for all ramp types."""

    def test_constant(self):
        from phast.config import compute_load_factor, LoadingConfig
        lc = LoadingConfig(ramp_type='constant')
        assert compute_load_factor(0, 1e-7, lc) == 1.0
        assert compute_load_factor(1000, 1e-7, lc) == 1.0

    def test_linear_ramp(self):
        from phast.config import compute_load_factor, LoadingConfig
        lc = LoadingConfig(ramp_type='linear', t_ramp=10e-6)
        assert compute_load_factor(0, 1e-7, lc) == pytest.approx(0.0)
        assert compute_load_factor(50, 1e-7, lc) == pytest.approx(0.5, rel=1e-6)
        assert compute_load_factor(100, 1e-7, lc) == pytest.approx(1.0, rel=1e-6)
        assert compute_load_factor(200, 1e-7, lc) == 1.0

    def test_linear_ramp_zero_tramp(self):
        from phast.config import compute_load_factor, LoadingConfig
        lc = LoadingConfig(ramp_type='linear', t_ramp=0.0)
        assert compute_load_factor(100, 1e-7, lc) == 1.0

    def test_smooth_ramp(self):
        from phast.config import compute_load_factor, LoadingConfig
        lc = LoadingConfig(ramp_type='smooth', t_ramp=10e-6)
        assert compute_load_factor(0, 1e-7, lc) == pytest.approx(0.0)
        assert compute_load_factor(50, 1e-7, lc) == pytest.approx(0.5, rel=1e-6)
        assert compute_load_factor(100, 1e-7, lc) == pytest.approx(1.0, rel=1e-6)

    def test_velocity_impact(self):
        from phast.config import compute_load_factor, LoadingConfig
        lc = LoadingConfig(ramp_type='velocity_impact', v0=16.5, t_ramp=1e-6)
        assert compute_load_factor(0, 1e-8, lc) == 0.0
        f = compute_load_factor(100, 1e-8, lc)
        assert f > 0
        f2 = compute_load_factor(200, 1e-8, lc)
        assert f2 > f

    def test_unknown_ramp_type(self):
        from phast.config import compute_load_factor, LoadingConfig
        lc = LoadingConfig(ramp_type='unknown_type')
        assert compute_load_factor(100, 1e-7, lc) == 1.0


class TestCyclicSchedule:
    """Test cyclic loading schedule builder."""

    def test_empty(self):
        from phast.config import build_cyclic_schedule, LoadingConfig
        lc = LoadingConfig(cyclic_phases='')
        assert build_cyclic_schedule(lc) == []

    def test_single_phase(self):
        from phast.config import build_cyclic_schedule, LoadingConfig
        lc = LoadingConfig(cyclic_phases='1.0:1')
        sched = build_cyclic_schedule(lc)
        assert len(sched) == 1
        assert sched[0] == pytest.approx(1.0)

    def test_multi_phase(self):
        from phast.config import build_cyclic_schedule, LoadingConfig
        lc = LoadingConfig(cyclic_phases='0.3:3,-0.2:2')
        sched = build_cyclic_schedule(lc)
        assert len(sched) == 5
        assert sched[2] == pytest.approx(0.3)
        assert sched[4] == pytest.approx(-0.2)

    def test_monotonicity_within_phase(self):
        from phast.config import build_cyclic_schedule, LoadingConfig
        lc = LoadingConfig(cyclic_phases='1.0:10')
        sched = build_cyclic_schedule(lc)
        for i in range(1, len(sched)):
            assert sched[i] > sched[i - 1]


class TestConfigLoadSave:
    """Test YAML round-trip of ProblemConfig."""

    def test_load_existing_yaml(self):
        from phast.config import load_config
        cfg = load_config('configs/B3_dynamic_sent.yaml')
        assert cfg.name == 'Dynamic SENT'
        assert cfg.loading.t_total == pytest.approx(5e-5)
        # B3 was migrated to the inline-DSL primitives vocabulary in
        # #192/#388, so ``geometry.type`` now falls through to the
        # dataclass default and the geometry is described by
        # ``geometry.primitives``. Assert the primitive vocabulary is
        # populated instead of the legacy generator name.
        assert cfg.geometry.primitives, (
            "B3 config should use the inline-DSL primitives vocabulary"
        )
        assert len(cfg.boundary_conditions) == 3

    def test_save_and_reload(self):
        from phast.config import load_config, save_config
        cfg = load_config('configs/B1_branching_glass.yaml')
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path)
            cfg2 = load_config(path)
            assert cfg2.name == cfg.name
            assert cfg2.geometry.type == cfg.geometry.type
            assert cfg2.loading.t_total == cfg.loading.t_total
        finally:
            os.unlink(path)

    def test_t_total_preserved(self):
        from phast.config import load_config, save_config
        cfg = load_config('configs/B2_kalthoff_winkler.yaml')
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            save_config(cfg, path)
            cfg2 = load_config(path)
            assert cfg2.loading.t_total == cfg.loading.t_total
            assert cfg2.loading.ramp_type == cfg.loading.ramp_type
        finally:
            os.unlink(path)

    def test_solver_settings_forwarded_to_solver_config(self, tmp_path, monkeypatch):
        from phast.config import load_config, resolve_config
        import phast.config as cfg_mod
        import phast.mesh as mesh_mod

        class FakeMesh:
            n_nodes = 4
            node_sets = {}

            def __init__(self, mesh_path, device=None, dtype=None):
                self.mesh_path = mesh_path

            def identify_boundaries(self):
                return {}

        monkeypatch.setattr(mesh_mod, 'FEMMesh', FakeMesh)
        monkeypatch.setattr(
            cfg_mod, 'get_geometry_registry',
            lambda: {'square_plate': lambda **kwargs: 'dummy.msh'})

        yaml_path = tmp_path / 'bridge.yaml'
        yaml_path.write_text("""
name: bridge-test
geometry:
  type: square_plate
  parameters:
    L: 1.0
    h: 0.75
    verbose: false
material:
  E: 1.0
  nu: 0.25
  Gc: 1.0
  l0: 0.2
  rho: 1.0
solver:
  solver_type: explicit
  adaptive_dt: true
  adaptive_dt_d_threshold: 0.123
  damping_ratio_max: 0.15
  use_multigrid: false
  preconditioner: jacobi
device:
  device: cpu
""")
        cfg = load_config(str(yaml_path))
        solver_cfg = resolve_config(cfg)['solver_config']

        assert solver_cfg.damping_ratio_max == pytest.approx(0.15)
        assert solver_cfg.adaptive_dt_d_threshold == pytest.approx(0.123)
        assert solver_cfg.dt_cutback_threshold == pytest.approx(0.123)

    def test_generalized_alpha_solver_settings_forwarded(
        self, tmp_path, monkeypatch
    ):
        from phast.config import load_config, resolve_config
        import phast.config as cfg_mod
        import phast.mesh as mesh_mod

        class FakeMesh:
            n_nodes = 4
            node_sets = {}

            def __init__(self, mesh_path, device=None, dtype=None):
                self.mesh_path = mesh_path

            def identify_boundaries(self):
                return {}

        monkeypatch.setattr(mesh_mod, 'FEMMesh', FakeMesh)
        monkeypatch.setattr(
            cfg_mod, 'get_geometry_registry',
            lambda: {'square_plate': lambda **kwargs: 'dummy.msh'})

        yaml_path = tmp_path / 'gen_alpha_bridge.yaml'
        yaml_path.write_text("""
name: gen-alpha-bridge
geometry:
  type: square_plate
  parameters:
    L: 1.0
    h: 0.75
    verbose: false
material:
  E: 1.0
  nu: 0.25
  Gc: 1.0
  l0: 0.2
  rho: 1.0
solver:
  solver_type: explicit
  time_integrator: generalized_alpha
  rho_inf: 0.5
  static_tol: 1.0e-7
device:
  device: cpu
""")
        cfg = load_config(str(yaml_path))
        solver_cfg = resolve_config(cfg)['solver_config']

        assert solver_cfg.time_integrator == 'generalized_alpha'
        assert solver_cfg.rho_inf == pytest.approx(0.5)
        assert solver_cfg.static_tol == pytest.approx(1.0e-7)

    def test_mps_spectral_resolves_to_cpu_float64(
        self, tmp_path, monkeypatch, capsys
    ):
        """Spectral split on MPS must route the whole run to CPU float64."""
        from phast.config import load_config, resolve_config
        import phast.config as cfg_mod
        import phast.mesh as mesh_mod

        class FakeMesh:
            n_nodes = 4
            node_sets = {}

            def __init__(self, mesh_path, device=None, dtype=None):
                self.mesh_path = mesh_path
                self.device = device
                self.dtype = dtype

            def identify_boundaries(self):
                return {}

        monkeypatch.setattr(mesh_mod, 'FEMMesh', FakeMesh)
        monkeypatch.setattr(
            cfg_mod, 'get_geometry_registry',
            lambda: {'square_plate': lambda **kwargs: 'dummy.msh'})

        yaml_path = tmp_path / 'mps_spectral.yaml'
        yaml_path.write_text("""
name: mps spectral fallback
geometry:
  type: square_plate
  parameters:
    L: 1.0
material:
  E: 210.0
  nu: 0.3
  Gc: 2.7
  l0: 0.1
  rho: 1.0
  energy_split: spectral
solver:
  solver_type: explicit
device:
  device: mps
""")

        cfg = load_config(str(yaml_path))
        objs = resolve_config(cfg)
        out = capsys.readouterr().out

        assert objs['ctx'].device.type == 'cpu'
        assert str(objs['ctx'].dtype) == 'torch.float64'
        assert objs['mesh'].device.type == 'cpu'
        assert objs['material'].energy_split == 'spectral'
        assert 'Routing the full solve to CPU float64' in out


class TestProblemBuilder:
    """Test the fluent Problem API."""

    def test_build_and_repr(self):
        from phast.problem import Problem
        p = (Problem('Test')
             .geometry('rectangular_sent', W=100, H=40, a=50, h_crack=4, h_coarse=10)
             .material('glass_borden', l0=1.0)
             .fix('left', dof='x')
             .neumann('top', dof='y', value=1.0))
        assert 'Test' in repr(p)
        assert p.config.geometry.type == 'rectangular_sent'
        assert len(p.config.boundary_conditions) == 2

    def test_dof_expansion(self):
        from phast.problem import Problem
        p = Problem('T').geometry('miehe_tension', L=1, a=0.5, h_crack=0.1, h_coarse=0.5)
        p.fix('bottom', dof='xy')
        assert len(p.config.boundary_conditions) == 2
        assert p.config.boundary_conditions[0].component == 0
        assert p.config.boundary_conditions[1].component == 1

    def test_from_yaml(self):
        from phast.problem import Problem
        p = Problem.from_yaml('configs/B3_dynamic_sent.yaml')
        assert p.config.name == 'Dynamic SENT'
        assert p.config.loading.t_total > 0

    def test_save_roundtrip(self):
        from phast.problem import Problem
        p = (Problem('RT Test')
             .geometry('rectangular_sent', W=50, H=20, a=25, h_crack=2, h_coarse=5)
             .material('steel_pf')
             .loading(protocol='simple', t_total=10e-6)
             .device('cpu'))
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            path = f.name
        try:
            p.save(path)
            p2 = Problem.from_yaml(path)
            assert p2.config.name == 'RT Test'
            assert p2.config.loading.t_total == pytest.approx(10e-6)
        finally:
            os.unlink(path)

    def test_explicit_run_num_steps_precedence_matches_cli(self, monkeypatch):
        """Explicit fluent runs should prefer num_steps over t_total."""
        from types import SimpleNamespace
        import torch
        import phast.problem as problem_mod
        from phast.problem import Problem

        class FakeBCs:
            load_factor = None

            def get_neumann_forces(self, mesh):
                return None

        class FakeSolver:
            def __init__(self, *args, config=None, **kwargs):
                self.config = config or args[3]
                self.dt = 0.1
                self.d = torch.zeros(1)
                self.steps = 0

            def step_full(self):
                self.steps += 1

        def fake_resolve_config(cfg):
            return {
                'mesh': SimpleNamespace(n_nodes=1, n_elems=1),
                'material': object(),
                'bcs': FakeBCs(),
                'solver_config': SimpleNamespace(solver_type='explicit'),
                'ctx': object(),
                'loading': None,
            }

        monkeypatch.setattr(problem_mod, 'resolve_config',
                            fake_resolve_config)
        monkeypatch.setattr(problem_mod, 'StaggeredSolver', FakeSolver)

        solver = (Problem('explicit precedence')
                  .geometry('rectangular_sent')
                  .material('glass_borden')
                  .loading(t_total=1.0, num_steps=3)
                  .solver('explicit')
                  .run(verbose=False))

        assert solver.config.num_steps == 3
        assert solver.steps == 3

    def test_explicit_loading_dt_propagates_to_solver_config(self, tmp_path,
                                                            monkeypatch):
        from phast.config import load_config, resolve_config
        import phast.config as cfg_mod
        import phast.mesh as mesh_mod

        class FakeMesh:
            n_nodes = 4
            node_sets = {}

            def __init__(self, mesh_path, device=None, dtype=None):
                self.mesh_path = mesh_path

            def identify_boundaries(self):
                return {}

        monkeypatch.setattr(mesh_mod, 'FEMMesh', FakeMesh)
        monkeypatch.setattr(
            cfg_mod, 'get_geometry_registry',
            lambda: {'square_plate': lambda **kwargs: 'dummy.msh'})
        yaml_text = f"""
name: explicit dt propagation
geometry:
  type: square_plate
  parameters:
    L: 1.0
material:
  E: 210.0
  nu: 0.3
  Gc: 2.7
  l0: 0.1
  rho: 1.0
solver:
  solver_type: explicit
loading:
  dt: 1.23e-7
  num_steps: 2
"""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml_text)

        cfg = load_config(str(cfg_path))
        assert resolve_config(cfg)['solver_config'].dt == pytest.approx(1.23e-7)

    def test_explicit_t_total_num_steps_zero_uses_cfl_dt(self, tmp_path,
                                                         monkeypatch):
        from phast.config import load_config, resolve_config
        import phast.config as cfg_mod
        import phast.mesh as mesh_mod

        class FakeMesh:
            n_nodes = 4
            node_sets = {}

            def __init__(self, mesh_path, device=None, dtype=None):
                self.mesh_path = mesh_path

            def identify_boundaries(self):
                return {}

        monkeypatch.setattr(mesh_mod, 'FEMMesh', FakeMesh)
        monkeypatch.setattr(
            cfg_mod, 'get_geometry_registry',
            lambda: {'square_plate': lambda **kwargs: 'dummy.msh'})
        yaml_text = """
name: explicit cfl dt
geometry:
  type: square_plate
  parameters:
    L: 1.0
material:
  E: 210.0
  nu: 0.3
  Gc: 2.7
  l0: 0.1
  rho: 1.0
solver:
  solver_type: explicit
loading:
  dt: 1.23e-7
  t_total: 1.0e-6
  num_steps: 0
"""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml_text)

        cfg = load_config(str(cfg_path))
        assert resolve_config(cfg)['solver_config'].dt is None

    def test_explicit_t_total_without_num_steps_uses_cfl_dt(self, tmp_path,
                                                            monkeypatch):
        from phast.config import load_config, resolve_config
        import phast.config as cfg_mod
        import phast.mesh as mesh_mod

        class FakeMesh:
            n_nodes = 4
            node_sets = {}

            def __init__(self, mesh_path, device=None, dtype=None):
                self.mesh_path = mesh_path

            def identify_boundaries(self):
                return {}

        monkeypatch.setattr(mesh_mod, 'FEMMesh', FakeMesh)
        monkeypatch.setattr(
            cfg_mod, 'get_geometry_registry',
            lambda: {'square_plate': lambda **kwargs: 'dummy.msh'})
        yaml_text = """
name: explicit cfl dt omitted num_steps
geometry:
  type: square_plate
  parameters:
    L: 1.0
material:
  E: 210.0
  nu: 0.3
  Gc: 2.7
  l0: 0.1
  rho: 1.0
solver:
  solver_type: explicit
loading:
  dt: 1.23e-7
  t_total: 1.0e-6
"""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml_text)

        cfg = load_config(str(cfg_path))
        assert cfg.loading.num_steps == 0
        assert resolve_config(cfg)['solver_config'].dt is None

    def test_precheck_config_uses_inline_material(self, tmp_path, monkeypatch):
        import phast.config as cfg_mod

        def fail_mesh_resolution(_cfg):
            raise RuntimeError("skip mesh build")

        monkeypatch.setattr(cfg_mod, 'resolve_config', fail_mesh_resolution)

        yaml_text = """
name: precheck inline material
material:
  E: "32 GPa"
  nu: 0.2
  Gc: "3 J/m^2"
  l0: "0.25 mm"
  rho: "2450 kg/m^3"
  energy_split: spectral
  pf_model: AT2
loading:
  t_total: 80 us
solver:
  solver_type: explicit
"""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml_text)

        from phast.precheck import _run_from_config
        params = _run_from_config(str(cfg_path))

        assert params['E'] == pytest.approx(32000.0)
        assert params['nu'] == pytest.approx(0.2)
        assert params['Gc'] == pytest.approx(3.0e-3)
        assert params['l0'] == pytest.approx(0.25)
        assert params['rho'] == pytest.approx(2.45e-9)


# ---------------------------------------------------------------------------
# Inline material spec (#137) — preset is no longer required.
# ---------------------------------------------------------------------------

def _build_material_from_yaml(yaml_text: str):
    """Helper: write yaml_text to tempfile, load, and resolve to Material."""
    from phast.config import load_config, MaterialConfig
    from phast.material import create_material
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        cfg = load_config(path)
    finally:
        os.unlink(path)

    # Mirror the resolution order used by config.resolve_config (without
    # touching mesh/BCs, which require gmsh and named node sets).
    mat_overrides = dict(cfg.material.overrides)
    inline_names = (
        'E', 'nu', 'Gc', 'l0', 'rho', 'eta_residual',
        'energy_split', 'pf_model', 'kinematics', 'plane_stress',
        'driving_force', 'cubic_s', 'sigma_ts', 'pfczm_p',
        'pfczm_softening',
    )
    for name in inline_names:
        v = getattr(cfg.material, name, None)
        if v is not None:
            mat_overrides[name] = v
    return cfg, create_material(preset=cfg.material.preset, **mat_overrides)


class TestInlineMaterial:
    """Inline material specification — issue #137."""

    INLINE_YAML = """
name: inline-only
material:
  E: 6.0e+9
  nu: 0.22
  Gc: 2280
  l0: 0.25e-3
  energy_split: spectral
  pf_model: AT2
  plane_stress: true
  eta_residual: 1.0e-7
  sigma_ts: 11.31
"""

    LEGACY_YAML = """
name: legacy preset+overrides
material:
  preset: glass_borden
  overrides:
    E: 6.0e+9
    nu: 0.22
    Gc: 2280
    l0: 0.25e-3
    energy_split: spectral
    pf_model: AT2
    plane_stress: true
    eta_residual: 1.0e-7
"""

    INLINE_OVERRIDES_YAML = """
name: inline beats overrides
material:
  preset: glass_borden
  overrides:
    l0: 1.0           # legacy override
    E: 1.0e+9          # should be beaten by inline
  l0: 0.25e-3         # inline wins over override
  E: 6.0e+9            # inline wins over override
"""

    def test_inline_only_no_preset_required(self):
        """A YAML block with only inline fields must produce a valid Material."""
        cfg, mat = _build_material_from_yaml(self.INLINE_YAML)
        assert cfg.material.preset is None
        assert mat.E == pytest.approx(6.0e+9)
        assert mat.nu == pytest.approx(0.22)
        assert mat.Gc == pytest.approx(2280)
        assert mat.l0 == pytest.approx(0.25e-3)
        assert mat.energy_split == 'spectral'
        assert mat.pf_model == 'AT2'
        assert mat.plane_stress is True
        assert mat.eta_residual == pytest.approx(1.0e-7)
        assert mat.sigma_ts == pytest.approx(11.31)

    def test_legacy_preset_overrides_still_works(self):
        """Existing preset+overrides syntax must keep working unchanged."""
        cfg, mat = _build_material_from_yaml(self.LEGACY_YAML)
        assert cfg.material.preset == 'glass_borden'
        assert mat.E == pytest.approx(6.0e+9)
        assert mat.l0 == pytest.approx(0.25e-3)
        assert mat.energy_split == 'spectral'
        assert mat.plane_stress is True

    def test_inline_and_legacy_produce_identical_material(self):
        """Same numeric inputs via either route should yield the same Material."""
        _, m_inline = _build_material_from_yaml(self.INLINE_YAML)
        _, m_legacy = _build_material_from_yaml(self.LEGACY_YAML)
        for attr in ('E', 'nu', 'Gc', 'l0', 'eta_residual',
                     'energy_split', 'pf_model', 'plane_stress'):
            assert getattr(m_inline, attr) == getattr(m_legacy, attr), attr

    def test_inline_beats_overrides_and_preset(self):
        """Resolution order: preset < overrides < inline."""
        _, mat = _build_material_from_yaml(self.INLINE_OVERRIDES_YAML)
        # Inline values must win over the override dict
        assert mat.E == pytest.approx(6.0e+9)
        assert mat.l0 == pytest.approx(0.25e-3)

    def test_existing_configs_still_load(self):
        """Every shipped config still loads without error."""
        import glob
        from phast.config import load_config
        files = sorted(glob.glob('configs/*.yaml'))
        assert files, 'No configs found — test setup is wrong'
        for path in files:
            cfg = load_config(path)
            assert cfg.material is not None, path


class TestLiuStructuredMesh:
    """Smoke tests for the Liu-style structured split-quad B1 mesh."""

    def test_generator_writes_triangle_mesh_with_crack_set(self, tmp_path):
        from phast.mesh import FEMMesh
        from phast.mesh_generator import rectangular_sent_liu_structured

        mesh_path = tmp_path / "liu_structured.msh"
        rectangular_sent_liu_structured(
            str(mesh_path),
            W=2.0,
            H=1.0,
            a=1.0,
            l0=0.25,
            h_crack=0.25,
            verbose=False,
        )
        mesh = FEMMesh(str(mesh_path), device='cpu')

        assert mesh.elements.shape[1] == 3
        assert mesh.n_elems == 2 * 8 * 4
        for name in ("bottom", "top", "left", "right", "crack"):
            assert name in mesh.node_sets
            assert mesh.node_sets[name].numel() > 0

    def test_liu_structured_config_loads(self):
        from phast.config import get_geometry_registry, load_config

        cfg = load_config('configs/B1_branching_glass_liu_structured.yaml')
        assert cfg.geometry.type == 'rectangular_sent_liu_structured'
        assert cfg.solver.damping_ratio_max == pytest.approx(0.10)
        assert 'rectangular_sent_liu_structured' in get_geometry_registry()

    def test_preseed_unit_strings_use_resolved_material(self, tmp_path):
        """Regression for unit-suffixed Gc/l0 with preseed_damage."""
        from phast.config import load_config, resolve_config
        from phast.initial_conditions import value_to_H_seed

        mesh_path = tmp_path / "liu_structured.msh"
        cfg_path = tmp_path / "preseed_units.yaml"
        cfg_path.write_text(f"""
schema_version: 1
problem: {{name: preseed units}}
geometry:
  type: rectangular_sent_liu_structured
  parameters:
    output_path: {mesh_path}
    W: 2.0
    H: 1.0
    a: 1.0
    l0: 0.25
    h_crack: 0.25
material:
  E: "32 GPa"
  nu: 0.2
  Gc: "3 J/m^2"
  l0: "0.25 mm"
  rho: "2450 kg/m^3"
boundary_conditions:
- {{nodes: left, type: fix, component: 0}}
- {{nodes: crack, type: pf_dirichlet, value: 1.0}}
initial_conditions:
  preseed_notch_nodesets: [crack]
solver:
  solver_type: explicit
device:
  device: cpu
acceptance:
  status: beta
  required_outputs: [run_lockfile.json]
""")
        cfg = load_config(str(cfg_path))
        assert cfg.acceptance == {
            'status': 'beta',
            'required_outputs': ['run_lockfile.json'],
        }
        mat = resolve_config(cfg)['material']

        assert isinstance(mat.Gc, float)
        assert isinstance(mat.l0, float)
        assert value_to_H_seed(1.0, mat.Gc, mat.l0) > 0.0


class TestComsolStructuredMesh:
    """Smoke tests for the COMSOL-style structured split-quad B7 mesh."""

    def test_generator_writes_half_plate_boundary_sets(self, tmp_path):
        from phast.mesh import FEMMesh
        from phast.mesh_generator import rectangular_sent_comsol_structured

        mesh_path = tmp_path / "comsol_structured.msh"
        rectangular_sent_comsol_structured(
            str(mesh_path),
            W=2.0,
            H=1.0,
            a=1.0,
            l0=0.5,
            h_crack=0.25,
            verbose=False,
        )
        mesh = FEMMesh(str(mesh_path), device='cpu')

        assert mesh.elements.shape[1] == 3
        assert mesh.n_elems == 2 * 8 * 4
        for name in ("bottom_sym", "top", "left", "right", "crack"):
            assert name in mesh.node_sets
            assert mesh.node_sets[name].numel() > 0

    def test_comsol_structured_config_loads(self):
        from phast.config import get_geometry_registry, load_config

        cfg = load_config('configs/B7_debug/B7_half_plate_comsol_structured.yaml')
        assert cfg.geometry.type == 'rectangular_sent_comsol_structured'
        assert cfg.material.energy_split == 'amor'
        assert cfg.solver.damage_every == 2
        assert 'rectangular_sent_comsol_structured' in get_geometry_registry()
