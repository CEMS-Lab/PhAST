"""
YAML-based configuration system for phast.

Provides dataclass-based configuration that can be loaded from / saved to YAML,
and a resolver that converts config into solver-ready objects (mesh, material,
BCs, SolverConfig, DeviceContext).

Usage::

    from phast.config import load_config, resolve_config

    cfg = load_config('benchmarks/miehe_tension.yaml')
    objs = resolve_config(cfg)
    solver = StaggeredSolver(objs['mesh'], objs['material'], objs['bcs'],
                             objs['solver_config'], objs['ctx'])

Dependencies: PyYAML (already present via h5py).
"""

import os
import yaml
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Dataclass config hierarchy
# ---------------------------------------------------------------------------

@dataclass
class GeometryConfig:
    """Mesh geometry specification.

    type : str
        Name of a mesh_generator function (e.g. 'miehe_tension').
        Ignored if ``mesh_path`` is set.
    parameters : dict
        Keyword arguments forwarded to the generator function.
    mesh_path : str or None
        Path to an external mesh file (.msh, .inp, .bdf, .vtu, etc.).
        If set, bypasses the generator and loads this file directly.
        Abaqus .inp files preserve node sets (*NSET).
    """
    type: str = 'miehe_tension'
    parameters: dict = field(default_factory=dict)
    mesh_path: Optional[str] = None

    # --- Phase 2.1 (issue #142): declarative primitive vocabulary -------
    # ``primitives`` carries the YAML mapping {name: {type, ...}, ...}
    # untouched; it is parsed lazily via geometry_dsl.parse_primitives so
    # that loading the config does not pull in the geometry parser.
    # ``units`` controls how primitive coordinates are interpreted on
    # parse (internally everything is stored in mm).
    # The classic ``type``/``parameters``/``mesh_path`` paths are
    # unchanged. ``primitives`` is mutually exclusive with ``type``
    # being explicitly set (see resolve_config).
    units: str = 'mm'
    primitives: Optional[dict] = None

    # --- Phase 2.2 (issue #143): boolean-op recipe over named primitives -
    # ``domain`` carries the YAML mapping {base, subtract, add, intersect}
    # untouched; it is parsed lazily via geometry_dsl.parse_domain in
    # resolve_config, which validates that every referenced primitive
    # name resolves and stashes the parsed Domain on the dataclass as
    # ``_parsed_domain`` for the geometry compiler in #146 to consume.
    domain: Optional[dict] = None

    # --- Phase 2.3 (issue #144): explicit named groups ------------------
    # ``named_groups`` is a YAML mapping of group_name -> spec, where
    # each spec is exactly one of:
    #   {primitive: <name>, kind: <boundary|interior|centre>}
    #   {point:   [x, y]}
    #   {region:  {<inline primitive spec>}}
    # Auto-exposed selectors (``<primitive>.<kind>``) are *implicit* and
    # need not appear here. Resolution against mesh entities lands in
    # issue #146; parse-time validation lives in geometry_dsl.
    named_groups: Optional[dict] = None

    # --- Phase 2.4 (issue #145): mesh refinement DSL --------------------
    # Optional ``mesh:`` block declaring a default element size and a
    # list of region-keyed refinement rules. The block is purely
    # additive: existing configs continue to work. The parsed
    # representation is stashed on the config as ``_parsed_mesh`` by
    # ``resolve_config`` for downstream consumers (the geometry compiler
    # in #146).
    mesh: Optional[dict] = None


@dataclass
class MaterialConfig:
    """Material specification — preset-based or fully inline.

    Three ways to define a material (highest priority wins):

    1. ``preset`` (legacy): named preset from ``create_material`` registry.
    2. ``overrides`` (legacy): dict layered on top of the preset.
    3. Inline top-level fields (E, nu, Gc, l0, energy_split, pf_model,
       plane_stress, eta_residual, rho, kinematics, ...): layered on top
       of ``overrides``.

    Either route — or a mix — is valid. ``preset`` is no longer required;
    a YAML block containing only inline fields produces a Material directly.

    Parameters
    ----------
    preset : str or None
        Optional preset name passed to ``create_material``. If ``None``,
        the material is built entirely from inline fields (and overrides,
        if any).
    overrides : dict
        Legacy keyword overrides applied on top of the preset.
    E, nu, Gc, l0, rho, eta_residual : float or None
        Inline material properties. ``None`` means "not specified inline".
    energy_split, pf_model, kinematics : str or None
        Inline string options.
    plane_stress : bool or None
        Inline plane-stress flag.
    """
    preset: Optional[str] = None
    overrides: dict = field(default_factory=dict)
    # Inline material fields (None = not specified, fall through to preset/overrides)
    E: Optional[float] = None
    nu: Optional[float] = None
    Gc: Optional[float] = None
    l0: Optional[float] = None
    rho: Optional[float] = None
    eta_residual: Optional[float] = None
    energy_split: Optional[str] = None
    pf_model: Optional[str] = None
    kinematics: Optional[str] = None
    plane_stress: Optional[bool] = None
    driving_force: Optional[str] = None  # 'strain_energy' | 'principal_stress' (issue #248)
    cubic_s: Optional[float] = None
    sigma_ts: Optional[float] = None
    pfczm_p: Optional[int] = None
    pfczm_softening: Optional[str] = None


@dataclass
class BoundaryConditionEntry:
    """A single boundary condition entry from the YAML file.

    The ``type`` field selects the dispatch path; not every other field is
    meaningful for every type.

    Common fields
    -------------
    nodes : str
        Node set name from the mesh (e.g. 'bottom', 'top').
    type : str
        One of: ``fix``, ``prescribe``, ``neumann``, ``traction``,
        ``symmetry``, ``rigid_connector``.
    component : int
        DOF component: 0 = x, 1 = y. Used by ``fix``/``prescribe``/
        ``neumann``/``traction``.
    value : float
        Prescribed displacement (``prescribe``, internal unit mm) or
        traction magnitude (``neumann``/``traction``, internal unit N/mm)
        on the chosen component. Unit-suffixed YAML strings such as
        ``"0.01 mm"`` and ``"1 MPa"`` are accepted; stress-like traction
        suffixes assume unit out-of-plane thickness.

    ``traction``-only fields
    ------------------------
    ramp_type : str
        ``constant`` | ``linear`` | ``smooth_step`` | ``cosine``.
        Default ``constant`` reproduces legacy ``neumann`` behaviour.
    t_ramp : float
        Ramp duration in seconds. Ignored for ``constant``.
    t_hold : float
        Optional hold time. Defaults to ``t_ramp``. For ``cosine`` this is
        the full ramp period.

    ``symmetry``-only fields
    ------------------------
    axis : str
        ``'x'`` or ``'y'``. Names the *normal* component to suppress: an
        edge parallel to x has axis ``y`` (fix v=0); an edge parallel to
        y has axis ``x`` (fix u=0). Shorthand for
        ``type: fix, component: <axis>``.

    ``rigid_connector``-only fields
    -------------------------------
    master : str
        Node-set name (typically containing a single node) for the master
        / control point. The first node of the set is used.
    dofs : list of str
        Components to lock, e.g. ``['x', 'y']``.
    prescribe : dict
        Per-component prescribed displacement, e.g. ``{y: 1.0}``. Missing
        components default to 0.
    rotation_free : bool
        If True (default), the master carries a free rotation DOF and
        slave nodes are tied via the linearised rigid-body constraint
        (full master-slave elimination). If False, falls back to the
        legacy "welded" behaviour from PR #155 where all slave + master
        DOFs are locked to the prescribed translation. The default
        flipped to True in #154; configs that depended on the welded
        behaviour must set ``rotation_free: false`` explicitly.

    Notes
    -----
    The full ``rigid_connector`` MPC (rotation_free=True) is enforced via
    T-matrix master-slave elimination across all three solver paths:
    static ``DirectSolver`` (PR #164), explicit-dynamic ``ExplicitDynamics``
    (PR #174), and iterative-CG ``SecantCGSolver`` (PR #182). The legacy
    welded behaviour (PR #155) is preserved as an opt-in fallback via
    ``rotation_free: false``.
    """
    nodes: str = ''
    type: str = 'fix'
    component: int = 0
    value: float = 0.0
    # traction-only
    ramp_type: str = 'constant'
    t_ramp: float = 0.0
    t_hold: Optional[float] = None
    # symmetry-only
    axis: Optional[str] = None
    # rigid_connector-only
    master: Optional[str] = None
    dofs: Optional[List[str]] = None
    prescribe: Optional[dict] = None
    rotation_free: bool = True


@dataclass
class LoadingConfig:
    """Loading protocol and time-varying BC schedule.

    protocol : str
        'simple', 'two_step_prestrain', or 'cyclic'.
    ramp_type : str
        How ``load_factor`` evolves in time (explicit dynamics):

        - ``'constant'``  — load_factor = 1.0 from step 0 (e.g. constant traction).
        - ``'linear'``    — load_factor = min(t / t_ramp, 1.0).
        - ``'smooth'``    — load_factor = 0.5*(1 - cos(pi * t / t_ramp)).
        - ``'smooth_step'`` — Hermite cubic 3*s^2 - 2*s^3 with
          s = clamp(t / t_ramp, 0, 1). Matches the COMSOL ``step1`` smooth
          step used in the Geomechanics dynamic-fracture tutorial; differs
          slightly from cosine in HF content at very short ramps (issue
          #246, B7 #213).
        - ``'velocity_impact'`` — displacement from velocity ramp:
          u(t) = v0*t^2/(2*t_ramp)  for t < t_ramp,
          u(t) = v0*(t - t_ramp/2)  for t >= t_ramp.
          (v0 in m/s, converted to mm/s internally since meshes are in mm.)

        For quasi-static solvers, load_factor is set from the loading
        schedule array and ramp_type is ignored.
    """
    protocol: str = 'simple'
    ramp_type: str = 'constant'
    num_steps: int = 100
    t_total: float = 0.0
    dt: float = 1e-5
    t_ramp: float = 0.0
    v0: float = 0.0
    disp_max: float = 0.0
    prestrain_displacement: float = 0.0
    coupled_prestrain: bool = False
    # Cyclic loading: comma-separated "target:steps" pairs.
    # Example: "0.3:100,-0.2:100,1.0:1200" means
    #   phase 1: ramp to 0.3 mm in 100 steps
    #   phase 2: ramp to -0.2 mm in 100 steps
    #   phase 3: ramp to 1.0 mm in 1200 steps
    cyclic_phases: str = ''


def compute_load_factor(step: int, dt: float, loading: LoadingConfig) -> float:
    """Compute the BC load_factor for a given time step.

    For explicit dynamics, this scales all prescribed displacement BCs.
    For constant Neumann (traction) problems, use ramp_type='constant'.
    """
    import math
    t = step * dt
    rt = loading.ramp_type

    if rt == 'constant':
        return 1.0

    elif rt == 'linear':
        if loading.t_ramp <= 0:
            return 1.0
        return min(t / loading.t_ramp, 1.0)

    elif rt == 'smooth':
        if loading.t_ramp <= 0 or t >= loading.t_ramp:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * t / loading.t_ramp))

    elif rt == 'smooth_step':
        # Hermite cubic 3*s^2 - 2*s^3 (COMSOL step1). C^1-continuous
        # at both endpoints; same monotone 0->1 shape as 'smooth' but
        # with slightly different HF content (matters at ~50 ns ramps,
        # see B7 issue #213).
        if loading.t_ramp <= 0 or t >= loading.t_ramp:
            return 1.0
        if t <= 0:
            return 0.0
        s = t / loading.t_ramp
        return s * s * (3.0 - 2.0 * s)

    elif rt == 'velocity_impact':
        v0_mm = loading.v0 * 1e3  # m/s → mm/s (mesh units)
        tr = loading.t_ramp
        if tr <= 0:
            return v0_mm * t
        if t < tr:
            return v0_mm / (2.0 * tr) * t * t
        return v0_mm * (t - tr / 2.0)

    return 1.0


def build_cyclic_schedule(loading: LoadingConfig) -> list:
    """Build a quasi-static loading schedule from cyclic_phases string.

    Format: "target1:steps1,target2:steps2,..."
    Returns a list of displacement values, one per step.
    """
    if not loading.cyclic_phases:
        return []
    schedule = []
    current = 0.0
    for phase_str in loading.cyclic_phases.split(','):
        target_str, steps_str = phase_str.strip().split(':')
        target = float(target_str)
        n = int(steps_str)
        for i in range(1, n + 1):
            val = current + (target - current) * i / n
            schedule.append(val)
        current = target
    return schedule


@dataclass
class SolverSettings:
    """Solver hyper-parameters.

    Maps closely to ``SolverConfig`` from staggered_solver but adds a few
    convenience fields (H_cap_factor, eta_residual forwarded to material).
    """
    solver_type: str = 'explicit'
    # Dynamic time integrator for solver_type='explicit'.  The default
    # central_difference path is the existing Velocity-Verlet / explicit
    # Newmark update.  generalized_alpha is the forward-only implicit
    # COMSOL-style option tracked in #570.
    time_integrator: str = 'central_difference'
    rho_inf: float = 0.5
    dt_safety: float = 0.8
    stagger_tol: float = 1e-6
    max_stagger: int = 500
    stagger_criterion: str = 'relative'
    # Norm for the relative stagger convergence check ('l2' default;
    # 'linf' uses max-norm, Bleyer & Roux-Langlois 2017 -- issue #244).
    stagger_norm: str = 'l2'
    anderson_depth: int = 0
    adaptive_stagger_tol: bool = False
    use_multigrid: bool = True
    preconditioner: Optional[str] = None
    H_cap_factor: float = 0.0
    damage_tol: float = 1e-5
    static_tol: float = 1e-8
    bounds_method: str = 'post_clamp'
    damage_every: int = 3
    # Issue #299 -- Verlet ordering: corrector damage freshness.
    # See SolverConfig docstring in staggered_solver.py. Default False
    # preserves the legacy "lagged d_n in corrector" Borden 2012 ordering
    # (bit-exact backward compatibility). Set True to opt into a PhaFiDyn-style
    # segregated "fresh d_{n+1} in corrector" ordering for dynamic-fracture
    # sensitivity checks.
    fresh_d_in_corrector: bool = False
    damage_max_iter: int = 5000
    static_max_iter: int = 5000
    softmax_H_beta: Optional[float] = None
    H_update_method: str = 'hard_max'
    enable_damage: bool = True
    fail_on_mechanics_nonconvergence: bool = True
    fail_on_stagger_nonconvergence: bool = True
    adaptive_dt: bool = False
    adaptive_dt_d_threshold: float = 0.01
    # Default mirrors ``Material.eta_residual`` (1e-7) — material.py is the
    # source of truth for the residual-stiffness floor. Audit T1.1 (W4
    # SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07) flagged a 10x discrepancy
    # between this default and ``Material.eta_residual`` that produced
    # different effective physics for direct-instantiated solvers vs
    # YAML-driven runs.
    eta_residual: float = 1e-7
    # Kelvin-Voigt stiffness-proportional damping ratio at omega_max.
    # Zero = pure velocity-Verlet (current default); >0 adds HF damping
    # that mirrors the numerical dissipation in Borden 2012's
    # generalized-alpha with rho_inf=0.5.  See ExplicitDynamics
    # docstring for details.
    damping_ratio_max: float = 0.0
    # Linear-solver backend for QuasiStaticSolver. Forwarded to
    # SolverConfig.backend -> QuasiStaticSolver(backend=...). One of
    # {'auto', 'scipy', 'mumps', 'cg'}. (#196)
    backend: str = 'auto'


@dataclass
class OutputConfig:
    """Output / IO settings."""
    output_dir: Optional[str] = None
    # Deprecated name kept for YAML/CLI compatibility. Prefer
    # ``trajectory: true`` plus ``trajectory_format`` for new configs.
    h5: bool = False
    # Public spelling for enabling trajectory snapshots.
    trajectory: bool = False
    # Preferred trajectory backend is Zarr; H5 is legacy compatibility.
    trajectory_format: str = 'zarr'
    # Snapshot cadence for trajectory outputs.
    h5_every: int = 20
    vtu: bool = False
    vtu_every: int = 10
    # Visualisation backend. 'vtu' is the ParaView-native default;
    # 'pv' uses the pyvista-zstd binary format (~43-90x faster writes,
    # but ParaView cannot open .pv directly). Requires the optional
    # viz-fast extras: pip install phast[viz-fast].
    viz_format: str = 'vtu'
    gif: bool = False
    gif_frames: int = 200
    # Comma-separated animation fields for postprocess_paper:
    # damage, stress/max_principal_stress, displacement.
    gif_fields: str = 'damage'
    # Animation container used by postprocess_paper for H5-derived
    # animations. MP4 + raster is the default screening path because it avoids
    # repeated Matplotlib triangle redraws and produces small review videos.
    # GIF/APNG and the Matplotlib renderer remain available for compatibility.
    animation_format: str = 'mp4'
    animation_renderer: str = 'raster'
    animation_raster_width: int = 960
    plots: bool = False
    profile: bool = False
    fast: bool = False
    print_every: int = 100
    # Reaction-force logging (quasi-static benchmarks). When set, the YAML
    # driver writes ``results.csv`` with columns ``step, time,
    # displacement, reaction_kN, max_d, max_H, stagger_iter, elapsed_ms``,
    # summing the internal
    # reaction at every node in ``reaction_node_set`` along
    # ``reaction_component`` (0 = x, 1 = y). The "displacement" column
    # records the prescribed BC value at the same node-set/component (i.e.
    # ``load_factor * value`` for that BC entry); the load-displacement
    # plot in compare.py reads this file. Set to None / unset to disable.
    reaction_node_set: Optional[str] = None
    reaction_component: int = 1


@dataclass
class DeviceConfig:
    """Device and compilation settings."""
    device: Optional[str] = None
    compile: bool = False


@dataclass
class InitialConditionsConfig:
    """Initial-state overrides applied after solver construction.

    preseed_notch_nodesets
        List of mesh node-set names (gmsh physical curves) whose
        one-ring of elements is pre-damaged to d = 1 at t = 0 via a
        large initial value of the history variable H. This is the
        standard phase-field convention for pre-existing cracks
        (Borden 2012 Fig 13, Bleyer 2017 etc.): the notch is real
        broken material, not just a geometric slit. Without it, AT2
        (which has no elastic threshold) accumulates damage on the
        free walls from reflected stress waves and produces
        spurious bands.

    preseed_damage
        General preseed list. Each entry is a dict with either
        ``nodes: <name>`` (named mesh node-set) or
        ``region: {type: ..., ...}`` (geometric predicate), plus an
        optional scalar ``value`` (default 1.0). Supported region types:

        - ``line_segment`` — ``from``, ``to``, optional ``thickness``
          (defaults to mean mesh edge length).
        - ``rectangle`` — ``origin`` (lower-left), ``size`` ``[w, h]``.
        - ``circle`` — ``center``, ``radius``.
        - ``polygon`` — ``vertices`` list (ray-cast point-in-polygon).

        Overlapping entries take the per-node maximum value. Legacy
        ``preseed_notch_nodesets`` is auto-converted to this form so
        existing configs keep working.
    """
    preseed_notch_nodesets: Optional[List[str]] = None
    preseed_damage: Optional[List[Dict[str, Any]]] = None


@dataclass
class ProblemConfig:
    """Top-level problem configuration.

    All sub-configs default to ``None`` and are filled with defaults during
    loading if absent from the YAML file.
    """
    schema_version: int = 1
    name: str = 'Phase-Field Problem'
    reference: str = ''
    geometry: Optional[GeometryConfig] = None
    material: Optional[MaterialConfig] = None
    boundary_conditions: Optional[List[BoundaryConditionEntry]] = None
    loading: Optional[LoadingConfig] = None
    solver: Optional[SolverSettings] = None
    output: Optional[OutputConfig] = None
    device: Optional[DeviceConfig] = None
    initial_conditions: Optional[InitialConditionsConfig] = None
    # Free-form sections used by example / inversion drivers. They are
    # not consumed by the core run_config pipeline but appear in
    # several shipped configs (autograd_kalthoff, spatial_gc_*, all
    # B*_*.yaml). Declared so schema validation accepts them.
    example: Optional[str] = None
    inversion: Optional[dict] = None
    acceptance: Optional[dict] = None


# ---------------------------------------------------------------------------
# Geometry registry  (mesh_generator function name -> callable)
# ---------------------------------------------------------------------------

def _build_geometry_registry() -> dict:
    """Lazily import mesh_generator functions and return name -> callable map."""
    from ..core import mesh_generator as mg
    registry = {}
    _names = [
        'miehe_tension',
        'miehe_shear',
        'square_plate',
        'three_point_bending',
        'l_shaped_panel',
        'plate_with_holes',
        'bazant_gap_test',
        'rectangular_sent',
        'rectangular_sent_comsol_structured',
        'rectangular_sent_liu_structured',
        'rectangular_sent_q4_structured',
        'kalthoff_winkler',
        'glass_impact_vnotch',
        'perforated_sent',
    ]
    for name in _names:
        fn = getattr(mg, name, None)
        if fn is not None:
            registry[name] = fn
    return registry


GEOMETRY_REGISTRY: Optional[dict] = None


def get_geometry_registry() -> dict:
    """Return (and cache) the geometry name -> function map."""
    global GEOMETRY_REGISTRY
    if GEOMETRY_REGISTRY is None:
        GEOMETRY_REGISTRY = _build_geometry_registry()
    return GEOMETRY_REGISTRY


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, d: dict):
    """Construct a dataclass instance from a dict, ignoring unknown keys.

    Coerces string values to float/int where the dataclass field expects them
    (handles YAML parsing '1e-3' as string instead of float). Where a field
    is registered as a unit-aware quantity (see ``units.py``), unit-suffixed
    strings (e.g. ``"1 us"``, ``"16.5 m/s"``, ``"0.01 mm"``, ``"1 MPa"``)
    are normalised to the solver's internal unit system before float
    coercion. Bare floats pass through unchanged so legacy configs keep
    bit-identical values.
    """
    if d is None:
        return cls()
    from ..utils.units import (
        parse_quantity,
        LOADING_QUANTITY_KINDS,
        BOUNDARY_VALUE_QUANTITY_KINDS,
        BOUNDARY_TIME_QUANTITY_KINDS,
    )
    known = {f.name: f for f in fields(cls)}
    coerced = {}
    bc_type = str(d.get('type', 'fix')) if cls is BoundaryConditionEntry else None
    for k, v in d.items():
        if k not in known:
            continue
        fld = known[k]
        # Coerce strings to numeric types. Match both raw types (float/int)
        # and stringified annotations (including 'Optional[float]', etc.)
        type_str = fld.type if isinstance(fld.type, str) else str(fld.type)
        # Unit-aware parse for known quantity fields (loading.t_total, etc.)
        # Runs first; on ValueError fall through to plain numeric coercion
        # (handles e.g. "1e-3").
        if isinstance(v, str) and k in LOADING_QUANTITY_KINDS and \
                (fld.type in (float, 'float') or 'float' in type_str):
            try:
                v = parse_quantity(v, LOADING_QUANTITY_KINDS[k])
            except ValueError:
                pass
        if cls is BoundaryConditionEntry:
            bc_kind = None
            if isinstance(v, str) and k == 'value':
                bc_kind = BOUNDARY_VALUE_QUANTITY_KINDS.get(bc_type)
            elif isinstance(v, str) and k in BOUNDARY_TIME_QUANTITY_KINDS:
                bc_kind = BOUNDARY_TIME_QUANTITY_KINDS[k]
            if bc_kind is not None:
                try:
                    v = parse_quantity(v, bc_kind)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid unit value for boundary_conditions.{k} "
                        f"on type {bc_type!r}: {v!r}"
                    ) from exc
            elif k == 'prescribe' and isinstance(v, dict):
                parsed = {}
                for dof, raw_value in v.items():
                    if isinstance(raw_value, str):
                        try:
                            parsed[dof] = parse_quantity(raw_value, 'length')
                        except ValueError as exc:
                            raise ValueError(
                                "Invalid unit value for "
                                f"boundary_conditions.prescribe[{dof!r}]: "
                                f"{raw_value!r}"
                            ) from exc
                    else:
                        parsed[dof] = raw_value
                v = parsed
        if isinstance(v, str):
            if fld.type in (float, 'float') or 'float' in type_str:
                try:
                    v = float(v)
                except ValueError:
                    pass
            elif fld.type in (int, 'int') or 'int' in type_str:
                try:
                    v = int(v)
                except ValueError:
                    pass
        coerced[k] = v
    return cls(**coerced)


def _ensure_defaults(cfg: ProblemConfig) -> ProblemConfig:
    """Fill in None sub-configs with their default instances."""
    if cfg.geometry is None:
        cfg.geometry = GeometryConfig()
    if cfg.material is None:
        cfg.material = MaterialConfig()
    if cfg.boundary_conditions is None:
        cfg.boundary_conditions = []
    if cfg.loading is None:
        cfg.loading = LoadingConfig()
    if cfg.solver is None:
        cfg.solver = SolverSettings()
    if cfg.output is None:
        cfg.output = OutputConfig()
    if cfg.device is None:
        cfg.device = DeviceConfig()
    return cfg


def load_config(yaml_path: str) -> ProblemConfig:
    """Load a ProblemConfig from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    -------
    ProblemConfig
        Fully populated configuration with defaults for missing sections.
    """
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}

    # Top-level scalars (support both flat and nested 'problem:' section)
    prob = raw.get('problem', {})
    cfg = ProblemConfig(
        schema_version=raw.get('schema_version', 1),
        name=prob.get('name', raw.get('name', 'Phase-Field Problem')),
        reference=prob.get('reference', raw.get('reference', '')),
    )

    # Nested sub-configs
    cfg.geometry = _dict_to_dataclass(GeometryConfig, raw.get('geometry'))
    # Track whether the user explicitly set ``geometry.type`` so we can
    # reliably distinguish the legacy generator path from the new
    # primitive-vocabulary path (issue #142). The dataclass default for
    # ``type`` is non-empty, so we can't infer this after the fact.
    _raw_geom = raw.get('geometry') or {}
    cfg.geometry._type_explicit = 'type' in _raw_geom  # type: ignore[attr-defined]
    cfg.material = _dict_to_dataclass(MaterialConfig, raw.get('material'))
    cfg.loading = _dict_to_dataclass(LoadingConfig, raw.get('loading'))
    cfg.solver = _dict_to_dataclass(SolverSettings, raw.get('solver'))
    output_raw = raw.get('output') or {}
    if isinstance(output_raw, dict) and output_raw.get('trajectory') and \
            'h5' not in output_raw:
        output_raw = dict(output_raw)
        output_raw['h5'] = output_raw['trajectory']
    cfg.output = _dict_to_dataclass(OutputConfig, output_raw)
    if cfg.output.trajectory:
        cfg.output.h5 = True
    cfg.output.trajectory = bool(cfg.output.h5)
    cfg.device = _dict_to_dataclass(DeviceConfig, raw.get('device'))
    cfg.initial_conditions = _dict_to_dataclass(
        InitialConditionsConfig, raw.get('initial_conditions'))
    if 'example' in raw:
        cfg.example = raw['example']
    if 'inversion' in raw:
        cfg.inversion = raw['inversion']
    if 'acceptance' in raw:
        cfg.acceptance = raw['acceptance']

    # Boundary conditions are a list of entries
    bc_list = raw.get('boundary_conditions', [])
    if bc_list:
        cfg.boundary_conditions = [
            _dict_to_dataclass(BoundaryConditionEntry, entry)
            for entry in bc_list
        ]
    else:
        cfg.boundary_conditions = []

    cfg = _ensure_defaults(cfg)

    # Explicit dynamics configs usually specify the physical end time and let
    # the solver derive a stable CFL step count. Preserve explicit
    # ``num_steps`` when present, but do not let the LoadingConfig dataclass
    # default of 100 silently override ``t_total`` for omitted fields.
    raw_loading = raw.get('loading') or {}
    raw_solver = raw.get('solver') or {}
    solver_type = raw_solver.get('solver_type', cfg.solver.solver_type)
    if (solver_type == 'explicit'
            and 'num_steps' not in raw_loading
            and cfg.loading.t_total > 0.0):
        cfg.loading.num_steps = 0

    # Resolve relative mesh_path against the YAML file's directory
    if cfg.geometry.mesh_path and not os.path.isabs(cfg.geometry.mesh_path):
        yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
        candidate = os.path.join(yaml_dir, cfg.geometry.mesh_path)
        if os.path.exists(candidate):
            cfg.geometry.mesh_path = candidate

    return cfg


def _dataclass_to_dict(obj):
    """Recursively convert a dataclass (or list thereof) to a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj


def save_config(config: ProblemConfig, yaml_path: str):
    """Save a ProblemConfig to a YAML file.

    Parameters
    ----------
    config : ProblemConfig
        The configuration to serialize.
    yaml_path : str
        Destination path.
    """
    d = _dataclass_to_dict(config)
    os.makedirs(os.path.dirname(yaml_path) or '.', exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Config-to-objects resolver
# ---------------------------------------------------------------------------

def _resolve_material(config: ProblemConfig):
    """Build the Material object from preset, overrides, and inline fields."""
    from ..physics.material import create_material

    # Resolution order (highest priority last):
    #   1. preset values (if preset given)
    #   2. overrides dict
    #   3. inline top-level material fields (E, nu, Gc, l0, ...)
    # Material overrides may include unit-suffixed strings ("32 GPa",
    # "3 J/m^2", "0.25 mm", "2450 kg/m^3"). create_material normalises
    # them to the internal unit system; bare floats pass through.
    mat_overrides = dict(config.material.overrides)
    _inline_field_names = (
        'E', 'nu', 'Gc', 'l0', 'rho', 'eta_residual',
        'energy_split', 'pf_model', 'kinematics', 'plane_stress',
        'driving_force', 'cubic_s', 'sigma_ts', 'pfczm_p',
        'pfczm_softening',
    )
    for name in _inline_field_names:
        v = getattr(config.material, name, None)
        if v is not None:
            mat_overrides[name] = v
    # Forward eta_residual from solver settings if not yet specified.
    if 'eta_residual' not in mat_overrides:
        mat_overrides['eta_residual'] = config.solver.eta_residual
    return create_material(preset=config.material.preset, **mat_overrides)


def _runtime_device_for_material(requested_device: Optional[str], material) -> Optional[str]:
    """Apply material-aware device fallbacks before tensors are allocated."""
    if requested_device is None:
        return requested_device

    try:
        import torch
        dev = torch.device(requested_device)
    except (TypeError, RuntimeError):
        return requested_device

    spectral_splits = {'spectral', 'spectral_stress'}
    if dev.type == 'mps' and material.energy_split in spectral_splits:
        print(
            "[device] energy_split="
            f"{material.energy_split!r} requested on MPS. "
            "Routing the full solve to CPU float64 because Apple MPS lacks "
            "float64 and spectral phase-field runs are sensitive to float32 "
            "eigenvalue-sign and history-field noise. Use a non-spectral "
            "split or CUDA for accelerated production runs.",
            flush=True,
        )
        return 'cpu'
    return requested_device


def resolve_config(config: ProblemConfig) -> dict:
    """Resolve a ProblemConfig into solver-ready objects.

    Returns
    -------
    dict with keys:
        'mesh'          : FEMMesh
        'material'      : Material
        'bcs'           : BoundaryConditions
        'solver_config' : SolverConfig
        'ctx'           : DeviceContext
        'loading'       : list of displacement values (QS) or None (explicit)
    """
    from ..core.mesh import FEMMesh
    from ..physics.boundary_conditions import BoundaryConditions
    from ..solvers.staggered_solver import SolverConfig
    from ..utils.device import DeviceContext

    config = _ensure_defaults(config)

    # Build material before DeviceContext so device selection can account
    # for split-specific numerical requirements (#249).
    material = _resolve_material(config)
    runtime_device = _runtime_device_for_material(config.device.device,
                                                  material)

    # --- Device context ---
    ctx = DeviceContext(
        device=runtime_device,
        compile_solvers=config.device.compile,
        profile=config.output.profile,
        energy_split=material.energy_split,
    )

    # --- Geometry / Mesh ---
    geom = config.geometry
    _type_explicit = getattr(geom, '_type_explicit', False)
    _has_primitives = bool(geom.primitives)

    # Phase 2.1 (issue #142): primitive vocabulary is mutually exclusive
    # with the legacy ``type`` generator path.
    if _has_primitives and _type_explicit:
        raise ValueError(
            "geometry.primitives and geometry.type are mutually exclusive. "
            "Use one or the other: 'type' selects a built-in mesh generator, "
            "'primitives' (issue #142) declares a primitive vocabulary that "
            "the geometry compiler (issue #146) will turn into a mesh."
        )

    if _has_primitives and not geom.mesh_path:
        # Validate the primitives now (raises ValueError on bad input).
        from ..core.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups, parse_mesh_dsl,
        )
        geom_dict = {
            'units': geom.units,
            'primitives': geom.primitives,
            'named_groups': geom.named_groups,
        }
        parsed = parse_primitives(geom_dict)
        # Stash the parsed result for downstream consumers and emitters.
        geom._parsed_primitives = parsed  # type: ignore[attr-defined]

        # Phase 2.2 (issue #143): if the user declared a ``domain`` block,
        # parse + validate it against the just-parsed primitives. The
        # compiler in #146 will consume ``_parsed_domain`` to emit the
        # corresponding Gmsh OCC boolean ops.
        if geom.domain is not None:
            parsed_domain = parse_domain(geom.domain, parsed)
            geom._parsed_domain = parsed_domain  # type: ignore[attr-defined]

        # Phase 2.3 (#144): parse the explicit named-group registry.
        # Auto-exposed <primitive>.<kind> selectors stay implicit and
        # are resolved on demand by validate_node_set_name.
        parsed_groups = parse_named_groups(geom_dict, parsed)
        geom._parsed_named_groups = parsed_groups  # type: ignore[attr-defined]

        # Phase 2.4 (issue #145): if a mesh refinement block is present,
        # parse + stash it before handing off to #146.
        parsed_mesh = None
        if geom.mesh is not None:
            parsed_mesh = parse_mesh_dsl(geom.mesh, parsed, units=geom.units)
            geom._parsed_mesh = parsed_mesh  # type: ignore[attr-defined]

        # Phase 2.5 (issue #146): compile to .geo + .msh with hash cache.
        # The compiler stashes the resulting absolute path so the mesh
        # loader below can pick it up via the standard mesh_path branch.
        from ..core.geometry_compiler import compile_geometry
        compiled_msh = compile_geometry(
            primitives=parsed,
            domain=getattr(geom, '_parsed_domain', None),
            named_groups=parsed_groups,
            mesh_dsl=parsed_mesh,
            verbose=False,
        )
        geom._compiled_mesh_path = str(compiled_msh)  # type: ignore[attr-defined]
        mesh = FEMMesh(str(compiled_msh), device=ctx.device, dtype=ctx.dtype)
    elif geom.mesh_path:
        mesh = FEMMesh(geom.mesh_path, device=ctx.device, dtype=ctx.dtype)
    else:
        registry = get_geometry_registry()
        if geom.type not in registry:
            raise ValueError(
                f"Unknown geometry type '{geom.type}'. "
                f"Available: {sorted(registry.keys())}"
            )
        mesh_fn = registry[geom.type]
        mesh_path = mesh_fn(**geom.parameters)
        mesh = FEMMesh(mesh_path, device=ctx.device, dtype=ctx.dtype)

    # Auto-detect boundaries if mesh has no named node sets
    if not mesh.node_sets:
        mesh.node_sets = mesh.identify_boundaries()

    # --- Boundary conditions ---
    bcs = BoundaryConditions(mesh.n_nodes, device=ctx.device, dtype=ctx.dtype)
    for entry in config.boundary_conditions:
        if entry.nodes not in mesh.node_sets:
            available = sorted(mesh.node_sets.keys())
            raise ValueError(
                f"Node set '{entry.nodes}' not found in mesh. "
                f"Available: {available}"
            )
        node_idx = mesh.node_sets[entry.nodes]

        if entry.type == 'fix':
            bcs.fix(node_idx, entry.component)
        elif entry.type == 'prescribe':
            bcs.add(node_idx, entry.component, entry.value)
        elif entry.type == 'neumann':
            # Legacy: constant traction (no time dependence).
            traction = [0.0, 0.0]
            traction[entry.component] = entry.value
            bcs.add_neumann(node_idx, traction)
        elif entry.type == 'traction':
            # New explicit traction with ramp_type / t_ramp / t_hold.
            traction = [0.0, 0.0]
            traction[entry.component] = entry.value
            bcs.add_traction(node_idx, traction,
                             ramp_type=entry.ramp_type,
                             t_ramp=entry.t_ramp,
                             t_hold=entry.t_hold)
        elif entry.type == 'symmetry':
            if entry.axis is None:
                raise ValueError(
                    f"symmetry BC on '{entry.nodes}' requires 'axis: x' or "
                    f"'axis: y' (the displacement component to suppress)."
                )
            bcs.add_symmetry(node_idx, entry.axis)
        elif entry.type == 'rigid_connector':
            if entry.master is None or entry.master not in mesh.node_sets:
                available = sorted(mesh.node_sets.keys())
                raise ValueError(
                    f"rigid_connector requires 'master' to name a mesh node "
                    f"set. Got master={entry.master!r}. Available: {available}"
                )
            master_set = mesh.node_sets[entry.master]
            master_node = int(master_set.flatten()[0].item())
            dof_map = {'x': 0, 'y': 1, 0: 0, 1: 1}
            dofs_in = entry.dofs if entry.dofs is not None else ['x', 'y']
            try:
                locked = [dof_map[d] for d in dofs_in]
            except KeyError as exc:
                raise ValueError(
                    f"rigid_connector dofs must be 'x' or 'y', got "
                    f"{entry.dofs!r}"
                ) from exc
            prescribe = {}
            if entry.prescribe:
                for k, v in entry.prescribe.items():
                    prescribe[dof_map[k]] = float(v)
            bcs.add_rigid_connector(master_node, node_idx, locked,
                                    prescribe=prescribe,
                                    rotation_free=entry.rotation_free)
        elif entry.type == 'pf_dirichlet':
            # Phase-field Dirichlet (issue #213): scalar damage lock.
            # The ``value`` field carries the prescribed phi (default
            # 0.0 from the dataclass — typical use is ``value: 1.0``
            # to model a sharp pre-existing crack).
            bcs.add_pf_dirichlet(node_idx, value=entry.value)
        else:
            raise ValueError(
                f"Unknown BC type '{entry.type}'. Expected one of: "
                f"fix, prescribe, neumann, traction, symmetry, "
                f"rigid_connector, pf_dirichlet."
            )

    # --- Solver config ---
    s = config.solver
    H_cap = None
    if s.H_cap_factor > 0 and material.Gc > 0 and material.l0 > 0:
        # Compute H_cap from factor * (Gc / (2 * l0))
        H_cap = s.H_cap_factor * material.Gc / (2.0 * material.l0)

    solver_dt = config.loading.dt
    if (s.solver_type == 'explicit' and config.loading.num_steps == 0
            and config.loading.t_total > 0):
        # Explicit dynamic validation configs commonly specify a physical
        # end time and ask the solver to derive the stable step count from
        # the CFL limit. In that mode, leave SolverConfig.dt unset so
        # ExplicitDynamics uses fem.dt_cfl * dt_safety (including damping
        # CFL shrinkage) instead of the LoadingConfig placeholder default.
        solver_dt = None

    solver_config = SolverConfig(
        solver_type=s.solver_type,
        time_integrator=s.time_integrator,
        rho_inf=s.rho_inf,
        dt=solver_dt,
        dt_safety=s.dt_safety,
        num_steps=config.loading.num_steps,
        damage_tol=s.damage_tol,
        static_tol=s.static_tol,
        H_cap=H_cap,
        stagger_tol=s.stagger_tol,
        max_stagger=s.max_stagger,
        stagger_criterion=s.stagger_criterion,
        stagger_norm=s.stagger_norm,
        use_multigrid=s.use_multigrid,
        bounds_method=s.bounds_method,
        preconditioner=s.preconditioner,
        anderson_depth=s.anderson_depth,
        adaptive_stagger_tol=s.adaptive_stagger_tol,
        damage_every=s.damage_every,
        fresh_d_in_corrector=s.fresh_d_in_corrector,
        damage_max_iter=s.damage_max_iter,
        static_max_iter=s.static_max_iter,
        softmax_H_beta=s.softmax_H_beta,
        H_update_method=s.H_update_method,
        enable_damage=s.enable_damage,
        fail_on_mechanics_nonconvergence=s.fail_on_mechanics_nonconvergence,
        fail_on_stagger_nonconvergence=s.fail_on_stagger_nonconvergence,
        adaptive_dt=s.adaptive_dt,
        adaptive_dt_d_threshold=s.adaptive_dt_d_threshold,
        dt_cutback_threshold=s.adaptive_dt_d_threshold,
        damping_ratio_max=s.damping_ratio_max,
        backend=s.backend,
        dump_every=config.output.vtu_every if config.output.vtu else 0,
        h5_every=config.output.h5_every if config.output.h5 else 0,
        viz_format=config.output.viz_format,
        print_every=config.output.print_every,
    )

    # --- Loading schedule ---
    loading = None
    lc = config.loading
    if s.solver_type in ('quasi_static', 'quasi_static_legacy',
                         'static', 'lbfgs'):
        if lc.protocol == 'cyclic' and lc.cyclic_phases:
            loading = build_cyclic_schedule(lc)
        elif lc.protocol == 'simple':
            loading = [lc.dt * (i + 1) for i in range(lc.num_steps)]
        elif lc.protocol == 'two_step_prestrain':
            loading = [lc.prestrain_displacement] + [
                lc.prestrain_displacement + lc.dt * (i + 1)
                for i in range(lc.num_steps - 1)
            ]

    return {
        'mesh': mesh,
        'material': material,
        'bcs': bcs,
        'solver_config': solver_config,
        'ctx': ctx,
        'loading': loading,
    }
