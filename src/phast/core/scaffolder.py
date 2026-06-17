"""
Config scaffolder — ``python -m phast new <name>``.

Generates a minimal, runnable starter YAML config under ``configs/`` (or
a user-chosen output directory), pre-filled with sensible defaults for
the chosen solver type, material preset, and geometry generator. The
generated stub is a *minimal* subset of ``configs/REFERENCE.yaml``;
inline comments point at REFERENCE.yaml for advanced options.

Issue #149, epic #136 phase 3.3.

Public API:
    generate_stub(name, solver_type, material, geometry, out_dir,
                  validate=True) -> Path
    main()  — CLI entry point invoked by ``python -m phast new``.

The YAML is hand-written with f-strings rather than going through
``yaml.safe_dump`` so we can preserve inline comments without taking on
a new ``ruamel.yaml`` dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Solver-type defaults
# ---------------------------------------------------------------------------

# Per-``solver_type`` block: each entry is a dict that gets formatted into
# the YAML template. Keep these small — they're only the *defaults*; the
# user is expected to edit them.
SOLVER_DEFAULTS = {
    'quasi_static': {
        'solver_block': (
            "solver:\n"
            "  solver_type: quasi_static\n"
            # eta_residual matches Material default (1e-7); see #272/#276.
            "  eta_residual: 1.0e-7\n"
            "  damage_every: 1\n"
            "  stagger_criterion: linf           # robust displacement/damage acceptance\n"
            "  stagger_tol: 1.0e-6\n"
            "  max_stagger: 500\n"
            "  anderson_depth: 3                 # acceleration; set 0 for strict baselines\n"
            "  preconditioner: jacobi             # QS-safe damage CG default\n"
            "  backend: auto                      # MUMPS > SciPy > CG after smoke tests\n"
        ),
        'loading_block': (
            "loading:\n"
            "  protocol: simple                 # 'simple' | 'cyclic' | 'two_step_prestrain'\n"
            "  ramp_type: linear\n"
            "  num_steps: 100\n"
            "  dt: 1.0e-3                       # quasi-static pseudo-time step\n"
            "  disp_max: 0.01                   # peak prescribed displacement [mm]\n"
        ),
    },
    # Legacy alias: same staggered loop dispatched to ``SecantCGSolver``
    # (iterative-CG path) instead of the default quasi-static solver.
    # See staggered_solver.py L297/L357. Kept for backwards compatibility
    # with older accepted quasi-static decks.
    'quasi_static_legacy': {
        'solver_block': (
            "solver:\n"
            "  solver_type: quasi_static_legacy\n"
            "  eta_residual: 1.0e-7\n"
            "  damage_every: 1\n"
            "  stagger_criterion: linf\n"
            "  stagger_tol: 1.0e-6\n"
            "  max_stagger: 500\n"
            "  preconditioner: jacobi             # QS-safe damage CG default\n"
            "  backend: auto                      # ignored by legacy mechanics CG path\n"
        ),
        'loading_block': (
            "loading:\n"
            "  protocol: simple                 # 'simple' | 'cyclic' | 'two_step_prestrain'\n"
            "  ramp_type: linear\n"
            "  num_steps: 100\n"
            "  dt: 1.0e-3                       # quasi-static pseudo-time step\n"
            "  disp_max: 0.01                   # peak prescribed displacement [mm]\n"
        ),
    },
    'static': {
        'solver_block': (
            "solver:\n"
            "  solver_type: static\n"
            # eta_residual matches Material default (1e-7); see #272/#276.
            "  eta_residual: 1.0e-7\n"
            "  static_max_iter: 5000\n"
            "  damage_every: 1\n"
            "  preconditioner: jacobi             # static-safe damage CG default\n"
            "  backend: auto                      # scipy direct solve when available\n"
        ),
        'loading_block': (
            "loading:\n"
            "  protocol: simple\n"
            "  ramp_type: linear\n"
            "  num_steps: 50\n"
            "  dt: 1.0e-3\n"
            "  disp_max: 0.01\n"
        ),
    },
    'lbfgs': {
        'solver_block': (
            "solver:\n"
            "  solver_type: lbfgs\n"
            # eta_residual matches Material default (1e-7); see #272/#276.
            "  eta_residual: 1.0e-7\n"
            "  damage_every: 1\n"
            "  preconditioner: jacobi             # implicit-safe damage CG default\n"
            "  backend: auto\n"
        ),
        'loading_block': (
            "loading:\n"
            "  protocol: simple\n"
            "  ramp_type: linear\n"
            "  num_steps: 100\n"
            "  dt: 1.0e-3\n"
            "  disp_max: 0.01\n"
        ),
    },
    # Experimental coupled implicit solve. The runtime still treats this as
    # an expert path, but the validator exposes the enum, so the scaffolder
    # needs a minimal stub instead of rejecting it.
    'monolithic': {
        'solver_block': (
            "solver:\n"
            "  solver_type: monolithic\n"
            "  eta_residual: 1.0e-7\n"
            "  damage_every: 1\n"
            "  max_stagger: 50\n"
            "  preconditioner: jacobi             # implicit-safe damage CG default\n"
            "  backend: auto\n"
        ),
        'loading_block': (
            "loading:\n"
            "  protocol: simple\n"
            "  ramp_type: linear\n"
            "  num_steps: 50\n"
            "  dt: 1.0e-3\n"
            "  disp_max: 0.01\n"
        ),
    },
    'explicit': {
        'solver_block': (
            "solver:\n"
            "  solver_type: explicit\n"
            "  dt_safety: 0.8                   # CFL safety factor (0 < x <= 1)\n"
            "  damage_every: 1                  # reference; use 2-3 only after validation\n"
            "  use_multigrid: true\n"
        ),
        'loading_block': (
            "loading:\n"
            "  protocol: simple                 # 'simple' | 'two_step_prestrain'\n"
            "  ramp_type: constant              # 'constant' | 'linear' | 'smooth' | 'velocity_impact'\n"
            "  num_steps: 0                     # 0 = derive from t_total + CFL dt\n"
            "  t_total: 80.0e-6                 # total simulated time [s]\n"
        ),
    },
}


# ---------------------------------------------------------------------------
# Material preset values (mirror of material.create_material's preset dict)
# ---------------------------------------------------------------------------
# Fetched lazily from material.py to keep the module decoupled. We do
# import it so a single source of truth is preserved.

def _get_preset_values(preset: str) -> Optional[Dict]:
    """Return the inline-field dict for a named preset, or ``None``.

    Re-uses the same preset dict that ``create_material`` ships, so the
    scaffolder cannot drift out of sync with the runtime registry.
    """
    if not preset:
        return None
    try:
        from ..physics import material as _m
    except ImportError:
        import material as _m  # pragma: no cover — for direct script runs
    # The preset registry lives inside ``create_material`` as a local
    # dict, so the cleanest way to fetch its values without duplicating
    # them here is to instantiate the Material once and read back the
    # inline-relevant attributes.
    try:
        mat = _m.create_material(preset)
    except Exception:
        return None
    inline = {
        'E': float(mat.E),
        'nu': float(mat.nu),
        'Gc': float(mat.Gc),
        'l0': float(mat.l0),
        'rho': float(mat.rho),
    }
    # Optional string fields if the preset overrode them
    for opt in ('energy_split', 'pf_model'):
        val = getattr(mat, opt, None)
        if val is not None and val != '':
            inline[opt] = val
    plane_stress = getattr(mat, 'plane_stress', None)
    if plane_stress:
        inline['plane_stress'] = bool(plane_stress)
    return inline


# ---------------------------------------------------------------------------
# Geometry placeholders
# ---------------------------------------------------------------------------

# Hand-crafted minimal parameter sets for the most common generators, so
# the stub is *runnable* (gmsh-buildable) out of the box. Generators not
# listed here fall back to {} and a TODO comment.
GEOMETRY_DEFAULTS = {
    'rectangular_sent': dict(W=100.0, H=40.0, a=50.0, h_crack=0.5, h_coarse=4.0),
    'kalthoff_winkler': dict(W=100.0, H=200.0, a=50.0, h_crack=0.25, h_coarse=5.0),
    'l_shaped_panel': dict(L=250.0, h_crack=2.0, h_coarse=10.0),
    'miehe_tension': dict(L=1.0, a=0.5, h_crack=0.005, h_coarse=0.05),
    'miehe_shear': dict(L=1.0, a=0.5, h_crack=0.005, h_coarse=0.05),
    'square_plate': dict(L=1.0, h_crack=0.01, h_coarse=0.05),
    'three_point_bending': dict(L=8.0, H=2.0, a=1.0, h_crack=0.05, h_coarse=0.5),
    'plate_with_holes': dict(W=100.0, H=100.0, h_crack=0.5, h_coarse=4.0),
    'perforated_sent': dict(W=100.0, H=100.0, h_crack=0.5, h_coarse=4.0),
    'glass_impact_vnotch': dict(W=100.0, H=100.0, h_crack=0.5, h_coarse=4.0),
    'bazant_gap_test': dict(L=100.0, h_crack=0.5, h_coarse=4.0),
}

VALID_GEOMETRIES = sorted(GEOMETRY_DEFAULTS.keys())
VALID_SOLVER_TYPES = sorted(SOLVER_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# YAML rendering helpers
# ---------------------------------------------------------------------------

def _fmt_scalar(v) -> str:
    """Format a Python scalar as a YAML scalar literal."""
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        # Use scientific notation for very small / very large; ``g`` keeps
        # things compact and avoids "1.0" being rendered as "1".
        if v == 0.0:
            return '0.0'
        if abs(v) < 1e-3 or abs(v) >= 1e6:
            return f"{v:.6e}"
        # Always include a decimal point so YAML reads it as float
        s = repr(v)
        return s
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        # Quote if contains special chars or starts with a non-letter
        if any(c in v for c in ":#{}[]&*!|>'\"%@`,") or v.strip() != v:
            return repr(v)
        return v
    return repr(v)


def _render_geometry_block(geometry: Optional[str]) -> str:
    """Render the geometry: ... block."""
    gen = geometry or 'rectangular_sent'
    params = GEOMETRY_DEFAULTS.get(gen, {})
    lines = ["geometry:"]
    lines.append(f"  type: {gen}                 # generator from mesh_generator.py")
    if params:
        lines.append("  parameters:")
        for k, v in params.items():
            lines.append(f"    {k}: {_fmt_scalar(v)}")
        lines.append(
            f"    # See mesh_generator.{gen} docstring for the full parameter list."
        )
    else:
        lines.append("  parameters: {}             # TODO: fill in generator-specific params")
    lines.append("")
    return "\n".join(lines)


def _render_material_block(material: Optional[str]) -> str:
    """Render the material: ... block.

    If ``material`` names a known preset, emit the preset's values
    *inline* (so the user can immediately see and tweak them) and also
    record the preset name as a YAML comment for traceability.
    """
    if not material:
        return (
            "material:\n"
            "  # Inline material — fill in your own constants, or set 'preset: <name>'.\n"
            "  E: 32000.0                       # Young's modulus [MPa]\n"
            "  nu: 0.2                          # Poisson ratio\n"
            "  Gc: 3.0e-3                       # Fracture toughness [N/mm]\n"
            "  l0: 0.25                         # Phase-field length scale [mm]\n"
            "  rho: 2.45e-9                     # Density [tonne/mm^3]\n"
            "  energy_split: spectral           # 'spectral' | 'amor' | 'volumetric_deviatoric' | 'isotropic'\n"
            "  pf_model: AT2                    # 'AT1' | 'AT2'\n"
            "\n"
        )
    inline = _get_preset_values(material)
    if inline is None:
        # Unknown preset — fall back to preset-only entry, comment out
        return (
            "material:\n"
            f"  preset: {material}                # WARNING: unknown preset; falls back to 'default'\n"
            "  # Override individual constants here if needed.\n"
            "\n"
        )
    lines = [
        "material:",
        f"  # Inline values from preset '{material}' (edit freely).",
    ]
    order = ['E', 'nu', 'Gc', 'l0', 'rho', 'energy_split', 'pf_model', 'plane_stress']
    units = {
        'E': 'Young\'s modulus [MPa]',
        'nu': 'Poisson ratio',
        'Gc': 'Fracture toughness [N/mm]',
        'l0': 'Phase-field length scale [mm]',
        'rho': 'Density [tonne/mm^3]',
        'energy_split': 'spectral | amor | volumetric_deviatoric | isotropic',
        'pf_model': 'AT1 | AT2',
        'plane_stress': 'plane-stress flag',
    }
    for k in order:
        if k in inline:
            comment = units.get(k, '')
            lines.append(f"  {k}: {_fmt_scalar(inline[k])}    # {comment}".rstrip())
    lines.append("")
    return "\n".join(lines)


def _render_bcs_block() -> str:
    return (
        "boundary_conditions:\n"
        "  # Edit node-set names to match those produced by the geometry generator.\n"
        "  - nodes: left\n"
        "    type: fix\n"
        "    component: 0                     # 0 = x, 1 = y\n"
        "  - nodes: bottom\n"
        "    type: fix\n"
        "    component: 1\n"
        "  - nodes: top\n"
        "    type: prescribe\n"
        "    component: 1\n"
        "    value: 1.0\n"
        "\n"
    )


def _render_output_block() -> str:
    return (
        "output:\n"
        "  h5: true\n"
        "  h5_every: 20\n"
        "  vtu: false\n"
        "  print_every: 50\n"
        "\n"
    )


def _render_device_block() -> str:
    return (
        "device:\n"
        "  device: cpu                        # 'cpu' | 'cuda' | 'mps' (mps works only with float32)\n"
        "  compile: false\n"
        "\n"
    )


def _render_footer(name: str) -> str:
    """Trailing comment block pointing at REFERENCE + README."""
    return (
        "# ----------------------------------------------------------------------\n"
        f"# Generated by `python -m phast new {name}`.\n"
        "#\n"
        "# Next steps:\n"
        "#   1. Adjust geometry / material / boundary_conditions / loading above.\n"
        f"#   2. Run:   python -m phast run configs/{name}.yaml\n"
        "#\n"
        "# More options:\n"
        "#   - configs/REFERENCE.yaml — every available field with defaults\n"
        "#   - configs/README.md       — quick start + CLI flag precedence\n"
        "# ----------------------------------------------------------------------\n"
    )


# ---------------------------------------------------------------------------
# Top-level template
# ---------------------------------------------------------------------------

def _render_yaml(name: str, solver_type: str,
                 material: Optional[str], geometry: Optional[str]) -> str:
    """Assemble the full YAML stub as a single string."""
    if solver_type not in SOLVER_DEFAULTS:
        raise ValueError(
            f"Unknown solver_type {solver_type!r}. "
            f"Choose from: {sorted(SOLVER_DEFAULTS.keys())}"
        )
    blocks = []
    blocks.append(
        "# ======================================================================\n"
        f"# {name} — phast config (scaffolded stub)\n"
        "# ======================================================================\n"
        "# Edit the placeholder values below to match your problem. See\n"
        "# configs/REFERENCE.yaml for every available field with defaults.\n"
        "\n"
        "problem:\n"
        f"  name: \"{name}\"\n"
        "  reference: \"\"                     # Citation, paper, or experiment label\n"
        "\n"
    )
    blocks.append(_render_geometry_block(geometry))
    blocks.append(_render_material_block(material))
    blocks.append(SOLVER_DEFAULTS[solver_type]['loading_block'] + "\n")
    blocks.append(SOLVER_DEFAULTS[solver_type]['solver_block'] + "\n")
    blocks.append(_render_bcs_block())
    blocks.append(_render_output_block())
    blocks.append(_render_device_block())
    blocks.append(_render_footer(name))
    return "".join(blocks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_stub(
    name: str,
    solver_type: str = 'quasi_static',
    material: Optional[str] = None,
    geometry: Optional[str] = None,
    out_dir: str = 'configs',
    validate: bool = True,
) -> Path:
    """Write a starter YAML config to ``<out_dir>/<name>.yaml`` and return its path.

    Parameters
    ----------
    name : str
        Stem of the output YAML file (without extension). Becomes the
        ``problem.name`` field too.
    solver_type : str
        One of ``'explicit'``, ``'quasi_static'``, ``'static'``,
        ``'lbfgs'``. Drives the ``solver:`` and ``loading:`` blocks.
    material : str, optional
        Name of a preset from ``material.create_material``. The
        preset's E / nu / Gc / l0 / rho / energy_split / pf_model are
        emitted *inline* so the user can see and tweak them. ``None``
        emits a generic inline material the user must fill in.
    geometry : str, optional
        Name of a generator function from ``mesh_generator``. ``None``
        defaults to ``'rectangular_sent'``.
    out_dir : str
        Output directory (default ``configs``). Created if missing.
    validate : bool
        If True (default), run ``config_validation.assert_valid`` on
        the generated stub and surface any errors before returning.

    Returns
    -------
    pathlib.Path
        Absolute path of the written YAML.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    yaml_text = _render_yaml(name, solver_type, material, geometry)
    target = out / f"{name}.yaml"
    target.write_text(yaml_text)
    if validate:
        try:
            from ..config.config_validation import assert_valid
        except ImportError:  # pragma: no cover
            from config_validation import assert_valid
        assert_valid(str(target))
    return target.resolve()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m phast new',
        description='Scaffold a starter YAML config for a new benchmark.',
    )
    p.add_argument('name', help='Benchmark name (becomes <name>.yaml)')
    p.add_argument(
        '--type', dest='solver_type', default='quasi_static',
        choices=VALID_SOLVER_TYPES,
        help='Solver type (default: quasi_static).',
    )
    p.add_argument(
        '--material', default=None,
        help=('Material preset name (e.g. pmma_bleyer, glass_borden). '
              'Inline values from the preset are written into the stub.'),
    )
    p.add_argument(
        '--geometry', default=None,
        help='Geometry generator name (e.g. rectangular_sent, kalthoff_winkler).',
    )
    p.add_argument(
        '--out', dest='out_dir', default='configs',
        help='Output directory (default: configs).',
    )
    p.add_argument(
        '--no-validate', dest='validate', action='store_false',
        help='Skip schema validation of the generated stub.',
    )
    p.add_argument(
        '--validate', dest='validate', action='store_true',
        help='(Default) Validate the generated stub against the schema.',
    )
    p.set_defaults(validate=True)
    return p


def main(argv=None):
    """CLI entry point: ``python -m phast new <name> [...]``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        path = generate_stub(
            name=args.name,
            solver_type=args.solver_type,
            material=args.material,
            geometry=args.geometry,
            out_dir=args.out_dir,
            validate=args.validate,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    rel = path
    try:
        rel = path.relative_to(Path.cwd())
    except ValueError:
        pass
    print(f"Created {rel} (filled with sensible defaults).")
    print("Edit geometry / BCs / loading to match your problem, then:")
    print(f"  python -m phast run {rel}")


if __name__ == '__main__':
    main()
