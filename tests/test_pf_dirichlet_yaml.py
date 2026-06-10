"""Issue #213 — schema parsing for the new ``pf_dirichlet`` BC type."""

import os
import tempfile
import textwrap

import pytest

from phast.config_validation import validate_config_file


def _validate_text(yaml_text: str):
    yaml_text = textwrap.dedent(yaml_text).lstrip('\n')
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write(yaml_text)
    try:
        return validate_config_file(path)
    finally:
        os.unlink(path)


def test_pf_dirichlet_validates():
    """``type: pf_dirichlet`` is accepted by the schema."""
    raw, errs = _validate_text("""
        boundary_conditions:
        - {nodes: notch_upper, type: pf_dirichlet, value: 1.0}
        - {nodes: notch_lower, type: pf_dirichlet, value: 1.0}
    """)
    bc_errs = [e for e in errs
               if e.path.startswith('boundary_conditions[')
               and e.path.endswith('.type')]
    assert bc_errs == [], (
        f"pf_dirichlet should validate cleanly; got {bc_errs}"
    )


def test_pf_dirichlet_typo_suggests_correction():
    """``pf_dirichelt`` (transposition) gets a did-you-mean hint."""
    raw, errs = _validate_text("""
        boundary_conditions:
        - {nodes: notch_upper, type: pf_dirichelt, value: 1.0}
    """)
    e = next(e for e in errs
             if e.path == 'boundary_conditions[0].type')
    assert 'invalid value' in e.message
    assert e.suggestion and 'pf_dirichlet' in e.suggestion


def test_pf_dirichlet_dispatch_via_config_entry(tmp_path):
    """End-to-end YAML -> ``BoundaryConditions`` dispatch.

    Avoids the geometry-registry path (which needs gmsh) by feeding a
    pre-built mesh via ``geometry.mesh_path``.
    """
    import torch
    from phast.config import load_config, resolve_config

    geo = """
Point(1) = {0,0,0,0.5};
Point(2) = {1,0,0,0.5};
Point(3) = {1,1,0,0.5};
Point(4) = {0,1,0,0.5};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Curve Loop(1)={1,2,3,4}; Plane Surface(1)={1};
Physical Surface("plate")={1};
Physical Curve("bottom")={1};
Physical Curve("top")={3};
Mesh.ElementOrder=1;
"""
    geo_file = tmp_path / "sq.geo"
    msh_file = tmp_path / "sq.msh"
    geo_file.write_text(geo)
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(geo_file))
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_file))
    finally:
        gmsh.finalize()

    yaml_text = textwrap.dedent(f"""
        geometry:
          mesh_path: {msh_file}
        material:
          E: 210e3
          nu: 0.3
          Gc: 2.7e-3
          l0: 0.1
          rho: 7.85e-9
        boundary_conditions:
        - {{nodes: bottom, type: fix,           component: 1}}
        - {{nodes: top,    type: pf_dirichlet,  value: 1.0}}
        loading:
          protocol: simple
          num_steps: 1
          dt: 1.0e-7
        solver:
          solver_type: explicit
        output: {{}}
    """).lstrip('\n')
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(yaml_text)
    cfg = load_config(str(yaml_path))
    out = resolve_config(cfg)
    bcs = out['bcs']
    assert len(bcs.pf_dirichlet_bcs) == 1
    assert bcs.pf_dirichlet_bcs[0].value == pytest.approx(1.0)
    mask, vals = bcs.get_pf_dirichlet_mask_values()
    assert int(mask.sum().item()) > 0
