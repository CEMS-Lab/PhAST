import subprocess
import sys

import pytest


pytestmark = pytest.mark.docs


def test_explain_config_reports_key_sections_without_resolving_mesh(tmp_path):
    cfg_path = tmp_path / "explain.yaml"
    cfg_path.write_text(
        """
name: explain smoke
schema_version: 1
example: phast.examples.dynamic_sent_tension.run
acceptance:
  status: beta
  required_outputs: [run_lockfile.json, damage.gif]
  metrics:
    crack_path:
      target: straight
      tolerance: visual
geometry:
  type: rectangular_sent
  parameters:
    W: 40.0
    H: 20.0
material:
  E: 32000.0
  nu: 0.2
  Gc: 0.003
  l0: 0.5
  pf_model: AT2
  energy_split: spectral
boundary_conditions:
  - nodes: bottom
    type: fix
    component: 1
    value: 0.0
loading:
  num_steps: 3
  dt: 1e-7
  t_total: 3e-7
  ramp_type: smooth_step
solver:
  solver_type: explicit
  use_multigrid: false
output:
  h5: true
  h5_every: 1
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Problem: explain smoke" in proc.stdout
    assert "Schema version: 1" in proc.stdout
    assert "Provenance example: phast.examples.dynamic_sent_tension.run" in proc.stdout
    assert "Acceptance metadata: yes" in proc.stdout
    assert "Acceptance" in proc.stdout
    assert "required_outputs: 2 entries" in proc.stdout
    assert "metrics: 1 fields (crack_path)" in proc.stdout
    assert "Geometry" in proc.stdout
    assert "built-in generator (rectangular_sent)" in proc.stdout
    assert "phase-field model: AT2" in proc.stdout
    assert "type: explicit" in proc.stdout
    assert "explicit time: dt=1e-07" in proc.stdout
    assert "enabled: zarr snapshots every 1 (--h5 alias)" in proc.stdout
    assert proc.stderr == ""


def test_explain_config_warns_for_experimental_monolithic(tmp_path):
    cfg_path = tmp_path / "mono.yaml"
    cfg_path.write_text(
        """
solver:
  solver_type: monolithic
output:
  h5: false
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "type: monolithic" in proc.stdout
    assert "Monolithic phase-field solve is experimental" in proc.stdout
    assert "No top-level schema_version is set" in proc.stdout


def test_explain_config_warns_for_coarse_h_over_l0(tmp_path):
    cfg_path = tmp_path / "coarse.yaml"
    cfg_path.write_text(
        """
geometry:
  type: rectangular_sent
  parameters:
    h_crack: "0.8 mm"
material:
  l0: "1.0 mm"
  pf_model: AT2
solver:
  solver_type: explicit
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "finest declared h/l0: 0.8" in proc.stdout
    assert "phase-field fracture validation usually needs h <= l0/2" in proc.stdout


def test_explain_config_reports_acceptable_h_over_l0_without_warning(tmp_path):
    cfg_path = tmp_path / "fine.yaml"
    cfg_path.write_text(
        """
geometry:
  mesh:
    element_size:
      default: 4.0
      refined:
        - size: 0.25
material:
  l0: 1.0
  pf_model: AT2
solver:
  solver_type: explicit
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "finest declared h/l0: 0.25" in proc.stdout
    assert "phase-field fracture validation usually needs" not in proc.stdout


def test_explain_config_reports_missing_external_mesh_validation_error(tmp_path):
    cfg_path = tmp_path / "missing_mesh.yaml"
    cfg_path.write_text(
        """
schema_version: 1
geometry:
  mesh_path: missing_mesh.msh
material:
  pf_model: AT2
solver:
  solver_type: quasi_static
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 2
    assert "geometry.mesh_path: mesh file does not exist" in proc.stderr
    assert "missing_mesh.msh" in proc.stderr


def test_explain_config_accepts_existing_external_mesh_relative_to_config(tmp_path):
    mesh_path = tmp_path / "existing_mesh.msh"
    mesh_path.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    cfg_path = tmp_path / "existing_mesh.yaml"
    cfg_path.write_text(
        """
schema_version: 1
geometry:
  mesh_path: existing_mesh.msh
material:
  pf_model: AT2
solver:
  solver_type: explicit
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "source: external mesh" in proc.stdout
    assert "existing_mesh.msh" in proc.stdout
    assert "geometry.mesh_path does not exist" not in proc.stdout


def test_explain_config_warns_for_implicit_multigrid_preconditioner(tmp_path):
    cfg_path = tmp_path / "implicit_amg.yaml"
    cfg_path.write_text(
        """
schema_version: 1
solver:
  solver_type: quasi_static
  preconditioner: amg
output:
  reaction_node_set: top
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "type: quasi_static" in proc.stdout
    assert "preconditioner: amg" in proc.stdout
    assert "current customer validation should use Jacobi" in proc.stdout


def test_explain_config_does_not_warn_for_explicit_multigrid_default(tmp_path):
    cfg_path = tmp_path / "explicit_default_multigrid.yaml"
    cfg_path.write_text(
        """
schema_version: 1
solver:
  solver_type: explicit
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "type: explicit" in proc.stdout
    assert "solver.use_multigrid is ignored" not in proc.stdout


def test_explain_config_warns_for_explicit_user_set_multigrid(tmp_path):
    cfg_path = tmp_path / "explicit_multigrid.yaml"
    cfg_path.write_text(
        """
schema_version: 1
solver:
  solver_type: explicit
  use_multigrid: true
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "type: explicit" in proc.stdout
    assert "solver.use_multigrid is ignored" in proc.stdout


def test_explain_config_warns_for_unreferenced_reaction_node_set(tmp_path):
    cfg_path = tmp_path / "reaction_typo.yaml"
    cfg_path.write_text(
        """
schema_version: 1
boundary_conditions:
  - nodes: top
    type: prescribe
    component: 1
    value: 0.01
solver:
  solver_type: quasi_static
output:
  reaction_node_set: tp
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "reaction logging: node_set=tp" in proc.stdout
    assert "output.reaction_node_set='tp' is not declared" in proc.stdout
    assert "Known YAML names: top" in proc.stdout


def test_explain_config_accepts_reaction_node_set_used_by_bc(tmp_path):
    cfg_path = tmp_path / "reaction_ok.yaml"
    cfg_path.write_text(
        """
schema_version: 1
boundary_conditions:
  - nodes: top
    type: prescribe
    component: 1
    value: 0.01
solver:
  solver_type: quasi_static
output:
  reaction_node_set: top
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "reaction logging: node_set=top" in proc.stdout
    assert "output.reaction_node_set='top' is not declared" not in proc.stdout


def test_explain_config_warns_when_at1_uses_post_clamp_bounds(tmp_path):
    cfg_path = tmp_path / "at1_bounds.yaml"
    cfg_path.write_text(
        """
schema_version: 1
material:
  pf_model: AT1
solver:
  solver_type: explicit
  bounds_method: post_clamp
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "phase-field model: AT1" in proc.stdout
    assert "AT1 damage requires projected bound enforcement" in proc.stdout


def test_explain_config_returns_validation_errors(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        """
solver:
  solver_type: turbo
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "phast", "explain-config", str(cfg_path)],
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 2
    assert "solver.solver_type" in proc.stderr
    assert "Allowed values" in proc.stderr
