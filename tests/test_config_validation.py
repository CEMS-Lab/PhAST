"""
Tests for the schema validator (issue #150).

Covers:

* Every shipped ``configs/*.yaml`` validates cleanly.
* Required-field / type-mismatch / enum / range / unknown-key / mutex
  errors all carry an accurate line number and an actionable message.
* Did-you-mean suggestions on unknown keys.
"""

import glob
import os
import textwrap
import tempfile
from pathlib import Path

import pytest
import phast.config_validation as _validation

from phast.config_validation import (
    validate_config,
    validate_config_file,
    validate_config_file_with_warnings,
    validate_config_warnings,
    format_warnings,
    format_errors,
    ConfigValidationError,
    assert_valid,
)

# Resolve root configs/ relative to the test tree, not cwd, so tests run from
# any directory after configs moved out of the package source tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / 'configs'
_NON_PROBLEM_CONFIGS = {
    # Manifest consumed by Slurm/rescue-visual orchestration, not by
    # phast.config_validation's problem schema.
    'QS_sens_tpb_rescue_visuals.yaml',
    'QS_sens_tpb_peak_window_corrected.yaml',
    'QS_mesh_convergence_arc_length.yaml',
}


def _shipped_problem_configs():
    return [
        p for p in sorted(glob.glob(str(_CONFIG_DIR / '*.yaml')))
        if Path(p).name not in _NON_PROBLEM_CONFIGS
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tmp(yaml_text: str) -> str:
    yaml_text = textwrap.dedent(yaml_text).lstrip('\n')
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write(yaml_text)
    return path


def _validate_text(yaml_text: str):
    path = _write_tmp(yaml_text)
    try:
        return validate_config_file(path)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Baseline: all shipped configs pass
# ---------------------------------------------------------------------------

class TestShippedConfigsValid:
    """Every YAML in configs/ must pass validation cleanly."""

    @pytest.mark.parametrize(
        'cfg_path',
        _shipped_problem_configs()
    )
    def test_config_validates(self, cfg_path):
        raw, errs = validate_config_file(cfg_path)
        assert errs == [], (
            f"{cfg_path} should validate cleanly but produced:\n"
            + format_errors(errs, cfg_path)
        )

    @pytest.mark.parametrize(
        'cfg_path',
        _shipped_problem_configs()
    )
    def test_config_declares_schema_version(self, cfg_path):
        raw, errs = validate_config_file(cfg_path)
        assert errs == []
        assert raw.get('schema_version') == 1


# ---------------------------------------------------------------------------
# Specific error paths
# ---------------------------------------------------------------------------

class TestEnumValidation:
    def test_invalid_solver_type(self):
        raw, errs = _validate_text("""
            solver:
              solver_type: turbo
        """)
        assert any(e.path == 'solver.solver_type' for e in errs)
        e = next(e for e in errs if e.path == 'solver.solver_type')
        assert 'invalid value' in e.message
        assert 'explicit' in e.allowed_values
        assert e.line_no == 2

    def test_invalid_energy_split(self):
        raw, errs = _validate_text("""
            material:
              preset: glass_borden
              overrides:
                energy_split: foo
        """)
        e = next(e for e in errs
                 if e.path == 'material.overrides.energy_split')
        assert 'spectral' in e.allowed_values

    def test_valid_enum_passes(self):
        raw, errs = _validate_text("""
            solver:
              solver_type: explicit
        """)
        assert errs == []

    @pytest.mark.parametrize(
        'criterion',
        ['relative', 'absolute', 'linf', 'residual', 'am_energy'],
    )
    def test_supported_stagger_criteria_pass(self, criterion):
        raw, errs = _validate_text(f"""
            solver:
              stagger_criterion: {criterion}
        """)
        assert errs == []

    def test_monolithic_solver_type_passes(self):
        raw, errs = _validate_text("""
            solver:
              solver_type: monolithic
        """)
        assert errs == []


class TestSchemaVersionValidation:
    def test_schema_version_accepts_positive_int(self):
        raw, errs = _validate_text("""
            schema_version: 1
        """)
        assert errs == []

    @pytest.mark.parametrize("value", ["0", "false", "one"])
    def test_schema_version_rejects_invalid_values(self, value):
        raw, errs = _validate_text(f"""
            schema_version: {value}
        """)
        assert any(e.path == 'schema_version' for e in errs)


class TestAcceptanceMetadata:
    def test_acceptance_block_is_validated_as_free_form_metadata(self):
        raw, errs = _validate_text("""
            schema_version: 1
            acceptance:
              status: beta
              required_outputs: [run_lockfile.json, force_displacement.csv]
              metrics:
                peak_force:
                  target: 123.4
                  tolerance: 0.05
        """)
        assert errs == []
        assert raw['acceptance']['metrics']['peak_force']['target'] == 123.4


class TestTypeValidation:
    def test_string_where_int_expected(self):
        raw, errs = _validate_text("""
            solver:
              max_stagger: not_a_number
        """)
        e = next(e for e in errs if e.path == 'solver.max_stagger')
        assert 'expected int' in e.message

    def test_string_where_bool_expected(self):
        raw, errs = _validate_text("""
            solver:
              use_multigrid: yes_please
        """)
        e = next(e for e in errs if e.path == 'solver.use_multigrid')
        assert 'expected bool' in e.message

    def test_numeric_string_accepted_for_float(self):
        # YAML 1.1 parses '1e-6' as str; validator must accept (loader coerces).
        raw, errs = _validate_text("""
            solver:
              stagger_tol: 1e-6
        """)
        assert errs == []


class TestRangeValidation:
    def test_dt_safety_above_one(self):
        raw, errs = _validate_text("""
            solver:
              dt_safety: 1.5
        """)
        e = next(e for e in errs if e.path == 'solver.dt_safety')
        assert 'out of range' in e.message
        assert e.line_no == 2

    def test_negative_Gc_in_overrides(self):
        raw, errs = _validate_text("""
            material:
              preset: glass_borden
              overrides:
                Gc: -1.0
        """)
        e = next(e for e in errs if e.path == 'material.overrides.Gc')
        assert 'out of range' in e.message

    def test_in_range_passes(self):
        raw, errs = _validate_text("""
            solver:
              dt_safety: 0.8
        """)
        assert errs == []


class TestUnknownKey:
    def test_unknown_top_level_key(self):
        raw, errs = _validate_text("""
            solvr:
              dt_safety: 0.5
        """)
        e = next(e for e in errs if e.path == 'solvr')
        assert 'unknown top-level key' in e.message
        assert e.suggestion and 'solver' in e.suggestion

    def test_unknown_nested_key(self):
        raw, errs = _validate_text("""
            solver:
              dt_saftey: 0.5
        """)
        e = next(e for e in errs if e.path == 'solver.dt_saftey')
        assert 'unknown field' in e.message
        assert e.suggestion and 'dt_safety' in e.suggestion

    def test_did_you_mean_bc_field(self):
        raw, errs = _validate_text("""
            boundary_conditions:
            - {nodez: top, type: fix, component: 1}
        """)
        e = next(e for e in errs
                 if e.path.endswith('.nodez'))
        assert e.suggestion and 'nodes' in e.suggestion

    def test_manifest_gets_explanatory_error(self):
        raw, errs = _validate_text("""
            schema_version: 1
            manifest_type: command_manifest
            cases: []
        """)
        assert raw["manifest_type"] == "command_manifest"
        e = next(e for e in errs if e.path == "manifest_type")
        assert "orchestration manifest" in e.message
        assert "configs/benchmarks/dynamic" in e.suggestion


class TestMutex:
    def test_geometry_type_and_mesh_path_both_set(self):
        raw, errs = _validate_text("""
            geometry:
              type: rectangular_sent
              mesh_path: foo.msh
        """)
        assert any('mutually exclusive' in e.message for e in errs)


class TestCrossFieldCompatibility:
    def test_time_integrator_rejected_for_implicit_solver(self):
        raw, errs = _validate_text("""
            solver:
              solver_type: quasi_static
              time_integrator: generalized_alpha
        """)
        e = next(e for e in errs if e.path == 'solver.time_integrator')
        assert "only used by solver_type='explicit'" in e.message
        assert e.suggestion and 'Remove solver.time_integrator' in e.suggestion

    def test_fresh_d_rejected_with_explicit_rigid_connector(self):
        raw, errs = _validate_text("""
            boundary_conditions:
            - {nodes: pin.boundary, type: rigid_connector, master: pin.centre}
            solver:
              solver_type: explicit
              fresh_d_in_corrector: true
        """)
        e = next(e for e in errs if e.path == 'solver.fresh_d_in_corrector')
        assert 'not supported with rigid_connector' in e.message

    def test_genalpha_rejected_with_explicit_rigid_connector(self):
        raw, errs = _validate_text("""
            boundary_conditions:
            - {nodes: pin.boundary, type: rigid_connector, master: pin.centre}
            solver:
              solver_type: explicit
              time_integrator: generalized_alpha
        """)
        e = next(e for e in errs if e.path == 'solver.time_integrator')
        assert 'does not yet support rigid_connector' in e.message


class TestValidationWarnings:
    def test_missing_schema_version_warning(self):
        raw, errs = _validate_text("""
            solver:
              solver_type: explicit
        """)
        warnings = validate_config_warnings(raw)
        w = next(w for w in warnings if w.path == 'schema_version')
        assert 'missing schema_version' in w.message

    def test_h_over_l0_warning_uses_unit_strings(self):
        raw, errs = _validate_text("""
            schema_version: 1
            geometry:
              parameters:
                h_crack: "0.75 mm"
            material:
              l0: "1 mm"
        """)
        warnings = validate_config_warnings(raw)
        w = next(w for w in warnings if w.path == 'geometry')
        assert 'h/l0=0.75' in w.message
        assert 'h <= l0/2' in w.suggestion

    def test_validate_config_file_with_warnings_keeps_errors_separate(self):
        path = _write_tmp("""
            geometry:
              parameters:
                h_crack: 0.75
            material:
              l0: 1.0
        """)
        try:
            raw, errs, warnings = validate_config_file_with_warnings(path)
        finally:
            os.unlink(path)

        assert errs == []
        assert any(w.path == 'schema_version' for w in warnings)
        assert any(w.path == 'geometry' for w in warnings)
        formatted = format_warnings(warnings, '/tmp/example.yaml')
        assert 'Warning in /tmp/example.yaml' in formatted
        assert 'schema_version' in formatted


class TestFileContextValidation:
    def test_missing_mesh_path_is_line_numbered_error(self):
        path = _write_tmp("""
            geometry:
              mesh_path: definitely_missing_mesh.msh
        """)
        try:
            raw, errs = validate_config_file(path)
            e = next(e for e in errs if e.path == 'geometry.mesh_path')
            assert 'mesh file does not exist' in e.message
            assert e.line_no > 0
            assert e.suggestion and 'relative to the YAML file' in e.suggestion
        finally:
            os.unlink(path)

    def test_existing_relative_mesh_path_validates(self, tmp_path, monkeypatch):
        mesh_path = tmp_path / "mesh.vtu"
        mesh_path.write_text("placeholder", encoding="utf-8")
        monkeypatch.setattr(
            _validation,
            "_node_sets_from_mesh_file",
            lambda _path: {"left", "top", "pin.master", "crack"},
        )
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            """
geometry:
  mesh_path: mesh.vtu
boundary_conditions:
  - {nodes: left, type: fix, component: 0}
  - {nodes: top, type: rigid_connector, master: pin.master}
initial_conditions:
  preseed_notch_nodesets: [crack]
output:
  reaction_node_set: left
""",
            encoding="utf-8",
        )

        raw, errs = validate_config_file(str(cfg_path))
        assert [e for e in errs if e.path == 'geometry.mesh_path'] == []
        assert [e for e in errs if 'node set' in e.message] == []

    def test_missing_external_mesh_node_set_is_validation_error(
            self, tmp_path, monkeypatch):
        mesh_path = tmp_path / "mesh.vtu"
        mesh_path.write_text("placeholder", encoding="utf-8")
        monkeypatch.setattr(
            _validation,
            "_node_sets_from_mesh_file",
            lambda _path: {"left", "top", "pin.master", "crack"},
        )
        cfg_path = tmp_path / "bad_node_set.yaml"
        cfg_path.write_text(
            """
geometry:
  mesh_path: mesh.vtu
boundary_conditions:
  - {nodes: rite, type: fix, component: 0}
initial_conditions:
  preseed_notch_nodesets: [crak]
output:
  reaction_node_set: botom
""",
            encoding="utf-8",
        )

        raw, errs = validate_config_file(str(cfg_path))

        bc_err = next(e for e in errs
                      if e.path == 'boundary_conditions[0].nodes')
        assert "node set 'rite' is not present" in bc_err.message
        assert bc_err.line_no > 0

        preseed_err = next(e for e in errs
                           if e.path == 'initial_conditions.preseed_notch_nodesets[0]')
        assert "node set 'crak' is not present" in preseed_err.message
        assert preseed_err.suggestion and 'crack' in preseed_err.suggestion

        reaction_err = next(e for e in errs
                            if e.path == 'output.reaction_node_set')
        assert "node set 'botom' is not present" in reaction_err.message


class TestLineNumbers:
    def test_line_number_matches_for_typo_on_known_line(self):
        # Typo at a specific line; line_no must point there.
        text = textwrap.dedent("""
            problem:
              name: x

            solver:
              solver_type: turbo
        """).lstrip('\n')
        path = _write_tmp(text)
        try:
            raw, errs = validate_config_file(path)
            e = next(e for e in errs if e.path == 'solver.solver_type')
            # Find line containing 'turbo' in the source
            with open(path) as f:
                src_lines = f.readlines()
            expected = next(i + 1 for i, ln in enumerate(src_lines)
                            if 'turbo' in ln)
            assert e.line_no == expected
        finally:
            os.unlink(path)


class TestErrorFormatting:
    def test_format_errors_includes_filename_and_message(self):
        raw, errs = _validate_text("""
            solver:
              solver_type: turbo
        """)
        out = format_errors(errs, '/tmp/foo.yaml')
        assert '/tmp/foo.yaml' in out
        assert 'solver_type' in out
        assert 'Allowed values' in out
        assert 'REFERENCE.yaml' in out

    def test_assert_valid_raises(self):
        path = _write_tmp("solver:\n  solver_type: turbo\n")
        try:
            with pytest.raises(ConfigValidationError):
                assert_valid(path)
        finally:
            os.unlink(path)

    def test_assert_valid_passes_for_good_config(self):
        # Use a real shipped config (any one).
        any_cfg = sorted(glob.glob(str(_CONFIG_DIR / 'B*.yaml')))[0]
        assert_valid(any_cfg)  # no exception


class TestBoundaryConditions:
    def test_invalid_bc_type(self):
        raw, errs = _validate_text("""
            boundary_conditions:
            - {nodes: top, type: ROLL, component: 1, value: 0.0}
        """)
        e = next(e for e in errs
                 if e.path == 'boundary_conditions[0].type')
        assert 'invalid value' in e.message
        assert 'fix' in e.allowed_values

    def test_invalid_bc_component(self):
        raw, errs = _validate_text("""
            boundary_conditions:
            - {nodes: top, type: fix, component: 5}
        """)
        e = next(e for e in errs
                 if e.path == 'boundary_conditions[0].component')
        assert 'invalid value' in e.message

    @pytest.mark.parametrize("bc_type",
                             ['traction', 'symmetry', 'rigid_connector'])
    def test_post_pr155_bc_types_accepted(self, bc_type):
        """Issue #181: traction/symmetry/rigid_connector must validate."""
        raw, errs = _validate_text(f"""
            boundary_conditions:
            - {{nodes: top, type: {bc_type}, component: 1, value: 0.0}}
        """)
        bc_errs = [e for e in errs
                   if e.path == 'boundary_conditions[0].type']
        assert bc_errs == [], (
            f"BC type {bc_type!r} should validate cleanly; got {bc_errs}"
        )

    def test_bc_type_typo_suggests_traction(self):
        """Issue #181: a typo on 'traction' should get a did-you-mean hint."""
        raw, errs = _validate_text("""
            boundary_conditions:
            - {nodes: top, type: tracton, component: 1, value: 0.0}
        """)
        e = next(e for e in errs
                 if e.path == 'boundary_conditions[0].type')
        assert 'invalid value' in e.message
        assert e.suggestion and 'traction' in e.suggestion

    def test_unit_suffixed_bc_values_validate(self):
        """Unit-suffixed BC quantities are accepted before loader parsing."""
        raw, errs = _validate_text("""
            boundary_conditions:
            - {nodes: top, type: prescribe, component: 1, value: "0.1 mm"}
            - {nodes: right, type: traction, component: 0,
               value: "5 MPa", t_ramp: "10 us", t_hold: "25 us"}
        """)
        bc_type_errors = [
            e for e in errs
            if e.path.startswith('boundary_conditions[')
            and 'expected' in e.message
        ]
        assert bc_type_errors == []
