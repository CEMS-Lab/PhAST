"""
Tests for ``phast.scaffolder`` (issue #149).

Covers:

* ``generate_stub`` produces syntactically valid YAML for every
  combination of ``--type`` / ``--material`` / ``--geometry`` we ship
  defaults for.
* The generated stub passes ``config_validation.assert_valid``.
* The default invocation (no flags) yields a config that round-trips
  through ``load_config`` to a populated ``ProblemConfig`` dataclass.
* ``--out`` is honoured (file lands where the user asked).
* The trailing comment block points at REFERENCE.yaml + the README.
* The CLI ``--no-validate`` flag suppresses validation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from phast.scaffolder import (
    GEOMETRY_DEFAULTS,
    SOLVER_DEFAULTS,
    VALID_GEOMETRIES,
    VALID_SOLVER_TYPES,
    generate_stub,
    main as scaffolder_main,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmpout(tmp_path):
    """Throwaway output directory."""
    out = tmp_path / "configs"
    out.mkdir()
    return out


# A representative cross-product (full Cartesian would be ~4 * 14 * 14 = 784
# tests; pick a sensible diagonal slice that still touches every option).
_COMBOS = []
_materials_to_test = [None, 'pmma_bleyer', 'glass_borden',
                      'l_shaped_concrete', 'maraging_steel_kw']
_geoms_to_test = [None, 'rectangular_sent', 'kalthoff_winkler',
                  'l_shaped_panel', 'three_point_bending']
for st in VALID_SOLVER_TYPES:
    for mat in _materials_to_test:
        for geom in _geoms_to_test:
            _COMBOS.append((st, mat, geom))


# ---------------------------------------------------------------------------
# 1. Combinations parse + validate
# ---------------------------------------------------------------------------

class TestCombinations:
    """Every (--type, --material, --geometry) combo yields valid YAML."""

    @pytest.mark.parametrize('solver_type,material,geometry', _COMBOS)
    def test_generates_valid_yaml(self, tmpout, solver_type, material, geometry):
        path = generate_stub(
            name='combo',
            solver_type=solver_type,
            material=material,
            geometry=geometry,
            out_dir=str(tmpout),
            validate=True,  # raises on failure
        )
        assert path.exists()
        # Re-parse to confirm syntactic YAML correctness
        with open(path) as f:
            doc = yaml.safe_load(f)
        assert isinstance(doc, dict)
        assert 'problem' in doc
        assert 'geometry' in doc
        assert 'material' in doc
        assert 'solver' in doc
        assert doc['solver']['solver_type'] == solver_type


# ---------------------------------------------------------------------------
# 2. Schema validation passes
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Generated stubs must pass the validator from #150."""

    def test_default_invocation_passes_validator(self, tmpout):
        from phast.config_validation import validate_config_file
        path = generate_stub('default_run', out_dir=str(tmpout))
        _, errs = validate_config_file(str(path))
        assert errs == [], (
            "Generated stub produced validation errors:\n"
            + "\n".join(e.format() for e in errs)
        )

    def test_explicit_passes_validator(self, tmpout):
        from phast.config_validation import validate_config_file
        path = generate_stub(
            'expl_run', solver_type='explicit',
            material='glass_borden', geometry='rectangular_sent',
            out_dir=str(tmpout),
        )
        _, errs = validate_config_file(str(path))
        assert errs == []

    def test_no_validate_flag_skips_validation(self, tmpout, monkeypatch):
        """``validate=False`` does not raise even on intentionally bad input."""
        # The internal renderer always produces valid YAML, but make sure
        # the validate=False path doesn't import config_validation.
        called = {}
        from phast import scaffolder as _s

        def _fake_assert_valid(p):
            called['hit'] = True

        # Monkeypatch in the local import path (it's done lazily inside
        # generate_stub).
        import phast.config_validation as _cv
        monkeypatch.setattr(_cv, 'assert_valid', _fake_assert_valid)
        generate_stub('skipv', out_dir=str(tmpout), validate=False)
        assert 'hit' not in called


# ---------------------------------------------------------------------------
# 3. Default stub is loadable as a ProblemConfig dataclass
# ---------------------------------------------------------------------------

class TestRunnableDefault:
    """Default invocation produces a config the loader can parse."""

    def test_default_loads_to_problem_config(self, tmpout):
        from phast.config import load_config, ProblemConfig
        path = generate_stub('runnable_default', out_dir=str(tmpout))
        cfg = load_config(str(path))
        assert isinstance(cfg, ProblemConfig)
        assert cfg.geometry is not None
        assert cfg.material is not None
        assert cfg.solver is not None
        assert cfg.loading is not None
        assert cfg.solver.solver_type == 'quasi_static'
        assert cfg.solver.preconditioner == 'jacobi'
        assert cfg.solver.backend == 'auto'
        # Default geometry is rectangular_sent
        assert cfg.geometry.type == 'rectangular_sent'

    def test_explicit_default_loads(self, tmpout):
        from phast.config import load_config
        path = generate_stub(
            'expl_default', solver_type='explicit',
            material='pmma_bleyer', geometry='rectangular_sent',
            out_dir=str(tmpout),
        )
        cfg = load_config(str(path))
        assert cfg.solver.solver_type == 'explicit'
        # pmma_bleyer preset values flowed inline
        assert cfg.material.E == pytest.approx(3090.0)
        assert cfg.material.pf_model == 'AT1'


# ---------------------------------------------------------------------------
# 4. --out honoured
# ---------------------------------------------------------------------------

class TestOutDir:
    def test_out_dir_honoured(self, tmp_path):
        sub = tmp_path / 'my_subdir' / 'nested'
        path = generate_stub('xyz', out_dir=str(sub))
        assert path == (sub / 'xyz.yaml').resolve()
        assert path.exists()

    def test_out_dir_created_if_missing(self, tmp_path):
        sub = tmp_path / 'created_on_demand'
        assert not sub.exists()
        generate_stub('foo', out_dir=str(sub))
        assert sub.is_dir()


# ---------------------------------------------------------------------------
# 5. Footer comment block
# ---------------------------------------------------------------------------

class TestFooter:
    def test_yaml_ends_with_reference_pointer(self, tmpout):
        path = generate_stub('footer_check', out_dir=str(tmpout))
        text = path.read_text()
        # Trailing comment block must point at REFERENCE.yaml + README
        # and show the user how to run the resulting config.
        tail = '\n'.join(text.rstrip().splitlines()[-15:])
        assert 'REFERENCE.yaml' in tail
        assert 'README' in tail
        assert 'python -m phast run' in tail
        # Final non-empty line is a comment (the closing dashed banner).
        assert text.rstrip().splitlines()[-1].lstrip().startswith('#')


# ---------------------------------------------------------------------------
# 6. CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_creates_file(self, tmp_path, capsys):
        out = tmp_path / 'configs'
        scaffolder_main([
            'cli_smoke',
            '--type', 'quasi_static',
            '--material', 'pmma_bleyer',
            '--geometry', 'rectangular_sent',
            '--out', str(out),
        ])
        captured = capsys.readouterr()
        assert 'Created' in captured.out
        assert (out / 'cli_smoke.yaml').exists()

    @pytest.mark.parametrize('solver_type', sorted(SOLVER_DEFAULTS))
    def test_cli_accepts_every_solver_default(self, tmp_path, solver_type):
        out = tmp_path / solver_type
        scaffolder_main([
            f'cli_{solver_type}',
            '--type', solver_type,
            '--out', str(out),
        ])
        assert (out / f'cli_{solver_type}.yaml').exists()

    def test_cli_unknown_type_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            scaffolder_main([
                'bad', '--type', 'not_a_solver',
                '--out', str(tmp_path),
            ])


# ---------------------------------------------------------------------------
# 7. Sanity: registries we depend on are non-empty
# ---------------------------------------------------------------------------

class TestRegistries:
    def test_geometry_defaults_cover_known_generators(self):
        # Every entry in the validator's geometry enum should have a
        # default parameter set we can render.
        from phast.config_validation import ENUMS
        for gen in ENUMS['geometry.type']:
            assert gen in GEOMETRY_DEFAULTS, (
                f"Scaffolder missing default params for generator {gen!r}; "
                f"add it to scaffolder.GEOMETRY_DEFAULTS."
            )

    def test_solver_defaults_cover_validator_enum(self):
        from phast.config_validation import ENUMS
        for st in ENUMS['solver.solver_type']:
            assert st in SOLVER_DEFAULTS
