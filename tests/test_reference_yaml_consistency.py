"""Consistency tests for ``configs/REFERENCE.yaml`` (issue #148).

The reference YAML is auto-generated from the dataclass schema in
``phast/config.py`` plus the side dicts (``ENUMS`` /
``RANGES`` / ``MUTEX``) in ``phast/config_validation.py``.

These tests guarantee:

1. The checked-in ``configs/REFERENCE.yaml`` matches the output of
   ``scripts/generate_reference_yaml.py``. Hand-edits to REFERENCE.yaml
   without rerunning the generator (or schema changes without
   regenerating) cause this test to fail.
2. New schema fields propagate: adding a ``Field`` to a section
   dataclass is reflected in the next regeneration.
3. New enum values propagate: adding a value to ``ENUMS`` is
   reflected in the regenerated content.
4. The generator is deterministic across runs (same content twice).
"""

from __future__ import annotations

import importlib
import os

import pytest
import yaml

pytestmark = pytest.mark.docs


# Resolve the repo root so the script and configs/ directory are
# discoverable regardless of where pytest is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_REFERENCE_PATH = os.path.join(_REPO_ROOT, 'configs', 'REFERENCE.yaml')
_TEMPLATE_PATH = os.path.join(_REPO_ROOT, 'configs', 'REFERENCE.template.yaml')


def _import_generator():
    """Import the generator module from ``scripts/``.

    The script lives outside the package, so we add it to ``sys.path``
    on demand. Cached after first call.
    """
    import sys
    scripts_dir = os.path.join(_REPO_ROOT, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module('generate_reference_yaml')


# ---------------------------------------------------------------------------
# (1) checked-in file matches generator output
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Whitespace-normalised text for tolerant comparison.

    Trailing whitespace on each line is stripped and a trailing newline
    is enforced. Inner spacing inside lines is preserved (column
    alignment matters in the generated comments).
    """
    lines = [line.rstrip() for line in text.splitlines()]
    return '\n'.join(lines).rstrip('\n') + '\n'


def test_reference_matches_generator_output():
    gen = _import_generator()
    generated = gen.generate()
    with open(_REFERENCE_PATH, 'r') as f:
        on_disk = f.read()
    if _normalise(on_disk) != _normalise(generated):
        import difflib
        diff = ''.join(difflib.unified_diff(
            on_disk.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile='configs/REFERENCE.yaml (checked-in)',
            tofile='generator output',
        ))
        pytest.fail(
            "configs/REFERENCE.yaml is out of sync with the dataclass "
            "schema.\nRerun: python scripts/generate_reference_yaml.py "
            "--write\n\n" + diff
        )


# ---------------------------------------------------------------------------
# (2) new dataclass field propagates
# ---------------------------------------------------------------------------

def test_new_dataclass_field_picked_up(monkeypatch):
    """Adding a ``Field`` to a section dataclass surfaces in the
    regenerated output."""
    import dataclasses

    gen = _import_generator()
    cfg = importlib.import_module('phast.config')

    # Build a sentinel dataclass cloned from OutputConfig with one extra
    # field. Patching the schema mapping points the generator at it.
    extra_field = dataclasses.field(default=42)
    NewOutput = dataclasses.make_dataclass(
        'OutputConfigSentinel',
        [('synthetic_extra', int, extra_field)],
        bases=(cfg.OutputConfig,),
    )
    # NewOutput's __doc__ defaults to None; set so renderer doesn't trip.
    NewOutput.__doc__ = cfg.OutputConfig.__doc__

    patched = dict(gen._SCHEMA_SECTIONS)
    patched['output'] = (NewOutput, '  ')
    monkeypatch.setattr(gen, '_SCHEMA_SECTIONS', patched)

    rendered = gen.generate()
    assert 'synthetic_extra: 42' in rendered, (
        "generator did not pick up new dataclass field 'synthetic_extra'"
    )


# ---------------------------------------------------------------------------
# (3) new enum value propagates
# ---------------------------------------------------------------------------

def test_new_enum_value_picked_up(monkeypatch):
    gen = _import_generator()
    val = importlib.import_module('phast.config_validation')

    # Add a sentinel choice to solver.solver_type.
    extended = dict(val.ENUMS)
    extended['solver.solver_type'] = (
        list(val.ENUMS['solver.solver_type']) + ['synthetic_solver']
    )
    monkeypatch.setattr(val, 'ENUMS', extended)

    rendered = gen.generate()
    assert 'synthetic_solver' in rendered, (
        "generator did not surface new ENUM entry for solver.solver_type"
    )


# ---------------------------------------------------------------------------
# (4) deterministic across runs
# ---------------------------------------------------------------------------

def test_generator_is_deterministic():
    gen = _import_generator()
    a = gen.generate()
    b = gen.generate()
    assert a == b, "generator output is not deterministic across runs"


# ---------------------------------------------------------------------------
# Sanity: the generated file is still a valid YAML mapping with the
# expected top-level sections.
# ---------------------------------------------------------------------------

def test_reference_is_valid_yaml():
    with open(_REFERENCE_PATH, 'r') as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict)
    expected = {
        'problem', 'geometry', 'material', 'boundary_conditions',
        'loading', 'solver', 'output', 'device', 'initial_conditions',
    }
    missing = expected - set(doc)
    assert not missing, f"missing top-level sections: {missing}"


def test_reference_passes_schema_validator():
    """The generated file itself must validate against the schema it
    claims to document."""
    from phast.config_validation import validate_config_file
    _, errors = validate_config_file(_REFERENCE_PATH)
    assert errors == [], (
        "generated REFERENCE.yaml does not validate against its own "
        "schema:\n" + '\n'.join(e.format() for e in errors)
    )
