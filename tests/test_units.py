"""
Tests for ``phast.units`` and the bit-exact equivalence of
existing material presets after wiring unit-suffix parsing into
``create_material``.

Two layers of coverage:

1. ``test_parse_quantity_*`` exercise the parser on representative
   suffix forms for each kind (stress, Gc, length, density, time,
   velocity).
2. ``test_preset_bit_exact_internal_values`` is the safety check
   demanded by issue #141: every preset must produce a Material object
   whose stored numeric fields match the *literal* values in the preset
   dict to machine precision. This proves no off-by-1000 unit-conversion
   bug crept into the migration; the literal preset values were left
   unchanged on purpose, so the assertion is trivially true if (and
   only if) ``create_material`` is still pass-through for bare floats.
"""

from __future__ import annotations

import math
import pytest

from phast.units import (
    parse_quantity,
    MATERIAL_OVERRIDE_KINDS,
    LOADING_QUANTITY_KINDS,
    BOUNDARY_VALUE_QUANTITY_KINDS,
    BOUNDARY_TIME_QUANTITY_KINDS,
)
from phast.material import create_material


# ---------------------------------------------------------------------------
# parse_quantity
# ---------------------------------------------------------------------------

class TestParseQuantityStress:
    def test_bare_float_passthrough(self):
        assert parse_quantity(32000.0, 'stress') == 32000.0

    def test_bare_int_passthrough(self):
        assert parse_quantity(7, 'stress') == 7.0

    def test_gpa_to_mpa(self):
        assert parse_quantity('32 GPa', 'stress') == pytest.approx(32000.0)

    def test_mpa_identity(self):
        assert parse_quantity('210000 MPa', 'stress') == 210000.0

    def test_pa_to_mpa(self):
        assert parse_quantity('6.0e9 Pa', 'stress') == pytest.approx(6000.0)

    def test_kpa(self):
        assert parse_quantity('1500 kPa', 'stress') == pytest.approx(1.5)

    def test_n_per_mm2(self):
        assert parse_quantity('210000 N/mm^2', 'stress') == 210000.0


class TestParseQuantityGc:
    def test_bare_float_passthrough(self):
        # 3.0e-3 N/mm is the glass_borden literal; must pass straight through.
        assert parse_quantity(3.0e-3, 'Gc') == 3.0e-3

    def test_jm2_to_npmm(self):
        # 3 J/m^2 = 3e-3 N/mm  (Borden glass)
        assert parse_quantity('3 J/m^2', 'Gc') == pytest.approx(3.0e-3)

    def test_kjm2_identity(self):
        assert parse_quantity('3.0e-3 kJ/m^2', 'Gc') == pytest.approx(3.0e-3)

    def test_npm_to_npmm(self):
        # 1 N/m = 1 J/m^2 = 1e-3 N/mm
        assert parse_quantity('1 N/m', 'Gc') == pytest.approx(1.0e-3)

    def test_npmm_identity(self):
        assert parse_quantity('2.7 N/mm', 'Gc') == 2.7


class TestParseQuantityLength:
    def test_bare_float(self):
        assert parse_quantity(0.25, 'length') == 0.25

    def test_mm_identity(self):
        assert parse_quantity('0.25 mm', 'length') == 0.25

    def test_m_to_mm(self):
        assert parse_quantity('0.001 m', 'length') == pytest.approx(1.0)

    def test_um(self):
        assert parse_quantity('5 um', 'length') == pytest.approx(5.0e-3)


class TestParseQuantityDensity:
    def test_bare_float(self):
        assert parse_quantity(2.45e-9, 'density') == 2.45e-9

    def test_kgm3_to_tpmm3(self):
        # 2450 kg/m^3 = 2.45e-9 tonne/mm^3  (Borden glass)
        assert parse_quantity('2450 kg/m^3', 'density') == pytest.approx(2.45e-9)

    def test_gcm3(self):
        # 2.45 g/cm^3 = 2.45e-9 tonne/mm^3
        assert parse_quantity('2.45 g/cm^3', 'density') == pytest.approx(2.45e-9)

    def test_tpmm3_identity(self):
        assert parse_quantity('7.8e-9 tonne/mm^3', 'density') == pytest.approx(7.8e-9)


class TestParseQuantityTraction:
    def test_bare_float(self):
        assert parse_quantity(1.0, 'traction') == 1.0

    def test_npmm_identity(self):
        assert parse_quantity('2.5 N/mm', 'traction') == pytest.approx(2.5)

    def test_kn_per_m_to_npmm(self):
        assert parse_quantity('1 kN/m', 'traction') == pytest.approx(1.0)

    def test_n_per_m_to_npmm(self):
        assert parse_quantity('1000 N/m', 'traction') == pytest.approx(1.0)

    def test_mpa_unit_thickness_convention(self):
        assert parse_quantity('5 MPa', 'traction') == pytest.approx(5.0)


class TestParseQuantityTime:
    def test_bare_float(self):
        assert parse_quantity(1.0e-6, 'time') == 1.0e-6

    def test_us(self):
        assert parse_quantity('1 us', 'time') == pytest.approx(1.0e-6)

    def test_ns(self):
        assert parse_quantity('500 ns', 'time') == pytest.approx(500e-9)


class TestParseQuantityVelocity:
    # NOTE: velocity normalises to m/s (NOT internal mm/s). Rationale:
    # the only YAML velocity field is loading.v0 whose legacy bare-float
    # semantics is m/s — see config.py:143 (compute_load_factor does
    # v0_mm = loading.v0 * 1e3 before passing to the solver).
    def test_bare_float(self):
        assert parse_quantity(16.5, 'velocity') == 16.5

    def test_mps_identity(self):
        assert parse_quantity('16.5 m/s', 'velocity') == 16.5

    def test_mmps_to_mps(self):
        # 1000 mm/s = 1 m/s
        assert parse_quantity('1000 mm/s', 'velocity') == pytest.approx(1.0)

    def test_kmps_to_mps(self):
        assert parse_quantity('2 km/s', 'velocity') == pytest.approx(2000.0)


class TestParseQuantityErrors:
    def test_unknown_kind(self):
        with pytest.raises(ValueError, match='unknown kind'):
            parse_quantity(1.0, 'temperature')

    def test_unknown_unit(self):
        with pytest.raises(ValueError, match='unknown unit'):
            parse_quantity('5 furlongs', 'length')

    def test_unparseable_string(self):
        with pytest.raises(ValueError):
            parse_quantity('not a number', 'stress')

    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_quantity('', 'stress')

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            parse_quantity(True, 'stress')


# ---------------------------------------------------------------------------
# Bit-exact preset equivalence — the issue #141 safety check
# ---------------------------------------------------------------------------
#
# Reference table: each preset's *internal-unit* values, manually
# transcribed from the literal Python dict in material.py. The migration
# in this PR adds unit-string parsing as an ergonomic input layer; it
# does NOT change any preset literal. If the test below fails, somebody
# accidentally divided/multiplied a value during the migration.

PRESET_REFERENCE = {
    'default':              dict(E=210000.0, nu=0.3,  Gc=2.7,    l0=0.005,  rho=7.8e-9),
    'steel_pf':             dict(E=210000.0, nu=0.3,  Gc=2.7,    l0=0.005,  rho=7.8e-9),
    'miehe_tension':        dict(E=210000.0, nu=0.3,  Gc=2.7,    l0=0.015,  rho=7.8e-9),
    'miehe_shear':          dict(E=210000.0, nu=0.3,  Gc=2.7,    l0=0.06,   rho=7.8e-9),
    'three_point_bending':  dict(E=20800.0,  nu=0.3,  Gc=0.5,    l0=0.06,   rho=1.2e-9),
    'l_shaped_glass':       dict(E=70000.0,  nu=0.23, Gc=0.008,  l0=0.4,    rho=2.5e-9),
    'l_shaped_concrete':    dict(E=25850.0,  nu=0.18, Gc=0.089,  l0=1.1875, rho=2.4e-9),
    'alumina_kumar':        dict(E=335000.0, nu=0.25, Gc=0.0268, l0=0.04,   rho=3.9e-9),
    'brittle_ceramic':      dict(E=370000.0, nu=0.22, Gc=0.042,  l0=0.01,   rho=3.9e-9),
    'pmma':                 dict(E=3000.0,   nu=0.35, Gc=0.3,    l0=0.02,   rho=1.18e-9),
    'glass_borden':         dict(E=32000.0,  nu=0.20, Gc=3.0e-3, l0=0.25,   rho=2.45e-9),
    'pmma_bleyer':          dict(E=3090.0,   nu=0.35, Gc=0.3,    l0=0.1,    rho=1.18e-9),
    'maraging_steel_kw':    dict(E=190000.0, nu=0.30, Gc=22.13,  l0=0.195,  rho=8.0e-9),
    'basalt_brazilian':     dict(E=20110.0,  nu=0.20, Gc=0.1,    l0=1.25,   rho=2.74e-9),
    'soda_lime_glass':      dict(E=72000.0,  nu=0.25, Gc=9.0,    l0=0.25,   rho=2.44e-9),
}


@pytest.mark.parametrize('preset', sorted(PRESET_REFERENCE.keys()))
def test_preset_bit_exact_internal_values(preset):
    """Every preset's resulting Material has fields == reference internal values.

    Tolerance: machine-precision (rel < 1e-15). Since no preset literal
    was changed in the migration and bare floats bypass parse_quantity,
    equality must be exact.
    """
    mat = create_material(preset)
    ref = PRESET_REFERENCE[preset]
    for field, expected in ref.items():
        actual = getattr(mat, field)
        assert actual == expected, (
            f"Preset {preset!r}: field {field!r} = {actual!r}, "
            f"expected bit-exact {expected!r}."
        )


# ---------------------------------------------------------------------------
# End-to-end: SI override strings round-trip to the legacy literal value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('preset,override_field,si_string,expected_internal', [
    # E: 32 GPa  (glass_borden)
    ('glass_borden', 'E', '32 GPa', 32000.0),
    # E: 6 GPa
    ('glass_borden', 'E', '6 GPa', 6000.0),
    # Gc: 3 J/m^2 (Borden) -> 3e-3 N/mm
    ('glass_borden', 'Gc', '3 J/m^2', 3.0e-3),
    # Gc: 2.28 kJ/m^2 -> 2.28 N/mm
    ('glass_borden', 'Gc', '2.28 kJ/m^2', 2.28),
    # l0: 0.25 mm -> 0.25
    ('glass_borden', 'l0', '0.25 mm', 0.25),
    # rho: 2450 kg/m^3 -> 2.45e-9 tonne/mm^3 (Borden)
    ('glass_borden', 'rho', '2450 kg/m^3', 2.45e-9),
    # rho: 2.45 g/cm^3 -> same
    ('glass_borden', 'rho', '2.45 g/cm^3', 2.45e-9),
    # Steel rho via SI string: 7800 kg/m^3 -> 7.8e-9
    ('steel_pf', 'rho', '7800 kg/m^3', 7.8e-9),
])
def test_si_override_strings_match_internal_literals(
        preset, override_field, si_string, expected_internal):
    """Issue #141 ergonomic input form converges to the legacy literal."""
    mat = create_material(preset, **{override_field: si_string})
    actual = getattr(mat, override_field)
    assert math.isclose(actual, expected_internal, rel_tol=1e-12), (
        f"{preset}.{override_field} = {si_string!r} -> {actual}, "
        f"expected {expected_internal}."
    )


def test_loading_t_total_string_us():
    """LoadingConfig accepts '80 us' and normalises to seconds."""
    from phast.config import _dict_to_dataclass, LoadingConfig
    lc = _dict_to_dataclass(LoadingConfig, {'t_total': '80 us'})
    assert lc.t_total == pytest.approx(80.0e-6)


def test_loading_v0_string_mps():
    """LoadingConfig.v0 accepts '16.5 m/s'. Stored value is m/s (matches
    the legacy bare-float convention; compute_load_factor *1e3 stays).
    """
    from phast.config import _dict_to_dataclass, LoadingConfig
    lc = _dict_to_dataclass(LoadingConfig, {'v0': '16.5 m/s'})
    assert lc.v0 == pytest.approx(16.5)


def test_loading_bare_float_unchanged():
    """Bare floats remain bit-identical (no normalisation applied)."""
    from phast.config import _dict_to_dataclass, LoadingConfig
    lc = _dict_to_dataclass(LoadingConfig, {'t_total': 80.0e-6, 'v0': 16.5})
    assert lc.t_total == 80.0e-6
    assert lc.v0 == 16.5


def test_loading_v0_then_compute_load_factor_roundtrip():
    """End-to-end: v0='16.5 m/s' produces same internal mm/s as legacy 16.5.

    Confirms that the m/s-canonical kind plus the unchanged *1e3 in
    compute_load_factor still yields the legacy mm-units displacement.
    """
    from phast.config import (
        _dict_to_dataclass, LoadingConfig, compute_load_factor)
    lc_str = _dict_to_dataclass(
        LoadingConfig, {'v0': '16.5 m/s', 'ramp_type': 'velocity_impact', 't_ramp': 0.0})
    lc_legacy = LoadingConfig(v0=16.5, ramp_type='velocity_impact', t_ramp=0.0)
    for step in (0, 10, 100):
        assert compute_load_factor(step, 1.0e-7, lc_str) == \
            compute_load_factor(step, 1.0e-7, lc_legacy)


def test_boundary_condition_prescribed_displacement_units():
    """Prescribed displacement strings normalise to internal mm."""
    from phast.config import _dict_to_dataclass, BoundaryConditionEntry
    bc = _dict_to_dataclass(
        BoundaryConditionEntry,
        {'type': 'prescribe', 'component': 1, 'value': '250 um'},
    )
    assert bc.value == pytest.approx(0.25)


def test_boundary_condition_traction_and_ramp_units():
    """Traction BC values use N/mm; ramp times use seconds."""
    from phast.config import _dict_to_dataclass, BoundaryConditionEntry
    bc = _dict_to_dataclass(
        BoundaryConditionEntry,
        {
            'type': 'traction',
            'component': 1,
            'value': '5 MPa',
            't_ramp': '10 us',
            't_hold': '25 us',
        },
    )
    assert bc.value == pytest.approx(5.0)
    assert bc.t_ramp == pytest.approx(10.0e-6)
    assert bc.t_hold == pytest.approx(25.0e-6)


def test_rigid_connector_prescribe_units():
    """Rigid connector prescribed translations accept length suffixes."""
    from phast.config import _dict_to_dataclass, BoundaryConditionEntry
    bc = _dict_to_dataclass(
        BoundaryConditionEntry,
        {
            'type': 'rigid_connector',
            'master': 'pin.centre',
            'prescribe': {'x': '0.1 mm', 'y': '250 um'},
        },
    )
    assert bc.prescribe['x'] == pytest.approx(0.1)
    assert bc.prescribe['y'] == pytest.approx(0.25)


def test_boundary_condition_bad_unit_fails_early():
    """Invalid BC unit strings fail at config load instead of reaching solvers."""
    from phast.config import _dict_to_dataclass, BoundaryConditionEntry
    with pytest.raises(ValueError, match='boundary_conditions.value'):
        _dict_to_dataclass(
            BoundaryConditionEntry,
            {'type': 'prescribe', 'component': 1, 'value': '1 MPa'},
        )


def test_known_kinds_consistent():
    """Sanity: every override-kind table maps to a known parser kind."""
    from phast.units import _TABLES
    for k, kind in MATERIAL_OVERRIDE_KINDS.items():
        assert kind in _TABLES, f"{k} -> {kind} not in parser tables"
    for k, kind in LOADING_QUANTITY_KINDS.items():
        assert kind in _TABLES, f"{k} -> {kind} not in parser tables"
    for k, kind in BOUNDARY_VALUE_QUANTITY_KINDS.items():
        assert kind in _TABLES, f"{k} -> {kind} not in parser tables"
    for k, kind in BOUNDARY_TIME_QUANTITY_KINDS.items():
        assert kind in _TABLES, f"{k} -> {kind} not in parser tables"
