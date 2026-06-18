"""
Unit parsing and normalisation for phast YAML inputs.

Reference internal convention (historical, mesh-driven)
-------------------------------------------------------
The solver internals run in a *consistent unit system* with these base units:

    length   : mm       (mesh coordinates)
    mass     : tonne    (= 1000 kg)
    time     : s
    force    : N        (= tonne * mm / s^2)
    stress   : MPa      (= N / mm^2)
    density  : tonne / mm^3
    energy   : N * mm   (= mJ)
    Gc       : N / mm   (= kJ / m^2 = 1000 J / m^2)
    traction : N / mm   (2D boundary force per unit length)
    velocity : mm / s   (impact velocities given in m/s are *1e3 internally)

This convention is dictated by the mesh generators (which produce
millimetre coordinates) and is preserved end-to-end through the solver,
post-processing, and IO layers. Refactoring to true SI (Pa, m, kg) would
require touching every numeric path and is intentionally out of scope.

Public API
----------
``parse_quantity(value, kind)`` — accept either:

  * a bare ``float`` / ``int`` (interpreted as already in the internal
    unit for ``kind``; status quo for legacy configs and presets), or
  * a ``str`` of the form ``"<number> <unit>"`` (SI-friendly suffixes),
    which is normalised down to the internal unit.

Supported ``kind`` values and the internal unit each maps to::

    'stress'    -> MPa            (E, sigma_ts, sigma_cs)
    'Gc'        -> N/mm           (= kJ/m^2)
    'length'    -> mm             (l0, mesh sizes, displacements)
    'density'   -> tonne/mm^3
    'traction'  -> N/mm           (2D boundary traction; stress suffixes
                                   assume unit out-of-plane thickness)
    'time'      -> s
    'velocity'  -> m/s            (loading.v0 only; converted *1e3 to mm/s
                                   inside compute_load_factor at runtime)

Examples::

    parse_quantity(32000.0, 'stress')       -> 32000.0      # MPa, status quo
    parse_quantity('32 GPa', 'stress')      -> 32000.0      # GPa -> MPa
    parse_quantity('3 J/m^2', 'Gc')         -> 3.0e-3       # J/m^2 -> N/mm
    parse_quantity('0.25 mm', 'length')     -> 0.25
    parse_quantity('2450 kg/m^3', 'density')-> 2.45e-9      # SI -> tonne/mm^3
    parse_quantity('5 MPa', 'traction')     -> 5.0          # unit thickness
    parse_quantity('16.5 m/s', 'velocity')  -> 16.5         # m/s, status quo
    parse_quantity('1 us', 'time')          -> 1.0e-6

Bare floats are passed through unchanged so that every existing config
and preset retains *bit-identical* numerical values. Suffixed strings
are purely an ergonomic addition.
"""

from __future__ import annotations

import re
from typing import Union

Number = Union[int, float]
QuantityInput = Union[int, float, str]


# ---------------------------------------------------------------------------
# Conversion tables: each maps a *suffix string* to the multiplicative
# factor that converts a value in that unit to the internal unit listed
# in the module docstring.
# ---------------------------------------------------------------------------

# stress -> MPa
_STRESS_TO_MPA = {
    'Pa':  1.0e-6,
    'kPa': 1.0e-3,
    'MPa': 1.0,
    'GPa': 1.0e3,
    'N/mm^2': 1.0,
    'N/mm2':  1.0,
}

# Gc / energy-release rate -> N/mm  (== kJ/m^2 == 1000 J/m^2)
_GC_TO_NPMM = {
    'N/mm':    1.0,
    'kN/mm':   1.0e3,
    'J/m^2':   1.0e-3,
    'J/m2':    1.0e-3,
    'kJ/m^2':  1.0,
    'kJ/m2':   1.0,
    'N/m':     1.0e-3,   # 1 N/m = 1 J/m^2
    'kN/m':    1.0,      # 1 kN/m = 1 kJ/m^2
}

# length -> mm
_LENGTH_TO_MM = {
    'm':   1.0e3,
    'cm':  10.0,
    'mm':  1.0,
    'um':  1.0e-3,
    'μm':  1.0e-3,
    'nm':  1.0e-6,
}

# density -> tonne/mm^3
# 1 kg/m^3  = 1e-12 tonne/mm^3
# 1 g/cm^3  = 1e-9  tonne/mm^3
_DENSITY_TO_TPMM3 = {
    'tonne/mm^3': 1.0,
    'tonne/mm3':  1.0,
    't/mm^3':     1.0,
    'kg/m^3':     1.0e-12,
    'kg/m3':      1.0e-12,
    'g/cm^3':     1.0e-9,
    'g/cm3':      1.0e-9,
    'g/mm^3':     1.0e-6,
}

# traction -> N/mm
#
# Boundary tractions are assembled as force per boundary length in the
# 2D weak form. The N/mm entries are the reference units. Stress-like
# suffixes are accepted as a convenience for plane-stress/plane-strain
# benchmark YAMLs where the implicit out-of-plane thickness is 1 mm.
_TRACTION_TO_NPMM = {
    'N/mm':    1.0,
    'kN/mm':   1.0e3,
    'N/m':     1.0e-3,
    'kN/m':    1.0,
    'Pa':      1.0e-6,
    'kPa':     1.0e-3,
    'MPa':     1.0,
    'GPa':     1.0e3,
    'N/mm^2':  1.0,
    'N/mm2':   1.0,
}

# time -> s
_TIME_TO_S = {
    's':   1.0,
    'ms':  1.0e-3,
    'us':  1.0e-6,
    'μs':  1.0e-6,
    'ns':  1.0e-9,
}

# velocity -> m/s
# NOTE: ``loading.v0`` is the *only* velocity field exposed in YAML and
# its legacy bare-float semantics is m/s (see config.py:143 — there is
# an explicit ``v0_mm = loading.v0 * 1e3`` conversion before the field
# reaches the solver). To keep bit-identical behaviour with legacy
# configs that supplied a bare float in m/s, the reference unit for
# ``kind='velocity'`` is m/s, not the internal mm/s. The runtime
# mm/s conversion in compute_load_factor is unchanged.
_VELOCITY_TO_MPS = {
    'm/s':   1.0,
    'mm/s':  1.0e-3,
    'cm/s':  1.0e-2,
    'km/s':  1.0e3,
    'mm/us': 1.0e3,    # 1 mm/us = 1000 m/s
    'm/ms':  1.0e3,
}

_TABLES = {
    'stress':   (_STRESS_TO_MPA,    'MPa'),
    'Gc':       (_GC_TO_NPMM,       'N/mm'),
    'length':   (_LENGTH_TO_MM,     'mm'),
    'density':  (_DENSITY_TO_TPMM3, 'tonne/mm^3'),
    'traction': (_TRACTION_TO_NPMM,  'N/mm'),
    'time':     (_TIME_TO_S,        's'),
    'velocity': (_VELOCITY_TO_MPS,  'm/s'),
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Number with optional sign and exponent, then optional whitespace, then unit.
_NUMBER_RE = re.compile(
    r'^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]+.*?)\s*$'
)


def _normalise_unit(raw: str) -> str:
    """Lightly normalise a unit string for table lookup.

    Strips whitespace, replaces unicode minus with ASCII, collapses ``**`` to
    ``^``. Does *not* lower-case (units are case-sensitive: ``mm`` != ``Mm``).
    """
    s = raw.strip()
    s = s.replace('**', '^')
    return s


def parse_quantity(value: QuantityInput, kind: str) -> float:
    """Parse a YAML-supplied quantity into the internal unit for ``kind``.

    Parameters
    ----------
    value : float | int | str
        Bare numeric: interpreted as already in the internal unit
        (status quo, no conversion). String: ``"<number> <unit>"``,
        converted via the table for ``kind``.
    kind : str
        One of: 'stress', 'Gc', 'length', 'density', 'traction', 'time',
        'velocity'.

    Returns
    -------
    float
        Value expressed in the internal unit for ``kind``.

    Raises
    ------
    ValueError
        If ``kind`` is not recognised, the string cannot be parsed, or
        the unit suffix is not in the conversion table.
    """
    if kind not in _TABLES:
        raise ValueError(
            f"parse_quantity: unknown kind '{kind}'. "
            f"Expected one of {sorted(_TABLES.keys())}."
        )

    if isinstance(value, bool):  # bool is a subclass of int — reject early
        raise ValueError(f"parse_quantity: bool is not a valid quantity ({value!r}).")

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(
            f"parse_quantity: expected float, int, or str for kind={kind!r}, "
            f"got {type(value).__name__}: {value!r}"
        )

    s = value.strip()
    if not s:
        raise ValueError(f"parse_quantity: empty string for kind={kind!r}.")

    # Try plain numeric string first ("3.0", "1e-3"). YAML's parser
    # generally already converts these to floats, but configs sometimes
    # quote them (e.g. ``Gc: "3.0"``); honour that as a bare float.
    try:
        return float(s)
    except ValueError:
        pass

    m = _NUMBER_RE.match(s)
    if m is None:
        raise ValueError(
            f"parse_quantity: could not parse '{value}' for kind={kind!r}. "
            f"Expected '<number> <unit>' (e.g. '32 GPa', '3 J/m^2')."
        )
    number_part, unit_part = m.group(1), _normalise_unit(m.group(2))
    try:
        n = float(number_part)
    except ValueError as exc:
        raise ValueError(
            f"parse_quantity: invalid number '{number_part}' in '{value}'."
        ) from exc

    table, internal = _TABLES[kind]
    if unit_part not in table:
        raise ValueError(
            f"parse_quantity: unknown unit '{unit_part}' for kind={kind!r}. "
            f"Internal unit is {internal!r}. "
            f"Supported: {sorted(table.keys())}."
        )
    return n * table[unit_part]


# Convenience: kind hints for material override fields. Centralised so
# that ``material.py`` and ``config.py`` agree on which override keys
# accept unit-suffixed strings.
MATERIAL_OVERRIDE_KINDS = {
    'E':         'stress',
    'sigma_ts':  'stress',
    'Gc':        'Gc',
    'l0':        'length',
    'rho':       'density',
}

# Loading fields that accept unit-suffixed strings.
LOADING_QUANTITY_KINDS = {
    't_total':  'time',
    'dt':       'time',
    't_ramp':   'time',
    'v0':       'velocity',
    'disp_max': 'length',
    'prestrain_displacement': 'length',
}

# Boundary-condition fields that accept unit-suffixed strings.
BOUNDARY_VALUE_QUANTITY_KINDS = {
    'prescribe': 'length',
    'neumann':   'traction',
    'traction':  'traction',
}

BOUNDARY_TIME_QUANTITY_KINDS = {
    't_ramp': 'time',
    't_hold': 'time',
}
