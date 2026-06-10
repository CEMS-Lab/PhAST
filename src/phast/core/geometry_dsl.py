"""
Geometry primitive vocabulary (Phase 2.1, issue #142), boolean-op recipe
(Phase 2.2, issue #143), and named-group registry (Phase 2.3, issue #144).

Declarative geometry primitives + a :class:`Domain` dataclass that composes
them via union / difference / intersection, plus user-named physical
groups that can be authored in YAML and parsed into lightweight Python
dataclasses. This module is intentionally self-contained and side-effect
free: it does **not** emit Gmsh ``.geo`` text, run the boolean
computation, or resolve groups onto mesh entities. Those land in
follow-up issues:

* Mesh refinement DSL:         issue #145
* Compiler -> ``.geo`` + group materialisation: issue #146
* Migrating ``configs/*.yaml``: issue #147

YAML form parsed by :func:`parse_primitives`::

    geometry:
      units: mm
      primitives:
        plate:    { type: rectangle, origin: [0, 0],     size: [65, 120] }
        big_hole: { type: circle,    center: [36.5, 51], radius: 10 }
        poly_a:   { type: polygon,   vertices: [[0,0], [10,0], [5,5]] }
        notch_pt: { type: point,     coords: [10, 65] }
        seg_top:  { type: line_segment, from: [0, 100], to: [65, 100] }
      domain:
        base: plate
        subtract: [big_hole]
        add: []
        intersect: []

Internal storage is always millimetres (mm), matching the solver's
mm-tonne-N-s-MPa unit convention. Inputs declared as ``units: m`` are
converted on parse.

Each primitive auto-exposes three selector properties --
``.boundary``, ``.interior``, ``.centre`` -- as lightweight
:class:`Selector` placeholders. Downstream issues (#143, #144, #146) are
expected to resolve these against the meshed domain. They are inert here.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# Internal canonical unit is millimetres; see PR #156 / mm-tonne-N-s-MPa.
_UNIT_TO_MM: Dict[str, float] = {
    'mm': 1.0,
    'millimeter': 1.0,
    'millimeters': 1.0,
    'm':  1000.0,
    'meter': 1000.0,
    'meters': 1000.0,
}


def _unit_scale(units: str) -> float:
    """Return the multiplicative factor that converts ``units`` into mm."""
    if units is None:
        return 1.0
    key = str(units).strip().lower()
    if key not in _UNIT_TO_MM:
        raise ValueError(
            f"Unsupported geometry units '{units}'. "
            f"Supported: {sorted(set(_UNIT_TO_MM))}."
        )
    return _UNIT_TO_MM[key]


# ---------------------------------------------------------------------------
# Selector placeholders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Selector:
    """A symbolic reference to a topological subset of a primitive.

    Selectors are *intentionally* inert in this issue: they carry the
    primitive's name and the kind of subset (``'boundary'``, ``'interior'``,
    ``'centre'``) but do not resolve to mesh entities. The geometry compiler
    (issue #146) and the named-group resolver (issue #144) will consume
    them.
    """
    primitive: str
    kind: str  # 'boundary' | 'interior' | 'centre'

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Selector({self.primitive}.{self.kind})"


# ---------------------------------------------------------------------------
# Primitive ABC + concrete dataclasses
# ---------------------------------------------------------------------------

class Primitive(ABC):
    """Abstract base for all geometry primitives.

    Subclasses are dataclasses carrying their geometric parameters in the
    canonical internal unit (mm). The instance attribute :attr:`name` is
    set by :func:`parse_primitives` once the primitive is keyed in the YAML
    mapping.
    """

    name: str = ''  # populated by parse_primitives

    # --- Auto-exposed selectors --------------------------------------------
    @property
    def boundary(self) -> Selector:
        """Selector for the primitive's boundary (1D entities)."""
        return Selector(primitive=self.name, kind='boundary')

    @property
    def interior(self) -> Selector:
        """Selector for the primitive's interior (2D entities)."""
        return Selector(primitive=self.name, kind='interior')

    @property
    def centre(self) -> Selector:
        """Selector for the primitive's geometric centre (point)."""
        return Selector(primitive=self.name, kind='centre')

    # --- Concrete-class hooks ----------------------------------------------
    @property
    @abstractmethod
    def kind(self) -> str:
        """Short string identifying the primitive type."""

    @abstractmethod
    def _scale(self, factor: float) -> 'Primitive':
        """Return a copy with all length quantities multiplied by ``factor``."""


@dataclass
class Rectangle(Primitive):
    """Axis-aligned rectangle defined by its bottom-left ``origin`` and
    ``size = (width, height)``. Stored in mm.
    """
    origin: Tuple[float, float] = (0.0, 0.0)
    size: Tuple[float, float] = (1.0, 1.0)
    name: str = ''

    @property
    def kind(self) -> str:
        return 'rectangle'

    def _scale(self, factor: float) -> 'Rectangle':
        ox, oy = self.origin
        sx, sy = self.size
        return Rectangle(
            origin=(ox * factor, oy * factor),
            size=(sx * factor, sy * factor),
            name=self.name,
        )


@dataclass
class Circle(Primitive):
    """Circle defined by its ``center`` and ``radius``. Stored in mm."""
    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    name: str = ''

    @property
    def kind(self) -> str:
        return 'circle'

    def _scale(self, factor: float) -> 'Circle':
        cx, cy = self.center
        return Circle(
            center=(cx * factor, cy * factor),
            radius=self.radius * factor,
            name=self.name,
        )


@dataclass
class Polygon(Primitive):
    """Closed polygon defined by an ordered list of ``vertices``. Stored
    in mm. The polygon is implicitly closed (last vertex auto-connects to
    first); user input must therefore have at least 3 distinct vertices.
    """
    vertices: List[Tuple[float, float]] = field(default_factory=list)
    name: str = ''

    @property
    def kind(self) -> str:
        return 'polygon'

    def _scale(self, factor: float) -> 'Polygon':
        return Polygon(
            vertices=[(x * factor, y * factor) for (x, y) in self.vertices],
            name=self.name,
        )


@dataclass
class Point(Primitive):
    """A single 2D point. Stored in mm."""
    coords: Tuple[float, float] = (0.0, 0.0)
    name: str = ''

    @property
    def kind(self) -> str:
        return 'point'

    def _scale(self, factor: float) -> 'Point':
        x, y = self.coords
        return Point(coords=(x * factor, y * factor), name=self.name)


@dataclass
class LineSegment(Primitive):
    """Straight line segment from ``start`` to ``end``. The YAML keys
    ``from`` / ``to`` map onto these (``from`` is reserved in Python).
    Stored in mm.
    """
    start: Tuple[float, float] = (0.0, 0.0)
    end: Tuple[float, float] = (1.0, 0.0)
    name: str = ''

    @property
    def kind(self) -> str:
        return 'line_segment'

    def _scale(self, factor: float) -> 'LineSegment':
        sx, sy = self.start
        ex, ey = self.end
        return LineSegment(
            start=(sx * factor, sy * factor),
            end=(ex * factor, ey * factor),
            name=self.name,
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _as_xy(value: Any, *, field_name: str, primitive_name: str) -> Tuple[float, float]:
    """Coerce a YAML 2-element sequence into a ``(float, float)`` tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"primitive '{primitive_name}': field '{field_name}' must be a "
            f"2-element [x, y] list, got {value!r}."
        )
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"primitive '{primitive_name}': field '{field_name}' contains "
            f"non-numeric values ({value!r})."
        ) from exc


def _require(spec: dict, key: str, primitive_name: str, prim_kind: str) -> Any:
    if key not in spec:
        raise ValueError(
            f"primitive '{primitive_name}' (type={prim_kind}) is missing "
            f"required field '{key}'."
        )
    return spec[key]


def _parse_one(name: str, spec: dict) -> Primitive:
    if not isinstance(spec, dict):
        raise ValueError(
            f"primitive '{name}' must be a mapping, got {type(spec).__name__}."
        )
    if 'type' not in spec:
        raise ValueError(f"primitive '{name}' is missing required field 'type'.")
    ptype = str(spec['type']).strip().lower()

    if ptype == 'rectangle':
        origin = _as_xy(_require(spec, 'origin', name, ptype),
                        field_name='origin', primitive_name=name)
        size = _as_xy(_require(spec, 'size', name, ptype),
                      field_name='size', primitive_name=name)
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(
                f"primitive '{name}' (rectangle): size must be strictly "
                f"positive, got {size}."
            )
        return Rectangle(origin=origin, size=size, name=name)

    if ptype == 'circle':
        center = _as_xy(_require(spec, 'center', name, ptype),
                        field_name='center', primitive_name=name)
        radius_raw = _require(spec, 'radius', name, ptype)
        try:
            radius = float(radius_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"primitive '{name}' (circle): radius must be numeric, "
                f"got {radius_raw!r}."
            ) from exc
        if radius <= 0:
            raise ValueError(
                f"primitive '{name}' (circle): radius must be strictly "
                f"positive, got {radius}."
            )
        return Circle(center=center, radius=radius, name=name)

    if ptype == 'polygon':
        verts_raw = _require(spec, 'vertices', name, ptype)
        if not isinstance(verts_raw, (list, tuple)) or len(verts_raw) < 3:
            raise ValueError(
                f"primitive '{name}' (polygon): vertices must be a list "
                f"of at least 3 [x, y] points, got {verts_raw!r}."
            )
        verts = [
            _as_xy(v, field_name=f'vertices[{i}]', primitive_name=name)
            for i, v in enumerate(verts_raw)
        ]
        return Polygon(vertices=verts, name=name)

    if ptype == 'point':
        coords = _as_xy(_require(spec, 'coords', name, ptype),
                        field_name='coords', primitive_name=name)
        return Point(coords=coords, name=name)

    if ptype == 'line_segment':
        start = _as_xy(_require(spec, 'from', name, ptype),
                       field_name='from', primitive_name=name)
        end = _as_xy(_require(spec, 'to', name, ptype),
                     field_name='to', primitive_name=name)
        if start == end:
            raise ValueError(
                f"primitive '{name}' (line_segment): 'from' and 'to' "
                f"coincide at {start}; segment has zero length."
            )
        return LineSegment(start=start, end=end, name=name)

    raise ValueError(
        f"primitive '{name}': unknown type '{spec['type']}'. "
        f"Supported: rectangle, circle, polygon, point, line_segment."
    )


def parse_primitives(geometry_dict: dict) -> Dict[str, Primitive]:
    """Parse the ``geometry`` block of a YAML config into a mapping of
    primitive name -> :class:`Primitive` instance.

    Parameters
    ----------
    geometry_dict : dict
        The ``geometry:`` sub-mapping of the YAML, expected to contain
        a ``primitives`` mapping and an optional ``units`` field
        (default: ``'mm'``).

    Returns
    -------
    dict[str, Primitive]
        Primitives keyed by their YAML name, with all length quantities
        converted to mm.

    Raises
    ------
    ValueError
        On any structural or validation problem (missing fields,
        unsupported primitive type, unsupported unit).
    """
    if geometry_dict is None:
        return {}
    if not isinstance(geometry_dict, dict):
        raise ValueError(
            f"'geometry' must be a mapping, got {type(geometry_dict).__name__}."
        )

    prims_raw = geometry_dict.get('primitives')
    if prims_raw is None:
        return {}
    if not isinstance(prims_raw, dict):
        raise ValueError(
            f"'geometry.primitives' must be a mapping of name -> spec, "
            f"got {type(prims_raw).__name__}."
        )

    units = geometry_dict.get('units', 'mm')
    scale = _unit_scale(units)

    out: Dict[str, Primitive] = {}
    for name, spec in prims_raw.items():
        prim = _parse_one(str(name), spec)
        if scale != 1.0:
            prim = prim._scale(scale)
        out[str(name)] = prim
    return out


# ---------------------------------------------------------------------------
# Helper geometric predicates (kept dependency-free; shapely not required)
# ---------------------------------------------------------------------------

def point_in_polygon(point: Tuple[float, float],
                     vertices: List[Tuple[float, float]]) -> bool:
    """Ray-cast point-in-polygon test (even-odd rule).

    Provided for downstream use (issues #143/#144); not invoked by the
    parser itself. Boundary points may return either True or False; this
    function does not aim to be robust on degenerate cases.
    """
    x, y = point
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        # Check if the horizontal ray from (x, y) crosses edge (i, j).
        if ((yi > y) != (yj > y)):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Domain: boolean-op tree over named primitives (Phase 2.2, issue #143)
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    """Boolean-op recipe defining the computational domain over named primitives.

    The data structure is intentionally a flat, declarative description of
    one composite domain built from ``base`` plus three optional name
    lists. The actual geometric computation -- mapping these onto Gmsh
    OCC ``BooleanUnion`` / ``BooleanDifference`` / ``BooleanIntersection``
    ops -- belongs to the geometry compiler in issue #146. This class is
    the validated input that compiler will consume.

    Attributes
    ----------
    base : str
        Name of the primitive that seeds the boolean tree. Required.
    subtract : list[str]
        Primitive names whose interior is removed from the running shape
        (Gmsh OCC ``BooleanDifference``). Default ``[]``.
    add : list[str]
        Primitive names whose interior is unioned onto the running shape
        (Gmsh OCC ``BooleanUnion``). Default ``[]``.
    intersect : list[str]
        Primitive names whose interior is intersected with the running
        shape (Gmsh OCC ``BooleanIntersection``). Default ``[]``.

    Notes
    -----
    Application order, when the compiler in #146 lands, is expected to be
    ``base -> add -> subtract -> intersect``. This is documented here so
    that #146 / #144 / #145 implement consistent semantics; the parser
    itself is order-agnostic and only does name-resolution validation.
    """
    base: str = ''
    subtract: List[str] = field(default_factory=list)
    add: List[str] = field(default_factory=list)
    intersect: List[str] = field(default_factory=list)

    def referenced_primitives(self) -> List[str]:
        """Return the full list of primitive names this domain refers to,
        in declaration order ``[base, *add, *subtract, *intersect]``.
        Useful for downstream compilers that need to walk the dependency set.
        """
        return [self.base, *self.add, *self.subtract, *self.intersect]


def _resolve_name(
    candidate: str,
    *,
    known: List[str],
    field_label: str,
) -> None:
    """Validate that ``candidate`` is a known primitive name. Raises
    :class:`ValueError` with a did-you-mean hint when it is not.
    """
    if candidate in known:
        return
    suggestions = difflib.get_close_matches(candidate, known, n=3, cutoff=0.6)
    hint = f" Did you mean: {suggestions}?" if suggestions else ''
    raise ValueError(
        f"geometry.domain.{field_label} references unknown primitive "
        f"'{candidate}'. Known primitives: {sorted(known)}.{hint}"
    )


def _as_name_list(
    value: Any,
    *,
    field_label: str,
) -> List[str]:
    """Coerce a YAML field into a list of primitive-name strings."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"geometry.domain.{field_label} must be a list of primitive "
            f"names, got {type(value).__name__} ({value!r})."
        )
    out: List[str] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, str) or not entry:
            raise ValueError(
                f"geometry.domain.{field_label}[{i}] must be a non-empty "
                f"primitive-name string, got {entry!r}."
            )
        out.append(entry)
    return out


def parse_domain(
    domain_dict: dict,
    primitives: Dict[str, Primitive],
) -> Domain:
    """Parse the ``geometry.domain`` block of a YAML config.

    Parameters
    ----------
    domain_dict : dict
        The ``domain:`` sub-mapping. Required keys: ``base``. Optional
        list-valued keys: ``subtract``, ``add``, ``intersect``.
    primitives : dict[str, Primitive]
        The already-parsed primitives keyed by name. Used to validate
        that every referenced name resolves.

    Returns
    -------
    Domain
        Validated boolean-op recipe.

    Raises
    ------
    ValueError
        On structural problems (missing ``base``, non-list ``subtract``
        etc.) or on any reference to an unknown primitive name.
    """
    if domain_dict is None:
        raise ValueError(
            "geometry.domain must be a mapping with at least a 'base' "
            "field, got None."
        )
    if not isinstance(domain_dict, dict):
        raise ValueError(
            f"geometry.domain must be a mapping, got "
            f"{type(domain_dict).__name__}."
        )
    if 'base' not in domain_dict:
        raise ValueError(
            "geometry.domain is missing required field 'base' (the "
            "primitive that seeds the boolean tree)."
        )
    base_raw = domain_dict['base']
    if not isinstance(base_raw, str) or not base_raw:
        raise ValueError(
            f"geometry.domain.base must be a non-empty primitive-name "
            f"string, got {base_raw!r}."
        )

    subtract = _as_name_list(domain_dict.get('subtract'), field_label='subtract')
    add = _as_name_list(domain_dict.get('add'), field_label='add')
    intersect = _as_name_list(domain_dict.get('intersect'), field_label='intersect')

    # Reject unknown top-level keys so typos surface early.
    allowed = {'base', 'subtract', 'add', 'intersect'}
    extra = set(domain_dict) - allowed
    if extra:
        raise ValueError(
            f"geometry.domain has unsupported field(s) {sorted(extra)}. "
            f"Allowed: {sorted(allowed)}."
        )

    known = list(primitives.keys())
    _resolve_name(base_raw, known=known, field_label='base')
    for n in add:
        _resolve_name(n, known=known, field_label='add')
    for n in subtract:
        _resolve_name(n, known=known, field_label='subtract')
    for n in intersect:
        _resolve_name(n, known=known, field_label='intersect')

    return Domain(
        base=base_raw,
        subtract=subtract,
        add=add,
        intersect=intersect,
    )


# ---------------------------------------------------------------------------
# Named groups (Phase 2.3, issue #144)
# ---------------------------------------------------------------------------
#
# A NamedGroup is a user-supplied label (or implicit default) that points at
# a topological subset of the meshed geometry. Three forms are supported in
# YAML:
#
#   notch_tip:        { point: [10, 65] }                 # PointGroup
#   crack_path_band:  { region: { type: rectangle, ... } } # RegionGroup
#   upper_pin_centre: { primitive: pin_top, kind: centre } # SelectorAlias
#
# Resolution to actual mesh node sets / boundary entities is deferred to the
# geometry compiler in issue #146; this module only carries the symbolic
# representation, the parser, and a name-resolution helper that the BC layer
# can call to validate references at config-load time.

_VALID_SELECTOR_KINDS = ('boundary', 'interior', 'centre')


class NamedGroup(ABC):
    """Abstract base for the three named-group forms.

    Subclasses are dataclasses carrying their parameters in the canonical
    internal unit (mm). The instance attribute :attr:`name` is set by
    :func:`parse_named_groups` once the group is keyed in the YAML mapping.
    """
    name: str = ''

    @property
    @abstractmethod
    def kind(self) -> str:
        """Short string identifying the group form (``selector_alias``,
        ``point``, ``region``)."""


@dataclass
class SelectorAliasGroup(NamedGroup):
    """User-named alias for an auto-exposed primitive selector.

    Equivalent to ``<primitive>.<selector_kind>`` (e.g.
    ``pin_top.centre``). Provided so the user can give domain-specific
    names ('upper_pin_centre') to commonly-referenced selectors.
    """
    primitive: str = ''
    selector_kind: str = 'boundary'  # one of _VALID_SELECTOR_KINDS
    name: str = ''

    @property
    def kind(self) -> str:
        return 'selector_alias'

    @property
    def selector(self) -> Selector:
        return Selector(primitive=self.primitive, kind=self.selector_kind)


@dataclass
class PointGroup(NamedGroup):
    """A named single 2D point (e.g. notch tip, probe location). Stored
    in mm. Resolves at #146-time to the closest mesh node.
    """
    coords: Tuple[float, float] = (0.0, 0.0)
    name: str = ''

    @property
    def kind(self) -> str:
        return 'point'


@dataclass
class RegionGroup(NamedGroup):
    """A named region defined inline by a primitive (typically rectangle,
    circle, or polygon). The wrapped :class:`Primitive` carries the
    region geometry; the compiler in #146 turns it into a node set.
    """
    region: Optional[Primitive] = None
    name: str = ''

    @property
    def kind(self) -> str:
        return 'region'


def _parse_named_group(name: str,
                       spec: dict,
                       primitives: Dict[str, Primitive],
                       *,
                       scale: float) -> NamedGroup:
    if not isinstance(spec, dict):
        raise ValueError(
            f"named group '{name}' must be a mapping, got "
            f"{type(spec).__name__}."
        )
    forms = [k for k in ('primitive', 'point', 'region') if k in spec]
    if len(forms) == 0:
        raise ValueError(
            f"named group '{name}' must declare exactly one of "
            f"'primitive' (selector alias), 'point', or 'region'; "
            f"got keys {sorted(spec)}."
        )
    if len(forms) > 1:
        raise ValueError(
            f"named group '{name}' declares multiple forms {forms}; "
            f"only one of 'primitive', 'point', or 'region' is allowed."
        )
    form = forms[0]

    if form == 'primitive':
        prim_name = str(spec['primitive'])
        if prim_name not in primitives:
            suggestion = _did_you_mean(prim_name, list(primitives))
            raise ValueError(
                f"named group '{name}': references unknown primitive "
                f"'{prim_name}'. Known primitives: "
                f"{sorted(primitives)}.{suggestion}"
            )
        sel_kind = str(spec.get('kind', 'boundary')).strip().lower()
        if sel_kind not in _VALID_SELECTOR_KINDS:
            raise ValueError(
                f"named group '{name}': selector 'kind' must be one of "
                f"{list(_VALID_SELECTOR_KINDS)}, got {sel_kind!r}."
            )
        return SelectorAliasGroup(primitive=prim_name,
                                  selector_kind=sel_kind,
                                  name=name)

    if form == 'point':
        coords = _as_xy(spec['point'],
                        field_name='point', primitive_name=name)
        if scale != 1.0:
            coords = (coords[0] * scale, coords[1] * scale)
        return PointGroup(coords=coords, name=name)

    # form == 'region'
    region_spec = spec['region']
    if not isinstance(region_spec, dict):
        raise ValueError(
            f"named group '{name}': 'region' must be an inline primitive "
            f"mapping, got {type(region_spec).__name__}."
        )
    region_prim = _parse_one(name, region_spec)
    if scale != 1.0:
        region_prim = region_prim._scale(scale)
    return RegionGroup(region=region_prim, name=name)


def parse_named_groups(geometry_dict: Optional[dict],
                       primitives: Dict[str, Primitive]
                       ) -> Dict[str, NamedGroup]:
    """Parse the ``geometry.named_groups`` block.

    Parameters
    ----------
    geometry_dict : dict or None
        The full ``geometry:`` sub-mapping; only the ``named_groups`` and
        ``units`` keys are consulted. Coordinates inside ``point`` /
        ``region`` forms are converted to mm using the same ``units``
        scale as :func:`parse_primitives`.
    primitives : dict[str, Primitive]
        The result of :func:`parse_primitives`. Used to validate
        ``primitive: <name>`` selector aliases and to surface
        did-you-mean suggestions.

    Returns
    -------
    dict[str, NamedGroup]
        Explicit named groups keyed by their YAML name. Auto-exposed
        ``<primitive>.<kind>`` selectors are *not* included here -- they
        are resolved on demand by :func:`resolve_node_set_name`.

    Raises
    ------
    ValueError
        On structural errors (multiple forms, unknown primitive in a
        selector alias, unsupported selector ``kind``, malformed inline
        region) or on a name collision with an auto-exposed selector.
    """
    if geometry_dict is None:
        return {}
    if not isinstance(geometry_dict, dict):
        raise ValueError(
            f"'geometry' must be a mapping, got "
            f"{type(geometry_dict).__name__}."
        )
    raw = geometry_dict.get('named_groups')
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"'geometry.named_groups' must be a mapping of name -> spec, "
            f"got {type(raw).__name__}."
        )

    units = geometry_dict.get('units', 'mm')
    scale = _unit_scale(units)

    out: Dict[str, NamedGroup] = {}
    for name, spec in raw.items():
        sname = str(name)
        # Reject names that shadow the auto-exposed <P>.<kind> namespace,
        # or that collide with a primitive name (which would make a bare
        # 'pin_top' reference ambiguous).
        if '.' in sname:
            head, _, tail = sname.partition('.')
            if head in primitives and tail in _VALID_SELECTOR_KINDS:
                raise ValueError(
                    f"named group '{sname}' collides with the auto-exposed "
                    f"selector '{head}.{tail}'. Pick a different name "
                    f"(auto-exposed selectors are always available without "
                    f"explicit declaration)."
                )
        if sname in primitives:
            raise ValueError(
                f"named group '{sname}' collides with the primitive of "
                f"the same name. Rename the group to avoid ambiguity."
            )
        out[sname] = _parse_named_group(sname, spec, primitives, scale=scale)
    return out


# ---------------------------------------------------------------------------
# Resolver helper
# ---------------------------------------------------------------------------

def _did_you_mean(needle: str, candidates: List[str]) -> str:
    """Return a ' Did you mean: ...' fragment, or '' if no close match."""
    matches = difflib.get_close_matches(needle, candidates, n=3, cutoff=0.6)
    if not matches:
        return ''
    return ' Did you mean: ' + ', '.join(repr(m) for m in matches) + '?'


def known_group_names(primitives: Dict[str, Primitive],
                      named_groups: Dict[str, NamedGroup]
                      ) -> List[str]:
    """Return all valid node-set names that BC / preseed entries may
    reference: explicit named groups + auto-exposed
    ``<primitive>.<kind>`` selectors.
    """
    out: List[str] = list(named_groups.keys())
    for pname in primitives:
        for k in _VALID_SELECTOR_KINDS:
            out.append(f'{pname}.{k}')
    return out


def validate_node_set_name(name: str,
                           primitives: Dict[str, Primitive],
                           named_groups: Dict[str, NamedGroup]) -> None:
    """Validate that ``name`` refers to a known node-set.

    Accepts either an explicit named group key or the auto-exposed
    ``<primitive>.<selector_kind>`` form. Raises :class:`ValueError`
    with a did-you-mean suggestion on failure.

    This is the validation primitive the BC / preseed layer should call
    at config-load time. Actual resolution to mesh entities is the
    geometry compiler's job (issue #146).
    """
    if name in named_groups:
        return
    if '.' in name:
        head, _, tail = name.partition('.')
        if head in primitives and tail in _VALID_SELECTOR_KINDS:
            return
        if head not in primitives:
            suggestion = _did_you_mean(head, list(primitives))
            raise ValueError(
                f"node-set reference '{name}': primitive '{head}' is "
                f"not defined. Known primitives: "
                f"{sorted(primitives)}.{suggestion}"
            )
        # head is a primitive but tail is not a valid selector kind
        raise ValueError(
            f"node-set reference '{name}': '{tail}' is not a valid "
            f"selector kind. Allowed: {list(_VALID_SELECTOR_KINDS)}."
        )
    suggestion = _did_you_mean(
        name, known_group_names(primitives, named_groups))
    raise ValueError(
        f"node-set reference '{name}' is not a declared named group "
        f"and does not match the '<primitive>.<selector_kind>' "
        f"auto-exposed pattern.{suggestion}"
    )


def resolve_node_set_name(name: str,
                          primitives: Dict[str, Primitive],
                          named_groups: Dict[str, NamedGroup],
                          mesh: Any) -> Any:
    """Resolve a node-set name to actual mesh entities.

    This is the entry point the BC / initial-condition layers will call
    at solve time. The signature is fixed now so the BC layer can be
    written against it; the **implementation** that walks the mesh and
    materialises node indices lands in issue #146 alongside the Gmsh
    physical-entity emitter.

    Always validates ``name`` first (raising :class:`ValueError` on a
    bad reference) before raising :class:`NotImplementedError`.
    """
    validate_node_set_name(name, primitives, named_groups)
    raise NotImplementedError(
        f"resolve_node_set_name('{name}', ...): name is valid but the "
        "geometry compiler that materialises named groups onto mesh "
        "entities lands in issue #146."
    )


# ---------------------------------------------------------------------------
# Mesh refinement DSL (Phase 2.4, issue #145)
# ---------------------------------------------------------------------------

# Primitive types that make sense as a refinement region (i.e. have an
# interior with non-zero area). ``ball`` is accepted as a natural alias for
# ``circle`` in the refinement context only -- ``parse_primitives`` itself
# does not advertise it.
#
# ``box`` is a refinement-only region type (issue #200). Unlike
# ``rectangle`` -- which the compiler turns into a Distance+Threshold band
# around the rectangle's *boundary* -- a ``box`` region asks the compiler
# for an interior-fill refinement (Gmsh ``Field[Box]``: fine size inside,
# coarse outside, with a smoothing thickness in between). The box is
# axis-aligned and parameterised by coordinate ranges ``x: [xmin, xmax]``
# and ``y: [ymin, ymax]``.
_REFINEMENT_REGION_TYPES = ('rectangle', 'circle', 'ball', 'polygon', 'box')


@dataclass
class ElementSizeRule:
    """A single mesh-refinement rule.

    Exactly one of ``region`` or ``primitive`` must be set.

    region : Primitive or None
        Ad-hoc region declared inline in the mesh block. Must be one of
        ``rectangle``, ``circle``, ``ball`` (alias for ``circle``),
        ``polygon``, or ``box``. Stored in mm with the same unit-conversion
        rules as :func:`parse_primitives`.
    primitive : str or None
        Name of an already-declared primitive in ``geometry.primitives``.
        Resolved by :func:`parse_mesh_dsl` against the supplied registry;
        unknown names raise a ``ValueError`` with a did-you-mean hint.
    size : float
        Target element edge length inside the region (mm).
    margin : float
        Distance (mm) by which the region is conceptually expanded for the
        Gmsh ``Threshold`` field's transition zone (#146 compiler maps this
        to ``DistMax``). Default 0.0; must be non-negative. Only meaningful
        for ``mode='threshold'`` rules; setting ``margin`` on a
        ``mode='box'`` rule is rejected at parse time.
    mode : str
        One of ``'threshold'`` (default) or ``'box'`` (#200). Threshold
        rules emit a Distance+Threshold field on the region's *boundary*
        curves -- they refine a band around the boundary. Box rules emit a
        Gmsh ``Field[Box]`` -- they refine the entire interior of an
        axis-aligned coordinate box and use ``thickness`` for the smooth
        transition outside the box.
    thickness : float
        Smoothing band width (mm) outside a ``mode='box'`` rule. Mapped
        to ``Field[Box].Thickness``. Must be strictly positive when
        explicitly supplied (a zero or negative thickness produces a sharp
        size discontinuity that Gmsh handles poorly with the OCC kernel
        and is rejected here). Defaults to the mesh DSL's
        ``element_size.default`` so an unspecified thickness gives a
        natural one-coarse-element-wide transition. Ignored for
        ``mode='threshold'`` rules.
    """
    region: Optional[Primitive] = None
    primitive: Optional[str] = None
    size: float = 0.0
    margin: float = 0.0
    mode: str = 'threshold'
    thickness: float = 0.0


@dataclass
class MeshDSL:
    """Parsed representation of the ``geometry.mesh`` block.

    default_size : float
        Element edge length used outside every refinement rule (mm).
    refined : list of ElementSizeRule
        Ordered list of refinement rules. The compiler (#146) is expected
        to combine them with ``Field[Min]`` so that the smallest size wins
        in overlapping zones.
    """
    default_size: float = 0.0
    refined: List[ElementSizeRule] = field(default_factory=list)


def _parse_one_region(name: str, spec: dict) -> Primitive:
    """Parse an ad-hoc refinement region.

    Accepts the same shapes as :func:`parse_primitives` but additionally
    recognises ``ball`` as an alias for ``circle`` and ``box`` as an
    axis-aligned coordinate box (``x: [xmin, xmax], y: [ymin, ymax]``).
    Rejects zero-area primitive types (``point``, ``line_segment``) --
    they would not yield a meaningful refinement zone.

    A ``box`` region is stored internally as a :class:`Rectangle` -- the
    bounding box is the natural representation -- but the rule it backs
    is flagged with ``mode='box'`` so the compiler emits a Gmsh
    ``Field[Box]`` (interior fill) rather than the default Distance +
    Threshold (boundary band) combination.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"mesh refinement region '{name}' must be a mapping, "
            f"got {type(spec).__name__}."
        )
    if 'type' not in spec:
        raise ValueError(
            f"mesh refinement region '{name}' is missing required field 'type'."
        )
    rtype = str(spec['type']).strip().lower()
    if rtype not in _REFINEMENT_REGION_TYPES:
        raise ValueError(
            f"mesh refinement region '{name}': unsupported type '{spec['type']}'. "
            f"Supported region types: {list(_REFINEMENT_REGION_TYPES)}."
        )
    if rtype == 'box':
        return _parse_box_region(name, spec)
    # Normalise ball -> circle for delegation to _parse_one.
    if rtype == 'ball':
        spec = dict(spec)
        spec['type'] = 'circle'
    return _parse_one(name, spec)


def _parse_box_region(name: str, spec: dict) -> Rectangle:
    """Parse a ``type: box`` refinement region into a Rectangle.

    Box regions are described by coordinate ranges, e.g.::

        region: { type: box, x: [0, 65], y: [45, 70] }

    rather than ``origin + size``; this matches Gmsh ``Field[Box]``'s
    ``XMin/XMax/YMin/YMax`` parameter set and reads more naturally for
    interior fill regions.
    """
    for key in ('x', 'y'):
        if key not in spec:
            raise ValueError(
                f"mesh refinement region '{name}' (type=box) is missing "
                f"required field '{key}' (expected a 2-element [min, max] "
                f"list)."
            )
        rng = spec[key]
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError(
                f"mesh refinement region '{name}' (type=box): field "
                f"'{key}' must be a 2-element [min, max] list, got "
                f"{rng!r}."
            )
        try:
            lo = float(rng[0])
            hi = float(rng[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mesh refinement region '{name}' (type=box): field "
                f"'{key}' contains non-numeric values ({rng!r})."
            ) from exc
        if hi <= lo:
            raise ValueError(
                f"mesh refinement region '{name}' (type=box): field "
                f"'{key}' must satisfy max > min, got {rng!r}."
            )
    x0, x1 = float(spec['x'][0]), float(spec['x'][1])
    y0, y1 = float(spec['y'][0]), float(spec['y'][1])
    return Rectangle(origin=(x0, y0), size=(x1 - x0, y1 - y0), name=name)


def _scale_rule(rule: ElementSizeRule, factor: float) -> ElementSizeRule:
    """Apply unit scaling to a rule's length quantities."""
    if factor == 1.0:
        return rule
    region = rule.region._scale(factor) if rule.region is not None else None
    return ElementSizeRule(
        region=region,
        primitive=rule.primitive,
        size=rule.size * factor,
        margin=rule.margin * factor,
        mode=rule.mode,
        thickness=rule.thickness * factor,
    )


def parse_mesh_dsl(mesh_dict: dict,
                   primitives: Optional[Dict[str, Primitive]] = None,
                   *,
                   units: str = 'mm') -> MeshDSL:
    """Parse a ``geometry.mesh`` YAML block into a :class:`MeshDSL`.

    Parameters
    ----------
    mesh_dict : dict
        The ``mesh:`` sub-mapping; must contain ``element_size.default``
        and may contain ``element_size.refined``.
    primitives : dict[str, Primitive] or None
        Registry of primitives already parsed from the same ``geometry``
        block. Used to resolve ``primitive:`` references in refinement
        rules. ``None`` is treated as an empty registry.
    units : str
        Unit hint applied to ``default``, ``size``, ``margin`` and any
        ad-hoc ``region`` lengths. Defaults to ``'mm'``.

    Raises
    ------
    ValueError
        On any structural or semantic problem (missing default size,
        unknown primitive name, mutex violation, negative quantity, etc.).
    """
    if mesh_dict is None:
        raise ValueError("mesh DSL: expected a mapping, got None.")
    if not isinstance(mesh_dict, dict):
        raise ValueError(
            f"mesh DSL: expected a mapping, got {type(mesh_dict).__name__}."
        )
    if primitives is None:
        primitives = {}

    es = mesh_dict.get('element_size')
    if es is None:
        raise ValueError(
            "mesh DSL: missing required 'element_size' block."
        )
    if not isinstance(es, dict):
        raise ValueError(
            f"mesh DSL: 'element_size' must be a mapping, "
            f"got {type(es).__name__}."
        )

    if 'default' not in es:
        raise ValueError(
            "mesh DSL: 'element_size.default' is required (the coarse "
            "element size used outside every refinement rule)."
        )
    try:
        default_size = float(es['default'])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"mesh DSL: 'element_size.default' must be numeric, "
            f"got {es['default']!r}."
        ) from exc
    if default_size <= 0:
        raise ValueError(
            f"mesh DSL: 'element_size.default' must be strictly positive, "
            f"got {default_size}."
        )

    refined_raw = es.get('refined', []) or []
    if not isinstance(refined_raw, (list, tuple)):
        raise ValueError(
            f"mesh DSL: 'element_size.refined' must be a list, "
            f"got {type(refined_raw).__name__}."
        )

    rules: List[ElementSizeRule] = []
    for i, entry in enumerate(refined_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"mesh DSL: refined[{i}] must be a mapping, "
                f"got {type(entry).__name__}."
            )

        has_region = 'region' in entry and entry['region'] is not None
        has_primitive = 'primitive' in entry and entry['primitive'] is not None
        if has_region and has_primitive:
            raise ValueError(
                f"mesh DSL: refined[{i}] sets both 'region' and "
                f"'primitive'; they are mutually exclusive (use one or "
                f"the other)."
            )
        if not has_region and not has_primitive:
            raise ValueError(
                f"mesh DSL: refined[{i}] must specify either 'region' "
                f"(an inline primitive spec) or 'primitive' (the name "
                f"of a registered primitive)."
            )

        if 'size' not in entry:
            raise ValueError(
                f"mesh DSL: refined[{i}] is missing required field 'size'."
            )
        try:
            size = float(entry['size'])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mesh DSL: refined[{i}].size must be numeric, "
                f"got {entry['size']!r}."
            ) from exc
        if size <= 0:
            raise ValueError(
                f"mesh DSL: refined[{i}].size must be strictly positive, "
                f"got {size}."
            )

        # Detect mode='box' (issue #200). A rule is box-mode iff its
        # inline region declares ``type: box``. Primitive-name references
        # cannot be box-mode (registered primitives are domain-relevant
        # shapes, not refinement-only coordinate ranges).
        is_box = False
        if has_region:
            region_spec = entry['region']
            if isinstance(region_spec, dict):
                rtype = str(region_spec.get('type', '')).strip().lower()
                is_box = (rtype == 'box')

        margin_raw = entry.get('margin', None)
        if is_box and margin_raw is not None:
            raise ValueError(
                f"mesh DSL: refined[{i}] sets 'margin' on a 'type: box' "
                f"region; box-mode refinement uses 'thickness' for the "
                f"smooth transition outside the box. Drop 'margin' or "
                f"switch the region type."
            )
        if margin_raw is None:
            margin_raw = 0.0
        try:
            margin = float(margin_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mesh DSL: refined[{i}].margin must be numeric, "
                f"got {margin_raw!r}."
            ) from exc
        if margin < 0:
            raise ValueError(
                f"mesh DSL: refined[{i}].margin must be non-negative, "
                f"got {margin}."
            )

        # Thickness: only meaningful for box-mode. If absent, default to
        # the coarse element size (one-coarse-element-wide smoothing).
        # An explicitly-supplied non-positive value is rejected (issue
        # acceptance: clean validation error).
        thickness_raw = entry.get('thickness', None)
        if not is_box:
            if thickness_raw is not None:
                raise ValueError(
                    f"mesh DSL: refined[{i}] sets 'thickness' on a "
                    f"non-box rule (region type is "
                    f"{entry.get('region', {}).get('type', 'primitive')!r}); "
                    f"thickness is only meaningful for 'type: box' "
                    f"regions."
                )
            thickness = 0.0
        else:
            if thickness_raw is None:
                thickness = default_size
            else:
                try:
                    thickness = float(thickness_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"mesh DSL: refined[{i}].thickness must be "
                        f"numeric, got {thickness_raw!r}."
                    ) from exc
                if thickness <= 0:
                    raise ValueError(
                        f"mesh DSL: refined[{i}].thickness must be "
                        f"strictly positive, got {thickness}."
                    )

        if has_region:
            region = _parse_one_region(f'refined[{i}].region', entry['region'])
            rule = ElementSizeRule(
                region=region, primitive=None,
                size=size, margin=margin,
                mode=('box' if is_box else 'threshold'),
                thickness=thickness,
            )
        else:
            pname = str(entry['primitive'])
            if pname not in primitives:
                suggestion = difflib.get_close_matches(
                    pname, list(primitives.keys()), n=1)
                hint = (f" Did you mean '{suggestion[0]}'?"
                        if suggestion else '')
                known = sorted(primitives.keys())
                raise ValueError(
                    f"mesh DSL: refined[{i}] references unknown primitive "
                    f"'{pname}'.{hint} Known primitives: {known}."
                )
            # Reject primitives whose interior is degenerate (point /
            # line_segment) -- they don't define a refinement zone.
            referenced = primitives[pname]
            if referenced.kind in ('point', 'line_segment'):
                raise ValueError(
                    f"mesh DSL: refined[{i}] references primitive "
                    f"'{pname}' of type '{referenced.kind}', which has "
                    f"no interior; refinement primitives must be one of "
                    f"{list(_REFINEMENT_REGION_TYPES)}."
                )
            rule = ElementSizeRule(region=None, primitive=pname,
                                   size=size, margin=margin,
                                   mode='threshold', thickness=0.0)

        rules.append(rule)

    scale = _unit_scale(units)
    if scale != 1.0:
        default_size = default_size * scale
        rules = [_scale_rule(r, scale) for r in rules]

    return MeshDSL(default_size=default_size, refined=rules)


__all__ = [
    'Primitive',
    'Rectangle',
    'Circle',
    'Polygon',
    'Point',
    'LineSegment',
    'Selector',
    'Domain',
    'NamedGroup',
    'SelectorAliasGroup',
    'PointGroup',
    'RegionGroup',
    'parse_primitives',
    'parse_domain',
    'parse_named_groups',
    'known_group_names',
    'validate_node_set_name',
    'resolve_node_set_name',
    'point_in_polygon',
    'ElementSizeRule',
    'MeshDSL',
    'parse_mesh_dsl',
]
