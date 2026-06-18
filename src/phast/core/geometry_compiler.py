"""
Geometry compiler (Phase 2.5, issue #146): YAML primitive vocabulary ->
Gmsh OCC ``.geo`` text -> meshed ``.msh`` file, with a SHA-256 hash cache.

Consumes the inert dataclasses produced by :mod:`geometry_dsl`
(:class:`Primitive`, :class:`Domain`, :class:`NamedGroup`, :class:`MeshDSL`)
and produces a ``.msh`` file ready to be loaded by :class:`mesh.FEMMesh`.

Cache layout (highest priority first):

* ``$TORCH_PF_MESH_CACHE_DIR`` -- explicit override
* ``$XDG_CACHE_HOME/phast/meshes/``
* ``~/.cache/phast/meshes/``

Each compiled mesh is keyed by a SHA-256 of the *normalized* parsed
geometry (so cosmetic YAML edits do not bust the cache, and ``units: m``
inputs hash identically to their mm-equivalent numeric form).

Scope of this issue
-------------------
The COMSOL holed-plate geometry (rectangle base + circular ``subtract``)
is the primary target and is fully supported. The remaining surface area
is intentionally stubbed so the headline workflow lands cleanly:

* ``Domain.add`` / ``Domain.intersect`` -> :class:`NotImplementedError`,
  follow-up tracked in this module's ``_unsupported_domain_op``.
* :class:`Polygon` subtracts: supported (issue #199). Both interior
  (``cut``) and edge-flush (``fragment``) configurations work; degenerate
  vertex sets (collinear, zero-length edges) are rejected with a clean
  error before reaching gmsh. Polygons in ``base`` / ``add`` /
  ``intersect`` are still deferred.
* :class:`LineSegment` primitives in the ``base`` / ``subtract`` lists.

Edge-flush slits / corner cutouts (issue #187)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Subtracting a primitive whose bounding box touches the base's bounding
box on any side (a thin slit flush with the plate edge, a corner cutout
that shares two edges, a hole tangent to the boundary) routes through
OCC ``fragment`` instead of ``cut``. ``cut`` produces a degenerate /
empty surface in those configurations; ``fragment`` splits the base
into N regions along all boolean boundaries and we keep the regions
that came from the base but did not also come from any tool, using the
``outDimTagsMap`` returned by gmsh as the exact bookkeeping (no
centroid heuristics). Pure-interior subtracts continue to use ``cut``,
which is faster and produces a single surface. The detection is
per-subtract: any single edge-flush tool flips the whole boolean to
fragment so the kept-region bookkeeping stays consistent.

Refinement rules (:class:`MeshDSL`) emit Gmsh ``Distance`` + ``Threshold``
fields combined with ``Field[Min]`` so the smallest size always wins.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .geometry_dsl import (
    Circle,
    Domain,
    ElementSizeRule,
    LineSegment,
    MeshDSL,
    NamedGroup,
    Point,
    PointGroup,
    Polygon,
    Primitive,
    Rectangle,
    RegionGroup,
    SelectorAliasGroup,
)


__all__ = [
    'compile_geometry',
    'cache_dir',
    'cache_key',
]


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """Return the directory where compiled ``.msh`` files are cached.

    Resolution order:

    1. ``$TORCH_PF_MESH_CACHE_DIR`` (explicit per-run / per-repo override)
    2. ``$XDG_CACHE_HOME/phast/meshes``
    3. ``~/.cache/phast/meshes``

    The directory is created on first call.
    """
    explicit = os.environ.get('TORCH_PF_MESH_CACHE_DIR')
    if explicit:
        d = Path(explicit).expanduser()
    else:
        xdg = os.environ.get('XDG_CACHE_HOME')
        base = Path(xdg).expanduser() if xdg else Path.home() / '.cache'
        d = base / 'phast' / 'meshes'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _normalize_primitive(p: Primitive) -> Dict[str, Any]:
    """Return a JSON-friendly reference dict for hashing.

    All length quantities are already in mm at this point (the geometry
    DSL parser does the unit scaling), so two YAML inputs that describe
    the same geometry but in different units hash identically.
    """
    if isinstance(p, Rectangle):
        return {'kind': 'rectangle', 'name': p.name,
                'origin': list(p.origin), 'size': list(p.size)}
    if isinstance(p, Circle):
        return {'kind': 'circle', 'name': p.name,
                'center': list(p.center), 'radius': p.radius}
    if isinstance(p, Polygon):
        return {'kind': 'polygon', 'name': p.name,
                'vertices': [list(v) for v in p.vertices]}
    if isinstance(p, Point):
        return {'kind': 'point', 'name': p.name,
                'coords': list(p.coords)}
    if isinstance(p, LineSegment):
        return {'kind': 'line_segment', 'name': p.name,
                'start': list(p.start), 'end': list(p.end)}
    raise TypeError(f'Unhandled primitive type {type(p).__name__}')


def _normalize_named_group(g: NamedGroup) -> Dict[str, Any]:
    if isinstance(g, SelectorAliasGroup):
        return {'kind': 'selector_alias', 'name': g.name,
                'primitive': g.primitive,
                'selector_kind': g.selector_kind}
    if isinstance(g, PointGroup):
        return {'kind': 'point', 'name': g.name,
                'coords': list(g.coords)}
    if isinstance(g, RegionGroup):
        return {'kind': 'region', 'name': g.name,
                'region': _normalize_primitive(g.region) if g.region else None}
    raise TypeError(f'Unhandled named-group type {type(g).__name__}')


def _normalize_rule(r: ElementSizeRule) -> Dict[str, Any]:
    return {
        'region': _normalize_primitive(r.region) if r.region is not None else None,
        'primitive': r.primitive,
        'size': r.size,
        'margin': r.margin,
        # Mode + thickness are part of the cache key (issue #200): a
        # box-mode rule produces a *different* mesh than a threshold-mode
        # rule with the same region geometry, so the reference payload
        # must distinguish them.
        'mode': getattr(r, 'mode', 'threshold'),
        'thickness': getattr(r, 'thickness', 0.0),
    }


def _normalized_payload(
    primitives: Dict[str, Primitive],
    domain: Optional[Domain],
    named_groups: Dict[str, NamedGroup],
    mesh_dsl: Optional[MeshDSL],
) -> str:
    """Build the reference JSON string used as the SHA-256 input."""
    payload = {
        'primitives': {k: _normalize_primitive(v)
                       for k, v in sorted(primitives.items())},
        'domain': asdict(domain) if domain is not None else None,
        'named_groups': {k: _normalize_named_group(v)
                         for k, v in sorted(named_groups.items())},
        'mesh': (
            {
                'default_size': mesh_dsl.default_size,
                'refined': [_normalize_rule(r) for r in mesh_dsl.refined],
            }
            if mesh_dsl is not None else None
        ),
        # Bump on every breaking change to .geo emission so cached meshes
        # built by an older compiler don't get reused after a code update.
        # v2 (issue #187): edge-flush subtracts now route through OCC
        # fragment instead of raising; this changes the resulting mesh
        # topology, so old caches must be invalidated.
        # v3 (issue #199): polygon subtracts (interior + edge-flush) are
        # now compiled via cut / fragment instead of raising
        # NotImplementedError. Old caches did not contain polygon
        # configurations, but bump for explicit history.
        # v4 (issue #200): the build path now explicitly resets gmsh's
        # mesh-size-control options at the start of every compile (so a
        # previous in-process compile can't leak ExtendFromBoundary /
        # FromPoints state into the next). Existing caches are still
        # *correct* for their inputs, but mesh topology near the box-mode
        # transition zone differs slightly enough that we invalidate to
        # avoid stale-cache surprises during the rollout.
        # v5 (issue #201): PointGroup uses OCC fragment (not mesh.embed)
        # and RegionGroup tags partial-overlap boundary sub-segments via
        # an OCC fragment + true-boundary filter. Mesh topology and the
        # set of physical groups can both change for affected configs;
        # bump invalidates v4 caches.
        'compiler_version': 5,
        # Per-subtract boolean strategy ('cut' | 'fragment' | 'noop'): part
        # of the reference form so a strategy change (e.g. a primitive
        # nudged onto the boundary) busts the cache deterministically.
        'boolean_strategy': (
            _boolean_strategies(domain, primitives) if domain is not None
            else None
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def cache_key(
    primitives: Dict[str, Primitive],
    domain: Optional[Domain],
    named_groups: Dict[str, NamedGroup],
    mesh_dsl: Optional[MeshDSL],
) -> str:
    """Return the SHA-256 hex digest of the reference geometry payload."""
    payload = _normalized_payload(primitives, domain, named_groups, mesh_dsl)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Boolean-strategy classification (issue #187)
# ---------------------------------------------------------------------------

def _bbox_touches_or_crosses(
    base_bbox: Tuple[float, float, float, float],
    tool_bbox: Tuple[float, float, float, float],
    tol: float,
) -> bool:
    """Return True if ``tool_bbox`` is flush with or crosses any side of
    ``base_bbox`` (within ``tol``).

    Touching = a tool side coincides with a base side; crossing = the
    tool extends past a base side. Either case requires OCC ``fragment``;
    a strictly-interior tool can use the faster ``cut``.
    """
    bx0, by0, bx1, by1 = base_bbox
    tx0, ty0, tx1, ty1 = tool_bbox
    return (
        tx0 <= bx0 + tol  # left edge touched/crossed
        or ty0 <= by0 + tol  # bottom edge touched/crossed
        or tx1 >= bx1 - tol  # right edge touched/crossed
        or ty1 >= by1 - tol  # top edge touched/crossed
    )


def _bbox_disjoint(
    base_bbox: Tuple[float, float, float, float],
    tool_bbox: Tuple[float, float, float, float],
    tol: float,
) -> bool:
    """Return True iff ``tool_bbox`` and ``base_bbox`` are disjoint."""
    bx0, by0, bx1, by1 = base_bbox
    tx0, ty0, tx1, ty1 = tool_bbox
    return (tx1 < bx0 - tol or tx0 > bx1 + tol
            or ty1 < by0 - tol or ty0 > by1 + tol)


def _classify_subtract(
    domain: Domain,
    primitives: Dict[str, Primitive],
) -> List[str]:
    """Return per-subtract strategy: ``'cut'``, ``'fragment'``, or ``'noop'``.

    * ``cut``: tool bbox is strictly interior to base bbox.
    * ``fragment``: tool bbox touches / crosses any base-bbox side.
    * ``noop``: tool bbox is disjoint from base bbox -> warn + skip.

    The "fully contains" case (tool bbox covers base bbox) is also
    classified as ``fragment``; the fragment driver detects an empty
    kept set after the operation and raises a clear error.
    """
    base_bbox = _bbox_of(primitives[domain.base])
    bx0, by0, bx1, by1 = base_bbox
    max_extent = max(bx1 - bx0, by1 - by0, 1.0)
    tol = max(1e-9 * max_extent, 1e-12)
    out: List[str] = []
    for n in domain.subtract:
        tool_bbox = _bbox_of(primitives[n])
        if _bbox_disjoint(base_bbox, tool_bbox, tol):
            out.append('noop')
        elif _bbox_touches_or_crosses(base_bbox, tool_bbox, tol):
            out.append('fragment')
        else:
            out.append('cut')
    return out


def _boolean_strategies(
    domain: Domain,
    primitives: Dict[str, Primitive],
) -> Optional[List[str]]:
    """Cache-key-friendly wrapper around :func:`_classify_subtract`.

    Returns ``None`` if the domain is degenerate enough that the regular
    scope-guard (``_unsupported_domain_op``) will reject it; in that
    case the cache key contributes nothing here and the error path
    fires before any compile work.
    """
    base_prim = primitives.get(domain.base)
    if base_prim is None or not isinstance(base_prim, (Rectangle, Circle)):
        return None
    for n in domain.subtract:
        p = primitives.get(n)
        if p is None or not isinstance(p, (Rectangle, Circle, Polygon)):
            return None
    try:
        return _classify_subtract(domain, primitives)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Domain validation (scope guard)
# ---------------------------------------------------------------------------

def _unsupported_domain_op(domain: Domain,
                           primitives: Dict[str, Primitive]) -> Optional[str]:
    """Return a human-readable reason if ``domain`` uses an op outside
    the issue #146 scope, else ``None``.

    Implemented subset:

    * ``base``: must reference a Rectangle or Circle.
    * ``subtract``: each entry must be a Rectangle or Circle.
    * ``add``, ``intersect``: not yet implemented.
    """
    base_prim = primitives.get(domain.base)
    if base_prim is None:
        return f"domain.base '{domain.base}' is not a known primitive."
    if not isinstance(base_prim, (Rectangle, Circle)):
        return (f"domain.base '{domain.base}' is a {base_prim.kind}; "
                "issue #146 implements rectangle and circle bases only.")
    for n in domain.subtract:
        p = primitives.get(n)
        if p is None:
            return f"domain.subtract references unknown primitive '{n}'."
        if not isinstance(p, (Rectangle, Circle, Polygon)):
            return (f"domain.subtract['{n}'] is a {p.kind}; "
                    "the geometry compiler subtracts rectangles, circles "
                    "and polygons. line_segment booleans are tracked as "
                    "follow-up work.")
        if isinstance(p, Polygon):
            reason = _polygon_validation_reason(p)
            if reason is not None:
                return f"domain.subtract['{n}']: {reason}"
    if domain.add:
        return ("domain.add is not implemented in issue #146; only "
                "base + subtract is supported. Track follow-up if you need "
                "a multi-piece domain.")
    if domain.intersect:
        return ("domain.intersect is not implemented in issue #146.")
    return None


# ---------------------------------------------------------------------------
# .geo text emitter (debug artifact)
# ---------------------------------------------------------------------------

def _emit_geo_text(
    primitives: Dict[str, Primitive],
    domain: Optional[Domain],
    named_groups: Dict[str, NamedGroup],
    mesh_dsl: Optional[MeshDSL],
) -> str:
    """Emit a Gmsh OCC-kernel ``.geo`` file for the given geometry.

    The compiler does *not* execute the .geo text; it drives gmsh via its
    Python API directly (see :func:`_build_in_gmsh`). This text is written
    alongside the .msh as a debug aid so the user can inspect / hand-tweak
    the geometry that was compiled.
    """
    lines: List[str] = []
    lines.append('// Auto-generated by phast.geometry_compiler')
    lines.append('// (issue #146). Do not edit by hand; regenerate via the')
    lines.append('// YAML geometry block.')
    lines.append('SetFactory("OpenCASCADE");')
    lines.append('')

    # Emit primitives that participate in the domain.
    base_name = domain.base if domain is not None else None
    subtract_names = list(domain.subtract) if domain is not None else []

    def _emit_prim(p: Primitive, tag_var: str) -> None:
        if isinstance(p, Rectangle):
            ox, oy = p.origin
            sx, sy = p.size
            lines.append(
                f'{tag_var} = news; '
                f'Rectangle({tag_var}) = {{{ox}, {oy}, 0, {sx}, {sy}}};'
            )
        elif isinstance(p, Circle):
            cx, cy = p.center
            lines.append(
                f'{tag_var} = news; '
                f'Disk({tag_var}) = {{{cx}, {cy}, 0, {p.radius}, {p.radius}}};'
            )
        else:
            lines.append(f'// {tag_var}: primitive type {p.kind!r} '
                         'not emitted by issue #146 compiler.')

    if base_name is not None:
        lines.append(f'// Base primitive: {base_name}')
        _emit_prim(primitives[base_name], 'base_tag')
        for n in subtract_names:
            lines.append(f'// Subtract: {n}')
            _emit_prim(primitives[n], f'sub_{n}')
        lines.append('')
        if subtract_names:
            cut_list = ', '.join(f'sub_{n}' for n in subtract_names)
            lines.append(
                'BooleanDifference{ Surface{base_tag}; Delete; }'
                '{ Surface{' + cut_list.replace('sub_', 'sub_') + '}; Delete; };'
            )

    # Mesh size + refinement (informational; Python API drives the real fields).
    if mesh_dsl is not None:
        lines.append('')
        lines.append(f'// default element size: {mesh_dsl.default_size} mm')
        for i, r in enumerate(mesh_dsl.refined):
            tag = (f'primitive={r.primitive!r}'
                   if r.primitive else f'inline {r.region.kind}')
            mode = getattr(r, 'mode', 'threshold')
            if mode == 'box':
                lines.append(
                    f'// refined[{i}]: {tag}, mode=box, size={r.size}, '
                    f'thickness={getattr(r, "thickness", 0.0)}'
                )
            else:
                lines.append(
                    f'// refined[{i}]: {tag}, mode=threshold, '
                    f'size={r.size}, margin={r.margin}'
                )

    # Named groups (informational; Python API will tag them as physical groups).
    if named_groups:
        lines.append('')
        lines.append('// Named groups (resolved at mesh time):')
        for name, g in named_groups.items():
            lines.append(f'//   {name}: {_normalize_named_group(g)}')

    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# Gmsh driver (Python API preferred; subprocess fallback)
# ---------------------------------------------------------------------------

def _bbox_of(p: Primitive) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box (xmin, ymin, xmax, ymax) of a primitive."""
    if isinstance(p, Rectangle):
        ox, oy = p.origin
        sx, sy = p.size
        return (ox, oy, ox + sx, oy + sy)
    if isinstance(p, Circle):
        cx, cy = p.center
        r = p.radius
        return (cx - r, cy - r, cx + r, cy + r)
    if isinstance(p, Polygon):
        xs = [v[0] for v in p.vertices]
        ys = [v[1] for v in p.vertices]
        return (min(xs), min(ys), max(xs), max(ys))
    if isinstance(p, Point):
        x, y = p.coords
        return (x, y, x, y)
    if isinstance(p, LineSegment):
        sx, sy = p.start
        ex, ey = p.end
        return (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
    raise TypeError(f'Unhandled primitive type {type(p).__name__}')


def _polygon_signed_area(vertices: List[Tuple[float, float]]) -> float:
    """Shoelace signed area. Positive for CCW vertex order, negative for CW.

    OCC ``addPlaneSurface`` accepts either orientation for a single closed
    loop, but downstream booleans are sensitive to orientation. We
    normalize to CCW in :func:`_add_occ_polygon` to keep the kept-region
    bookkeeping deterministic.
    """
    n = len(vertices)
    s = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def _polygon_validation_reason(p: Polygon) -> Optional[str]:
    """Return a human-readable rejection reason for a polygon subtract,
    or ``None`` if the polygon is geometrically usable.

    Rejects degenerate polygons (< 3 distinct vertices, all-collinear
    vertex set, ~zero signed area) before the OCC builder gets a chance
    to emit cryptic gmsh errors.
    """
    verts = list(p.vertices)
    if len(verts) < 3:
        return (f"polygon must have at least 3 vertices, got {len(verts)}.")
    # Reject duplicate consecutive vertices (zero-length edges).
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        if a == b:
            return (f"polygon has a zero-length edge between vertex "
                    f"{i} and vertex {(i + 1) % n} ({a!r} == {b!r}).")
    # Reject all-collinear vertex sets (zero signed area).
    bbox = (min(v[0] for v in verts), min(v[1] for v in verts),
            max(v[0] for v in verts), max(v[1] for v in verts))
    diag2 = max(
        (bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2,
        1.0,
    )
    area = _polygon_signed_area(verts)
    if abs(area) < 1e-12 * diag2:
        return ("polygon vertices are collinear (zero signed area); a "
                "polygon subtract must enclose a non-degenerate area.")
    return None


def _midpoints(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


def _add_occ_polygon(gmsh, p: Polygon) -> int:
    """Build a closed polygon as an OCC plane surface and return its tag.

    OCC has no single ``addPolygon`` entry point; we assemble it from
    points, lines, a closed curve loop, and a plane surface. Vertices are
    normalized to CCW orientation so downstream booleans (cut /
    fragment) see a consistent outward normal regardless of the YAML
    declaration order.
    """
    verts = list(p.vertices)
    # Normalize to CCW so the OCC plane surface has a positive area
    # under the standard right-hand-rule orientation.
    if _polygon_signed_area(verts) < 0.0:
        verts = list(reversed(verts))
    point_tags: List[int] = [
        gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in verts
    ]
    line_tags: List[int] = []
    n = len(point_tags)
    for i in range(n):
        a = point_tags[i]
        b = point_tags[(i + 1) % n]
        line_tags.append(gmsh.model.occ.addLine(a, b))
    loop_tag = gmsh.model.occ.addCurveLoop(line_tags)
    surf_tag = gmsh.model.occ.addPlaneSurface([loop_tag])
    return surf_tag


def _build_in_gmsh(
    primitives: Dict[str, Primitive],
    domain: Optional[Domain],
    named_groups: Dict[str, NamedGroup],
    mesh_dsl: Optional[MeshDSL],
    msh_path: Path,
    *,
    verbose: bool = False,
) -> Tuple[int, int]:
    """Drive gmsh via its Python API to mesh the given geometry.

    Returns ``(n_nodes, n_triangles)``.
    """
    import gmsh  # imported lazily so the module stays importable without gmsh
    import atexit

    # Reuse the mesh_generator atexit pattern: initialise once, clear() between
    # runs, finalise on interpreter shutdown.
    if not gmsh.isInitialized():
        gmsh.initialize()
        from .mesh_generator import _safe_gmsh_finalize
        atexit.register(_safe_gmsh_finalize)

    try:
        gmsh.option.setNumber('General.Verbosity', 2 if verbose else 0)
        # Reset size-control options to defaults: previous calls into the
        # same gmsh process may have flipped these (e.g. a refinement-DSL
        # build sets ExtendFromBoundary=0 / FromPoints=0 / FromCurvature=0
        # to give the field path full authority; a later non-refinement
        # build would otherwise inherit those toggles). The
        # ``CharacteristicLengthMin/Max`` knobs are also restored to a
        # neutral [0, 1e22] so a stale value from a previous build doesn't
        # silently override the new field stack.
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 1)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMin', 0.0)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', 1e22)
        gmsh.model.add('phast_compiled')

        # ------------------------------------------------------------------
        # 1. Build OCC primitives.
        # ------------------------------------------------------------------
        prim_tags: Dict[str, int] = {}  # primitive name -> OCC surface tag

        def _add_occ(p: Primitive) -> int:
            if isinstance(p, Rectangle):
                ox, oy = p.origin
                sx, sy = p.size
                return gmsh.model.occ.addRectangle(ox, oy, 0.0, sx, sy)
            if isinstance(p, Circle):
                cx, cy = p.center
                return gmsh.model.occ.addDisk(cx, cy, 0.0, p.radius, p.radius)
            if isinstance(p, Polygon):
                return _add_occ_polygon(gmsh, p)
            raise NotImplementedError(
                f"primitive '{p.name}' of type {p.kind!r} cannot participate "
                "in boolean ops (rectangle / circle / polygon only). "
                "File a follow-up issue if you need it.")

        if domain is None:
            # No explicit domain: union all 2D primitives (defensive default).
            # In practice config.py guarantees a domain is present whenever
            # primitives are; this branch is just for direct API use.
            for name, p in primitives.items():
                prim_tags[name] = _add_occ(p)
            gmsh.model.occ.synchronize()
            domain_dim_tags = [(2, t) for t in prim_tags.values()]
        else:
            base_tag = _add_occ(primitives[domain.base])
            prim_tags[domain.base] = base_tag

            # Per-subtract strategy classification (issue #187):
            #   'cut'      -> strictly interior; use OCC cut (faster).
            #   'fragment' -> bbox flush with / crossing base; use OCC
            #                 fragment with map-based kept-region rule.
            #   'noop'     -> bbox disjoint from base; warn and skip.
            strategies = _classify_subtract(domain, primitives)

            cut_names: List[str] = []
            cut_tags: List[int] = []
            frag_names: List[str] = []
            frag_tags: List[int] = []
            for n, strat in zip(domain.subtract, strategies):
                if strat == 'noop':
                    import warnings
                    warnings.warn(
                        f"geometry_compiler: subtract primitive '{n}' is "
                        f"disjoint from base '{domain.base}'; skipping. "
                        "Check the YAML: this is almost certainly a bug "
                        "in the primitive coordinates.",
                        stacklevel=2,
                    )
                    continue
                t = _add_occ(primitives[n])
                prim_tags[n] = t
                if strat == 'fragment':
                    frag_names.append(n)
                    frag_tags.append(t)
                else:
                    cut_names.append(n)
                    cut_tags.append(t)

            # If any subtract is edge-flush we run cut FIRST (interior
            # holes) and then fragment (edge-flush slits). This keeps the
            # two operations cleanly separated and the kept-region map
            # simple. Order matters: fragment after cut so the base it
            # fragments is the already-holed surface.
            current_base = base_tag
            if cut_tags:
                out, _ = gmsh.model.occ.cut(
                    [(2, current_base)],
                    [(2, t) for t in cut_tags],
                    removeObject=True, removeTool=True,
                )
                surfaces = [t for (d, t) in out if d == 2]
                if not surfaces:
                    raise RuntimeError(
                        "geometry_compiler: OCC cut produced no surfaces "
                        f"for base '{domain.base}' minus "
                        f"{cut_names!r}. This usually means a subtract "
                        "primitive fully contains the base; check your "
                        "YAML primitive coordinates.")
                current_base = surfaces[0]

            if frag_tags:
                out, omap = gmsh.model.occ.fragment(
                    [(2, current_base)],
                    [(2, t) for t in frag_tags],
                    removeObject=True, removeTool=True,
                )
                # omap[0] -> children that came from the base.
                # omap[1..N] -> children that came from each tool.
                # Kept set = base children NOT in any tool's children.
                base_children = {tag for (d, tag) in omap[0] if d == 2}
                tool_children: set = set()
                for tm in omap[1:]:
                    tool_children.update(tag for (d, tag) in tm if d == 2)
                kept = sorted(base_children - tool_children)
                drop = sorted(base_children & tool_children) + [
                    tag for (d, tag) in out
                    if d == 2 and tag not in base_children
                ]
                if not kept:
                    raise RuntimeError(
                        "geometry_compiler: OCC fragment produced no kept "
                        f"surfaces for base '{domain.base}' minus "
                        f"{frag_names!r}. This usually means a subtract "
                        "primitive fully contains the base, or every "
                        "fragmented region was claimed by a tool; check "
                        "your YAML primitive coordinates.")
                # Remove the unwanted regions (the slit / corner cutouts
                # AND any orphan tool-only fragments that don't overlap
                # the base at all -- defensive: noop tools are filtered
                # earlier but a tool that only barely touches the base
                # corner can still leave orphan output entities).
                if drop:
                    gmsh.model.occ.remove(
                        [(2, t) for t in drop], recursive=True)
                if len(kept) > 1:
                    # Multiple kept regions => the boolean split the
                    # base into disconnected pieces. Defer that case.
                    raise NotImplementedError(
                        "geometry_compiler: OCC fragment of base "
                        f"'{domain.base}' produced {len(kept)} "
                        "disconnected kept regions; multi-component "
                        "domains are not yet supported by the inline "
                        "compiler. File a follow-up issue if needed.")
                current_base = kept[0]

            domain_dim_tags = [(2, current_base)]
            # NOTE: synchronize is deferred until *after* named-group OCC
            # fragmenting below so curve / surface tags stabilise once
            # before we snapshot them for bbox matching (issue #201).

        # ------------------------------------------------------------------
        # 2a. Named-group OCC fragmenting (issue #201).
        #
        # Two named-group forms now go through OCC ``fragment`` against
        # the domain surface, BEFORE the post-boolean synchronize:
        #
        #   * PointGroup -- previously this called ``mesh.embed`` after
        #     synchronize, which embeds the point as a meshing constraint
        #     but does NOT split the underlying OCC curve. When the point
        #     happens to lie ON an existing boundary curve (e.g. the pin
        #     centre of a hole, or any point on a domain edge), the
        #     embedded vertex breaks meshability and produces a degenerate
        #     mesh (zero triangle cells). Fragmenting the surface with the
        #     point splits the curve cleanly and preserves meshability.
        #     Points outside the surface gracefully fall through (OCC
        #     leaves them disjoint; we tag them as standalone Physical
        #     Points, matching PR #207's external-master-node workflow).
        #
        #   * RegionGroup whose region only partially overlaps the base
        #     domain (the L-shaped panel ``load_segment`` BC: a thin
        #     rectangle straddling part of the top edge). The legacy
        #     full-bbox-containment match in :func:`_curves_in_bbox` finds
        #     no curves to tag because the underlying boundary is a single
        #     unsplit curve. Fragmenting the surface with the region as a
        #     2D tool splits the boundary curves at the region's edges; we
        #     keep all base-derived children as part of the domain (so the
        #     mesh area is unchanged) and later filter for boundary
        #     sub-curves whose midpoint lies inside the region.
        # ------------------------------------------------------------------

        named_point_tags: Dict[str, int] = {}      # gname -> OCC point tag
        named_region_info: Dict[str, Tuple[float, float, float, float]] = {}
        # gname -> region bbox (post-fragment, used to filter curves)

        if domain_dim_tags:
            current_surfs = list(domain_dim_tags)  # list[(2, tag)]

            # PointGroups: fragment surface with each point one-at-a-time
            # so the per-point output map is unambiguous.
            for gname, g in named_groups.items():
                if not isinstance(g, PointGroup):
                    continue
                px, py = g.coords
                pt_tag = gmsh.model.occ.addPoint(px, py, 0.0)
                try:
                    out, omap = gmsh.model.occ.fragment(
                        current_surfs, [(0, pt_tag)],
                        removeObject=True, removeTool=True,
                    )
                except Exception:
                    # OCC failed to fragment (rare; point geometry-degenerate).
                    # Fall back to a standalone Physical Point.
                    named_point_tags[gname] = pt_tag
                    continue
                # The point may have been merged into an existing vertex of
                # the surface (point-on-boundary case): the tool entry of
                # the omap then refers to the merged vertex tag rather than
                # the original. Either way, use the omap entry as the
                # reference post-fragment point tag.
                tool_pts = [t for (d, t) in omap[1] if d == 0]
                if tool_pts:
                    named_point_tags[gname] = tool_pts[0]
                else:
                    named_point_tags[gname] = pt_tag
                # Surface entities can change tag if the point lay on an
                # interior of the surface (it generally does not, but be
                # defensive).
                kept = [(d, t) for (d, t) in omap[0] if d == 2]
                if kept:
                    current_surfs = kept

            # RegionGroups: fragment surface with each region's primitive.
            for gname, g in named_groups.items():
                if not isinstance(g, RegionGroup):
                    continue
                region = g.region
                if isinstance(region, Rectangle):
                    rx, ry = region.origin
                    sx, sy = region.size
                    region_tag = gmsh.model.occ.addRectangle(
                        rx, ry, 0.0, sx, sy)
                elif isinstance(region, Circle):
                    cx, cy = region.center
                    region_tag = gmsh.model.occ.addDisk(
                        cx, cy, 0.0, region.radius, region.radius)
                else:
                    # Polygon / unsupported region kind: skip (legacy bbox
                    # path will still try below).
                    named_region_info[gname] = _bbox_of(region)
                    continue
                try:
                    out, omap = gmsh.model.occ.fragment(
                        current_surfs, [(2, region_tag)],
                        removeObject=True, removeTool=True,
                    )
                except Exception:
                    named_region_info[gname] = _bbox_of(region)
                    continue
                # Keep all base-derived surfaces; drop tool-only children
                # (the region's portion that did not overlap the base).
                base_children = {tag for tm in omap[:len(current_surfs)]
                                 for (d, tag) in tm if d == 2}
                tool_children = {tag for (d, tag) in omap[-1] if d == 2}
                kept_tags = sorted(base_children)
                drop_tags = sorted(tool_children - base_children)
                if drop_tags:
                    gmsh.model.occ.remove(
                        [(2, t) for t in drop_tags], recursive=True)
                if kept_tags:
                    current_surfs = [(2, t) for t in kept_tags]
                named_region_info[gname] = _bbox_of(region)

            domain_dim_tags = current_surfs

        gmsh.model.occ.synchronize()

        # ------------------------------------------------------------------
        # 2b. Tag physical groups.
        #    - The composite surface gets "domain".
        #    - Each surviving primitive boundary gets <name>.boundary as a
        #      Physical Curve. Boundaries are matched by axis-aligned
        #      bounding-box overlap with the primitive's source bbox.
        #    - Region named-groups become Physical Curve sets restricted
        #      to true boundary sub-curves whose midpoint lies inside the
        #      region (issue #201).
        #    - PointGroup -> Physical Point at the (possibly-merged) vertex.
        # ------------------------------------------------------------------

        # Composite domain surface (may now be multiple base-derived pieces
        # if a RegionGroup straddled a boundary; they're all domain).
        surf_tags = [t for (_, t) in domain_dim_tags] if domain_dim_tags else []
        surf_tag = surf_tags[0] if surf_tags else None
        if surf_tags:
            gmsh.model.addPhysicalGroup(2, surf_tags, name='domain')

        # Map primitive boundaries -> surviving curves via bbox match.
        # OCC curves expose getBoundingBox(dim, tag) -> (xmin, ymin, zmin, xmax, ymax, zmax).
        all_curves = gmsh.model.occ.getEntities(1)
        curve_bboxes: Dict[int, Tuple[float, float, float, float]] = {}
        for (_, tag) in all_curves:
            bb = gmsh.model.getBoundingBox(1, tag)
            curve_bboxes[tag] = (bb[0], bb[1], bb[3], bb[4])

        # True-boundary filter: a curve is on the domain boundary iff it
        # is adjacent to exactly one of our kept surfaces. Curves adjacent
        # to two kept surfaces are internal seams created by named-group
        # fragmenting (issue #201) and must not be tagged as boundary.
        kept_surf_set = set(surf_tags)

        def _is_true_boundary(curve_tag: int) -> bool:
            try:
                up, _ = gmsh.model.getAdjacencies(1, curve_tag)
            except Exception:
                return True
            adj_kept = [s for s in up if int(s) in kept_surf_set]
            return len(adj_kept) == 1

        def _on_outline(base_prim: Primitive, curve_tag: int,
                        region_bbox: Tuple[float, float, float, float],
                        *, tol: float = 1e-4) -> bool:
            """True iff the curve lies on the outline of ``base_prim`` in
            an orientation consistent with the RegionGroup's intent.

            For a Rectangle base, edges are horizontal or vertical. The
            user pattern for a boundary-segment RegionGroup is a thin
            strip that crosses ONE edge of the base. We disambiguate the
            target edge by the strip's aspect ratio: a strip wider than
            tall (long axis horizontal) selects a sub-segment of a
            horizontal base edge; tall-and-thin selects a vertical edge.
            This rejects the orthogonal stub curves that the fragment
            inevitably creates where the strip's perpendicular sides
            cross other edges (issue #201).
            """
            cx0, cy0, cx1, cy1 = curve_bboxes[curve_tag]
            mx = 0.5 * (cx0 + cx1)
            my = 0.5 * (cy0 + cy1)
            rx0, ry0, rx1, ry1 = region_bbox
            r_w = rx1 - rx0
            r_h = ry1 - ry0
            if isinstance(base_prim, Rectangle):
                ox, oy = base_prim.origin
                sx, sy = base_prim.size
                on_top_or_bot = (abs(my - oy) < tol
                                 or abs(my - (oy + sy)) < tol)
                on_left_or_rt = (abs(mx - ox) < tol
                                 or abs(mx - (ox + sx)) < tol)
                # Aspect-ratio gate for ambiguous strips. A square-ish
                # region accepts both; only when the region is clearly
                # wider than tall (or vice versa) do we reject the
                # orthogonal edge.
                if r_w > 1.5 * r_h:
                    return on_top_or_bot
                if r_h > 1.5 * r_w:
                    return on_left_or_rt
                return on_top_or_bot or on_left_or_rt
            if isinstance(base_prim, Circle):
                cx, cy = base_prim.center
                d = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
                return abs(d - base_prim.radius) < tol
            # Polygon / other: be permissive.
            return True

        def _curves_in_bbox(target_bbox: Tuple[float, float, float, float],
                            *, tol: float = 1e-4) -> List[int]:
            """Return curve tags whose bbox is fully inside ``target_bbox``
            (with a small tolerance). For a primitive that survived a
            boolean op, the boundary curves of the *hole* it punched (or
            of the part of its own outline that survived) lie inside its
            source bbox.
            """
            xmin, ymin, xmax, ymax = target_bbox
            out: List[int] = []
            for tag, (cx0, cy0, cx1, cy1) in curve_bboxes.items():
                if (cx0 >= xmin - tol and cy0 >= ymin - tol
                        and cx1 <= xmax + tol and cy1 <= ymax + tol):
                    out.append(tag)
            return out

        # Auto-exposed <primitive>.boundary physical groups.
        for name, p in primitives.items():
            bbox = _bbox_of(p)
            curves = _curves_in_bbox(bbox)
            if not curves:
                # Primitive didn't survive (e.g. fully consumed by a cut).
                continue
            # For the *base* primitive we further restrict: its bbox
            # contains everything; only emit boundary curves that are NOT
            # contained in any subtract-primitive's bbox.
            if domain is not None and name == domain.base and domain.subtract:
                excluded: set = set()
                for sub_name in domain.subtract:
                    sub_bbox = _bbox_of(primitives[sub_name])
                    for ct in _curves_in_bbox(sub_bbox):
                        excluded.add(ct)
                curves = [c for c in curves if c not in excluded]
            # Drop internal seams introduced by named-group fragmenting.
            curves = [c for c in curves if _is_true_boundary(c)]
            if not curves:
                continue
            gmsh.model.addPhysicalGroup(1, curves, name=f'{name}.boundary')

        # Explicit named groups.
        for gname, g in named_groups.items():
            if isinstance(g, SelectorAliasGroup):
                # Alias: emit the same curve set under the user-friendly name.
                src = primitives[g.primitive]
                bbox = _bbox_of(src)
                curves = _curves_in_bbox(bbox)
                if (g.primitive == (domain.base if domain else None)
                        and domain and domain.subtract):
                    excluded = set()
                    for sub_name in domain.subtract:
                        sub_bbox = _bbox_of(primitives[sub_name])
                        for ct in _curves_in_bbox(sub_bbox):
                            excluded.add(ct)
                    curves = [c for c in curves if c not in excluded]
                curves = [c for c in curves if _is_true_boundary(c)]
                if g.selector_kind == 'boundary' and curves:
                    gmsh.model.addPhysicalGroup(1, curves, name=gname)
                # 'centre' / 'interior' selectors would need point-/surface-
                # level tagging; deferred in #146 (auto-exposed
                # <prim>.boundary already covers the BC layer's needs).
            elif isinstance(g, PointGroup):
                # Surface was already fragmented with the point above; use
                # the (possibly-merged) post-fragment point tag.
                pt_tag = named_point_tags.get(gname)
                if pt_tag is None:
                    # Defensive: shouldn't happen, but tag a fresh point.
                    px, py = g.coords
                    pt_tag = gmsh.model.occ.addPoint(px, py, 0.0)
                    gmsh.model.occ.synchronize()
                gmsh.model.addPhysicalGroup(0, [pt_tag], name=gname)
            elif isinstance(g, RegionGroup):
                # The surface was fragmented with the region above so any
                # boundary curves of the base are split at the region's
                # edges. The semantic of a RegionGroup is "tag the curves
                # that lie on the original base outline AND inside the
                # region" -- i.e. the boundary sub-segment(s) of the
                # underlying domain that fall within the region. We
                # cannot just keep all true-boundary curves whose bbox
                # falls inside the region: fragmenting with a 2D region
                # that straddles a base edge creates new boundary curves
                # along the region's other sides (the "load_segment"
                # strip's underside in the L-shaped panel test case);
                # those new curves are post-fragment artefacts, not part
                # of the original domain outline. Filter geometrically
                # by checking that each candidate curve's midpoint lies
                # on the base primitive's outline (issue #201).
                bbox = named_region_info.get(gname, _bbox_of(g.region))
                base_prim = (primitives[domain.base]
                             if domain is not None else None)
                curves = _curves_in_bbox(bbox)
                curves = [c for c in curves if _is_true_boundary(c)]
                if base_prim is not None:
                    curves = [c for c in curves
                              if _on_outline(base_prim, c, bbox)]
                if curves:
                    gmsh.model.addPhysicalGroup(1, curves, name=gname)

        # ------------------------------------------------------------------
        # 3. Mesh refinement: emit Distance + Threshold per rule, combine
        #    with Field[Min].
        # ------------------------------------------------------------------
        default_size = (mesh_dsl.default_size if mesh_dsl is not None
                        else _heuristic_default_size(primitives))

        field_ids: List[int] = []
        if mesh_dsl is not None and mesh_dsl.refined:
            for rule in mesh_dsl.refined:
                if rule.region is not None:
                    bbox = _bbox_of(rule.region)
                else:
                    src = primitives.get(rule.primitive)
                    if src is None:
                        continue
                    bbox = _bbox_of(src)

                if getattr(rule, 'mode', 'threshold') == 'box':
                    # Issue #200: interior-fill refinement via Gmsh
                    # Field[Box]. VIn applies inside the coordinate box,
                    # VOut elsewhere; Thickness controls a smooth
                    # transition outside the box. A box that lies
                    # entirely outside the meshed domain is a no-op:
                    # Gmsh evaluates the field at every node but only
                    # nodes inside the box receive VIn, so an empty
                    # intersection silently leaves the mesh uniform.
                    bx0, by0, bx1, by1 = bbox
                    box_id = gmsh.model.mesh.field.add('Box')
                    gmsh.model.mesh.field.setNumber(box_id, 'VIn', rule.size)
                    gmsh.model.mesh.field.setNumber(box_id, 'VOut', default_size)
                    gmsh.model.mesh.field.setNumber(box_id, 'XMin', bx0)
                    gmsh.model.mesh.field.setNumber(box_id, 'XMax', bx1)
                    gmsh.model.mesh.field.setNumber(box_id, 'YMin', by0)
                    gmsh.model.mesh.field.setNumber(box_id, 'YMax', by1)
                    gmsh.model.mesh.field.setNumber(
                        box_id, 'Thickness', rule.thickness)
                    field_ids.append(box_id)
                    continue

                target_curves = _curves_in_bbox(bbox)
                dist_id = gmsh.model.mesh.field.add('Distance')
                if target_curves:
                    gmsh.model.mesh.field.setNumbers(
                        dist_id, 'CurvesList', target_curves)
                else:
                    # Fall back to point-distance from the bbox centre.
                    cx, cy = _midpoints(bbox)
                    pt_tag = gmsh.model.occ.addPoint(cx, cy, 0.0)
                    gmsh.model.occ.synchronize()
                    gmsh.model.mesh.field.setNumbers(
                        dist_id, 'PointsList', [pt_tag])
                gmsh.model.mesh.field.setNumber(dist_id, 'Sampling', 50)

                thr_id = gmsh.model.mesh.field.add('Threshold')
                gmsh.model.mesh.field.setNumber(thr_id, 'InField', dist_id)
                gmsh.model.mesh.field.setNumber(thr_id, 'SizeMin', rule.size)
                gmsh.model.mesh.field.setNumber(thr_id, 'SizeMax', default_size)
                gmsh.model.mesh.field.setNumber(thr_id, 'DistMin', 0.0)
                gmsh.model.mesh.field.setNumber(thr_id, 'DistMax',
                                                rule.margin if rule.margin > 0
                                                else 0.0)
                field_ids.append(thr_id)

        if field_ids:
            min_id = gmsh.model.mesh.field.add('Min')
            gmsh.model.mesh.field.setNumbers(min_id, 'FieldsList', field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_id)
            gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
            gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
            gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        else:
            # No refinement DSL: rely on a uniform CharacteristicLength.
            gmsh.option.setNumber('Mesh.CharacteristicLengthMin', default_size)
            gmsh.option.setNumber('Mesh.CharacteristicLengthMax', default_size)

        gmsh.option.setNumber('Mesh.Algorithm', 6)  # Frontal-Delaunay
        gmsh.option.setNumber('Mesh.ElementOrder', 1)

        gmsh.model.mesh.generate(2)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, elem_tags, _ = gmsh.model.mesh.getElements(2)
        n_nodes = len(node_tags)
        n_tri = sum(len(t) for t in elem_tags)

        gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        gmsh.write(str(msh_path))
    finally:
        gmsh.clear()

    return n_nodes, n_tri


def _heuristic_default_size(primitives: Dict[str, Primitive]) -> float:
    """Fallback element size when the user supplied no mesh DSL.

    Uses ~5% of the smallest primitive dimension; biased small to avoid
    silently delivering an unusable coarse mesh.
    """
    best = None
    for p in primitives.values():
        if isinstance(p, Rectangle):
            d = min(p.size)
        elif isinstance(p, Circle):
            d = 2.0 * p.radius
        else:
            continue
        if best is None or d < best:
            best = d
    if best is None:
        return 1.0
    return max(0.05 * best, 1e-3)


def _run_gmsh_subprocess(geo_path: Path, msh_path: Path,
                         *, verbose: bool = False) -> None:
    """Subprocess fallback for environments without the gmsh Python module.

    The compiler prefers the in-process Python API (it lets us programmatically
    tag physical groups by bounding-box matching after the boolean op);
    this fallback is a thin convenience for users who only have the gmsh
    CLI installed.
    """
    cmd = ['gmsh', '-2', '-format', 'msh2', '-o', str(msh_path), str(geo_path)]
    if not verbose:
        cmd.insert(1, '-v')
        cmd.insert(2, '0')
    subprocess.run(cmd, check=True,
                   stdout=(None if verbose else subprocess.DEVNULL),
                   stderr=(None if verbose else subprocess.DEVNULL))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compile_geometry(
    primitives: Dict[str, Primitive],
    domain: Optional[Domain],
    named_groups: Optional[Dict[str, NamedGroup]] = None,
    mesh_dsl: Optional[MeshDSL] = None,
    *,
    verbose: bool = False,
    force: bool = False,
) -> Path:
    """Compile a parsed YAML geometry block into a cached ``.msh`` file.

    Parameters
    ----------
    primitives : dict[str, Primitive]
        Output of :func:`geometry_dsl.parse_primitives`.
    domain : Domain or None
        Output of :func:`geometry_dsl.parse_domain`. May be ``None`` only
        when ``primitives`` contains a single 2D primitive (in which case
        that primitive *is* the domain).
    named_groups : dict[str, NamedGroup] or None
        Output of :func:`geometry_dsl.parse_named_groups`.
    mesh_dsl : MeshDSL or None
        Output of :func:`geometry_dsl.parse_mesh_dsl`. If ``None`` the
        compiler picks a heuristic uniform element size.
    force : bool
        If ``True``, ignore any cache hit and always rebuild.

    Returns
    -------
    Path
        Absolute path to the cached ``.msh`` file. The path is stable
        across runs for the same reference geometry.
    """
    if named_groups is None:
        named_groups = {}

    if domain is None and len(primitives) > 1:
        raise ValueError(
            "compile_geometry: when more than one primitive is declared, a "
            "geometry.domain block is required to disambiguate the active "
            "computational region."
        )
    if domain is not None:
        reason = _unsupported_domain_op(domain, primitives)
        if reason is not None:
            raise NotImplementedError(
                f"geometry compiler (issue #146) does not yet support this "
                f"domain: {reason}"
            )

    key = cache_key(primitives, domain, named_groups, mesh_dsl)
    cdir = cache_dir()
    msh_path = cdir / f'{key}.msh'
    geo_path = cdir / f'{key}.geo'

    if msh_path.exists() and not force:
        if verbose:
            print(f"[geometry_compiler] cache hit: {msh_path}", flush=True)
        return msh_path

    # Write the .geo debug artifact (compiler does not consume it).
    geo_text = _emit_geo_text(primitives, domain, named_groups, mesh_dsl)
    geo_path.write_text(geo_text)

    # Build the mesh into a temporary file in the same dir (atomic replace).
    # Gmsh determines the writer from the file extension, so the temp must
    # also end in .msh; we just prepend a dot to the basename.
    tmp = msh_path.with_name('.' + msh_path.name + '.tmp.msh')
    try:
        try:
            n_nodes, n_tri = _build_in_gmsh(
                primitives, domain, named_groups, mesh_dsl,
                tmp, verbose=verbose,
            )
        except ImportError:
            # Python gmsh module unavailable; try CLI fallback against the
            # debug .geo. The CLI path will NOT carry our programmatic
            # physical-group tagging -- emit a clear warning so the user
            # knows the BC layer may not see <name>.boundary node sets.
            import warnings
            warnings.warn(
                "Python 'gmsh' module not importable; falling back to the "
                "gmsh CLI on the auto-generated .geo file. Named-group "
                "physical entities will NOT be present -- install the gmsh "
                "Python package for full functionality.",
                stacklevel=2,
            )
            _run_gmsh_subprocess(geo_path, tmp, verbose=verbose)
            n_nodes, n_tri = -1, -1
        os.replace(tmp, msh_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    if verbose:
        print(f"[geometry_compiler] cached {msh_path} "
              f"({n_nodes} nodes, {n_tri} triangles)", flush=True)
    return msh_path
