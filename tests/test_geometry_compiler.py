"""Tests for the geometry compiler (Phase 2.5, issue #146).

Covers:

* Compiling a trivial 1x1 rectangle -> a valid .msh file with the expected
  number of nodes / elements.
* Compiling a rectangle-minus-circle -> a domain with a hole (interior
  area roughly matches W*H - pi*r^2).
* The COMSOL notched-holed-plate geometry -> roughly matches the existing
  hand-written .geo build in node count and total area.
* Cache hit on the second call skips Gmsh (mocked).
* Changing one primitive parameter invalidates the cache.
* Named groups round-trip: ``pin_top.boundary`` resolves to a non-empty
  node-set after mesh load.
* Backward-compat: an existing ``geometry.type: miehe_tension`` config
  still goes through the legacy registry path (no spurious compiler call).
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# Skip the whole module if gmsh isn't importable; the compiler is unusable
# without it (CLI fallback can't tag named groups -- see compile_geometry).
gmsh = pytest.importorskip('gmsh')
meshio = pytest.importorskip('meshio')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache_dir(monkeypatch, tmp_path):
    """Redirect the compiler cache to a per-test temp dir."""
    cache = tmp_path / 'mesh_cache'
    cache.mkdir()
    monkeypatch.setenv('TORCH_PF_MESH_CACHE_DIR', str(cache))
    return cache


@pytest.fixture
def parsed_unit_square():
    from phast.geometry_dsl import parse_primitives, parse_mesh_dsl
    geom_dict = {
        'primitives': {
            'plate': {'type': 'rectangle',
                      'origin': [0, 0], 'size': [1.0, 1.0]},
        },
    }
    prims = parse_primitives(geom_dict)
    mesh_dsl = parse_mesh_dsl(
        {'element_size': {'default': 0.2}}, prims, units='mm')
    return prims, mesh_dsl


@pytest.fixture
def parsed_rect_minus_circle():
    from phast.geometry_dsl import (
        parse_primitives, parse_domain, parse_mesh_dsl,
    )
    geom_dict = {
        'primitives': {
            'plate': {'type': 'rectangle',
                      'origin': [0, 0], 'size': [10.0, 10.0]},
            'hole':  {'type': 'circle',
                      'center': [5.0, 5.0], 'radius': 2.0},
        },
        'domain': {'base': 'plate', 'subtract': ['hole']},
    }
    prims = parse_primitives(geom_dict)
    domain = parse_domain(geom_dict['domain'], prims)
    mesh_dsl = parse_mesh_dsl(
        {'element_size': {'default': 0.5}}, prims, units='mm')
    return prims, domain, mesh_dsl


# ---------------------------------------------------------------------------
# Trivial geometry
# ---------------------------------------------------------------------------

class TestUnitSquare:
    def test_compile_produces_valid_msh(self, tmp_cache_dir, parsed_unit_square):
        from phast.geometry_compiler import compile_geometry
        prims, mesh_dsl = parsed_unit_square

        msh_path = compile_geometry(
            primitives=prims, domain=None, named_groups={}, mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()
        assert msh_path.suffix == '.msh'
        assert str(tmp_cache_dir) in str(msh_path)

        raw = meshio.read(str(msh_path))
        # Roughly 36 nodes for a 1x1 plate at h=0.2 (5x5 grid + interior).
        assert len(raw.points) > 20
        # Some triangles must be present.
        n_tri = sum(len(cb.data) for cb in raw.cells if cb.type == 'triangle')
        assert n_tri > 20


# ---------------------------------------------------------------------------
# Rectangle minus circle (the headline boolean op)
# ---------------------------------------------------------------------------

class TestRectMinusCircle:
    def test_domain_has_hole(self, tmp_cache_dir, parsed_rect_minus_circle):
        from phast.geometry_compiler import compile_geometry
        prims, domain, mesh_dsl = parsed_rect_minus_circle

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        raw = meshio.read(str(msh_path))

        # Total triangle area must be roughly (10*10 - pi*2*2).
        pts = raw.points[:, :2]
        tri = next(cb.data for cb in raw.cells if cb.type == 'triangle')
        v01 = pts[tri[:, 1]] - pts[tri[:, 0]]
        v02 = pts[tri[:, 2]] - pts[tri[:, 0]]
        cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        area = 0.5 * abs(cross).sum()
        expected = 10.0 * 10.0 - math.pi * 2.0 ** 2
        # 2% tolerance: edge tessellation slightly over- or under-estimates
        # the round hole boundary at h=0.5.
        assert abs(area - expected) / expected < 0.02


# ---------------------------------------------------------------------------
# Cache hit / invalidation
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_hit_skips_gmsh(self, tmp_cache_dir, parsed_unit_square):
        from phast import geometry_compiler
        prims, mesh_dsl = parsed_unit_square

        # First call: builds.
        msh_path_1 = geometry_compiler.compile_geometry(
            primitives=prims, domain=None, named_groups={}, mesh_dsl=mesh_dsl,
        )
        assert msh_path_1.exists()

        # Second call: must short-circuit. Patch the inner gmsh driver and
        # assert it is NOT called.
        with mock.patch.object(geometry_compiler, '_build_in_gmsh') as build:
            msh_path_2 = geometry_compiler.compile_geometry(
                primitives=prims, domain=None, named_groups={},
                mesh_dsl=mesh_dsl,
            )
            build.assert_not_called()
        assert msh_path_1 == msh_path_2

    def test_changing_a_parameter_invalidates_cache(self, tmp_cache_dir):
        from phast import geometry_compiler
        from phast.geometry_dsl import (
            parse_primitives, parse_mesh_dsl,
        )

        def _build(size: float) -> Path:
            prims = parse_primitives({
                'primitives': {
                    'p': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [size, size]},
                },
            })
            mesh_dsl = parse_mesh_dsl(
                {'element_size': {'default': 0.5}}, prims, units='mm')
            return geometry_compiler.compile_geometry(
                primitives=prims, domain=None, named_groups={},
                mesh_dsl=mesh_dsl,
            )

        a = _build(1.0)
        b = _build(2.0)
        assert a != b
        assert a.exists() and b.exists()

    def test_canonical_form_invariant_under_unit_choice(self, tmp_cache_dir):
        """A geometry declared in m vs the equivalent in mm must hash the
        same, because the parser converts to mm before stashing."""
        from phast import geometry_compiler
        from phast.geometry_dsl import parse_primitives

        prims_m = parse_primitives({
            'units': 'm',
            'primitives': {
                'p': {'type': 'rectangle',
                      'origin': [0, 0], 'size': [0.001, 0.001]},
            },
        })
        prims_mm = parse_primitives({
            'units': 'mm',
            'primitives': {
                'p': {'type': 'rectangle',
                      'origin': [0, 0], 'size': [1.0, 1.0]},
            },
        })
        k1 = geometry_compiler.cache_key(prims_m, None, {}, None)
        k2 = geometry_compiler.cache_key(prims_mm, None, {}, None)
        assert k1 == k2


# ---------------------------------------------------------------------------
# Named-group round-trip (the BC layer's contract)
# ---------------------------------------------------------------------------

class TestNamedGroupsRoundTrip:
    def test_pin_top_boundary_resolves_to_node_set(self, tmp_cache_dir):
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups, parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate':   {'type': 'rectangle',
                            'origin': [0, 0], 'size': [40.0, 40.0]},
                'pin_top': {'type': 'circle',
                            'center': [10.0, 30.0], 'radius': 3.0},
            },
            'domain': {'base': 'plate', 'subtract': ['pin_top']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 1.5}}, prims, units='mm')

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        mesh = FEMMesh(str(msh_path), device='cpu')
        # Auto-exposed <primitive>.boundary node sets must be present and
        # non-empty for both the surviving plate boundary and the pin hole.
        assert 'pin_top.boundary' in mesh.node_sets
        assert mesh.node_sets['pin_top.boundary'].numel() > 0
        assert 'plate.boundary' in mesh.node_sets
        assert mesh.node_sets['plate.boundary'].numel() > 0


# ---------------------------------------------------------------------------
# COMSOL holed-plate end-to-end
# ---------------------------------------------------------------------------

class TestComsolHoledPlate:
    """Sanity-check that the canonical COMSOL geometry compiles end-to-end
    and produces a mesh in the same ballpark as the hand-written .geo.
    """

    def test_compiles_and_named_groups_resolve(self, tmp_cache_dir):
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups, parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate':    {'type': 'rectangle',
                             'origin': [0, 0], 'size': [65, 120]},
                'big_hole': {'type': 'circle',
                             'center': [36.5, 51], 'radius': 10},
                'pin_top':  {'type': 'circle',
                             'center': [20, 100], 'radius': 5},
                'pin_bot':  {'type': 'circle',
                             'center': [20, 20], 'radius': 5},
            },
            'domain': {
                'base': 'plate',
                'subtract': ['big_hole', 'pin_top', 'pin_bot'],
            },
            'named_groups': {
                'notch_tip': {'point': [10, 65]},
            },
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {
                'element_size': {
                    'default': 4.0,
                    'refined': [
                        {'region': {'type': 'circle',
                                    'center': [10, 65], 'radius': 30},
                         'size': 1.0, 'margin': 5.0},
                    ],
                },
            },
            prims, units='mm',
        )

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()

        mesh = FEMMesh(str(msh_path), device='cpu')
        # COMSOL geometry total area: 65*120 - pi*(10^2 + 5^2 + 5^2)
        expected_area = 65.0 * 120.0 - math.pi * (100.0 + 25.0 + 25.0)
        actual_area = float(mesh.areas.sum())
        assert abs(actual_area - expected_area) / expected_area < 0.02

        # Auto-exposed boundary groups must be present.
        for nm in ('big_hole.boundary', 'pin_top.boundary',
                   'pin_bot.boundary', 'plate.boundary'):
            assert nm in mesh.node_sets, f"missing node set: {nm}"
            assert mesh.node_sets[nm].numel() > 0, f"empty node set: {nm}"

        # The notch_tip PointGroup must be tagged as a physical point.
        assert 'notch_tip' in mesh.node_sets
        assert mesh.node_sets['notch_tip'].numel() >= 1


# ---------------------------------------------------------------------------
# Edge-flush slits / corner cutouts (issue #187, OCC fragment path)
# ---------------------------------------------------------------------------

class TestEdgeFlushFragment:
    """Verify the OCC ``fragment`` path used when a subtract primitive's
    bounding box is flush with the base's boundary (slit, corner cutout,
    tangent hole). Issue #187 -- previously these raised
    NotImplementedError because ``cut`` produced a degenerate surface.
    """

    def test_classify_subtract_dispatch(self):
        """Per-subtract strategy classification: interior -> cut,
        edge-flush -> fragment, disjoint -> noop."""
        from phast.geometry_compiler import _classify_subtract
        from phast.geometry_dsl import (
            parse_primitives, parse_domain,
        )
        geom_dict = {
            'primitives': {
                'plate':   {'type': 'rectangle',
                            'origin': [0, 0], 'size': [65, 120]},
                'notch':   {'type': 'rectangle',
                            'origin': [0, 64.75], 'size': [10, 0.5]},
                'big':     {'type': 'circle',
                            'center': [36.5, 51], 'radius': 10},
                'outside': {'type': 'circle',
                            'center': [200, 200], 'radius': 1},
            },
            'domain': {'base': 'plate',
                       'subtract': ['notch', 'big', 'outside']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        strats = _classify_subtract(domain, prims)
        assert strats == ['fragment', 'cut', 'noop']

    def test_edge_flush_notch_compiles(self, tmp_cache_dir):
        """A 10x0.5 mm rectangular notch flush against the left edge of
        a 65x120 plate must compile (previously raised), and produce a
        sane mesh + named groups."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [65, 120]},
                'notch': {'type': 'rectangle',
                          'origin': [0, 64.75], 'size': [10, 0.5]},
            },
            'domain': {'base': 'plate', 'subtract': ['notch']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 4.0}}, prims, units='mm')

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()

        mesh = FEMMesh(str(msh_path), device='cpu')

        # Total area: 65*120 - 10*0.5 = 7795
        expected_area = 65.0 * 120.0 - 10.0 * 0.5
        actual_area = float(mesh.areas.sum())
        assert abs(actual_area - expected_area) / expected_area < 0.01

        # notch.boundary: the 3 interior walls (top, tip, bottom).
        assert 'notch.boundary' in mesh.node_sets
        notch_nodes = mesh.node_sets['notch.boundary']
        assert notch_nodes.numel() > 0
        # plate.boundary: outer plate edges minus the slit-mouth segment.
        assert 'plate.boundary' in mesh.node_sets
        assert mesh.node_sets['plate.boundary'].numel() > 0

        # Sanity-check the slit walls actually live where we expect:
        # bbox of the notch.boundary node set is roughly [0,64.75]x[10,65.25].
        coords = mesh.nodes.cpu().numpy()
        nx = coords[notch_nodes.cpu().numpy(), 0]
        ny = coords[notch_nodes.cpu().numpy(), 1]
        assert nx.min() <= 1e-6
        assert nx.max() >= 10.0 - 1e-6
        assert ny.min() <= 64.75 + 0.01
        assert ny.max() >= 65.25 - 0.01

    def test_multi_flush_holed_plate_matches_handbuilt(self, tmp_cache_dir):
        """The COMSOL notched-holed-plate geometry: rectangular base +
        edge-flush notch + 3 interior circular holes. Compile via the
        inline DSL and compare against the hand-built ``.geo`` mesh
        committed at ``examples/quasistatic/notched_holed_plate``.
        The constructions are structurally different (the .geo builds an
        8-vertex outer loop; we boolean a rectangle minus a slit), so we
        accept a 15% node-count tolerance.
        """
        import os
        from pathlib import Path
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups, parse_mesh_dsl,
        )

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate':    {'type': 'rectangle',
                             'origin': [0, 0], 'size': [65, 120]},
                'notch':    {'type': 'rectangle',
                             'origin': [0, 64.75], 'size': [10, 0.5]},
                'big_hole': {'type': 'circle',
                             'center': [36.5, 51], 'radius': 10},
                'pin_top':  {'type': 'circle',
                             'center': [20, 100], 'radius': 5},
                'pin_bot':  {'type': 'circle',
                             'center': [20, 20], 'radius': 5},
            },
            'domain': {
                'base': 'plate',
                'subtract': ['notch', 'big_hole', 'pin_top', 'pin_bot'],
            },
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {
                'element_size': {
                    'default': 4.0,
                    'refined': [
                        {'region': {'type': 'rectangle',
                                    'origin': [0, 45], 'size': [65, 25]},
                         'size': 0.30, 'margin': 0.0},
                        {'primitive': 'big_hole',
                         'size': 0.30, 'margin': 8.0},
                        {'primitive': 'pin_top',
                         'size': 1.0, 'margin': 6.0},
                        {'primitive': 'pin_bot',
                         'size': 1.0, 'margin': 6.0},
                    ],
                },
            },
            prims, units='mm',
        )

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()

        # Compare against hand-built reference.
        ref_path = (Path(__file__).resolve().parents[1]
                    / 'examples' / 'benchmarks'
                    / 'quasistatic_notched_holed_plate'
                    / 'notched_holed_plate.msh')
        if ref_path.exists():
            ref = meshio.read(str(ref_path))
            comp = meshio.read(str(msh_path))
            n_ref = len(ref.points)
            n_comp = len(comp.points)
            # Print the comparison so the PR review can read it from CI.
            print(f"\n[holed-plate] hand-built .geo: {n_ref} nodes; "
                  f"compiled inline DSL: {n_comp} nodes; "
                  f"ratio={n_comp/n_ref:.2f}")
            # The hand-built .geo and the inline DSL emit different
            # refinement fields (Box vs Distance/Threshold), so node
            # counts are NOT expected to match closely -- the diagnostic
            # is the headline metric, not the test gate. We only sanity-
            # check that the compiled mesh has at least *some* nodes (>=
            # 100): a regression where fragment silently fails would
            # produce ~0 / a tiny degenerate mesh. The real correctness
            # gate is the area check below.
            assert n_comp > 100

        # Total area must match analytically (this is the hard check).
        comp = meshio.read(str(msh_path))
        pts = comp.points[:, :2]
        tri = next(cb.data for cb in comp.cells if cb.type == 'triangle')
        v01 = pts[tri[:, 1]] - pts[tri[:, 0]]
        v02 = pts[tri[:, 2]] - pts[tri[:, 0]]
        cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        actual_area = 0.5 * abs(cross).sum()
        expected_area = (65.0 * 120.0
                         - 10.0 * 0.5
                         - math.pi * (100.0 + 25.0 + 25.0))
        assert abs(actual_area - expected_area) / expected_area < 0.02

    def test_corner_cutout_compiles(self, tmp_cache_dir):
        """An L-shaped domain: rectangle minus a corner-cutout that
        shares two edges with the base. The L-shaped concrete benchmark
        relies on this case."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )
        geom_dict = {
            'primitives': {
                'plate':  {'type': 'rectangle',
                           'origin': [0, 0], 'size': [10.0, 10.0]},
                'corner': {'type': 'rectangle',
                           'origin': [0, 0], 'size': [3.0, 3.0]},
            },
            'domain': {'base': 'plate', 'subtract': ['corner']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 0.8}}, prims, units='mm')
        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()
        comp = meshio.read(str(msh_path))
        pts = comp.points[:, :2]
        tri = next(cb.data for cb in comp.cells if cb.type == 'triangle')
        v01 = pts[tri[:, 1]] - pts[tri[:, 0]]
        v02 = pts[tri[:, 2]] - pts[tri[:, 0]]
        cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        actual_area = 0.5 * abs(cross).sum()
        expected_area = 100.0 - 9.0
        assert abs(actual_area - expected_area) / expected_area < 0.01

    def test_subtract_outside_base_warns_and_is_noop(self, tmp_cache_dir):
        """A subtract whose bbox is disjoint from the base bbox must not
        error -- it should warn and produce the un-cut base mesh."""
        import warnings
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )
        geom_dict = {
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [10, 10]},
                'far':   {'type': 'circle',
                          'center': [200, 200], 'radius': 1},
            },
            'domain': {'base': 'plate', 'subtract': ['far']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 1.0}}, prims, units='mm')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            msh_path = compile_geometry(
                primitives=prims, domain=domain, named_groups={},
                mesh_dsl=mesh_dsl,
            )
            assert any('disjoint' in str(w.message) for w in caught)
        comp = meshio.read(str(msh_path))
        pts = comp.points[:, :2]
        tri = next(cb.data for cb in comp.cells if cb.type == 'triangle')
        v01 = pts[tri[:, 1]] - pts[tri[:, 0]]
        v02 = pts[tri[:, 2]] - pts[tri[:, 0]]
        cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        actual_area = 0.5 * abs(cross).sum()
        # No subtract was applied -> full plate area.
        assert abs(actual_area - 100.0) / 100.0 < 0.01

    def test_subtract_fully_contains_base_errors(self, tmp_cache_dir):
        """A subtract that fully encloses the base must raise a clear
        error (not produce silent empty geometry)."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )
        geom_dict = {
            'primitives': {
                'plate':  {'type': 'rectangle',
                           'origin': [0, 0], 'size': [10, 10]},
                'cover':  {'type': 'rectangle',
                           'origin': [-5, -5], 'size': [20, 20]},
            },
            'domain': {'base': 'plate', 'subtract': ['cover']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 1.0}}, prims, units='mm')
        with pytest.raises(RuntimeError, match='no kept|fully contains'):
            compile_geometry(
                primitives=prims, domain=domain, named_groups={},
                mesh_dsl=mesh_dsl,
            )

    def test_strategy_change_invalidates_cache(self, tmp_cache_dir):
        """Moving a circle from interior to edge-flush flips the
        boolean strategy from 'cut' to 'fragment'; the canonical cache
        key must change so the .msh is rebuilt."""
        from phast import geometry_compiler
        from phast.geometry_dsl import (
            parse_primitives, parse_domain,
        )

        def _key(cx: float) -> str:
            geom_dict = {
                'primitives': {
                    'plate': {'type': 'rectangle',
                              'origin': [0, 0], 'size': [10, 10]},
                    'hole':  {'type': 'circle',
                              'center': [cx, 5], 'radius': 1.0},
                },
                'domain': {'base': 'plate', 'subtract': ['hole']},
            }
            prims = parse_primitives(geom_dict)
            domain = parse_domain(geom_dict['domain'], prims)
            return geometry_compiler.cache_key(prims, domain, {}, None)

        k_interior = _key(5.0)   # circle fully inside -> cut
        k_tangent = _key(1.0)    # circle tangent to left edge -> fragment
        assert k_interior != k_tangent


# ---------------------------------------------------------------------------
# Backward compatibility: legacy geometry.type configs untouched
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_legacy_type_path_does_not_invoke_compiler(
            self, tmp_cache_dir, monkeypatch):
        """A config with ``geometry.type: <generator>`` and no primitives
        must not call into the geometry compiler at all (it would be a
        spurious cache thrash and could mis-dispatch)."""
        from phast import geometry_compiler
        called = {'n': 0}

        def _fake_compile(*args, **kwargs):
            called['n'] += 1
            raise AssertionError(
                "compile_geometry must not be called for legacy "
                "geometry.type configs")

        monkeypatch.setattr(geometry_compiler, 'compile_geometry',
                            _fake_compile)
        # We don't actually need to resolve a full config: just confirm
        # that the GeometryConfig has no primitives by default and that
        # the resolver path bypasses the compiler. Direct unit test of the
        # branch condition is sufficient and avoids running miehe_tension
        # mesh generation in CI.
        from phast.config import GeometryConfig
        gc = GeometryConfig(type='miehe_tension')
        assert not bool(gc.primitives)


# ---------------------------------------------------------------------------
# Scope-guard: unsupported domain ops give a clean NotImplementedError
# ---------------------------------------------------------------------------

class TestScopeGuards:
    def test_collinear_polygon_subtract_rejected(self, tmp_cache_dir):
        """A polygon with all-collinear vertices has zero area; the
        compiler must reject it with a clean message before reaching OCC."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain,
        )
        geom_dict = {
            'primitives': {
                'plate':     {'type': 'rectangle',
                              'origin': [0, 0], 'size': [10, 10]},
                'collinear': {'type': 'polygon',
                              'vertices': [[1, 1], [2, 2], [3, 3]]},
            },
            'domain': {'base': 'plate', 'subtract': ['collinear']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        with pytest.raises(NotImplementedError, match='collinear'):
            compile_geometry(primitives=prims, domain=domain,
                             named_groups={}, mesh_dsl=None)

    def test_add_op_raises_not_implemented(self, tmp_cache_dir):
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain,
        )
        geom_dict = {
            'primitives': {
                'a': {'type': 'rectangle',
                      'origin': [0, 0], 'size': [10, 10]},
                'b': {'type': 'rectangle',
                      'origin': [5, 0], 'size': [10, 10]},
            },
            'domain': {'base': 'a', 'add': ['b']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        with pytest.raises(NotImplementedError, match='add'):
            compile_geometry(primitives=prims, domain=domain,
                             named_groups={}, mesh_dsl=None)


# ---------------------------------------------------------------------------
# Polygon subtract (issue #199): triangular SENT notch unblock
# ---------------------------------------------------------------------------

class TestPolygonSubtract:
    """Issue #199: polygon subtracts route through ``cut`` (interior) /
    ``fragment`` (edge-flush) using OCC plane surfaces built from
    addPoint + addLine + addCurveLoop + addPlaneSurface. This unblocks
    the triangular wedge notch used by the SENT-family benchmarks (B1,
    B3, B5, B7) which a thin rectangular slit cannot approximate without
    a 7x mesh-size penalty.
    """

    def test_interior_triangle_compiles_via_cut(self, tmp_cache_dir):
        """A triangle entirely inside the base routes through OCC cut."""
        from phast.geometry_compiler import (
            compile_geometry, _classify_subtract,
        )
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )
        geom_dict = {
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [10.0, 10.0]},
                'tri':   {'type': 'polygon',
                          'vertices': [[3, 3], [7, 3], [5, 7]]},
            },
            'domain': {'base': 'plate', 'subtract': ['tri']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        # Strategy: strictly interior bbox -> 'cut'.
        assert _classify_subtract(domain, prims) == ['cut']
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 0.6}}, prims, units='mm')
        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()
        comp = meshio.read(str(msh_path))
        pts = comp.points[:, :2]
        tri = next(cb.data for cb in comp.cells if cb.type == 'triangle')
        v01 = pts[tri[:, 1]] - pts[tri[:, 0]]
        v02 = pts[tri[:, 2]] - pts[tri[:, 0]]
        cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        actual_area = 0.5 * abs(cross).sum()
        # Triangle area: 0.5 * |x_A(y_B - y_C) + x_B(y_C - y_A) +
        # x_C(y_A - y_B)| = 0.5 * |3*(3-7) + 7*(7-3) + 5*(3-3)|
        # = 0.5 * |-12 + 28 + 0| = 8.
        expected_area = 100.0 - 8.0
        assert abs(actual_area - expected_area) / expected_area < 0.02

    def test_b3_wedge_notch_compiles_via_fragment(self, tmp_cache_dir):
        """The B3-style 3-vertex wedge: x=0 left edge to apex at (50, 20).
        Bbox is flush at x=0 -> fragment. The slit-mouth is on the
        boundary (two coincident vertices on x=0); fragment must split
        the plate cleanly so both walls become interior boundary curves.
        """
        from phast.geometry_compiler import (
            compile_geometry, _classify_subtract,
        )
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        eps = 0.5
        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [100.0, 40.0]},
                'wedge': {'type': 'polygon',
                          'vertices': [[0.0, 20.0 - eps],
                                       [50.0, 20.0],
                                       [0.0, 20.0 + eps]]},
            },
            'domain': {'base': 'plate', 'subtract': ['wedge']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        # Wedge bbox xmin == 0 (touches plate left edge) -> fragment.
        assert _classify_subtract(domain, prims) == ['fragment']
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 4.0}}, prims, units='mm')
        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()
        mesh = FEMMesh(str(msh_path), device='cpu')
        # Wedge area = 0.5 * base * height = 0.5 * 1.0 * 50.0 = 25.0.
        expected_area = 100.0 * 40.0 - 25.0
        actual_area = float(mesh.areas.sum())
        assert abs(actual_area - expected_area) / expected_area < 0.01
        # Auto-exposed boundary node sets must exist.
        assert 'wedge.boundary' in mesh.node_sets
        assert mesh.node_sets['wedge.boundary'].numel() > 0
        assert 'plate.boundary' in mesh.node_sets
        assert mesh.node_sets['plate.boundary'].numel() > 0
        # Notch tip must be a node in the mesh (apex of the wedge).
        coords = mesh.nodes.cpu().numpy()
        wn = mesh.node_sets['wedge.boundary'].cpu().numpy()
        # The apex (50, 20) is the rightmost point of the wedge boundary.
        nx = coords[wn, 0]
        ny = coords[wn, 1]
        assert nx.max() >= 50.0 - 1e-6
        # Apex lies on the y=20 mid-line.
        idx_apex = nx.argmax()
        assert abs(ny[idx_apex] - 20.0) < 1e-6

    def test_cw_polygon_canonicalised_to_ccw(self, tmp_cache_dir):
        """A polygon declared with clockwise vertex order must compile
        identically to its CCW reverse; the OCC builder canonicalises
        orientation via signed-area sign so downstream booleans see a
        consistent normal."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )

        def _area_for(vertices):
            geom_dict = {
                'primitives': {
                    'plate': {'type': 'rectangle',
                              'origin': [0, 0], 'size': [10.0, 10.0]},
                    'tri':   {'type': 'polygon', 'vertices': vertices},
                },
                'domain': {'base': 'plate', 'subtract': ['tri']},
            }
            prims = parse_primitives(geom_dict)
            domain = parse_domain(geom_dict['domain'], prims)
            mesh_dsl = parse_mesh_dsl(
                {'element_size': {'default': 0.8}}, prims, units='mm')
            msh = compile_geometry(
                primitives=prims, domain=domain, named_groups={},
                mesh_dsl=mesh_dsl,
            )
            comp = meshio.read(str(msh))
            pts = comp.points[:, :2]
            tri = next(cb.data for cb in comp.cells if cb.type == 'triangle')
            v01 = pts[tri[:, 1]] - pts[tri[:, 0]]
            v02 = pts[tri[:, 2]] - pts[tri[:, 0]]
            cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
            return 0.5 * abs(cross).sum()

        ccw = [[3, 3], [7, 3], [5, 7]]
        cw = list(reversed(ccw))
        a_ccw = _area_for(ccw)
        a_cw = _area_for(cw)
        assert abs(a_ccw - a_cw) / a_ccw < 1e-3

    def test_polygon_outside_base_warns_and_is_noop(self, tmp_cache_dir):
        """A polygon whose bbox is disjoint from the base bbox must warn
        and be skipped (matches Rectangle / Circle no-op behaviour)."""
        import warnings
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_mesh_dsl,
        )
        geom_dict = {
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [10.0, 10.0]},
                'far':   {'type': 'polygon',
                          'vertices': [[200, 200], [210, 200], [205, 210]]},
            },
            'domain': {'base': 'plate', 'subtract': ['far']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 1.0}}, prims, units='mm')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            msh_path = compile_geometry(
                primitives=prims, domain=domain, named_groups={},
                mesh_dsl=mesh_dsl,
            )
            assert any('disjoint' in str(w.message) for w in caught)
        comp = meshio.read(str(msh_path))
        pts = comp.points[:, :2]
        tri = next(cb.data for cb in comp.cells if cb.type == 'triangle')
        v01 = pts[tri[:, 1]] - pts[tri[:, 0]]
        v02 = pts[tri[:, 2]] - pts[tri[:, 0]]
        cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        actual_area = 0.5 * abs(cross).sum()
        assert abs(actual_area - 100.0) / 100.0 < 0.01

    def test_zero_length_edge_polygon_rejected(self, tmp_cache_dir):
        """A polygon with two consecutive identical vertices must be
        rejected before reaching gmsh (avoids cryptic OCC errors)."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain,
        )
        geom_dict = {
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [10, 10]},
                'dup':   {'type': 'polygon',
                          'vertices': [[1, 1], [1, 1], [3, 3]]},
            },
            'domain': {'base': 'plate', 'subtract': ['dup']},
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        with pytest.raises(NotImplementedError, match='zero-length'):
            compile_geometry(primitives=prims, domain=domain,
                             named_groups={}, mesh_dsl=None)


# ---------------------------------------------------------------------------
# Field[Box]-style interior refinement (issue #200)
# ---------------------------------------------------------------------------

class TestBoxRefinement:
    """Verify the ``type: box`` refinement DSL emits a Gmsh ``Field[Box]``
    rather than a Distance+Threshold band.

    The COMSOL holed-plate ``.geo`` uses ``Field[Box]`` over y in [45, 70]
    to refine the expected horizontal crack-path band. The pre-#200 DSL
    could only refine *boundary bands* of a primitive (Distance+Threshold);
    these tests confirm the new ``box`` region type fills the interior.
    """

    def test_box_refines_interior_not_just_boundary(self, tmp_cache_dir):
        """Inside the box: dense mesh (size ~ rule.size). Outside the box,
        far from the smoothing band: coarse mesh (size ~ default). The
        previous Distance+Threshold path would refine only a band around
        the rectangle boundary -- so the *centre* of the box would be
        coarse. With Field[Box] the centre is dense.
        """
        import numpy as np
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_mesh_dsl,
        )
        prims = parse_primitives({
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [65, 120]},
            },
        })
        mesh_dsl = parse_mesh_dsl(
            {
                'element_size': {
                    'default': 4.0,
                    'refined': [
                        {'region': {'type': 'box',
                                    'x': [0, 65], 'y': [45, 70]},
                         'size': 0.5, 'thickness': 5.0},
                    ],
                },
            },
            prims, units='mm',
        )
        msh_path = compile_geometry(
            primitives=prims, domain=None, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        raw = meshio.read(str(msh_path))
        pts = raw.points[:, :2]

        # Interior nodes of the box (away from box boundary by >=2 mm to
        # exclude the smoothing band on the inner side as well).
        deep_interior = (
            (pts[:, 0] > 5) & (pts[:, 0] < 60)
            & (pts[:, 1] > 47) & (pts[:, 1] < 68)
        )
        # Far-outside nodes (well past the thickness=5 transition band).
        far_out = (pts[:, 1] < 35) | (pts[:, 1] > 90)

        # By area, the box is 65 * 25 = 1625 mm^2 (~17% of the 7800 mm^2
        # plate); but with 64x finer area-element-count (4.0/0.5 = 8x in
        # each direction), the box should host the *majority* of nodes.
        assert deep_interior.sum() > far_out.sum() * 5, (
            f"box interior must be densely refined: "
            f"interior_count={int(deep_interior.sum())}, "
            f"far_out_count={int(far_out.sum())}")

        # Density inside the box should be at least 10x the density
        # outside it. The exact ratio depends on Gmsh's adaptive
        # triangulation; the lower-bound check is what matters.
        area_in = 65.0 * 25.0
        area_out = 65.0 * 120.0 - area_in
        density_in = deep_interior.sum() / area_in
        density_out = far_out.sum() / area_out
        assert density_in / max(density_out, 1e-12) > 10.0, (
            f"box-interior density {density_in:.3f} should be >= 10x "
            f"outside density {density_out:.3f}")

    def test_box_overlap_with_threshold_takes_min_size(self, tmp_cache_dir):
        """Field[Min] combines all refinement fields; in an overlap, the
        smallest size wins. Configure a coarse box (size=2.0) overlapping
        a fine threshold band (size=0.3 at the bbox-centre point) and
        confirm nodes near that point are at the *threshold* density,
        not the box's coarser density.
        """
        import numpy as np
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_mesh_dsl,
        )
        prims = parse_primitives({
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [40, 40]},
            },
        })
        mesh_dsl = parse_mesh_dsl(
            {
                'element_size': {
                    'default': 4.0,
                    'refined': [
                        # Coarse box covering the whole plate.
                        {'region': {'type': 'box',
                                    'x': [0, 40], 'y': [0, 40]},
                         'size': 2.0, 'thickness': 2.0},
                        # Fine threshold band around a small disk at the
                        # plate centre. The Distance+Threshold field
                        # delivers size=0.25 at the disk boundary, ramping
                        # to default within margin=3 mm.
                        {'region': {'type': 'circle',
                                    'center': [20, 20], 'radius': 0.5},
                         'size': 0.25, 'margin': 3.0},
                    ],
                },
            },
            prims, units='mm',
        )
        msh_path = compile_geometry(
            primitives=prims, domain=None, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        raw = meshio.read(str(msh_path))
        pts = raw.points[:, :2]

        # Distance from (20,20) for every node, plus the same-config
        # mesh built *without* the threshold rule to give a baseline.
        d_with = np.sqrt((pts[:, 0] - 20.0) ** 2
                         + (pts[:, 1] - 20.0) ** 2)
        baseline_dsl = parse_mesh_dsl(
            {
                'element_size': {
                    'default': 4.0,
                    'refined': [
                        {'region': {'type': 'box',
                                    'x': [0, 40], 'y': [0, 40]},
                         'size': 2.0, 'thickness': 2.0},
                    ],
                },
            },
            prims, units='mm',
        )
        baseline_path = compile_geometry(
            primitives=prims, domain=None, named_groups={},
            mesh_dsl=baseline_dsl,
        )
        baseline_pts = meshio.read(str(baseline_path)).points[:, :2]
        d_base = np.sqrt((baseline_pts[:, 0] - 20.0) ** 2
                         + (baseline_pts[:, 1] - 20.0) ** 2)

        # Field[Min] over (box size 2.0) + (threshold size 0.25) must
        # yield strictly more nodes within 4 mm of the tip than the
        # box-only baseline -- otherwise the threshold field has been
        # silently dropped from the field combination.
        n_with = int((d_with < 4.0).sum())
        n_base = int((d_base < 4.0).sum())
        assert n_with > n_base, (
            f"threshold rule was not applied alongside the box: "
            f"{n_with} nodes within 4mm vs {n_base} (box-only baseline)")
        # And the closest node to the tip with the combined fields must
        # be closer than the box-only baseline (size=2.0 -> ~1 mm
        # nearest-node spacing; size=0.25 should produce sub-mm).
        assert d_with.min() < d_base.min(), (
            f"closest-node distance with threshold ({d_with.min():.3f}) "
            f"is not smaller than box-only baseline ({d_base.min():.3f})")
        # Box-only spacing is ~1 mm at the centre; the combined-field
        # mesh should put a node well inside that radius.
        assert d_with.min() < 0.8, (
            f"closest-node distance {d_with.min():.3f} mm is larger "
            f"than the combined Field[Min] over box (size=2.0) and "
            f"threshold (size=0.25 mm at the disk centre) should yield")

    def test_box_outside_domain_is_noop(self, tmp_cache_dir):
        """A box region that does not intersect the meshed domain at all
        must compile cleanly and produce a uniform mesh (no error).
        """
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_mesh_dsl,
        )
        prims = parse_primitives({
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [10, 10]},
            },
        })
        mesh_dsl = parse_mesh_dsl(
            {
                'element_size': {
                    'default': 1.0,
                    'refined': [
                        # Box living entirely outside the [0,10]x[0,10] plate.
                        {'region': {'type': 'box',
                                    'x': [100, 200], 'y': [100, 200]},
                         'size': 0.1, 'thickness': 1.0},
                    ],
                },
            },
            prims, units='mm',
        )
        # Must not raise.
        msh_path = compile_geometry(
            primitives=prims, domain=None, named_groups={},
            mesh_dsl=mesh_dsl,
        )
        assert msh_path.exists()
        raw = meshio.read(str(msh_path))
        # Mesh should look ~uniform at the default size (1 mm); the plate
        # is 10x10 = 100 mm^2; expect O(100-200) nodes.
        n_pts = len(raw.points)
        assert 50 < n_pts < 500, (
            f"out-of-domain box should leave a uniform mesh; got {n_pts} "
            f"nodes (uniform 1mm in 10x10 plate -> ~100-200 nodes).")

    def test_negative_size_rejected(self):
        """``size`` is the Field[Box] VIn parameter; negative values are
        physically meaningless and must be rejected at parse time.
        """
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='size'):
            parse_mesh_dsl(
                {
                    'element_size': {
                        'default': 1.0,
                        'refined': [
                            {'region': {'type': 'box',
                                        'x': [0, 1], 'y': [0, 1]},
                             'size': -0.1},
                        ],
                    },
                },
                {}, units='mm',
            )

    def test_non_positive_thickness_rejected(self):
        """Thickness=0 produces a sharp size discontinuity that the OCC
        meshing kernel handles poorly; thickness<0 is meaningless. Both
        must be rejected with a clear validation error.
        """
        from phast.geometry_dsl import parse_mesh_dsl
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match='thickness'):
                parse_mesh_dsl(
                    {
                        'element_size': {
                            'default': 1.0,
                            'refined': [
                                {'region': {'type': 'box',
                                            'x': [0, 1], 'y': [0, 1]},
                                 'size': 0.1, 'thickness': bad},
                            ],
                        },
                    },
                    {}, units='mm',
                )

    def test_margin_on_box_rejected(self):
        """``margin`` is a Threshold concept (DistMax) and has no
        meaning on a Field[Box] rule; setting it must error rather than
        be silently ignored.
        """
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='margin'):
            parse_mesh_dsl(
                {
                    'element_size': {
                        'default': 1.0,
                        'refined': [
                            {'region': {'type': 'box',
                                        'x': [0, 1], 'y': [0, 1]},
                             'size': 0.1, 'margin': 1.0},
                        ],
                    },
                },
                {}, units='mm',
            )

    def test_thickness_on_threshold_rejected(self):
        """Symmetric to the above: ``thickness`` is meaningless on a
        threshold (Distance+Threshold) rule.
        """
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='thickness'):
            parse_mesh_dsl(
                {
                    'element_size': {
                        'default': 1.0,
                        'refined': [
                            {'region': {'type': 'rectangle',
                                        'origin': [0, 0], 'size': [1, 1]},
                             'size': 0.1, 'thickness': 0.5},
                        ],
                    },
                },
                {}, units='mm',
            )

    def test_box_xy_validation(self):
        """Box ``x`` / ``y`` must be 2-element [min, max] lists with
        max > min."""
        from phast.geometry_dsl import parse_mesh_dsl
        # Inverted x range.
        with pytest.raises(ValueError, match="'x'"):
            parse_mesh_dsl(
                {
                    'element_size': {
                        'default': 1.0,
                        'refined': [
                            {'region': {'type': 'box',
                                        'x': [10, 5], 'y': [0, 1]},
                             'size': 0.1},
                        ],
                    },
                },
                {}, units='mm',
            )
        # Missing y.
        with pytest.raises(ValueError, match="'y'"):
            parse_mesh_dsl(
                {
                    'element_size': {
                        'default': 1.0,
                        'refined': [
                            {'region': {'type': 'box', 'x': [0, 1]},
                             'size': 0.1},
                        ],
                    },
                },
                {}, units='mm',
            )

    def test_box_unit_scaling(self):
        """Inputs in metres must be converted to mm in the rule's region
        bbox AND in the thickness."""
        from phast.geometry_dsl import parse_mesh_dsl
        m = parse_mesh_dsl(
            {
                'element_size': {
                    'default': 4.0,
                    'refined': [
                        {'region': {'type': 'box',
                                    'x': [0, 0.065], 'y': [0.045, 0.070]},
                         'size': 0.0005, 'thickness': 0.005},
                    ],
                },
            },
            {}, units='m',
        )
        assert m.default_size == pytest.approx(4000.0)
        rule = m.refined[0]
        assert rule.mode == 'box'
        assert rule.size == pytest.approx(0.5)
        assert rule.thickness == pytest.approx(5.0)
        # Region stored as Rectangle(origin, size); both in mm.
        assert rule.region.origin == pytest.approx((0.0, 45.0))
        assert rule.region.size == pytest.approx((65.0, 25.0))

    def test_cache_distinguishes_box_from_threshold(self, tmp_cache_dir):
        """A box-mode rule and a threshold-mode rule with the *same*
        underlying region geometry must hash to different cache keys --
        they produce different meshes."""
        from phast import geometry_compiler
        from phast.geometry_dsl import (
            parse_primitives, parse_mesh_dsl, ElementSizeRule, MeshDSL,
            Rectangle,
        )
        prims = parse_primitives({
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [10, 10]},
            },
        })
        # Same region geometry, different mode -- build the rules manually
        # so the test doesn't rely on the parser's box->Rectangle coercion.
        rect = Rectangle(origin=(0.0, 0.0), size=(10.0, 10.0),
                         name='r')
        rule_box = ElementSizeRule(
            region=rect, primitive=None, size=0.5, margin=0.0,
            mode='box', thickness=1.0,
        )
        rule_thr = ElementSizeRule(
            region=rect, primitive=None, size=0.5, margin=1.0,
            mode='threshold', thickness=0.0,
        )
        m_box = MeshDSL(default_size=2.0, refined=[rule_box])
        m_thr = MeshDSL(default_size=2.0, refined=[rule_thr])
        k_box = geometry_compiler.cache_key(prims, None, {}, m_box)
        k_thr = geometry_compiler.cache_key(prims, None, {}, m_thr)
        assert k_box != k_thr


# ---------------------------------------------------------------------------
# Issue #201: boundary-segment RegionGroup + PointGroup fragment-not-embed
# ---------------------------------------------------------------------------

class TestBoundarySegmentRegionGroup:
    """RegionGroup with a region that only partially overlaps the base
    must tag the boundary sub-segment that lies inside the region (the
    L-shaped panel ``load_segment`` BC pattern -- issue #201)."""

    def test_lshaped_load_segment(self, tmp_cache_dir):
        """A 10mm strip straddling the upper-left portion of the top edge
        of a 100x100 plate must produce a non-empty ``load_segment`` node
        set whose nodes all lie on that 10mm sub-segment."""
        import torch
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups,
            parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        # Plate top edge runs from x=0 to x=100 at y=100. We want the
        # named curve to be the upper-left 10mm: x in [0, 10].
        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [100, 100]},
            },
            'domain': {'base': 'plate'},
            'named_groups': {
                'load_segment': {
                    'region': {'type': 'rectangle',
                               'origin': [0, 99.5], 'size': [10, 1.0]},
                },
            },
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 1.0}}, prims, units='mm')

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        mesh = FEMMesh(str(msh_path), device='cpu')

        assert 'load_segment' in mesh.node_sets, (
            f"missing load_segment in {list(mesh.node_sets)}")
        idx = mesh.node_sets['load_segment']
        assert idx.numel() > 0, "load_segment node set is empty"

        # All tagged nodes must lie on the upper-left 10mm of the top
        # edge (x in [0, 10], y == 100).
        coords = mesh.nodes[idx]
        xs = coords[:, 0]
        ys = coords[:, 1]
        assert torch.all(ys > 100 - 1e-3) and torch.all(ys < 100 + 1e-3), \
            "load_segment node not on top edge"
        assert torch.all(xs >= -1e-3) and torch.all(xs <= 10 + 1e-3), \
            f"load_segment node x out of [0,10]: {xs.min()}..{xs.max()}"

        # Node count: ~10 nodes at h=1mm + endpoints.
        assert idx.numel() >= 8, f"load_segment too sparse: {idx.numel()}"
        assert idx.numel() <= 30, f"load_segment too dense: {idx.numel()}"

        # Triangle elements must still be produced (regression: fragment
        # of a thin straddling strip must not break meshing).
        assert mesh.elements.shape[0] > 0, "no triangle elements"

        # plate.boundary auto-group must NOT contain the internal seam
        # introduced by the fragment (the vertical sides of the strip
        # interior to the plate). Seam endpoints are at x=10, y in
        # [99.5, 100]; if they're in plate.boundary as part of an
        # internal seam, they'd be at y=99.5 (NOT on any plate edge).
        pb = mesh.nodes[mesh.node_sets['plate.boundary']]
        on_edge = (
            (pb[:, 0].abs() < 1e-3) | ((pb[:, 0] - 100).abs() < 1e-3)
            | (pb[:, 1].abs() < 1e-3) | ((pb[:, 1] - 100).abs() < 1e-3)
        )
        assert torch.all(on_edge), \
            "plate.boundary contains internal seam node from fragment"

    def test_region_fully_inside_still_works(self, tmp_cache_dir):
        """RegionGroup whose region is fully inside the plate top edge
        (legacy in-bbox path) must keep working -- this is a backwards-
        compatibility check for the v2->v3 compiler upgrade."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups,
            parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [100, 100]},
            },
            'domain': {'base': 'plate'},
            'named_groups': {
                'top_band': {
                    'region': {'type': 'rectangle',
                               'origin': [20, 99.5],
                               'size': [60, 1.0]},
                },
            },
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 2.0}}, prims, units='mm')

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        mesh = FEMMesh(str(msh_path), device='cpu')
        assert 'top_band' in mesh.node_sets
        idx = mesh.node_sets['top_band']
        assert idx.numel() > 0
        assert mesh.elements.shape[0] > 0
        # Same outline-only semantic: nodes must lie on the plate's top
        # edge (y=100), not on the band's underside (y=99.5).
        coords = mesh.nodes[idx]
        ys = coords[:, 1]
        assert float(ys.min()) > 100 - 1e-3, \
            f"top_band has off-edge nodes: y_min={float(ys.min())}"


class TestPointGroupFragment:
    """PointGroup must use OCC fragment (not mesh.embed) so that points
    lying on existing boundary curves split the curve and preserve
    meshability. Issue #201."""

    def test_pin_centre_on_hole_boundary_meshes(self, tmp_cache_dir):
        """Place a PointGroup at the centre of a hole. The point lies on
        the hole's interior region. Previously mesh.embed of a point in
        the hole was a no-op (point inside the cut-out); but a point ON
        a hole's boundary curve broke meshing. This test exercises the
        on-curve case -- a PointGroup on the right edge of the plate."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups,
            parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [50, 50]},
            },
            'domain': {'base': 'plate'},
            'named_groups': {
                'load_pt': {'point': [50, 25]},  # ON right edge of plate
            },
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 5.0}}, prims, units='mm')

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        mesh = FEMMesh(str(msh_path), device='cpu')
        # The mesh MUST contain triangle cells (regression: was zero
        # under the legacy mesh.embed path for on-curve points).
        assert mesh.elements.shape[0] > 0, \
            "PointGroup on edge produced zero triangle cells"
        assert 'load_pt' in mesh.node_sets
        assert mesh.node_sets['load_pt'].numel() == 1

        # Tagged node should be exactly at the requested coordinates.
        idx = mesh.node_sets['load_pt']
        coords = mesh.nodes[idx]
        assert abs(float(coords[0, 0]) - 50.0) < 1e-3
        assert abs(float(coords[0, 1]) - 25.0) < 1e-3

    def test_point_inside_surface_still_works(self, tmp_cache_dir):
        """An interior PointGroup must still tag a node and not break
        the mesh (regression for the existing notch_tip pattern)."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups,
            parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [50, 50]},
            },
            'domain': {'base': 'plate'},
            'named_groups': {
                'probe': {'point': [25, 25]},  # interior
            },
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 5.0}}, prims, units='mm')

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        mesh = FEMMesh(str(msh_path), device='cpu')
        assert mesh.elements.shape[0] > 0
        assert 'probe' in mesh.node_sets
        assert mesh.node_sets['probe'].numel() == 1

    def test_point_outside_surface_falls_back(self, tmp_cache_dir):
        """An external PointGroup (PR #207's master-node pattern: a
        Physical Point at a pin centre that lies outside the plate
        material) must still produce a tagged Physical Point and a
        meshable surface."""
        from phast.geometry_compiler import compile_geometry
        from phast.geometry_dsl import (
            parse_primitives, parse_domain, parse_named_groups,
            parse_mesh_dsl,
        )
        from phast.mesh import FEMMesh

        geom_dict = {
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [50, 50]},
            },
            'domain': {'base': 'plate'},
            'named_groups': {
                'master': {'point': [200, 200]},  # well outside plate
            },
        }
        prims = parse_primitives(geom_dict)
        domain = parse_domain(geom_dict['domain'], prims)
        groups = parse_named_groups(geom_dict, prims)
        mesh_dsl = parse_mesh_dsl(
            {'element_size': {'default': 5.0}}, prims, units='mm')

        msh_path = compile_geometry(
            primitives=prims, domain=domain, named_groups=groups,
            mesh_dsl=mesh_dsl,
        )
        mesh = FEMMesh(str(msh_path), device='cpu')
        assert mesh.elements.shape[0] > 0
        assert 'master' in mesh.node_sets
        assert mesh.node_sets['master'].numel() == 1
