"""Tests for the geometry primitive vocabulary (issue #142, Phase 2.1).

Covers:

* Parsing of each primitive type (rectangle, circle, polygon, point,
  line_segment) -> expected internal dataclass representation.
* The auto-exposed ``.boundary`` / ``.interior`` / ``.centre`` selector API.
* Unit conversion (``mm`` no-op, ``m`` -> mm).
* Validation errors for missing required fields and malformed inputs.
* Integration with :mod:`config`: backward-compat for the legacy
  ``type``-based path, the mutually-exclusive guard, and the
  ``NotImplementedError`` "compiler lands in #146" hand-off.
"""

import os
import tempfile
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Direct primitive parsing
# ---------------------------------------------------------------------------

class TestParsePrimitives:
    def test_rectangle(self):
        from phast.geometry_dsl import parse_primitives, Rectangle
        prims = parse_primitives({
            'units': 'mm',
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [65, 120]},
            },
        })
        assert set(prims) == {'plate'}
        plate = prims['plate']
        assert isinstance(plate, Rectangle)
        assert plate.origin == (0.0, 0.0)
        assert plate.size == (65.0, 120.0)
        assert plate.name == 'plate'
        assert plate.kind == 'rectangle'

    def test_circle(self):
        from phast.geometry_dsl import parse_primitives, Circle
        prims = parse_primitives({
            'primitives': {
                'big_hole': {'type': 'circle',
                             'center': [36.5, 51], 'radius': 10},
            },
        })
        c = prims['big_hole']
        assert isinstance(c, Circle)
        assert c.center == (36.5, 51.0)
        assert c.radius == 10.0
        assert c.kind == 'circle'

    def test_polygon(self):
        from phast.geometry_dsl import parse_primitives, Polygon
        prims = parse_primitives({
            'primitives': {
                'poly_a': {'type': 'polygon',
                           'vertices': [[0, 0], [10, 0], [5, 5]]},
            },
        })
        p = prims['poly_a']
        assert isinstance(p, Polygon)
        assert p.vertices == [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)]
        assert p.kind == 'polygon'

    def test_point(self):
        from phast.geometry_dsl import parse_primitives, Point
        prims = parse_primitives({
            'primitives': {
                'notch_pt': {'type': 'point', 'coords': [10, 65]},
            },
        })
        pt = prims['notch_pt']
        assert isinstance(pt, Point)
        assert pt.coords == (10.0, 65.0)
        assert pt.kind == 'point'

    def test_line_segment(self):
        from phast.geometry_dsl import parse_primitives, LineSegment
        prims = parse_primitives({
            'primitives': {
                'seg_top': {'type': 'line_segment',
                            'from': [0, 100], 'to': [65, 100]},
            },
        })
        seg = prims['seg_top']
        assert isinstance(seg, LineSegment)
        assert seg.start == (0.0, 100.0)
        assert seg.end == (65.0, 100.0)
        assert seg.kind == 'line_segment'

    def test_full_canonical_yaml_block(self):
        """End-to-end: the exact YAML block in the issue description parses."""
        from phast.geometry_dsl import parse_primitives
        prims = parse_primitives({
            'units': 'mm',
            'primitives': {
                'plate':    {'type': 'rectangle', 'origin': [0, 0],     'size': [65, 120]},
                'big_hole': {'type': 'circle',    'center': [36.5, 51], 'radius': 10},
                'poly_a':   {'type': 'polygon',
                             'vertices': [[0, 0], [10, 0], [5, 5]]},
                'notch_pt': {'type': 'point',     'coords': [10, 65]},
                'seg_top':  {'type': 'line_segment',
                             'from': [0, 100], 'to': [65, 100]},
            },
        })
        assert set(prims) == {'plate', 'big_hole', 'poly_a',
                              'notch_pt', 'seg_top'}

    def test_empty_or_none_returns_empty(self):
        from phast.geometry_dsl import parse_primitives
        assert parse_primitives(None) == {}
        assert parse_primitives({}) == {}
        assert parse_primitives({'units': 'mm'}) == {}


# ---------------------------------------------------------------------------
# Selector API
# ---------------------------------------------------------------------------

class TestSelectors:
    @pytest.mark.parametrize('spec,name', [
        ({'type': 'rectangle', 'origin': [0, 0], 'size': [1, 1]}, 'r'),
        ({'type': 'circle', 'center': [0, 0], 'radius': 1}, 'c'),
        ({'type': 'polygon', 'vertices': [[0, 0], [1, 0], [0, 1]]}, 'p'),
        ({'type': 'point', 'coords': [0, 0]}, 'pt'),
        ({'type': 'line_segment', 'from': [0, 0], 'to': [1, 0]}, 's'),
    ])
    def test_each_primitive_exposes_the_three_selectors(self, spec, name):
        from phast.geometry_dsl import parse_primitives, Selector
        prims = parse_primitives({'primitives': {name: spec}})
        prim = prims[name]
        for kind in ('boundary', 'interior', 'centre'):
            sel = getattr(prim, kind)
            assert isinstance(sel, Selector)
            assert sel.primitive == name
            assert sel.kind == kind


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

class TestUnits:
    def test_meters_scale_to_millimeters(self):
        """``units: m`` -> all length quantities stored as mm internally."""
        from phast.geometry_dsl import parse_primitives
        prims = parse_primitives({
            'units': 'm',
            'primitives': {
                'plate':    {'type': 'rectangle',
                             'origin': [0, 0], 'size': [0.065, 0.120]},
                'big_hole': {'type': 'circle',
                             'center': [0.0365, 0.051], 'radius': 0.01},
                'poly_a':   {'type': 'polygon',
                             'vertices': [[0, 0], [0.01, 0], [0.005, 0.005]]},
                'notch_pt': {'type': 'point', 'coords': [0.01, 0.065]},
                'seg_top':  {'type': 'line_segment',
                             'from': [0, 0.1], 'to': [0.065, 0.1]},
            },
        })
        # 65 mm = 0.065 m
        assert prims['plate'].size == pytest.approx((65.0, 120.0))
        assert prims['big_hole'].center == pytest.approx((36.5, 51.0))
        assert prims['big_hole'].radius == pytest.approx(10.0)
        # Polygon vertices each scaled
        assert prims['poly_a'].vertices == [
            pytest.approx((0.0, 0.0)),
            pytest.approx((10.0, 0.0)),
            pytest.approx((5.0, 5.0)),
        ]
        assert prims['notch_pt'].coords == pytest.approx((10.0, 65.0))
        assert prims['seg_top'].start == pytest.approx((0.0, 100.0))
        assert prims['seg_top'].end == pytest.approx((65.0, 100.0))

    def test_default_units_is_mm(self):
        """No units key -> mm (no scaling)."""
        from phast.geometry_dsl import parse_primitives
        prims = parse_primitives({
            'primitives': {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [65, 120]},
            },
        })
        assert prims['plate'].size == (65.0, 120.0)

    def test_unsupported_unit_raises(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match='Unsupported geometry units'):
            parse_primitives({
                'units': 'inch',
                'primitives': {
                    'p': {'type': 'point', 'coords': [0, 0]},
                },
            })


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_type(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'type'"):
            parse_primitives({'primitives': {'p': {'origin': [0, 0]}}})

    def test_unknown_type(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="unknown type 'ellipse'"):
            parse_primitives({'primitives':
                              {'p': {'type': 'ellipse'}}})

    def test_rectangle_missing_origin(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'origin'"):
            parse_primitives({'primitives':
                              {'r': {'type': 'rectangle', 'size': [1, 1]}}})

    def test_rectangle_missing_size(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'size'"):
            parse_primitives({'primitives':
                              {'r': {'type': 'rectangle', 'origin': [0, 0]}}})

    def test_rectangle_negative_size(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match='size must be strictly positive'):
            parse_primitives({'primitives': {
                'r': {'type': 'rectangle', 'origin': [0, 0], 'size': [-1, 1]}
            }})

    def test_circle_missing_center(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'center'"):
            parse_primitives({'primitives':
                              {'c': {'type': 'circle', 'radius': 1}}})

    def test_circle_missing_radius(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'radius'"):
            parse_primitives({'primitives':
                              {'c': {'type': 'circle', 'center': [0, 0]}}})

    def test_circle_zero_radius(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match='radius must be strictly positive'):
            parse_primitives({'primitives': {
                'c': {'type': 'circle', 'center': [0, 0], 'radius': 0}
            }})

    def test_polygon_missing_vertices(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'vertices'"):
            parse_primitives({'primitives': {'p': {'type': 'polygon'}}})

    def test_polygon_too_few_vertices(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match='at least 3'):
            parse_primitives({'primitives': {
                'p': {'type': 'polygon', 'vertices': [[0, 0], [1, 1]]}
            }})

    def test_point_missing_coords(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'coords'"):
            parse_primitives({'primitives': {'pt': {'type': 'point'}}})

    def test_line_segment_missing_from(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'from'"):
            parse_primitives({'primitives': {
                's': {'type': 'line_segment', 'to': [1, 0]}
            }})

    def test_line_segment_missing_to(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match="missing required field 'to'"):
            parse_primitives({'primitives': {
                's': {'type': 'line_segment', 'from': [0, 0]}
            }})

    def test_line_segment_zero_length(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match='zero length'):
            parse_primitives({'primitives': {
                's': {'type': 'line_segment',
                      'from': [1, 1], 'to': [1, 1]}
            }})

    def test_xy_must_be_two_element_list(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match='2-element'):
            parse_primitives({'primitives': {
                'r': {'type': 'rectangle', 'origin': [0], 'size': [1, 1]}
            }})

    def test_xy_must_be_numeric(self):
        from phast.geometry_dsl import parse_primitives
        with pytest.raises(ValueError, match='non-numeric'):
            parse_primitives({'primitives': {
                'pt': {'type': 'point', 'coords': ['a', 0]}
            }})


# ---------------------------------------------------------------------------
# Helper: point_in_polygon
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    def test_inside_unit_square(self):
        from phast.geometry_dsl import point_in_polygon
        sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert point_in_polygon((0.5, 0.5), sq) is True
        assert point_in_polygon((-0.1, 0.5), sq) is False
        assert point_in_polygon((1.5, 0.5), sq) is False

    def test_inside_triangle(self):
        from phast.geometry_dsl import point_in_polygon
        tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)]
        assert point_in_polygon((5.0, 1.0), tri) is True
        assert point_in_polygon((5.0, 6.0), tri) is False


# ---------------------------------------------------------------------------
# Integration with config / YAML loader
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def _write_yaml(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        with open(path, 'w') as f:
            f.write(textwrap.dedent(body))
        return path

    def test_legacy_type_path_still_loads(self):
        """Existing configs/* with geometry.type: <name> still parse fine."""
        from phast.config import load_config
        path = self._write_yaml("""
            geometry:
              type: miehe_tension
              parameters: {}
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.type == 'miehe_tension'
            assert cfg.geometry.primitives is None
            assert getattr(cfg.geometry, '_type_explicit', False) is True
        finally:
            os.unlink(path)

    def test_primitives_parse_via_loader(self):
        from phast.config import load_config
        path = self._write_yaml("""
            geometry:
              units: mm
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.primitives == {
                'plate': {'type': 'rectangle',
                          'origin': [0, 0], 'size': [65, 120]},
            }
            assert cfg.geometry.units == 'mm'
            assert getattr(cfg.geometry, '_type_explicit', False) is False
        finally:
            os.unlink(path)

    def test_primitives_resolve_raises_not_implemented(self):
        """resolve_config without a domain block + multiple primitives now
        raises a clean error from the geometry compiler (issue #146).

        Historically this test asserted the placeholder NotImplementedError.
        With #146 in place the resolver builds a real mesh whenever a
        valid domain is supplied; the multi-primitive / no-domain case
        below is the disambiguation guard.
        """
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              units: mm
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
                hole:  {type: circle,    center: [32.5, 60], radius: 5}
        """)
        try:
            cfg = load_config(path)
            with pytest.raises(ValueError, match='domain'):
                resolve_config(cfg)
            # The parse side-effect should have populated _parsed_primitives.
            parsed = getattr(cfg.geometry, '_parsed_primitives', None)
            assert parsed is not None
            assert set(parsed) == {'plate', 'hole'}
        finally:
            os.unlink(path)

    def test_type_and_primitives_are_mutually_exclusive(self):
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              type: miehe_tension
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
        """)
        try:
            cfg = load_config(path)
            with pytest.raises(ValueError, match='mutually exclusive'):
                resolve_config(cfg)
        finally:
            os.unlink(path)

    def test_invalid_primitive_surfaces_at_resolve(self):
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              primitives:
                bad: {type: rectangle, origin: [0, 0]}
        """)
        try:
            cfg = load_config(path)
            with pytest.raises(ValueError, match="missing required field 'size'"):
                resolve_config(cfg)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Domain (boolean ops) — issue #143
# ---------------------------------------------------------------------------

def _holed_plate_primitives():
    """Helper: the canonical YAML-block primitives from issue #143."""
    from phast.geometry_dsl import parse_primitives
    return parse_primitives({
        'primitives': {
            'plate':    {'type': 'rectangle', 'origin': [0, 0],
                         'size': [65, 120]},
            'notch':    {'type': 'rectangle', 'origin': [0, 64.75],
                         'size': [10, 0.5]},
            'big_hole': {'type': 'circle', 'center': [36.5, 51], 'radius': 10},
            'pin_top':  {'type': 'circle', 'center': [20, 100], 'radius': 5},
            'pin_bot':  {'type': 'circle', 'center': [20, 20], 'radius': 5},
        },
    })


def _basic_primitives():
    """Two primitives used across the named-group tests."""
    from phast.geometry_dsl import parse_primitives
    return parse_primitives({
        'units': 'mm',
        'primitives': {
            'plate':   {'type': 'rectangle', 'origin': [0, 0], 'size': [65, 120]},
            'pin_top': {'type': 'circle', 'center': [32.5, 110], 'radius': 5},
        },
    })


def _ps():
    """Convenience: primitive registry used in mesh refinement tests."""
    from phast.geometry_dsl import parse_primitives
    return parse_primitives({
        'primitives': {
            'plate': {'type': 'rectangle', 'origin': [0, 0], 'size': [65, 120]},
            'notch': {'type': 'rectangle', 'origin': [0, 64.75],
                      'size': [10, 0.5]},
            'tip':   {'type': 'point', 'coords': [10, 65]},
        },
    })


class TestParseDomain:
    def test_subtract_only_full_holed_plate(self):
        """The canonical issue #143 example parses end-to-end."""
        from phast.geometry_dsl import parse_domain, Domain
        prims = _holed_plate_primitives()
        dom = parse_domain({
            'base': 'plate',
            'subtract': ['notch', 'big_hole', 'pin_top', 'pin_bot'],
            'add': [],
            'intersect': [],
        }, prims)
        assert isinstance(dom, Domain)
        assert dom.base == 'plate'
        assert dom.subtract == ['notch', 'big_hole', 'pin_top', 'pin_bot']
        assert dom.add == []
        assert dom.intersect == []

    def test_add_only(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        dom = parse_domain({
            'base': 'plate',
            'add': ['big_hole', 'pin_top'],
        }, prims)
        assert dom.base == 'plate'
        assert dom.add == ['big_hole', 'pin_top']
        assert dom.subtract == []
        assert dom.intersect == []

    def test_intersect_only(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        dom = parse_domain({
            'base': 'plate',
            'intersect': ['big_hole'],
        }, prims)
        assert dom.intersect == ['big_hole']
        assert dom.add == []
        assert dom.subtract == []

    def test_base_only_defaults_empty_lists(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        dom = parse_domain({'base': 'plate'}, prims)
        assert dom.base == 'plate'
        assert dom.add == [] and dom.subtract == [] and dom.intersect == []

    def test_referenced_primitives_walks_full_set(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        dom = parse_domain({
            'base': 'plate',
            'add': ['big_hole'],
            'subtract': ['notch'],
            'intersect': ['pin_top'],
        }, prims)
        assert dom.referenced_primitives() == [
            'plate', 'big_hole', 'notch', 'pin_top',
        ]

    # --- Validation errors -------------------------------------------------

    def test_missing_base_raises(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError, match="missing required field 'base'"):
            parse_domain({'subtract': ['notch']}, prims)

    def test_none_domain_raises(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_domain(None, prims)

    def test_non_dict_domain_raises(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_domain(['plate'], prims)

    def test_empty_base_string_raises(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError, match='non-empty primitive-name'):
            parse_domain({'base': ''}, prims)

    def test_unknown_base_raises_with_did_you_mean(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError) as exc:
            parse_domain({'base': 'plat'}, prims)  # typo
        msg = str(exc.value)
        assert 'unknown primitive' in msg
        assert "'plat'" in msg
        assert "'plate'" in msg  # did-you-mean suggestion fired

    def test_unknown_subtract_raises_with_did_you_mean(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError) as exc:
            parse_domain({
                'base': 'plate',
                'subtract': ['big_holee'],  # typo
            }, prims)
        msg = str(exc.value)
        assert 'subtract' in msg
        assert 'big_holee' in msg
        assert 'big_hole' in msg

    def test_unknown_add_raises_with_did_you_mean(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError) as exc:
            parse_domain({
                'base': 'plate',
                'add': ['pin_topp'],
            }, prims)
        msg = str(exc.value)
        assert 'add' in msg
        assert 'pin_top' in msg

    def test_unknown_intersect_raises_with_did_you_mean(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError) as exc:
            parse_domain({
                'base': 'plate',
                'intersect': ['notchx'],
            }, prims)
        msg = str(exc.value)
        assert 'intersect' in msg
        assert 'notch' in msg

    def test_unknown_no_close_match_no_hint(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError) as exc:
            parse_domain({'base': 'plate',
                          'subtract': ['zzzzzzzzzzzz']}, prims)
        msg = str(exc.value)
        assert 'unknown primitive' in msg
        # No close match -> no did-you-mean phrase.
        assert 'Did you mean' not in msg

    def test_subtract_must_be_list(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError, match='must be a list'):
            parse_domain({'base': 'plate', 'subtract': 'notch'}, prims)

    def test_subtract_entries_must_be_strings(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError, match='primitive-name string'):
            parse_domain({'base': 'plate', 'subtract': [123]}, prims)

    def test_unknown_top_level_field_rejected(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        with pytest.raises(ValueError, match='unsupported field'):
            parse_domain({'base': 'plate', 'minus': ['notch']}, prims)

    def test_empty_lists_are_fine(self):
        from phast.geometry_dsl import parse_domain
        prims = _holed_plate_primitives()
        dom = parse_domain({
            'base': 'plate', 'add': [], 'subtract': [], 'intersect': [],
        }, prims)
        assert dom.add == [] and dom.subtract == [] and dom.intersect == []


# ---------------------------------------------------------------------------
# Named groups (issue #144)
# ---------------------------------------------------------------------------

class TestParseNamedGroups:
    def test_canonical_yaml_block(self):
        from phast.geometry_dsl import (
            parse_named_groups, SelectorAliasGroup, PointGroup, RegionGroup,
            Rectangle,
        )
        prims = _basic_primitives()
        groups = parse_named_groups({
            'units': 'mm',
            'named_groups': {
                'notch_tip':        {'point': [10, 65]},
                'crack_path_band':  {'region': {'type': 'rectangle',
                                                'origin': [0, 64.5],
                                                'size': [55, 1]}},
                'upper_pin_centre': {'primitive': 'pin_top', 'kind': 'centre'},
            },
        }, prims)
        assert set(groups) == {'notch_tip', 'crack_path_band',
                               'upper_pin_centre'}

        nt = groups['notch_tip']
        assert isinstance(nt, PointGroup)
        assert nt.coords == (10.0, 65.0)
        assert nt.kind == 'point'
        assert nt.name == 'notch_tip'

        band = groups['crack_path_band']
        assert isinstance(band, RegionGroup)
        assert isinstance(band.region, Rectangle)
        assert band.region.origin == (0.0, 64.5)
        assert band.region.size == (55.0, 1.0)
        assert band.kind == 'region'

        upc = groups['upper_pin_centre']
        assert isinstance(upc, SelectorAliasGroup)
        assert upc.primitive == 'pin_top'
        assert upc.selector_kind == 'centre'
        assert upc.selector.primitive == 'pin_top'
        assert upc.selector.kind == 'centre'
        assert upc.kind == 'selector_alias'

    def test_selector_alias_default_kind_is_boundary(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        groups = parse_named_groups({
            'named_groups': {'pt_bnd': {'primitive': 'pin_top'}},
        }, prims)
        assert groups['pt_bnd'].selector_kind == 'boundary'

    def test_point_coords_scaled_with_units(self):
        from phast.geometry_dsl import parse_named_groups
        from phast.geometry_dsl import parse_primitives
        prims = parse_primitives({
            'units': 'm',
            'primitives': {'plate': {'type': 'rectangle',
                                     'origin': [0, 0],
                                     'size': [0.065, 0.120]}},
        })
        groups = parse_named_groups({
            'units': 'm',
            'named_groups': {'tip': {'point': [0.01, 0.065]}},
        }, prims)
        assert groups['tip'].coords == pytest.approx((10.0, 65.0))

    def test_region_inline_primitive_scaled_with_units(self):
        from phast.geometry_dsl import (
            parse_named_groups, parse_primitives,
        )
        prims = parse_primitives({
            'units': 'm',
            'primitives': {'plate': {'type': 'rectangle',
                                     'origin': [0, 0],
                                     'size': [0.065, 0.120]}},
        })
        groups = parse_named_groups({
            'units': 'm',
            'named_groups': {
                'band': {'region': {'type': 'rectangle',
                                    'origin': [0, 0.0645],
                                    'size': [0.055, 0.001]}},
            },
        }, prims)
        band = groups['band']
        assert band.region.origin == pytest.approx((0.0, 64.5))
        assert band.region.size == pytest.approx((55.0, 1.0))

    def test_empty_or_missing_named_groups_returns_empty(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        assert parse_named_groups(None, prims) == {}
        assert parse_named_groups({}, prims) == {}
        assert parse_named_groups({'units': 'mm'}, prims) == {}


class TestNamedGroupValidation:
    def test_multiple_forms_rejected(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError, match='multiple forms'):
            parse_named_groups({
                'named_groups': {
                    'bad': {'point': [0, 0],
                            'region': {'type': 'rectangle',
                                       'origin': [0, 0], 'size': [1, 1]}},
                },
            }, prims)

    def test_no_form_rejected(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError, match='exactly one of'):
            parse_named_groups({
                'named_groups': {'bad': {'foo': 'bar'}},
            }, prims)

    def test_alias_to_unknown_primitive_with_did_you_mean(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError) as exc:
            parse_named_groups({
                'named_groups': {
                    'g': {'primitive': 'pin_tap', 'kind': 'boundary'},
                },
            }, prims)
        msg = str(exc.value)
        assert 'unknown primitive' in msg
        assert 'pin_top' in msg  # did-you-mean suggestion

    def test_alias_invalid_selector_kind(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError, match="must be one of"):
            parse_named_groups({
                'named_groups': {
                    'g': {'primitive': 'pin_top', 'kind': 'edge'},
                },
            }, prims)

    def test_region_must_be_mapping(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError, match="must be an inline primitive"):
            parse_named_groups({
                'named_groups': {'g': {'region': 'not_a_dict'}},
            }, prims)

    def test_region_inline_primitive_validated(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError,
                           match="missing required field 'size'"):
            parse_named_groups({
                'named_groups': {
                    'g': {'region': {'type': 'rectangle', 'origin': [0, 0]}},
                },
            }, prims)

    def test_name_collision_with_auto_exposed_selector(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError,
                           match='collides with the auto-exposed selector'):
            parse_named_groups({
                'named_groups': {
                    'pin_top.boundary': {'point': [0, 0]},
                },
            }, prims)

    def test_name_collision_with_primitive(self):
        from phast.geometry_dsl import parse_named_groups
        prims = _basic_primitives()
        with pytest.raises(ValueError, match='collides with the primitive'):
            parse_named_groups({
                'named_groups': {'pin_top': {'point': [0, 0]}},
            }, prims)


class TestValidateNodeSetName:
    """Auto-exposed defaults: ``<P>.<kind>`` is implicit and need not be
    declared. This is the headline API decision for issue #144."""

    def test_explicit_named_group_resolves(self):
        from phast.geometry_dsl import (
            parse_named_groups, validate_node_set_name,
        )
        prims = _basic_primitives()
        groups = parse_named_groups({
            'named_groups': {'notch_tip': {'point': [10, 65]}},
        }, prims)
        # Does not raise.
        validate_node_set_name('notch_tip', prims, groups)

    def test_auto_exposed_selector_resolves_without_declaration(self):
        from phast.geometry_dsl import validate_node_set_name
        prims = _basic_primitives()
        # No explicit named_groups; auto-exposed defaults must work.
        for k in ('boundary', 'interior', 'centre'):
            validate_node_set_name(f'pin_top.{k}', prims, {})
            validate_node_set_name(f'plate.{k}', prims, {})

    def test_auto_exposed_with_unknown_primitive_errors_with_suggestion(self):
        from phast.geometry_dsl import validate_node_set_name
        prims = _basic_primitives()
        with pytest.raises(ValueError) as exc:
            validate_node_set_name('pin_tap.boundary', prims, {})
        msg = str(exc.value)
        assert "primitive 'pin_tap' is not defined" in msg
        assert 'pin_top' in msg

    def test_auto_exposed_with_invalid_kind_errors(self):
        from phast.geometry_dsl import validate_node_set_name
        prims = _basic_primitives()
        with pytest.raises(ValueError, match='not a valid selector kind'):
            validate_node_set_name('pin_top.edge', prims, {})

    def test_unknown_bare_name_errors_with_did_you_mean(self):
        from phast.geometry_dsl import (
            parse_named_groups, validate_node_set_name,
        )
        prims = _basic_primitives()
        groups = parse_named_groups({
            'named_groups': {'notch_tip': {'point': [10, 65]}},
        }, prims)
        with pytest.raises(ValueError) as exc:
            validate_node_set_name('notch_tipp', prims, groups)
        msg = str(exc.value)
        assert 'not a declared named group' in msg
        assert 'notch_tip' in msg  # did-you-mean

    def test_resolve_node_set_name_stub_raises_pointing_at_146(self):
        from phast.geometry_dsl import resolve_node_set_name
        prims = _basic_primitives()
        # Valid name -> NotImplementedError naming #146.
        with pytest.raises(NotImplementedError, match='#146'):
            resolve_node_set_name('pin_top.boundary', prims, {}, mesh=None)
        # Invalid name -> ValueError before NotImplementedError.
        with pytest.raises(ValueError):
            resolve_node_set_name('does_not_exist', prims, {}, mesh=None)


class TestKnownGroupNames:
    def test_includes_explicit_and_auto_exposed(self):
        from phast.geometry_dsl import (
            parse_named_groups, known_group_names,
        )
        prims = _basic_primitives()
        groups = parse_named_groups({
            'named_groups': {'notch_tip': {'point': [10, 65]}},
        }, prims)
        names = set(known_group_names(prims, groups))
        # Explicit group present.
        assert 'notch_tip' in names
        # All auto-exposed selectors present without being declared.
        for p in ('plate', 'pin_top'):
            for k in ('boundary', 'interior', 'centre'):
                assert f'{p}.{k}' in names


# ---------------------------------------------------------------------------
# Mesh refinement DSL (Phase 2.4, issue #145)
# ---------------------------------------------------------------------------

class TestMeshDSLParsing:
    def test_default_only(self):
        from phast.geometry_dsl import parse_mesh_dsl, MeshDSL
        m = parse_mesh_dsl({'element_size': {'default': 2.0}}, _ps())
        assert isinstance(m, MeshDSL)
        assert m.default_size == 2.0
        assert m.refined == []

    def test_region_rectangle(self):
        from phast.geometry_dsl import parse_mesh_dsl, Rectangle
        m = parse_mesh_dsl({
            'element_size': {
                'default': 2.0,
                'refined': [{
                    'region': {'type': 'rectangle',
                               'origin': [0, 64.5], 'size': [55, 1]},
                    'size': 0.1,
                }],
            },
        }, _ps())
        assert len(m.refined) == 1
        rule = m.refined[0]
        assert isinstance(rule.region, Rectangle)
        assert rule.region.origin == (0.0, 64.5)
        assert rule.region.size == (55.0, 1.0)
        assert rule.primitive is None
        assert rule.size == 0.1
        assert rule.margin == 0.0

    def test_region_circle(self):
        from phast.geometry_dsl import parse_mesh_dsl, Circle
        m = parse_mesh_dsl({
            'element_size': {
                'default': 2.0,
                'refined': [{
                    'region': {'type': 'circle',
                               'center': [10, 65], 'radius': 30},
                    'size': 0.25,
                }],
            },
        }, _ps())
        assert isinstance(m.refined[0].region, Circle)
        assert m.refined[0].region.radius == 30.0

    def test_region_ball_alias(self):
        """``ball`` is accepted in the refinement DSL as an alias for circle."""
        from phast.geometry_dsl import parse_mesh_dsl, Circle
        m = parse_mesh_dsl({
            'element_size': {
                'default': 2.0,
                'refined': [{
                    'region': {'type': 'ball',
                               'center': [10, 65], 'radius': 30},
                    'size': 0.25,
                }],
            },
        }, _ps())
        assert isinstance(m.refined[0].region, Circle)
        assert m.refined[0].region.center == (10.0, 65.0)
        assert m.refined[0].region.radius == 30.0

    def test_region_polygon(self):
        from phast.geometry_dsl import parse_mesh_dsl, Polygon
        m = parse_mesh_dsl({
            'element_size': {
                'default': 2.0,
                'refined': [{
                    'region': {'type': 'polygon',
                               'vertices': [[0, 0], [10, 0], [5, 5]]},
                    'size': 0.5,
                }],
            },
        }, _ps())
        assert isinstance(m.refined[0].region, Polygon)
        assert m.refined[0].region.vertices == [
            (0.0, 0.0), (10.0, 0.0), (5.0, 5.0)]

    def test_primitive_reference(self):
        from phast.geometry_dsl import parse_mesh_dsl
        m = parse_mesh_dsl({
            'element_size': {
                'default': 2.0,
                'refined': [{'primitive': 'notch', 'size': 0.25,
                             'margin': 5.0}],
            },
        }, _ps())
        rule = m.refined[0]
        assert rule.region is None
        assert rule.primitive == 'notch'
        assert rule.size == 0.25
        assert rule.margin == 5.0

    def test_full_canonical_yaml_block(self):
        """The example block from issue #145 parses end-to-end."""
        from phast.geometry_dsl import parse_mesh_dsl
        m = parse_mesh_dsl({
            'element_size': {
                'default': 2.0,
                'refined': [
                    {'region': {'type': 'ball',
                                'center': [10, 65], 'radius': 30},
                     'size': 0.25},
                    {'region': {'type': 'rectangle',
                                'origin': [0, 64.5], 'size': [55, 1]},
                     'size': 0.1},
                    {'primitive': 'notch', 'size': 0.25, 'margin': 5.0},
                ],
            },
        }, _ps())
        assert m.default_size == 2.0
        assert len(m.refined) == 3
        kinds = [
            (r.region.kind if r.region else f"prim:{r.primitive}")
            for r in m.refined
        ]
        assert kinds == ['circle', 'rectangle', 'prim:notch']

    def test_units_meters_scales_sizes_and_margin(self):
        from phast.geometry_dsl import parse_mesh_dsl
        m = parse_mesh_dsl({
            'element_size': {
                'default': 0.002,  # 2 mm
                'refined': [{
                    'region': {'type': 'circle',
                               'center': [0.01, 0.065], 'radius': 0.03},
                    'size': 0.00025,    # 0.25 mm
                    'margin': 0.005,    # 5 mm
                }],
            },
        }, _ps(), units='m')
        assert m.default_size == pytest.approx(2.0)
        assert m.refined[0].size == pytest.approx(0.25)
        assert m.refined[0].margin == pytest.approx(5.0)
        assert m.refined[0].region.radius == pytest.approx(30.0)
        assert m.refined[0].region.center == pytest.approx((10.0, 65.0))


class TestMeshDSLValidation:
    def test_default_size_required(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match="'element_size.default' is required"):
            parse_mesh_dsl({'element_size': {'refined': []}}, _ps())

    def test_element_size_block_required(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match="missing required 'element_size'"):
            parse_mesh_dsl({}, _ps())

    def test_default_size_non_positive(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='strictly positive'):
            parse_mesh_dsl({'element_size': {'default': 0.0}}, _ps())

    def test_negative_size(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='strictly positive'):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{'primitive': 'notch', 'size': -0.1}],
                },
            }, _ps())

    def test_negative_margin(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='non-negative'):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{'primitive': 'notch',
                                 'size': 0.1, 'margin': -1.0}],
                },
            }, _ps())

    def test_unknown_primitive_with_did_you_mean(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError) as exc:
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{'primitive': 'notc', 'size': 0.25}],
                },
            }, _ps())
        msg = str(exc.value)
        assert 'unknown primitive' in msg
        assert "'notc'" in msg
        assert "Did you mean 'notch'" in msg

    def test_unknown_primitive_no_close_match_lists_known(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError) as exc:
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{'primitive': 'qqqq', 'size': 0.25}],
                },
            }, _ps())
        msg = str(exc.value)
        assert 'unknown primitive' in msg
        assert 'plate' in msg and 'notch' in msg

    def test_region_and_primitive_mutex(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='mutually exclusive'):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{
                        'region': {'type': 'circle',
                                   'center': [0, 0], 'radius': 1},
                        'primitive': 'notch',
                        'size': 0.25,
                    }],
                },
            }, _ps())

    def test_neither_region_nor_primitive(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match="either 'region'"):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{'size': 0.25}],
                },
            }, _ps())

    def test_size_required(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match="missing required field 'size'"):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{'primitive': 'notch'}],
                },
            }, _ps())

    def test_zero_area_primitive_rejected(self):
        """``primitive: tip`` (a point) cannot be a refinement region."""
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='no interior'):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{'primitive': 'tip', 'size': 0.25}],
                },
            }, _ps())

    def test_ad_hoc_region_rejects_point_type(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='unsupported type'):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{
                        'region': {'type': 'point', 'coords': [0, 0]},
                        'size': 0.25,
                    }],
                },
            }, _ps())

    def test_ad_hoc_region_rejects_line_segment_type(self):
        from phast.geometry_dsl import parse_mesh_dsl
        with pytest.raises(ValueError, match='unsupported type'):
            parse_mesh_dsl({
                'element_size': {
                    'default': 2.0,
                    'refined': [{
                        'region': {'type': 'line_segment',
                                   'from': [0, 0], 'to': [1, 0]},
                        'size': 0.25,
                    }],
                },
            }, _ps())


class TestMeshDSLConfigIntegration:
    def _write_yaml(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        with open(path, 'w') as f:
            f.write(textwrap.dedent(body))
        return path

    def test_mesh_block_loads_and_resolves(self):
        """Mesh DSL parses; without a domain block + multiple primitives
        the compiler raises a clean disambiguation error (#146)."""
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              units: mm
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
                notch: {type: rectangle, origin: [0, 64.75], size: [10, 0.5]}
              mesh:
                element_size:
                  default: 2.0
                  refined:
                    - region: {type: ball, center: [10, 65], radius: 30}
                      size: 0.25
                    - primitive: notch
                      size: 0.25
                      margin: 5.0
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.mesh is not None
            with pytest.raises(ValueError, match='domain'):
                resolve_config(cfg)
            parsed_mesh = getattr(cfg.geometry, '_parsed_mesh', None)
            assert parsed_mesh is not None
            assert parsed_mesh.default_size == 2.0
            assert len(parsed_mesh.refined) == 2
        finally:
            os.unlink(path)

    def test_legacy_config_without_mesh_block_unchanged(self):
        """``mesh:`` is purely additive: existing configs still load."""
        from phast.config import load_config
        path = self._write_yaml("""
            geometry:
              type: miehe_tension
              parameters: {}
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.mesh is None
        finally:
            os.unlink(path)

    def test_mesh_block_validation_surfaces_at_resolve(self):
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
              mesh:
                element_size:
                  default: 2.0
                  refined:
                    - primitive: nonsuch
                      size: 0.25
        """)
        try:
            cfg = load_config(path)
            with pytest.raises(ValueError, match='unknown primitive'):
                resolve_config(cfg)
        finally:
            os.unlink(path)


class TestDomainConfigIntegration:
    def _write_yaml(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        with open(path, 'w') as f:
            f.write(textwrap.dedent(body))
        return path

    def test_domain_parses_via_resolve_and_is_stashed(self):
        """Loading the canonical issue-#143 YAML stashes _parsed_domain."""
        from phast.config import load_config, resolve_config
        from phast.geometry_dsl import Domain
        path = self._write_yaml("""
            geometry:
              units: mm
              primitives:
                plate:    {type: rectangle, origin: [0, 0],     size: [65, 120]}
                notch:    {type: rectangle, origin: [0, 64.75], size: [10, 0.5]}
                big_hole: {type: circle,    center: [36.5, 51], radius: 10}
                pin_top:  {type: circle,    center: [20, 100],  radius: 5}
                pin_bot:  {type: circle,    center: [20, 20],   radius: 5}
              domain:
                base: plate
                subtract: [notch, big_hole, pin_top, pin_bot]
                add: []
                intersect: []
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.domain is not None
            # The 'notch' primitive in this fixture is a Rectangle that
            # the issue-#146 compiler subtracts cleanly; the domain
            # parses + stashes regardless.
            #
            # We only check the parse-and-stash here; full end-to-end
            # mesh compilation is exercised by the geometry-compiler
            # test module so we don't pay the gmsh runtime cost twice.
            from phast.geometry_dsl import (
                parse_primitives, parse_domain,
            )
            parsed = parse_primitives({
                'units': cfg.geometry.units,
                'primitives': cfg.geometry.primitives,
            })
            dom = parse_domain(cfg.geometry.domain, parsed)
            assert isinstance(dom, Domain)
            assert dom.base == 'plate'
            assert dom.subtract == ['notch', 'big_hole', 'pin_top', 'pin_bot']
        finally:
            os.unlink(path)

    def test_domain_absent_back_compat(self):
        """A primitives-only config (single primitive, no 'domain') now
        compiles directly: that single primitive *is* the domain."""
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
              mesh:
                element_size: {default: 10.0}
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.domain is None
            objs = resolve_config(cfg)
            # No explicit domain -> _parsed_domain stays unset.
            assert getattr(cfg.geometry, '_parsed_domain', None) is None
            # But the compiler still produced a real mesh.
            assert getattr(cfg.geometry, '_compiled_mesh_path', None)
            assert objs['mesh'].n_nodes > 0
        finally:
            os.unlink(path)

    def test_domain_unknown_primitive_surfaces_at_resolve(self):
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
              domain:
                base: plate
                subtract: [holex]
        """)
        try:
            cfg = load_config(path)
            with pytest.raises(ValueError, match='unknown primitive'):
                resolve_config(cfg)
        finally:
            os.unlink(path)

    def test_legacy_type_path_unaffected_by_domain_field(self):
        """Configs that use geometry.type continue to ignore the domain
        machinery entirely (the parser only kicks in when primitives are
        declared)."""
        from phast.config import load_config
        path = self._write_yaml("""
            geometry:
              type: miehe_tension
              parameters: {}
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.domain is None
            assert getattr(cfg.geometry, '_parsed_domain', None) is None
        finally:
            os.unlink(path)


class TestNamedGroupConfigIntegration:
    def _write_yaml(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        with open(path, 'w') as f:
            f.write(textwrap.dedent(body))
        return path

    def test_named_groups_round_trip_through_loader(self):
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              units: mm
              primitives:
                plate:   {type: rectangle, origin: [0, 0],     size: [65, 120]}
                pin_top: {type: circle,    center: [32.5, 110], radius: 5}
              named_groups:
                notch_tip:        {point: [10, 65]}
                crack_path_band:  {region: {type: rectangle, origin: [0, 64.5], size: [55, 1]}}
                upper_pin_centre: {primitive: pin_top, kind: centre}
        """)
        try:
            cfg = load_config(path)
            assert set(cfg.geometry.named_groups) == {
                'notch_tip', 'crack_path_band', 'upper_pin_centre'}
            # Multi-primitive without a domain block -> compiler asks for
            # explicit domain disambiguation. The named-group parse +
            # stash side-effect still happens before that point.
            with pytest.raises(ValueError, match='domain'):
                resolve_config(cfg)
            parsed_groups = getattr(
                cfg.geometry, '_parsed_named_groups', None)
            assert parsed_groups is not None
            assert set(parsed_groups) == {
                'notch_tip', 'crack_path_band', 'upper_pin_centre'}
        finally:
            os.unlink(path)

    def test_named_groups_validation_surfaces_at_resolve(self):
        from phast.config import load_config, resolve_config
        path = self._write_yaml("""
            geometry:
              units: mm
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
              named_groups:
                bad: {primitive: not_a_thing, kind: boundary}
        """)
        try:
            cfg = load_config(path)
            with pytest.raises(ValueError, match='unknown primitive'):
                resolve_config(cfg)
        finally:
            os.unlink(path)

    def test_no_named_groups_block_is_fine(self):
        """Backward compat: configs without named_groups still parse."""
        from phast.config import load_config
        path = self._write_yaml("""
            geometry:
              units: mm
              primitives:
                plate: {type: rectangle, origin: [0, 0], size: [65, 120]}
        """)
        try:
            cfg = load_config(path)
            assert cfg.geometry.named_groups is None
        finally:
            os.unlink(path)
