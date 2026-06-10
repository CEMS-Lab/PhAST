"""Tests for issue #139: preseed damage by region or named node-set."""
import torch
import pytest

from phast.mesh import FEMMesh
from phast.initial_conditions import (
    resolve_preseed_specs,
    normalise_legacy_preseed,
    value_to_H_seed,
    _mask_line_segment,
    _mask_rectangle,
    _mask_circle,
    _mask_polygon,
)


def _grid_mesh(nx: int = 5, ny: int = 5,
               extras: dict = None) -> FEMMesh:
    """Build a regular nx*ny triangulated unit-square mesh."""
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append([i / (nx - 1), j / (ny - 1)])
    nodes = torch.tensor(pts, dtype=torch.float64)

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + i + 1
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + i + 1
            elems.append([n00, n10, n11])
            elems.append([n00, n11, n01])
    elements = torch.tensor(elems, dtype=torch.long)

    node_sets = {}
    if extras:
        node_sets.update({k: torch.as_tensor(v, dtype=torch.long)
                          for k, v in extras.items()})
    return FEMMesh.from_tensors(nodes, elements,
                                node_sets=node_sets, device='cpu')


# ---------------------------------------------------------------------------
# Region predicates
# ---------------------------------------------------------------------------

def test_line_segment_horizontal():
    mesh = _grid_mesh(11, 11)  # spacing 0.1
    mask = _mask_line_segment(mesh.nodes, [0.0, 0.5], [0.5, 0.5],
                              thickness=1e-6)
    # Expected: nodes on y=0.5 with 0 <= x <= 0.5 -> 6 nodes.
    expected = ((mesh.nodes[:, 1] - 0.5).abs() < 1e-9) & \
               (mesh.nodes[:, 0] <= 0.5 + 1e-9)
    assert torch.equal(mask, expected)
    assert int(mask.sum()) == 6


def test_line_segment_thickness_tube():
    mesh = _grid_mesh(11, 11)
    mask = _mask_line_segment(mesh.nodes, [0.0, 0.5], [1.0, 0.5],
                              thickness=0.05)
    # Tube of width 0.1 around y=0.5; only nodes on y=0.5 line qualify.
    expected = (mesh.nodes[:, 1] - 0.5).abs() <= 0.05 + 1e-12
    assert torch.equal(mask, expected)


def test_rectangle_inclusive():
    mesh = _grid_mesh(11, 11)
    mask = _mask_rectangle(mesh.nodes, [0.2, 0.3], [0.4, 0.4])
    expected = ((mesh.nodes[:, 0] >= 0.2 - 1e-12) &
                (mesh.nodes[:, 0] <= 0.6 + 1e-12) &
                (mesh.nodes[:, 1] >= 0.3 - 1e-12) &
                (mesh.nodes[:, 1] <= 0.7 + 1e-12))
    assert torch.equal(mask, expected)


def test_circle_radius():
    mesh = _grid_mesh(11, 11)
    mask = _mask_circle(mesh.nodes, [0.5, 0.5], 0.15)
    # Reference computation
    expected = (mesh.nodes - torch.tensor([0.5, 0.5],
                                          dtype=torch.float64)).norm(dim=1) <= 0.15
    assert torch.equal(mask, expected)
    assert int(mask.sum()) >= 1


def test_polygon_triangle():
    mesh = _grid_mesh(11, 11)
    verts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    mask = _mask_polygon(mesh.nodes, verts)
    # Strict interior of triangle x+y<1; ray-cast may include some edges.
    # Just sanity-check: corner (0,0) and (1,0),(0,1) edges; (0.9,0.9) outside.
    n00 = (mesh.nodes - torch.tensor([0.0, 0.0])).norm(dim=1).argmin()
    n_far = (mesh.nodes - torch.tensor([0.9, 0.9])).norm(dim=1).argmin()
    n_mid = (mesh.nodes - torch.tensor([0.2, 0.2])).norm(dim=1).argmin()
    # Inside-ish point must be in mask
    assert bool(mask[n_mid])
    # Far outside corner must not be
    assert not bool(mask[n_far])
    # All masked points must satisfy x+y<=1 (within tolerance)
    s = mesh.nodes[mask].sum(dim=1)
    assert bool((s <= 1.0 + 1e-9).all())


# ---------------------------------------------------------------------------
# resolve_preseed_specs
# ---------------------------------------------------------------------------

def test_resolve_named_nodes():
    extras = {'notch': [0, 1, 2]}
    mesh = _grid_mesh(5, 5, extras=extras)
    specs = [{'nodes': 'notch', 'value': 1.0}]
    mask, vals = resolve_preseed_specs(mesh, specs)
    assert int(mask.sum()) == 3
    assert torch.equal(mask.nonzero().flatten(), torch.tensor([0, 1, 2]))
    assert float(vals[0]) == 1.0


def test_resolve_unknown_nodeset_raises():
    mesh = _grid_mesh(3, 3)
    with pytest.raises(RuntimeError, match="no node set"):
        resolve_preseed_specs(mesh, [{'nodes': 'missing', 'value': 1.0}])


def test_resolve_overlapping_takes_max():
    mesh = _grid_mesh(11, 11)
    specs = [
        {'region': {'type': 'rectangle',
                    'origin': [0.0, 0.0], 'size': [1.0, 1.0]},
         'value': 0.3},
        {'region': {'type': 'circle',
                    'center': [0.5, 0.5], 'radius': 0.1},
         'value': 0.9},
    ]
    mask, vals = resolve_preseed_specs(mesh, specs)
    assert bool(mask.all())
    # Inside the circle: value=0.9; outside: value=0.3
    in_circle = ((mesh.nodes - torch.tensor([0.5, 0.5],
                                            dtype=torch.float64)).norm(dim=1)
                 <= 0.1)
    assert torch.allclose(vals[in_circle],
                          torch.full((int(in_circle.sum()),), 0.9,
                                     dtype=torch.float64))
    assert torch.allclose(vals[~in_circle],
                          torch.full((int((~in_circle).sum()),), 0.3,
                                     dtype=torch.float64))


def test_resolve_lower_value_does_not_overwrite():
    """A later spec with lower value must not lower the existing seed."""
    mesh = _grid_mesh(11, 11)
    specs = [
        {'region': {'type': 'circle',
                    'center': [0.5, 0.5], 'radius': 0.1},
         'value': 0.9},
        {'region': {'type': 'rectangle',
                    'origin': [0.0, 0.0], 'size': [1.0, 1.0]},
         'value': 0.3},
    ]
    _, vals = resolve_preseed_specs(mesh, specs)
    in_circle = ((mesh.nodes - torch.tensor([0.5, 0.5],
                                            dtype=torch.float64)).norm(dim=1)
                 <= 0.1)
    assert torch.allclose(vals[in_circle],
                          torch.full((int(in_circle.sum()),), 0.9,
                                     dtype=torch.float64))


# ---------------------------------------------------------------------------
# Backward-compat
# ---------------------------------------------------------------------------

def test_legacy_preseed_normalised():
    out = normalise_legacy_preseed(['a', 'b'])
    assert out == [{'nodes': 'a', 'value': 1.0},
                   {'nodes': 'b', 'value': 1.0}]


def test_backward_compat_legacy_matches_new_form():
    """Legacy preseed_notch_nodesets must produce identical (mask, value)
    as the equivalent new-form preseed_damage list."""
    extras = {'notchA': [0, 1], 'notchB': [3, 4]}
    mesh = _grid_mesh(5, 5, extras=extras)

    legacy_specs = normalise_legacy_preseed(['notchA', 'notchB'])
    new_specs = [{'nodes': 'notchA', 'value': 1.0},
                 {'nodes': 'notchB', 'value': 1.0}]

    m1, v1 = resolve_preseed_specs(mesh, legacy_specs)
    m2, v2 = resolve_preseed_specs(mesh, new_specs)
    assert torch.equal(m1, m2)
    assert torch.allclose(v1, v2)
    # And the union mask should hit all named nodes
    expected_idx = torch.tensor([0, 1, 3, 4])
    assert torch.equal(m1.nonzero().flatten(), expected_idx)


# ---------------------------------------------------------------------------
# value_to_H_seed
# ---------------------------------------------------------------------------

def test_value_to_H_saturation_is_legacy_sentinel():
    Gc, l0 = 2.7, 0.015
    H = value_to_H_seed(1.0, Gc, l0)
    assert H == pytest.approx(1.0e4 * Gc / l0)


def test_value_to_H_partial_solves_at2_equilibrium():
    """For partial value, AT2 zero-gradient eq gives d = 2H/(Gc/l0+2H)."""
    Gc, l0 = 2.7, 0.015
    for v in (0.1, 0.3, 0.5, 0.8):
        H = value_to_H_seed(v, Gc, l0)
        d = 2 * H / (Gc / l0 + 2 * H)
        assert d == pytest.approx(v, abs=1e-12)


def test_value_to_H_zero():
    assert value_to_H_seed(0.0, 2.7, 0.015) == 0.0


# ---------------------------------------------------------------------------
# Bad input
# ---------------------------------------------------------------------------

def test_bad_value_range_raises():
    mesh = _grid_mesh(3, 3)
    with pytest.raises(ValueError):
        resolve_preseed_specs(mesh,
                              [{'region': {'type': 'circle',
                                           'center': [0, 0], 'radius': 0.1},
                                'value': 1.5}])


def test_unknown_region_type_raises():
    mesh = _grid_mesh(3, 3)
    with pytest.raises(ValueError, match="Unknown preseed_damage region"):
        resolve_preseed_specs(mesh,
                              [{'region': {'type': 'banana'}, 'value': 1.0}])


def test_missing_region_key_raises():
    mesh = _grid_mesh(3, 3)
    with pytest.raises(ValueError, match="must specify"):
        resolve_preseed_specs(mesh, [{'value': 1.0}])
