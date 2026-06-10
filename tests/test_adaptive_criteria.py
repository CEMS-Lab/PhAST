"""Tests for refinement-criterion helpers integrated into adaptive.py (#111)."""

import torch

from phast.adaptive import (
    crack_tip_neighborhood_criterion,
    damage_gradient_criterion,
    union_refine_set,
)
from phast.mesh import FEMMesh


def _structured_grid(nx: int, ny: int) -> FEMMesh:
    """Build a unit-square structured triangular mesh (nx*ny cells, 2 tris each)."""
    xs = torch.linspace(0.0, 1.0, nx + 1, dtype=torch.float64)
    ys = torch.linspace(0.0, 1.0, ny + 1, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    nodes = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    def nid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            tris.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)])
            tris.append([nid(i, j), nid(i + 1, j + 1), nid(i, j + 1)])
    elems = torch.tensor(tris, dtype=torch.long)
    return FEMMesh.from_tensors(nodes, elems, node_sets=None,
                                device='cpu', dtype=torch.float64)


def test_damage_gradient_criterion_synthetic():
    """A step in d at x = 0.5 marks (and only marks) the elements straddling it."""
    mesh = _structured_grid(5, 5)
    d = (mesh.nodes[:, 0] > 0.5 - 1e-9).to(torch.float64)
    flagged = damage_gradient_criterion(d, mesh, threshold=0.1)

    assert len(flagged) > 0, "step in d should flag at least one element"

    # Every flagged element must straddle the step: x_min < 0.5 and x_max > 0.5.
    elems = mesh.elements
    xs = mesh.nodes[:, 0][elems]  # (E, 3)
    for ei in flagged:
        x_e = xs[ei]
        assert x_e.min().item() < 0.5 + 1e-9
        assert x_e.max().item() > 0.5 - 1e-9

    # Far-from-the-step elements (entirely left or entirely right) must not be flagged.
    flagged_set = set(flagged)
    for ei in range(mesh.n_elems):
        x_e = xs[ei]
        if x_e.max().item() < 0.5 - 1e-9 or x_e.min().item() > 0.5 + 1e-9:
            assert ei not in flagged_set


def test_crack_tip_neighborhood_criterion():
    """A single hot element flags its nearby ring; far elements are excluded."""
    mesh = _structured_grid(7, 7)
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)

    # Pick the element whose centroid is closest to (0.5, 0.5) and damage all 3 nodes.
    centroids = mesh.nodes[mesh.elements].mean(dim=1)
    centre_elem = int(torch.linalg.norm(centroids - 0.5, dim=1).argmin().item())
    for ni in mesh.elements[centre_elem].tolist():
        d[ni] = 0.7

    flagged = crack_tip_neighborhood_criterion(
        d, mesh, radius=2.0, d_tip_threshold=0.5)

    assert centre_elem in flagged, "centre element must be in its own neighbourhood"
    assert 1 < len(flagged) < mesh.n_elems, (
        f"neighbourhood should be a proper subset, got {len(flagged)} of {mesh.n_elems}")

    # Every flagged element must be within radius * h_min of *some* tip element.
    # Damaging the 3 nodes of centre_elem also lifts neighbouring elements
    # (which share those nodes) above the d_tip_threshold, so the tip set is
    # the star of centre_elem, not a singleton.
    cutoff = 2.0 * mesh.h_min
    d_elem_max = (torch.zeros_like(d) + d)[mesh.elements].max(dim=1).values
    tip_centroids = centroids[d_elem_max > 0.5]
    for ei in flagged:
        diffs = centroids[ei].unsqueeze(0) - tip_centroids
        min_d = torch.linalg.norm(diffs, dim=1).min().item()
        assert min_d <= cutoff + 1e-12

    # No-damage case: empty result.
    d_zero = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    assert crack_tip_neighborhood_criterion(d_zero, mesh) == []


def test_union_helper():
    assert union_refine_set([[1, 2, 3], [3, 4]]) == [1, 2, 3, 4]
    assert union_refine_set([]) == []
    assert union_refine_set([[5], [5], [5]]) == [5]
