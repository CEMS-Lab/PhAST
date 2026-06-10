"""
J-integral and stress intensity factor (SIF) computation for phase-field fracture.

Implements the equivalent domain integral (EDI) method on unstructured
triangular meshes with mode decomposition via the interaction integral.

Usage::

    from phast.fracture_mechanics import compute_j_integral, compute_sif

    J = compute_j_integral(mesh, u, d, material, fem_ops)
    K_I, K_II, J = compute_sif(mesh, u, d, material, fem_ops)
"""

import math
import torch
from typing import Optional, Tuple


def _farthest_isoline_point_geodesic(crack_path: torch.Tensor,
                                     src: torch.Tensor,
                                     edge_threshold: float) -> int:
    """Return index of the crack_path point at maximum graph-distance from
    ``src`` along the isoline radius graph (edges between crossings within
    ``edge_threshold``).

    Falls back to Euclidean argmax if the radius graph cannot reach any
    crack-path point from ``src`` (disconnected, empty, etc.). The fallback
    preserves the legacy behaviour for cases where the geodesic graph is
    degenerate.

    Implementation: O(M^2) Dijkstra; M is the number of isoline crossings,
    typically a few hundred for paper-scale meshes -- fast enough.
    """
    M = crack_path.shape[0]
    if M == 0:
        return 0

    src_dists = (crack_path - src.unsqueeze(0)).norm(dim=1)
    if M == 1:
        return 0

    src_idx = int(src_dists.argmin().item())

    # Pairwise distances; mask edges by radius threshold
    pdist = torch.cdist(crack_path, crack_path)
    eye = torch.eye(M, dtype=torch.bool, device=crack_path.device)
    adj = (pdist <= edge_threshold) & ~eye

    # Dense Dijkstra: dist[i] = walked path length from src_idx to i
    inf = float('inf')
    dist = torch.full((M,), inf, dtype=crack_path.dtype,
                      device=crack_path.device)
    visited = torch.zeros(M, dtype=torch.bool, device=crack_path.device)
    dist[src_idx] = 0.0

    for _ in range(M):
        # Pick smallest-tentative-distance unvisited node
        cand = torch.where(visited, torch.full_like(dist, inf), dist)
        u = int(cand.argmin().item())
        if cand[u].item() == inf:
            break
        visited[u] = True
        # Relax outgoing edges
        nbr = adj[u]
        if nbr.any():
            new_dist = dist[u] + pdist[u]
            update = nbr & (new_dist < dist)
            dist = torch.where(update, new_dist, dist)

    finite = ~torch.isinf(dist)
    if not finite.any() or finite.sum().item() == 1:
        # Disconnected / source-only-reachable; fall back to Euclidean.
        return int(src_dists.argmax().item())

    masked = torch.where(finite, dist, torch.full_like(dist, -1.0))
    return int(masked.argmax().item())


def find_crack_tip(mesh, d: torch.Tensor, direction: str = 'x',
                   threshold: float = 0.5,
                   nucleation: Optional[torch.Tensor] = None,
                   path_metric: str = 'euclidean',
                   ) -> Tuple[torch.Tensor, float]:
    """Find crack tip by edge-interpolation of the d=threshold isoline.

    Returns (tip_xy, alpha) where tip_xy is (2,) coordinates and alpha
    is the crack tangent angle in radians.

    Parameters
    ----------
    direction : {'x', '-x', 'y', 'auto'}
        Cartesian extremum along the chosen axis. Fine for primary-
        direction cracks (e.g. horizontal SENT), but unsafe when the
        crack curls (Kalthoff shear band, Y-branching under mixed
        mode): the point of max-x may be on a retrograde branch rather
        than the true tip. Use ``nucleation`` instead in those cases.
    nucleation : torch.Tensor or None, shape (2,)
        If provided, the tip is selected from the isoline using
        ``path_metric`` (see below). When both ``nucleation`` and
        ``direction`` are given, ``nucleation`` wins.
    path_metric : {'euclidean', 'geodesic'}
        Distance used to score isoline points relative to ``nucleation``.

        ``'euclidean'`` (default, backward-compatible): straight-line
        distance from the nucleation source. Robust for monotone-
        propagation cracks (SENT, B5/Borden branching, Kalthoff before
        the shear band curls); can mis-pick the tip on J/U-shaped or
        re-entrant paths where the Euclidean-farthest isoline point
        lies on a retrograde curl rather than the leading tip.

        ``'geodesic'``: graph-distance along the isoline. Builds a
        radius graph on the crossings (edges between points within
        ``2 * mesh.h_min``) and runs Dijkstra from the source's nearest
        crack-path node. The farthest node by walked path length is
        returned. Falls back to Euclidean if the graph is empty or
        disconnects the nucleation. Closes issue #99 for curved-crack
        post-processing (J-integral / SIF on Y-branched or curling
        cracks).
    """
    nodes = mesh.nodes     # (N, 2)
    elems = mesh.elements  # (E, 3)
    d_e = d[elems]         # (E, 3)

    crossings = []
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        da, db = d_e[:, a], d_e[:, b]
        mask = (da - threshold) * (db - threshold) < 0
        if not mask.any():
            continue
        t = (threshold - da[mask]) / (db[mask] - da[mask] + 1e-30)
        xa = nodes[elems[mask, a]]
        xb = nodes[elems[mask, b]]
        pts = xa + t.unsqueeze(1) * (xb - xa)
        crossings.append(pts)

    if not crossings:
        return torch.tensor([0.0, 0.0], dtype=nodes.dtype, device=nodes.device), 0.0

    crack_path = torch.cat(crossings, dim=0)  # (M, 2)

    if nucleation is not None:
        src = nucleation.to(dtype=crack_path.dtype, device=crack_path.device)
        if path_metric == 'geodesic':
            idx = _farthest_isoline_point_geodesic(
                crack_path, src, edge_threshold=2.0 * mesh.h_min)
        elif path_metric == 'euclidean':
            idx = (crack_path - src.unsqueeze(0)).norm(dim=1).argmax()
        else:
            raise ValueError(
                f"path_metric must be 'euclidean' or 'geodesic', "
                f"got {path_metric!r}")
    elif direction == 'x':
        idx = crack_path[:, 0].argmax()
    elif direction == '-x':
        idx = crack_path[:, 0].argmin()
    elif direction == 'y':
        idx = crack_path[:, 1].argmax()
    else:
        idx = crack_path[:, 0].argmax()

    tip = crack_path[idx]

    # Estimate tangent from nearby isoline points
    dist_to_tip = (crack_path - tip.unsqueeze(0)).norm(dim=1)
    r_local = dist_to_tip.median().item() * 2.0
    nearby = crack_path[dist_to_tip < max(r_local, 1e-6)]

    if len(nearby) >= 3:
        dx = nearby[:, 0] - tip[0]
        dy = nearby[:, 1] - tip[1]
        alpha = math.atan2(
            (dx * dy).sum().item(),
            (dx * dx).sum().item()
        )
    else:
        alpha = 0.0

    return tip, alpha


def _build_q_function(mesh, tip: torch.Tensor, l0: float,
                      r_inner_factor: float = 5.0,
                      r_outer_factor: float = 10.0) -> torch.Tensor:
    """Build smooth weight function q for the domain integral.

    q=1 inside r_inner, q=0 outside r_outer, cosine ramp between.
    """
    r_inner = r_inner_factor * l0
    r_outer = r_outer_factor * l0
    r = (mesh.nodes - tip.unsqueeze(0)).norm(dim=1)  # (N,)

    q = torch.zeros_like(r)
    inner = r <= r_inner
    outer = r >= r_outer
    ramp = ~inner & ~outer

    q[inner] = 1.0
    if ramp.any():
        s = (r[ramp] - r_inner) / (r_outer - r_inner)
        q[ramp] = 0.5 * (1.0 + torch.cos(math.pi * s))

    return q


def compute_j_integral(
    mesh, u: torch.Tensor, d: torch.Tensor,
    material, fem_ops,
    tip: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    r_inner_factor: float = 5.0,
    r_outer_factor: float = 10.0,
    d_process_threshold: float = 0.01,
) -> float:
    """Compute J-integral using the equivalent domain integral method.

    Parameters
    ----------
    mesh : FEMMesh
    u : (N, 2) displacement
    d : (N,) damage
    material : Material
    fem_ops : FEMOperators
    tip : (2,) crack tip coordinates, or None to auto-detect
    alpha : crack tangent angle [rad], or None to auto-detect
    r_inner_factor, r_outer_factor : q-function radii as multiples of l0
    d_process_threshold : elements with d_avg > this are excluded

    Returns
    -------
    J : float, J-integral value [N/mm = kJ/m^2 in consistent units]
    """
    if tip is None or alpha is None:
        tip_auto, alpha_auto = find_crack_tip(mesh, d)
        if tip is None:
            tip = tip_auto
        if alpha is None:
            alpha = alpha_auto

    q = _build_q_function(mesh, tip, material.l0, r_inner_factor, r_outer_factor)

    elems = mesh.elements   # (E, 3)
    areas = mesh.areas      # (E,)
    gp = mesh.grad_phi      # (E, 3, 2)

    # q gradient (element-level)
    q_e = q[elems]           # (E, 3)
    dq_dx = (gp[:, :, 0] * q_e).sum(1)  # (E,)
    dq_dy = (gp[:, :, 1] * q_e).sum(1)

    # Full displacement gradient (asymmetric — 4 components)
    u_e = u[elems]           # (E, 3, 2)
    du_x_dx = (gp[:, :, 0] * u_e[:, :, 0]).sum(1)
    du_x_dy = (gp[:, :, 1] * u_e[:, :, 0]).sum(1)
    du_y_dx = (gp[:, :, 0] * u_e[:, :, 1]).sum(1)
    du_y_dy = (gp[:, :, 1] * u_e[:, :, 1]).sum(1)

    # Rotate to crack direction
    c, s = math.cos(alpha), math.sin(alpha)
    du_x_dx1 = du_x_dx * c + du_x_dy * s
    du_y_dx1 = du_y_dx * c + du_y_dy * s
    dq_dx1 = dq_dx * c + dq_dy * s

    # Degraded stress
    strain = fem_ops.compute_strain(u)
    sxx, syy, sxy = fem_ops.compute_stress(u, d, strain=strain)

    # Strain energy density W = g(d) * psi+ + psi-
    psi_plus = fem_ops.compute_psi_plus(u, strain=strain)
    exx, eyy, gxy = strain
    lam = material.lam
    mu = material.mu
    tr = exx + eyy
    psi_full = 0.5 * lam * tr**2 + mu * (exx**2 + eyy**2 + 0.5 * gxy**2)
    psi_minus = psi_full - psi_plus
    d_avg = d[elems].mean(dim=1)
    g_d = material.degradation(d_avg)
    W = g_d * psi_plus + psi_minus

    # Integrand: sigma_ij * du_i/dx_1 * dq/dxj - W * dq/dx_1
    integrand = (
        sxx * du_x_dx1 * dq_dx + sxy * du_x_dx1 * dq_dy
        + sxy * du_y_dx1 * dq_dx + syy * du_y_dx1 * dq_dy
        - W * dq_dx1
    )

    # Mask: exclude process zone
    mask = d_avg < d_process_threshold
    J = (integrand * areas * mask.float()).sum().item()

    return J


def compute_sif(
    mesh, u: torch.Tensor, d: torch.Tensor,
    material, fem_ops,
    tip: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    r_inner_factor: float = 5.0,
    r_outer_factor: float = 10.0,
) -> Tuple[float, float, float]:
    """Compute stress intensity factors K_I, K_II via interaction integral.

    Uses Williams auxiliary fields for mode decomposition.

    Returns (K_I, K_II, J) in consistent units [MPa*sqrt(mm), MPa*sqrt(mm), N/mm].
    """
    if tip is None or alpha is None:
        tip_auto, alpha_auto = find_crack_tip(mesh, d)
        if tip is None:
            tip = tip_auto
        if alpha is None:
            alpha = alpha_auto

    J = compute_j_integral(mesh, u, d, material, fem_ops,
                           tip=tip, alpha=alpha,
                           r_inner_factor=r_inner_factor,
                           r_outer_factor=r_outer_factor)

    E = material.E
    nu = material.nu
    if material.plane_stress:
        E_prime = E
        kappa = (3.0 - nu) / (1.0 + nu)
    else:
        E_prime = E / (1.0 - nu**2)
        kappa = 3.0 - 4.0 * nu

    mu = material.mu

    q = _build_q_function(mesh, tip, material.l0, r_inner_factor, r_outer_factor)
    elems = mesh.elements
    areas = mesh.areas
    gp = mesh.grad_phi

    q_e = q[elems]
    dq_dx = (gp[:, :, 0] * q_e).sum(1)
    dq_dy = (gp[:, :, 1] * q_e).sum(1)
    c, s = math.cos(alpha), math.sin(alpha)
    dq_dx1 = dq_dx * c + dq_dy * s

    # Actual stress and displacement gradient
    strain = fem_ops.compute_strain(u)
    sxx, syy, sxy = fem_ops.compute_stress(u, d, strain=strain)
    u_e = u[elems]
    du_x_dx = (gp[:, :, 0] * u_e[:, :, 0]).sum(1)
    du_x_dy = (gp[:, :, 1] * u_e[:, :, 0]).sum(1)
    du_y_dx = (gp[:, :, 0] * u_e[:, :, 1]).sum(1)
    du_y_dy = (gp[:, :, 1] * u_e[:, :, 1]).sum(1)
    du_x_dx1 = du_x_dx * c + du_x_dy * s
    du_y_dx1 = du_y_dx * c + du_y_dy * s

    # Element centroids in local crack coordinates
    centroids = mesh.nodes[elems].mean(dim=1)  # (E, 2)
    dx = centroids[:, 0] - tip[0]
    dy = centroids[:, 1] - tip[1]
    x_loc = dx * c + dy * s
    y_loc = -dx * s + dy * c
    r = torch.sqrt(x_loc**2 + y_loc**2 + 1e-30)
    theta = torch.atan2(y_loc, x_loc)

    d_avg = d[elems].mean(dim=1)
    mask = (d_avg < 0.01).float()

    inv_sqrt_2pir = 1.0 / torch.sqrt(2.0 * math.pi * r)
    ct2 = torch.cos(theta / 2)
    st2 = torch.sin(theta / 2)
    c3t2 = torch.cos(1.5 * theta)
    s3t2 = torch.sin(1.5 * theta)

    K_values = []
    for mode in ('I', 'II'):
        if mode == 'I':
            # Williams Mode I auxiliary stress (K_I^aux = 1)
            s_xx_a = inv_sqrt_2pir * ct2 * (1 - st2 * s3t2)
            s_yy_a = inv_sqrt_2pir * ct2 * (1 + st2 * s3t2)
            s_xy_a = inv_sqrt_2pir * ct2 * st2 * c3t2
            # Auxiliary strain (Hooke's law inverse)
            e_xx_a = (s_xx_a - nu * s_yy_a) / E
            e_yy_a = (s_yy_a - nu * s_xx_a) / E
            g_xy_a = s_xy_a / mu
            # Auxiliary du/dx_1 (symmetric + rotation for mode I)
            omega_xy = -0.5 * inv_sqrt_2pir * st2
            du_x_dx1_a = e_xx_a * c - (0.5 * g_xy_a - omega_xy) * s
            du_y_dx1_a = (0.5 * g_xy_a + omega_xy) * c + e_yy_a * s
        else:
            # Williams Mode II auxiliary stress (K_II^aux = 1)
            s_xx_a = -inv_sqrt_2pir * st2 * (2 + ct2 * c3t2)
            s_yy_a = inv_sqrt_2pir * st2 * ct2 * c3t2
            s_xy_a = inv_sqrt_2pir * ct2 * (1 - st2 * s3t2)
            e_xx_a = (s_xx_a - nu * s_yy_a) / E
            e_yy_a = (s_yy_a - nu * s_xx_a) / E
            g_xy_a = s_xy_a / mu
            omega_xy = -0.5 * inv_sqrt_2pir * ct2
            du_x_dx1_a = e_xx_a * c - (0.5 * g_xy_a - omega_xy) * s
            du_y_dx1_a = (0.5 * g_xy_a + omega_xy) * c + e_yy_a * s

        # Interaction energy: W_int = sigma_ij * eps_ij^aux
        W_int = sxx * e_xx_a + syy * e_yy_a + sxy * g_xy_a

        # Interaction integral
        M_integrand = (
            (sxx * du_x_dx1_a + sxy * du_y_dx1_a) * dq_dx
            + (sxy * du_x_dx1_a + syy * du_y_dx1_a) * dq_dy
            + (s_xx_a * du_x_dx1 + s_xy_a * du_y_dx1) * dq_dx
            + (s_xy_a * du_x_dx1 + s_yy_a * du_y_dx1) * dq_dy
            - W_int * dq_dx1
        )

        M = (M_integrand * areas * mask).sum().item()
        K = E_prime / 2.0 * M
        K_values.append(K)

    K_I, K_II = K_values
    return K_I, K_II, J
