"""
J-integral and stress intensity factor (SIF) computation for phase-field fracture.

Implements the equivalent domain integral (EDI) method on unstructured
triangular meshes with mode decomposition via the interaction integral
(Yau, Wang and Corten, J. Appl. Mech. 47 (1980) 335-341,
doi:10.1115/1.3153665).

Both integrals leave out the cells whose mean damage reaches
``d_process_threshold`` (default 0.9). A smaller cutoff also removes the
damage halo that a regularised crack face carries inside the weight
function, and J then depends on the domain radius.

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
    d_process_threshold: float = 0.9,
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
    d_process_threshold : cells whose mean damage is at least this value are
        left out of the integral (default 0.9). A cell with d = 0.9 keeps
        (1 - d)^2 = 1 % of its stiffness, so the cells above the cutoff carry
        no load; a much smaller value (the former default was 0.01) also
        removes the damage halo of a regularised crack face, which is 4 to 5
        l0 wide and lies inside the weight ramp, and J then changes with the
        domain radius. Pass a small value only to restrict the integral to
        undamaged material.

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

    # Mask: exclude the cells that carry no stiffness (d_avg >= threshold)
    mask = d_avg < d_process_threshold
    J = (integrand * areas * mask.float()).sum().item()

    return J


def _williams_unit_fields(r: torch.Tensor, theta: torch.Tensor, mu: float,
                          kappa: float, mode: str):
    """Williams auxiliary field of unit stress intensity factor, crack frame.

    The crack tip is the origin, the crack lies on theta = +-pi, ``r`` and
    ``theta`` are the polar coordinates of the evaluation points (``theta``
    in (-pi, pi]). ``mode`` is ``'I'`` or ``'II'``; ``kappa`` is the Kolosov
    constant, ``3 - 4 nu`` in plane strain and ``(3 - nu)/(1 + nu)`` in plane
    stress (not the bulk modulus ``Material.kappa``).

    Returns ``(stress, grad_u)``: ``stress = (sxx, syy, sxy)`` and
    ``grad_u = (dux_dx, dux_dy, duy_dx, duy_dy)``, all in the crack frame and
    for K = 1 [MPa mm^0.5], so they scale with 1/sqrt(r).

    The stresses are those of Williams, J. Appl. Mech. 24 (1957) 109-114,
    doi:10.1115/1.4011454. The displacement is
    ``u_i = K/(2 mu) sqrt(r/(2 pi)) f_i(theta)`` with::

        mode I   f_x =  cos(t/2) (kappa - 1 + 2 sin^2(t/2))
                 f_y =  sin(t/2) (kappa + 1 - 2 cos^2(t/2))
        mode II  f_x =  sin(t/2) (kappa + 1 + 2 cos^2(t/2))
                 f_y = -cos(t/2) (kappa - 1 - 2 sin^2(t/2))

    and its gradient follows from ``u = A sqrt(r) f(theta)``::

        du/dx = A r^(-1/2) (cos(t) f/2 - sin(t) f')
        du/dy = A r^(-1/2) (sin(t) f/2 + cos(t) f'),   A = 1/(2 mu sqrt(2 pi)).

    The auxiliary strain is the symmetric part of ``grad_u``, so it is
    compatible with the displacement by construction; ``sxx, syy, sxy`` are
    the textbook Williams stresses, which satisfy Hooke's law for the plane
    law that ``kappa`` belongs to (checked in ``tests/test_fracture_sif.py``).
    """
    inv_sqrt_2pir = 1.0 / torch.sqrt(2.0 * math.pi * r)
    h = 0.5 * theta
    c, sn = torch.cos(h), torch.sin(h)
    c3, s3 = torch.cos(1.5 * theta), torch.sin(1.5 * theta)
    if mode == 'I':
        sxx = inv_sqrt_2pir * c * (1.0 - sn * s3)
        syy = inv_sqrt_2pir * c * (1.0 + sn * s3)
        sxy = inv_sqrt_2pir * c * sn * c3
        fx = c * (kappa - 1.0 + 2.0 * sn ** 2)
        fy = sn * (kappa + 1.0 - 2.0 * c ** 2)
        dfx = 0.5 * sn * (-(kappa - 1.0) - 2.0 * sn ** 2 + 4.0 * c ** 2)
        dfy = 0.5 * c * (kappa + 1.0 - 2.0 * c ** 2 + 4.0 * sn ** 2)
    elif mode == 'II':
        sxx = -inv_sqrt_2pir * sn * (2.0 + c * c3)
        syy = inv_sqrt_2pir * sn * c * c3
        sxy = inv_sqrt_2pir * c * (1.0 - sn * s3)
        fx = sn * (kappa + 1.0 + 2.0 * c ** 2)
        fy = -c * (kappa - 1.0 - 2.0 * sn ** 2)
        dfx = 0.5 * c * (kappa + 1.0 + 2.0 * c ** 2 - 4.0 * sn ** 2)
        dfy = 0.5 * sn * (kappa - 1.0 - 2.0 * sn ** 2 + 4.0 * c ** 2)
    else:
        raise ValueError(f"mode must be 'I' or 'II', got {mode!r}")

    a = inv_sqrt_2pir / (2.0 * mu)           # = A / sqrt(r)
    ct, st = torch.cos(theta), torch.sin(theta)
    dux_dx = a * (0.5 * ct * fx - st * dfx)
    dux_dy = a * (0.5 * st * fx + ct * dfx)
    duy_dx = a * (0.5 * ct * fy - st * dfy)
    duy_dy = a * (0.5 * st * fy + ct * dfy)
    return (sxx, syy, sxy), (dux_dx, dux_dy, duy_dx, duy_dy)


def compute_sif(
    mesh, u: torch.Tensor, d: torch.Tensor,
    material, fem_ops,
    tip: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    r_inner_factor: float = 5.0,
    r_outer_factor: float = 10.0,
    d_process_threshold: float = 0.9,
) -> Tuple[float, float, float]:
    """Compute stress intensity factors K_I, K_II via the interaction integral.

    The interaction integral of the actual state with the Williams field of
    unit K_I (resp. unit K_II) is::

        M = int [ sigma_ij u^a_i,1 + sigma^a_ij u_i,1 - sigma_ij eps^a_ij delta_1j ] q_,j dA,
        K = E'/2 M,      E' = E (plane stress), E/(1 - nu^2) (plane strain),

    where ``1`` is the crack direction ``alpha`` and ``q`` is the weight
    function of ``compute_j_integral``. The auxiliary stress and displacement
    gradient are built in the crack frame and rotated to the global axes, so
    ``alpha != 0`` is supported. Signs: K_I > 0 opens the crack (sigma_yy > 0
    ahead of the tip), K_II > 0 for sigma_xy > 0 ahead of the tip, both in the
    crack frame (x along ``alpha``, y 90 degrees counter-clockwise from it).

    ``find_crack_tip`` returns a tangent angle in [-pi/2, pi/2] without an
    orientation. For a crack that runs towards -x (or any direction that is
    not within 90 degrees of +x), pass ``alpha`` along the direction the crack
    points from its mouth to the tip, otherwise the crack frame is reversed and
    both K change sign or mix.

    Parameters
    ----------
    tip, alpha, r_inner_factor, r_outer_factor : as in ``compute_j_integral``
    d_process_threshold : cells whose mean damage is at least this value are
        left out of both integrals (default 0.9, see ``compute_j_integral``).

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
                           r_outer_factor=r_outer_factor,
                           d_process_threshold=d_process_threshold)

    nu = material.nu
    if material.plane_stress:
        E_prime = material.E
        kappa = (3.0 - nu) / (1.0 + nu)
    else:
        E_prime = material.E / (1.0 - nu ** 2)
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

    # Actual stress and displacement gradient (global axes)
    strain = fem_ops.compute_strain(u)
    sxx, syy, sxy = fem_ops.compute_stress(u, d, strain=strain)
    u_e = u[elems]
    du_x_dx = (gp[:, :, 0] * u_e[:, :, 0]).sum(1)
    du_x_dy = (gp[:, :, 1] * u_e[:, :, 0]).sum(1)
    du_y_dx = (gp[:, :, 0] * u_e[:, :, 1]).sum(1)
    du_y_dy = (gp[:, :, 1] * u_e[:, :, 1]).sum(1)
    du_x_dx1 = du_x_dx * c + du_x_dy * s
    du_y_dx1 = du_y_dx * c + du_y_dy * s

    # Element centroids in the crack frame
    centroids = mesh.nodes[elems].mean(dim=1)  # (E, 2)
    dx = centroids[:, 0] - tip[0]
    dy = centroids[:, 1] - tip[1]
    x_loc = dx * c + dy * s
    y_loc = -dx * s + dy * c
    r = torch.sqrt(x_loc ** 2 + y_loc ** 2 + 1e-30)
    theta = torch.atan2(y_loc, x_loc)

    d_avg = d[elems].mean(dim=1)
    mask = (d_avg < d_process_threshold).to(r.dtype)

    # R maps crack-frame components to global ones: v_g = R v_l
    R = torch.stack([torch.stack([torch.full_like(r, c), torch.full_like(r, -s)], -1),
                     torch.stack([torch.full_like(r, s), torch.full_like(r, c)], -1)], -2)   # (E, 2, 2)
    Rt = R.transpose(-1, -2)

    K_values = []
    for mode in ('I', 'II'):
        (s_xx_l, s_yy_l, s_xy_l), (gxx, gxy, gyx, gyy) = _williams_unit_fields(
            r, theta, mu, kappa, mode)
        sig_l = torch.stack([torch.stack([s_xx_l, s_xy_l], -1),
                             torch.stack([s_xy_l, s_yy_l], -1)], -2)
        grad_l = torch.stack([torch.stack([gxx, gxy], -1),
                              torch.stack([gyx, gyy], -1)], -2)
        sig_a = R @ sig_l @ Rt          # auxiliary stress, global axes
        grad_a = R @ grad_l @ Rt        # auxiliary du_i/dx_j, global axes
        s_xx_a, s_yy_a, s_xy_a = sig_a[:, 0, 0], sig_a[:, 1, 1], sig_a[:, 0, 1]
        # derivative of the auxiliary displacement along the crack direction
        du_x_dx1_a = grad_a[:, 0, 0] * c + grad_a[:, 0, 1] * s
        du_y_dx1_a = grad_a[:, 1, 0] * c + grad_a[:, 1, 1] * s
        # auxiliary strain (symmetric part of grad u^a): eps_xx, eps_yy, tensor eps_xy
        e_xx_a, e_yy_a = grad_a[:, 0, 0], grad_a[:, 1, 1]
        e_xy_a = 0.5 * (grad_a[:, 0, 1] + grad_a[:, 1, 0])

        # Interaction energy: W_int = sigma_ij eps^a_ij
        W_int = sxx * e_xx_a + syy * e_yy_a + 2.0 * sxy * e_xy_a

        M_integrand = (
            (sxx * du_x_dx1_a + sxy * du_y_dx1_a) * dq_dx
            + (sxy * du_x_dx1_a + syy * du_y_dx1_a) * dq_dy
            + (s_xx_a * du_x_dx1 + s_xy_a * du_y_dx1) * dq_dx
            + (s_xy_a * du_x_dx1 + s_yy_a * du_y_dx1) * dq_dy
            - W_int * dq_dx1
        )

        M = (M_integrand * areas * mask).sum().item()
        K_values.append(0.5 * E_prime * M)

    K_I, K_II = K_values
    return K_I, K_II, J
