"""
Visualization utilities for phase-field fracture simulations.

Provides:
  - Side-by-side tricontour plots of stress, strain, damage
  - GIF animation from simulation history
  - Initial conditions summary PNG (geometry, BCs, material)
  - Final state summary PNG
"""

import math

import torch
import numpy as np
import json
import os
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _build_triangulation(mesh) -> mtri.Triangulation:
    """Create matplotlib Triangulation from FEMMesh."""
    nodes = mesh.nodes.cpu().numpy()
    elems_t = mesh.elements
    if getattr(mesh, 'element_type', 'T3') == 'Q4':
        from phast.quad_elements import q4_to_triangles
        elems_t = q4_to_triangles(elems_t)
    elems = elems_t.cpu().numpy()
    return mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)


def apply_publication_style(font_size: int = 10) -> None:
    """Apply the repository publication/demo Matplotlib style."""
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "legend.fontsize": max(font_size - 1, 8),
        "xtick.labelsize": max(font_size - 1, 8),
        "ytick.labelsize": max(font_size - 1, 8),
    })


def plot_field(mesh, field, title: str = '', cmap: str = 'inferno',
               vmin: float = None, vmax: float = None,
               ax=None, show_mesh: bool = False,
               colorbar: bool = True,
               colorbar_label: str | None = None) -> plt.Axes:
    """Plot a nodal scalar field on the mesh.

    Parameters
    ----------
    mesh : FEMMesh
    field : (N,) tensor or numpy array
    title : str
    cmap : str
    vmin, vmax : float or None
    ax : matplotlib Axes or None
    show_mesh : bool — draw element edges
    colorbar : bool

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))

    tri = _build_triangulation(mesh)
    f = field.detach().cpu().numpy() if torch.is_tensor(field) else field

    import numpy as np
    f = np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)

    if vmin is None:
        vmin = f.min()
    if vmax is None:
        vmax = f.max()
    if vmin == vmax:
        vmax = vmin + 1e-10

    levels = np.linspace(vmin, vmax, 64)
    tcf = ax.tricontourf(tri, f, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    if show_mesh:
        ax.triplot(tri, 'k-', linewidth=0.1, alpha=0.3)
    if colorbar:
        cbar = plt.colorbar(tcf, ax=ax, shrink=0.8)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x (mm)', fontsize=8)
    ax.set_ylabel('y (mm)', fontsize=8)
    ax.tick_params(labelsize=7)
    return ax


def save_field_plot(mesh, field, save_path: str | os.PathLike,
                    *, title: str = '', cmap: str = 'inferno',
                    vmin: float = None, vmax: float = None,
                    colorbar_label: str | None = None,
                    show_mesh: bool = False, dpi: int = 180,
                    figsize: tuple[float, float] = (4.8, 3.2)) -> None:
    """Save a nodal scalar field with the standard PhAST visual style."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plot_field(mesh, field, title=title, cmap=cmap, vmin=vmin, vmax=vmax,
               ax=ax, show_mesh=show_mesh, colorbar=True,
               colorbar_label=colorbar_label)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def save_deformed_shape(mesh, u, save_path: str | os.PathLike,
                        *, title: str = 'Deformed shape',
                        scale: float | None = None, dpi: int = 180,
                        figsize: tuple[float, float] = (5.0, 3.0)) -> float:
    """Save reference/deformed mesh overlay and return the displacement scale."""
    apply_publication_style()
    nodes = mesh.nodes.detach().cpu().numpy()
    elems_t = mesh.elements
    if getattr(mesh, 'element_type', 'T3') == 'Q4':
        from phast.quad_elements import q4_to_triangles
        elems_t = q4_to_triangles(elems_t)
    elems = elems_t.detach().cpu().numpy()
    disp = u.detach().cpu().numpy() if torch.is_tensor(u) else np.asarray(u)
    if scale is None:
        span = max(float(np.ptp(nodes[:, 0])), float(np.ptp(nodes[:, 1])), 1.0)
        umax = max(float(np.linalg.norm(disp, axis=1).max()), 1.0e-30)
        scale = 0.12 * span / umax
    deformed = nodes + float(scale) * disp

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.triplot(nodes[:, 0], nodes[:, 1], elems, color='0.75', lw=0.6,
               label='reference')
    ax.triplot(deformed[:, 0], deformed[:, 1], elems, color='#d62728',
               lw=0.8, label=f'deformed x{float(scale):.2g}')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return float(scale)


def write_visual_manifest(output_dir: str | os.PathLike,
                          files: list[str],
                          *,
                          visual_scope: str = 'review') -> list[dict]:
    """Write PhAST's lightweight visual-manifest JSON for promoted outputs."""
    from PIL import Image

    root = os.fspath(output_dir)
    manifest = []
    for name in files:
        path = os.path.join(root, name)
        suffix = os.path.splitext(name)[1].lower()
        row = {
            'file': name,
            'size_bytes': int(os.path.getsize(path)),
            'visual_scope': visual_scope,
        }
        if suffix in {'.mp4', '.mov', '.webm'}:
            row.update({
                'artifact_type': 'video',
                'review_dimension_passed': True,
            })
            manifest.append(row)
            continue
        with Image.open(path) as img:
            width, height = img.size
        manifest.append({
            **row,
            'artifact_type': 'image',
            'width_px': int(width),
            'height_px': int(height),
            'review_dimension_passed': bool(max(width, height) < 2000),
        })
    with open(os.path.join(root, 'visual_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    return manifest


def plot_quasistatic_convergence(history, save_path: str, dpi: int = 150):
    """Plot PhaseFieldX-style staggered convergence diagnostics.

    ``history`` is the standalone quasi-static benchmark record list. It
    contains the same quantities persisted to ``solver_telemetry.csv``:
    load step, imposed displacement, stagger iterations, residual, and
    mechanics/phase-field linear iterations.
    """
    if not history:
        return

    step = np.asarray([row['step'] for row in history], dtype=float)
    stagger = np.asarray([row['stagger_iter'] for row in history], dtype=float)
    residual = np.asarray(
        [row.get('residual', np.nan) for row in history], dtype=float)
    pcg_mech = np.asarray(
        [row.get('pcg_iters_mech', 0) for row in history], dtype=float)
    pcg_pf = np.asarray(
        [row.get('pcg_iters_pf', 0) for row in history], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)

    axes[0].plot(step, stagger, 'k.-', lw=1.4, ms=3)
    axes[0].set_ylabel('Stagger iters')
    axes[0].set_title('Quasi-static solver convergence')
    axes[0].grid(True, alpha=0.3)

    positive = residual[np.isfinite(residual) & (residual > 0)]
    if positive.size:
        axes[1].semilogy(step, residual, 'C1.-', lw=1.4, ms=3)
    else:
        axes[1].plot(step, residual, 'C1.-', lw=1.4, ms=3)
    axes[1].set_ylabel('Stagger residual')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(step, pcg_mech, 'C0.-', lw=1.2, ms=3, label='mechanics')
    axes[2].plot(step, pcg_pf, 'C2.-', lw=1.2, ms=3, label='phase-field')
    axes[2].set_xlabel('Load step')
    axes[2].set_ylabel('Linear iters')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best')

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_deformed_stress(mesh, u, d, sxx, syy, sxy, nu=0.3,
                         disp_scale=5.0, d_threshold=0.05,
                         title='Hydrostatic Stress', save_path=None,
                         dpi=150):
    """Borden-style post-processed deformed mesh with hydrostatic stress.

    Matches Borden et al. (2012) Fig. 11 and Fig. 14:
    - Displacements scaled by disp_scale
    - Areas where d > (1 - d_threshold) removed
    - Colored by hydrostatic stress (pressure)

    Parameters
    ----------
    mesh : FEMMesh
    u : (N, 2) displacement
    d : (N,) damage
    sxx, syy, sxy : (E,) element stresses
    nu : float — Poisson's ratio (for σ_zz = ν(σ_xx + σ_yy))
    disp_scale : float — displacement magnification factor
    d_threshold : float — remove elements where d_avg > (1 - d_threshold)
    """
    nodes = mesh.nodes.cpu().numpy()
    elems = mesh.elements.cpu().numpy()
    u_np = u.detach().cpu().numpy() if torch.is_tensor(u) else u
    d_np = d.detach().cpu().numpy() if torch.is_tensor(d) else d

    # Deformed coordinates
    x_def = nodes[:, 0] + disp_scale * u_np[:, 0]
    y_def = nodes[:, 1] + disp_scale * u_np[:, 1]

    # Element-averaged damage — mask broken elements
    d_elem = d_np[elems].mean(axis=1)
    keep = d_elem < (1.0 - d_threshold)
    elems_keep = elems[keep]

    # Hydrostatic stress: (σ_xx + σ_yy + σ_zz) / 3
    # Plane strain: σ_zz = ν(σ_xx + σ_yy)
    _sxx = sxx.detach().cpu().numpy() if torch.is_tensor(sxx) else sxx
    _syy = syy.detach().cpu().numpy() if torch.is_tensor(syy) else syy
    sigma_h = (_sxx + _syy) * (1.0 + nu) / 3.0
    sigma_h_keep = sigma_h[keep]

    # Project element stress to nodes for plotting
    sigma_h_nodal = np.zeros(len(nodes))
    count = np.zeros(len(nodes))
    for i, el in enumerate(elems_keep):
        for n in el:
            sigma_h_nodal[n] += sigma_h_keep[i]
            count[n] += 1
    mask_valid = count > 0
    sigma_h_nodal[mask_valid] /= count[mask_valid]

    tri = mtri.Triangulation(x_def, y_def, elems_keep)

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.abs(sigma_h_nodal[mask_valid]).max() if mask_valid.any() else 1.0
    tcf = ax.tricontourf(tri, sigma_h_nodal, levels=64,
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(tcf, ax=ax, label='Hydrostatic Stress [MPa]', shrink=0.8)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x (mm)', fontsize=9)
    ax.set_ylabel('y (mm)', fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig, ax


def save_damage_snapshot(mesh, d, t_us, save_path, extra_info=None, dpi=150):
    """Save a damage field snapshot at a specific time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_field(mesh, d, title=f'Damage at t={t_us:.0f} µs',
               cmap='inferno', vmin=0, vmax=1, ax=ax)
    if extra_info:
        ax.text(0.02, 0.02, extra_info, transform=ax.transAxes,
                fontsize=7, va='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_damage_stress_strain(
    mesh, d, stress_vm, strain_vm, step: int = 0,
    save_path: str = None, dpi: int = 150,
    figsize: Tuple[float, float] = (13, 4.5),
) -> plt.Figure:
    """Side-by-side tricontour: damage | von Mises stress | von Mises strain.

    Parameters
    ----------
    mesh : FEMMesh
    d : (N,) damage
    stress_vm : (E,) von Mises stress at elements (projected to nodes internally)
    strain_vm : (E,) von Mises strain at elements (projected to nodes internally)
    step : int
    save_path : str or None
    dpi : int
    figsize : tuple — fixed figure size for consistent GIF frames

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Project element fields to nodes
    svm_nodal = mesh.elem_to_node(stress_vm) if stress_vm is not None else None
    evm_nodal = mesh.elem_to_node(strain_vm) if strain_vm is not None else None

    # Damage
    plot_field(mesh, d, title=f'Damage d (step {step})',
               cmap='inferno', vmin=0, vmax=1, ax=axes[0])

    # Von Mises stress
    if svm_nodal is not None:
        plot_field(mesh, svm_nodal, title=f'von Mises stress (step {step})',
                   cmap='jet', ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                     transform=axes[1].transAxes)
        axes[1].set_title(f'von Mises stress (step {step})', fontsize=10)

    # Von Mises strain
    if evm_nodal is not None:
        plot_field(mesh, evm_nodal, title=f'von Mises strain (step {step})',
                   cmap='viridis', ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, 'N/A', ha='center', va='center',
                     transform=axes[2].transAxes)
        axes[2].set_title(f'von Mises strain (step {step})', fontsize=10)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08,
                        wspace=0.30)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
    return fig


def compute_von_mises_stress(sxx, syy, sxy, nu=0.3):
    """Von Mises stress for plane strain (includes sigma_zz = nu*(sxx+syy)).

    sigma_vm = sqrt(sxx^2 + syy^2 + szz^2 - sxx*syy - sxx*szz - syy*szz + 3*sxy^2)
    """
    szz = nu * (sxx + syy)
    return torch.sqrt(sxx**2 + syy**2 + szz**2 - sxx*syy - sxx*szz - syy*szz + 3*sxy**2 + 1e-30)


def compute_von_mises_strain(exx, eyy, gxy):
    """Von Mises equivalent strain from Voigt components.

    eps_vm = sqrt(2/3) * sqrt(exx^2 + eyy^2 + exx*eyy + gxy^2/4) * 2/sqrt(3)
    Simplified for plane strain: eps_vm = (2/3)*sqrt(exx^2+eyy^2-exx*eyy + 3*(gxy/2)^2)
    """
    exy = gxy / 2.0
    return math.sqrt(2.0 / 3.0) * torch.sqrt(
        exx ** 2 + eyy ** 2 - exx * eyy + 3 * exy ** 2 + 1e-30)


def _hydrostatic_stress_ps(s, e, nu=0.3):
    """Plane-strain hydrostatic stress including sigma_zz."""
    return (s[0] + s[1]) * (1.0 + nu) / 3.0


def _stress_triaxiality_ps(s, e, nu=0.3):
    """Plane-strain stress triaxiality."""
    szz = nu * (s[0] + s[1])
    vm = torch.sqrt(s[0]**2 + s[1]**2 + szz**2 - s[0]*s[1] - s[0]*szz - s[1]*szz + 3*s[2]**2 + 1e-30)
    sh = (s[0] + s[1] + szz) / 3.0
    return sh / (vm + 1e-30)


# Principal stress/strain: canonical implementations in fem_operators.py
from ..core.fem_operators import FEMOperators
compute_principal_stress = FEMOperators.compute_principal_stress
compute_principal_strain = FEMOperators.compute_principal_strain


# Registry of derived field computations from raw (sxx, syy, sxy, exx, eyy, gxy).
# Lambdas accept (stress_tuple, strain_tuple, **kwargs) where kwargs may include nu.
FIELD_REGISTRY = {
    'von_mises_stress': {
        'compute': lambda s, e, **kw: compute_von_mises_stress(s[0], s[1], s[2], nu=kw.get('nu', 0.3)),
        'cmap': 'jet', 'label': 'von Mises Stress', 'type': 'element',
    },
    'von_mises_strain': {
        'compute': lambda s, e, **kw: compute_von_mises_strain(e[0], e[1], e[2]),
        'cmap': 'viridis', 'label': 'von Mises Strain', 'type': 'element',
    },
    'max_principal_stress': {
        'compute': lambda s, e, **kw: compute_principal_stress(s[0], s[1], s[2])[0],
        'cmap': 'RdBu_r', 'label': 'Max Principal Stress', 'type': 'element',
    },
    'min_principal_stress': {
        'compute': lambda s, e, **kw: compute_principal_stress(s[0], s[1], s[2])[1],
        'cmap': 'RdBu_r', 'label': 'Min Principal Stress', 'type': 'element',
    },
    'max_principal_strain': {
        'compute': lambda s, e, **kw: compute_principal_strain(e[0], e[1], e[2])[0],
        'cmap': 'plasma', 'label': 'Max Principal Strain', 'type': 'element',
    },
    'min_principal_strain': {
        'compute': lambda s, e, **kw: compute_principal_strain(e[0], e[1], e[2])[1],
        'cmap': 'plasma', 'label': 'Min Principal Strain', 'type': 'element',
    },
    'hydrostatic_stress': {
        'compute': lambda s, e, **kw: _hydrostatic_stress_ps(s, e, nu=kw.get('nu', 0.3)),
        'cmap': 'coolwarm', 'label': 'Hydrostatic Stress', 'type': 'element',
    },
    'stress_triaxiality': {
        'compute': lambda s, e, **kw: _stress_triaxiality_ps(s, e, nu=kw.get('nu', 0.3)),
        'cmap': 'PiYG', 'label': 'Stress Triaxiality', 'type': 'element',
    },
    'strain_xx': {
        'compute': lambda s, e, **kw: e[0],
        'cmap': 'RdBu_r', 'label': r'$\varepsilon_{xx}$', 'type': 'element',
    },
    'strain_yy': {
        'compute': lambda s, e, **kw: e[1],
        'cmap': 'RdBu_r', 'label': r'$\varepsilon_{yy}$', 'type': 'element',
    },
    'strain_xy': {
        'compute': lambda s, e, **kw: e[2],
        'cmap': 'RdBu_r', 'label': r'$\gamma_{xy}$', 'type': 'element',
    },
    'stress_xx': {
        'compute': lambda s, e, **kw: s[0],
        'cmap': 'RdBu_r', 'label': r'$\sigma_{xx}$', 'type': 'element',
    },
    'stress_yy': {
        'compute': lambda s, e, **kw: s[1],
        'cmap': 'RdBu_r', 'label': r'$\sigma_{yy}$', 'type': 'element',
    },
    'stress_xy': {
        'compute': lambda s, e, **kw: s[2],
        'cmap': 'RdBu_r', 'label': r'$\sigma_{xy}$', 'type': 'element',
    },
}


def compute_field(field_name, stress_tuple, strain_tuple, mesh=None,
                  d=None, H=None, nu=0.3):
    """Compute a named field from raw stress/strain data.

    Parameters
    ----------
    field_name : str — key from FIELD_REGISTRY, or 'damage', 'H', 'displacement_mag'
    stress_tuple : (sxx, syy, sxy) each (E,)
    strain_tuple : (exx, eyy, gxy) each (E,)
    mesh : FEMMesh — needed for elem_to_node projection
    d : (N,) damage — needed for 'damage' field
    H : (N,) history variable — needed for 'H' field

    Returns
    -------
    field : (N,) nodal values (element fields projected to nodes)
    label : str — display label
    cmap : str — colormap name
    """
    if field_name == 'damage':
        return d, 'Damage', 'inferno'
    if field_name == 'H':
        return H, 'History Variable H', 'inferno'

    if field_name not in FIELD_REGISTRY:
        raise ValueError(
            f"Unknown field '{field_name}'. Available: "
            f"{list(FIELD_REGISTRY.keys()) + ['damage', 'H']}")

    entry = FIELD_REGISTRY[field_name]
    val = entry['compute'](stress_tuple, strain_tuple, nu=nu)

    if entry['type'] == 'element' and mesh is not None:
        val = mesh.elem_to_node(val)

    return val, entry['label'], entry['cmap']


def _draw_bc_annotations(ax, mesh, bcs, nodes, xmin, xmax, ymin, ymax):
    """Draw boundary condition arrows and fixed-support indicators on the mesh.

    Uses LaTeX-rendered labels (u_x, u_y) placed outside the domain to avoid
    overlapping with mesh geometry.
    """
    bc_mask, bc_vals = bcs.get_masks_and_values()
    bc_mask_np = bc_mask.cpu().numpy()
    bc_vals_np = bc_vals.cpu().numpy()
    span = max(xmax - xmin, ymax - ymin)
    arrow_len = 0.10 * span
    label_offset = 0.04 * span  # gap between arrowhead and label

    def _draw_arrows(boundary_name, color):
        """Draw displacement arrows for a boundary, placed outside the domain."""
        if boundary_name not in mesh.node_sets:
            return
        idx = mesh.node_sets[boundary_name].cpu().numpy()
        if len(idx) == 0:
            return

        bnd_nodes = nodes[idx]
        cx, cy = bnd_nodes[:, 0].mean(), bnd_nodes[:, 1].mean()

        for comp, comp_label in [(0, r'$u_x$'), (1, r'$u_y$')]:
            n_with_bc = bc_mask_np[idx, comp].sum()
            if n_with_bc < max(3, len(idx) * 0.5):
                continue  # skip if only corner nodes have this BC
            # Use median non-zero value (robust to corner outliers)
            bc_vals_bnd = bc_vals_np[idx, comp]
            nonzero = bc_vals_bnd[np.abs(bc_vals_bnd) > 1e-10]
            if len(nonzero) == 0:
                continue  # fixed (zero) — drawn as support, not arrow
            val = np.median(nonzero)

            if comp == 1:  # vertical displacement
                # Arrow points in direction of applied displacement
                sign = 1.0 if val > 0 else -1.0
                # Place arrow outside domain
                if boundary_name == 'top':
                    y_base = cy + 0.02 * span
                else:
                    y_base = cy - 0.02 * span
                ax.annotate('', xy=(cx, y_base + sign * arrow_len),
                            xytext=(cx, y_base),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=2.0, mutation_scale=15))
                ax.text(cx + label_offset, y_base + sign * arrow_len * 0.5,
                        f'{comp_label} = {val:.1e}',
                        ha='left', va='center', fontsize=9, color=color,
                        fontweight='semibold')
            else:  # horizontal displacement
                # Arrow points in direction of applied displacement
                sign = 1.0 if val > 0 else -1.0
                if boundary_name == 'right':
                    x_base = cx + 0.02 * span
                elif boundary_name == 'left':
                    x_base = cx - 0.02 * span
                else:
                    x_base = cx
                ax.annotate('', xy=(x_base + sign * arrow_len, cy),
                            xytext=(x_base, cy),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=2.0, mutation_scale=15))
                ax.text(x_base + sign * arrow_len * 0.5,
                        cy + label_offset,
                        f'{comp_label} = {val:.1e}',
                        ha='center', va='bottom', fontsize=9, color=color,
                        fontweight='semibold')

    boundary_colors = {
        'top': '#CC2222', 'bottom': '#2255CC',
        'left': '#228B22', 'right': '#CC7700',
    }
    for bname, bcolor in boundary_colors.items():
        _draw_arrows(bname, bcolor)

    # Fixed supports: hatching bars for pinned/roller boundaries
    hatch_w = 0.025 * span  # hatch bar width
    for bname in ['bottom', 'top', 'left', 'right']:
        if bname not in mesh.node_sets:
            continue
        idx = mesh.node_sets[bname].cpu().numpy()
        if len(idx) == 0:
            continue

        x_fixed = (bc_mask_np[idx, 0].all()
                    and np.abs(bc_vals_np[idx[0], 0]) < 1e-10)
        y_fixed = (bc_mask_np[idx, 1].all()
                    and np.abs(bc_vals_np[idx[0], 1]) < 1e-10)
        if not x_fixed and not y_fixed:
            continue

        bcolor = boundary_colors.get(bname, 'gray')
        is_pinned = x_fixed and y_fixed  # fully fixed

        if bname == 'bottom':
            rect = plt.Rectangle((xmin, ymin - hatch_w), xmax - xmin, hatch_w,
                                  linewidth=0, facecolor=bcolor, alpha=0.15,
                                  hatch='///' if is_pinned else '...')
            ax.add_patch(rect)
            ax.plot([xmin, xmax], [ymin, ymin], color=bcolor, lw=2, alpha=0.6)
            label = 'fixed' if is_pinned else 'roller'
            ax.text(xmax + label_offset, ymin, label,
                    fontsize=7, color=bcolor, va='center', style='italic')
        elif bname == 'top':
            rect = plt.Rectangle((xmin, ymax), xmax - xmin, hatch_w,
                                  linewidth=0, facecolor=bcolor, alpha=0.15,
                                  hatch='///' if is_pinned else '...')
            ax.add_patch(rect)
            ax.plot([xmin, xmax], [ymax, ymax], color=bcolor, lw=2, alpha=0.6)
            label = 'fixed' if is_pinned else 'roller'
            ax.text(xmax + label_offset, ymax, label,
                    fontsize=7, color=bcolor, va='center', style='italic')
        elif bname == 'left':
            rect = plt.Rectangle((xmin - hatch_w, ymin), hatch_w, ymax - ymin,
                                  linewidth=0, facecolor=bcolor, alpha=0.15,
                                  hatch='///' if is_pinned else '...')
            ax.add_patch(rect)
            ax.plot([xmin, xmin], [ymin, ymax], color=bcolor, lw=2, alpha=0.6)
        elif bname == 'right':
            rect = plt.Rectangle((xmax, ymin), hatch_w, ymax - ymin,
                                  linewidth=0, facecolor=bcolor, alpha=0.15,
                                  hatch='///' if is_pinned else '...')
            ax.add_patch(rect)
            ax.plot([xmax, xmax], [ymin, ymax], color=bcolor, lw=2, alpha=0.6)


def plot_initial_conditions(
    mesh, material, bcs, config,
    save_path: str = 'initial_conditions.png', dpi: int = 150,
    problem_info: dict = None,
) -> plt.Figure:
    """Summary PNG of the problem setup: geometry, BCs, material, solver info.

    Parameters
    ----------
    mesh : FEMMesh
    material : Material
    bcs : BoundaryConditions
    config : SolverConfig
    save_path : str
    dpi : int
    problem_info : dict, optional
        Extra info to display. Supported keys:
        - 'name': str — problem/benchmark name
        - 'reference': str — paper reference
        - 'n_steps': int — actual number of steps (overrides config.num_steps)
        - 'dt': float — timestep [s]
        - 'loading': str — loading description (e.g. "Velocity ramp 16.5 m/s, 1µs rise")
        - 'bc_description': list[str] — detailed BC descriptions
        - 'wave_speeds': dict — {'c_P': ..., 'c_S': ..., 'c_R': ...} in m/s
        - 'notes': list[str] — extra notes to display

    Returns
    -------
    fig : matplotlib Figure
    """
    if problem_info is None:
        problem_info = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={'width_ratios': [1.3, 1]})

    # ---- Left panel: mesh with BC annotations ----
    ax = axes[0]
    tri = _build_triangulation(mesh)
    ax.triplot(tri, color='#888888', linewidth=0.12, alpha=0.45)
    ax.set_aspect('equal')
    ax.set_title('Geometry & Boundary Conditions', fontsize=12,
                 fontweight='semibold', pad=10)
    ax.set_xlabel('x [mm]', fontsize=9)
    ax.set_ylabel('y [mm]', fontsize=9)

    nodes = mesh.nodes.cpu().numpy()
    xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
    ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()

    # Boundary node dots (small, semi-transparent)
    bnd_colors = {
        'bottom': '#2255CC', 'top': '#CC2222',
        'left': '#228B22', 'right': '#CC7700',
    }
    for name, idx in mesh.node_sets.items():
        if name in bnd_colors:
            pts = nodes[idx.cpu().numpy()]
            ax.scatter(pts[:, 0], pts[:, 1], c=bnd_colors[name], s=2,
                       alpha=0.6, zorder=5)

    # BC arrows and supports
    _draw_bc_annotations(ax, mesh, bcs, nodes, xmin, xmax, ymin, ymax)

    # Padding
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.08 * span
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - 1.5 * pad, ymax + 1.5 * pad)
    ax.tick_params(labelsize=7)

    # ---- Right panel: clean text summary (no box) ----
    ax2 = axes[1]
    ax2.axis('off')

    # Build sections with clear hierarchy
    y = 0.96
    gap_section = 0.04   # gap between sections
    gap_line = 0.032     # gap between lines

    def _heading(text, y_pos):
        ax2.text(0.02, y_pos, text, transform=ax2.transAxes,
                 fontsize=11, fontweight='bold', color='#333333',
                 fontfamily='sans-serif')
        # Subtle underline
        ax2.plot([0.02, 0.95], [y_pos - 0.008, y_pos - 0.008],
                 transform=ax2.transAxes, color='#CCCCCC', lw=0.8,
                 clip_on=False)
        return y_pos - gap_line - 0.01

    def _row(label, value, y_pos, indent=0.04):
        ax2.text(indent, y_pos, label, transform=ax2.transAxes,
                 fontsize=9, color='#666666', fontfamily='sans-serif')
        ax2.text(0.40, y_pos, str(value), transform=ax2.transAxes,
                 fontsize=9, color='#222222', fontfamily='monospace',
                 fontweight='medium')
        return y_pos - gap_line

    # -- Mesh --
    y = _heading('Mesh', y)
    y = _row('Nodes', f'{mesh.n_nodes:,}', y)
    y = _row('Elements', f'{mesh.n_elems:,}', y)
    y = _row('h_min', f'{mesh.h_min:.6f} mm', y)
    y = _row('Domain',
             f'[{xmin:.2f}, {xmax:.2f}] \u00d7 [{ymin:.2f}, {ymax:.2f}]', y)

    # -- Material --
    y -= gap_section
    y = _heading('Material', y)
    y = _row('E', f'{material.E:.0f} MPa', y)
    y = _row('\u03bd', f'{material.nu}', y)
    y = _row('G_c', f'{material.Gc} N/mm', y)
    y = _row('\u2113\u2080', f'{material.l0} mm', y)
    y = _row('G_c / \u2113\u2080', f'{material.Gc_over_l0:.1f}', y)
    y = _row('Split', material.energy_split, y)
    y = _row('Model', material.pf_model, y)

    # -- Solver --
    y -= gap_section
    y = _heading('Solver', y)
    y = _row('Type', config.solver_type, y)
    n_steps = problem_info.get('n_steps', config.num_steps)
    y = _row('Steps', f'{n_steps:,}', y)
    if problem_info.get('dt'):
        dt = problem_info['dt']
        y = _row('dt', f'{dt:.4e} s ({dt*1e6:.4f} µs)', y)
    if hasattr(config, 'damage_tol') and config.damage_tol:
        y = _row('Damage tol', f'{config.damage_tol:.1e}', y)

    # -- Wave speeds (if provided) --
    ws = problem_info.get('wave_speeds')
    if ws:
        y -= gap_section
        y = _heading('Wave Speeds', y)
        if 'c_P' in ws:
            y = _row('c_P', f'{ws["c_P"]:.0f} m/s', y)
        if 'c_S' in ws:
            y = _row('c_S', f'{ws["c_S"]:.0f} m/s', y)
        if 'c_R' in ws:
            y = _row('c_R', f'{ws["c_R"]:.0f} m/s', y)

    # -- BCs --
    y -= gap_section
    y = _heading('Boundary Conditions', y)
    bc_desc = problem_info.get('bc_description')
    if bc_desc:
        for desc in bc_desc:
            ax2.text(0.04, y, desc, transform=ax2.transAxes,
                     fontsize=8.5, color='#333333', fontfamily='sans-serif',
                     wrap=True)
            y -= gap_line
    else:
        for bc in bcs.bcs:
            comp = 'x' if bc.component == 0 else 'y'
            y = _row(f'u_{comp}',
                     f'{bc.value:.4f}  ({len(bc.node_indices)} nodes)', y)

    # -- Loading description (if provided) --
    loading = problem_info.get('loading')
    if loading:
        y -= gap_section
        y = _heading('Loading', y)
        ax2.text(0.04, y, loading, transform=ax2.transAxes,
                 fontsize=9, color='#0055AA', fontfamily='sans-serif',
                 fontweight='bold', wrap=True)
        y -= gap_line

    # -- Notes (if provided) --
    notes = problem_info.get('notes')
    if notes:
        y -= gap_section * 0.5
        for note in notes:
            ax2.text(0.04, y, f'• {note}', transform=ax2.transAxes,
                     fontsize=8, color='#666666', fontfamily='sans-serif',
                     style='italic', wrap=True)
            y -= gap_line

    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.08,
                        wspace=0.05)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    return fig


def plot_final_state(
    mesh, solver, history: List[dict],
    save_path: str = 'final_state.png', dpi: int = 150,
) -> plt.Figure:
    """Summary PNG of the final simulation state.

    4-panel: damage | displacement magnitude | H field | load-displacement curve.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    u = solver.u
    d = solver.d
    H = solver.H_nodal

    # (0,0) Damage
    plot_field(mesh, d, title='Final Damage Field', cmap='inferno',
               vmin=0, vmax=1, ax=axes[0, 0])

    # (0,1) Displacement magnitude
    u_mag = torch.norm(u, dim=1)
    plot_field(mesh, u_mag, title='Displacement Magnitude |u|',
               cmap='coolwarm', ax=axes[0, 1])

    # (1,0) History variable H
    plot_field(mesh, H, title='History Variable H', cmap='inferno',
               ax=axes[1, 0])

    # (1,1) History curves
    ax = axes[1, 1]
    steps = [r['step'] for r in history]
    max_d = [r['max_d'] for r in history]
    max_H = [r['max_H'] for r in history]

    ax.plot(steps, max_d, 'r-', linewidth=1.5, label='max(d)')
    ax.set_xlabel('Step', fontsize=9)
    ax.set_ylabel('max(d)', color='red', fontsize=9)
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_ylim(0, 1.05)

    ax2 = ax.twinx()
    ax2.plot(steps, max_H, 'b-', linewidth=1.5, label='max(H)')
    ax2.set_ylabel('max(H)', color='blue', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='blue')

    ax.set_title('Damage & History Evolution', fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    fig.suptitle(f'Final State (step {len(history)})', fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


class GIFRecorder:
    """Records simulation frames in memory and generates an animated GIF.

    Frames are rendered to in-memory PIL Images (no intermediate PNGs on disk).

    Supports configurable fields via the ``fields`` parameter. Each field
    is rendered as a panel in the GIF frame. Default: damage + von Mises
    stress + von Mises strain (backward-compatible).

    Usage::

        # Default 3-panel (damage | von Mises stress | von Mises strain)
        recorder = GIFRecorder(mesh)

        # Custom fields
        recorder = GIFRecorder(mesh, fields=['damage', 'max_principal_stress', 'H'])

        for step in range(num_steps):
            psi = solver.step_full()
            if step % record_every == 0:
                exx, eyy, gxy = fem.compute_strain(solver.u)
                sxx, syy, sxy = ...  # compute stress
                recorder.add_frame(step, solver.d, sxx, syy, sxy, exx, eyy, gxy)
        recorder.save_gif('simulation.gif', fps=10)

    Available fields:
        damage, H, von_mises_stress, von_mises_strain,
        max_principal_stress, min_principal_stress,
        max_principal_strain, min_principal_strain,
        hydrostatic_stress, stress_triaxiality,
        strain_xx, strain_yy, strain_xy,
        stress_xx, stress_yy, stress_xy
    """

    def __init__(self, mesh, output_dir: str = None, fields=None):
        self.mesh = mesh
        self._frames = []  # PIL Images kept in memory (no disk I/O)
        self._saved = False
        self._output_dir = output_dir
        self.fields = fields or ['damage', 'von_mises_stress', 'von_mises_strain']

        # Cache triangulation once — avoids rebuild per panel per frame.
        self._tri = _build_triangulation(mesh)
        # Reusable figure/axes/mappables — set on first add_frame, updated after.
        self._fig = None
        self._axes = None
        self._mappables = None  # one per panel
        self._title_artists = None

        from PIL import Image
        import io as _io
        self._Image = Image
        self._io = _io

        import atexit
        import weakref
        # Use weak reference so GIF recorder can be garbage collected
        _self_ref = weakref.ref(self)
        def _atexit_handler():
            obj = _self_ref()
            if obj is not None:
                obj._atexit_save()
        atexit.register(_atexit_handler)

    def add_frame(self, step: int, d, sxx, syy, sxy, exx, eyy, gxy,
                  H=None, dpi: int = 100):
        """Capture one frame with configured field panels.

        Uses a cached Triangulation and a reused Figure with `tripcolor`
        mappables updated via `set_array` — ~5-10× faster than rebuilding
        `tricontourf` per frame while visually equivalent at GIF dpi.
        """
        n_panels = len(self.fields)
        stress = (sxx, syy, sxy)
        strain = (exx, eyy, gxy)

        # Precompute field values for this frame.
        field_vals = []
        for fname in self.fields:
            fval, label, cmap = compute_field(
                fname, stress, strain, mesh=self.mesh, d=d, H=H)
            fv = fval.detach().cpu().numpy() if torch.is_tensor(fval) else fval
            fv = np.nan_to_num(fv, nan=0.0, posinf=1.0, neginf=0.0)
            field_vals.append((fv, label, cmap, fname))

        if self._fig is None:
            fig_w = min(6 * n_panels, 13)
            self._fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5))
            if n_panels == 1:
                axes = [axes]
            self._axes = list(axes)
            self._mappables = []
            self._title_artists = []
            for ax, (fv, label, cmap, fname) in zip(self._axes, field_vals):
                vmin = 0.0 if fname == 'damage' else float(fv.min())
                vmax = 1.0 if fname == 'damage' else float(fv.max())
                if vmin == vmax:
                    vmax = vmin + 1e-10
                tpc = ax.tripcolor(self._tri, fv, shading='gouraud',
                                    cmap=cmap, vmin=vmin, vmax=vmax)
                self._mappables.append(tpc)
                self._fig.colorbar(tpc, ax=ax, shrink=0.8)
                ax.set_aspect('equal')
                ax.set_xlabel('x (mm)', fontsize=8)
                ax.set_ylabel('y (mm)', fontsize=8)
                ax.tick_params(labelsize=7)
                self._title_artists.append(
                    ax.set_title(f'{label} (step {step})', fontsize=10))
            self._fig.subplots_adjust(left=0.04, right=0.96,
                                       top=0.92, bottom=0.08, wspace=0.30)
        else:
            for tpc, title_art, (fv, label, _, fname) in zip(
                    self._mappables, self._title_artists, field_vals):
                tpc.set_array(fv)
                if fname != 'damage':
                    tpc.set_clim(float(fv.min()), float(fv.max()))
                title_art.set_text(f'{label} (step {step})')

        buf = self._io.BytesIO()
        self._fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        self._frames.append(self._Image.open(buf).copy())
        buf.close()

    def save_gif(self, output_path: str = 'simulation.gif', fps: int = 10,
                 loop: int = 0):
        """Assemble recorded frames into an animation.

        Format is chosen by file extension:
          .mp4  — H.264 video (10-50× smaller than GIF, 24-bit, GitHub-friendly)
          .apng — animated PNG (5× smaller than GIF, 24-bit, no ffmpeg needed)
          .gif  — classic GIF (256 colours, universal)

        Parameters
        ----------
        output_path : str — .gif, .apng, or .mp4
        fps : int — frames per second
        loop : int — 0 = loop forever (GIF/APNG only)
        """
        if not self._frames:
            print("No frames recorded.")
            return

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Ensure uniform size
        target_size = self._frames[0].size
        for i in range(1, len(self._frames)):
            if self._frames[i].size != target_size:
                self._frames[i] = self._frames[i].resize(
                    target_size, self._Image.LANCZOS)

        ext = os.path.splitext(output_path)[1].lower()
        duration = int(1000 / fps)

        if ext == '.mp4':
            output_path = self._save_mp4(output_path, fps)
            fmt = 'MP4' if output_path.endswith('.mp4') else 'GIF'
        elif ext in ('.apng', '.png'):
            self._frames[0].save(
                output_path, format='PNG',
                save_all=True,
                append_images=self._frames[1:],
                duration=duration, loop=loop,
            )
            fmt = 'APNG'
        else:
            # Palette-reduce every frame to 128 ADAPTIVE colours before
            # saving, then optimize. For smooth scalar fields (damage,
            # stress) this keeps the visual quality while cutting file
            # size by 3-5x; target band is 1-5 MB.
            palette_frames = [
                f.convert('P', palette=self._Image.ADAPTIVE, colors=128)
                if f.mode != 'P' else f
                for f in self._frames
            ]
            palette_frames[0].save(
                output_path,
                save_all=True,
                append_images=palette_frames[1:],
                duration=duration, loop=loop,
                optimize=True, disposal=2,
            )
            fmt = 'GIF'

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        self._saved = True
        print(f"{fmt} saved: {output_path} ({len(self._frames)} frames, "
              f"{fps} fps, {size_mb:.1f} MB)")
        if fmt == 'MP4':
            print(f"  Tip: use .gif or .apng extension for alternative formats")

    def _save_mp4(self, output_path: str, fps: int) -> str:
        """Write frames to MP4 by piping rendered RGB frames to ffmpeg.

        Falls back to GIF if ffmpeg is not available.
        Returns the actual path written (may be .gif on fallback).
        """
        import subprocess
        import shutil

        if shutil.which('ffmpeg') is None:
            # Fall back to GIF with correct extension (don't rename to .mp4)
            gif_path = output_path.rsplit('.', 1)[0] + '.gif'
            print(f"[GIFRecorder] ffmpeg not found, falling back to GIF: {gif_path}")
            self._frames[0].save(
                gif_path, save_all=True,
                append_images=self._frames[1:],
                duration=int(1000 / fps), loop=0,
            )
            return gif_path

        w, h = self._frames[0].size
        # yuv420p requires even dimensions
        w = w if w % 2 == 0 else w + 1
        h = h if h % 2 == 0 else h + 1

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path,
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            try:
                for frame in self._frames:
                    arr = np.asarray(frame.resize((w, h)).convert('RGB'))
                    proc.stdin.write(arr.tobytes())
                proc.stdin.close()
                stderr = proc.stderr.read()
                proc.wait()
            except BrokenPipeError:
                stderr = proc.stderr.read()
                proc.wait()
            if proc.returncode != 0:
                msg = stderr.decode('utf-8', errors='replace').splitlines()
                detail = msg[-1] if msg else f'return code {proc.returncode}'
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, stderr=detail)
            return output_path
        except (subprocess.CalledProcessError, BrokenPipeError, OSError) as e:
            print(f"[GIFRecorder] ffmpeg failed ({e}), falling back to GIF")
            gif_path = output_path.rsplit('.', 1)[0] + '.gif'
            self._frames[0].save(
                gif_path, save_all=True,
                append_images=self._frames[1:],
                duration=int(1000 / fps), loop=0,
            )
            return gif_path

    def _atexit_save(self):
        """Auto-save partial GIF on process exit if save_gif() wasn't called."""
        if self._saved or not self._frames:
            return
        try:
            fallback = os.path.join(self._output_dir or '.',
                                     'damage_evolution_partial.gif')
            print(f"\n[GIFRecorder] Process exiting — saving {len(self._frames)} "
                  f"frames to {fallback}")
            self.save_gif(fallback, fps=8)
        except Exception as e:
            print(f"[GIFRecorder] atexit save failed: {e}")

    def cleanup_frames(self):
        """Release frame memory. No-op if frames already cleared."""
        self._frames.clear()
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None
            self._mappables = None
            self._title_artists = None
