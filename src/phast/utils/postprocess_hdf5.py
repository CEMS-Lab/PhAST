"""
Offline post-processing for phast HDF5 output files.

Generates GIFs, plots, and field snapshots from previously completed
simulation runs — no re-simulation needed.

Usage (CLI)::

    # GIF of default fields (damage | von Mises stress | von Mises strain)
    python -m phast.postprocess_hdf5 path/to/training_data.h5

    # GIF of specific fields
    python -m phast.postprocess_hdf5 data.h5 --fields damage max_principal_stress H

    # Single snapshot PNG at a specific step
    python -m phast.postprocess_hdf5 data.h5 --snapshot 50 --fields damage stress_xx

    # Energy evolution plot
    python -m phast.postprocess_hdf5 data.h5 --energy_plot

    # List available steps and fields
    python -m phast.postprocess_hdf5 data.h5 --info

Usage (Python API)::

    from phast.postprocess_hdf5 import PostProcessor
    pp = PostProcessor('training_data.h5')
    pp.info()
    pp.make_gif(fields=['damage', 'max_principal_stress'], fps=15)
    pp.snapshot(step=50, fields=['damage', 'stress_xx'])
    pp.energy_plot()
"""

import torch
import numpy as np
import os
import argparse
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def _load_h5(h5_path: str):
    """Load HDF5 file, return h5py File handle."""
    import h5py
    return h5py.File(h5_path, 'r')


class PostProcessor:
    """Offline post-processor for phast H5 files.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file produced by phast.
    output_dir : str or None
        Output directory for generated files. Defaults to same dir as h5_path.
    """

    def __init__(self, h5_path: str, output_dir: str = None):
        self.h5_path = h5_path
        self.output_dir = output_dir or os.path.dirname(h5_path) or '.'
        os.makedirs(self.output_dir, exist_ok=True)

        self._h5 = _load_h5(h5_path)
        sim = self._h5['simulation_data']

        # Load mesh
        mesh_grp = sim['mesh']
        self.nodes = np.array(mesh_grp['node_coordinates'])
        self.elements = np.array(mesh_grp['element_connectivity'])
        self.n_nodes = len(self.nodes)
        self.n_elems = len(self.elements)

        # Triangulation for plotting
        elems_for_plot = self.elements
        if elems_for_plot.ndim == 2 and elems_for_plot.shape[1] == 4:
            elems_for_plot = np.vstack([
                elems_for_plot[:, [0, 1, 2]],
                elems_for_plot[:, [0, 2, 3]],
            ])
        self._tri = mtri.Triangulation(
            self.nodes[:, 0], self.nodes[:, 1], elems_for_plot)

        # Material metadata
        if 'metadata' in sim:
            meta = sim['metadata']
            self.material = {k: float(meta.attrs[k])
                             for k in ['E', 'nu', 'Gc', 'l0']
                             if k in meta.attrs}
            self.energy_split = str(meta.attrs.get('energy_split', 'unknown'))
            # H5 files written before v0.16.3 lack the plane_stress attr;
            # default to False (plane strain) for back-compat.
            self.plane_stress = bool(meta.attrs.get('plane_stress', False))
        else:
            self.material = {}
            self.energy_split = 'unknown'
            self.plane_stress = False

        # Discover available steps
        steps_grp = sim['steps']
        self._step_names = sorted(steps_grp.keys(),
                                  key=lambda s: int(s.split('_')[-1]))
        self._steps_grp = steps_grp

        # Pre-compute elem_to_node projection weights
        self._precompute_projection()

    def _precompute_projection(self):
        """Precompute area-weighted element-to-node projection."""
        # Compute element areas: 0.5 * |cross(v1-v0, v2-v0)|
        p = self.nodes[self.elements]  # (E, 3, 2)
        v1 = p[:, 1] - p[:, 0]
        v2 = p[:, 2] - p[:, 0]
        self._elem_areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

        # Area weight per node: sum of (area/3) over contributing elements
        self._node_area_weight = np.zeros(self.n_nodes, dtype=np.float64)
        area_third = self._elem_areas / 3.0
        for i in range(3):
            np.add.at(self._node_area_weight, self.elements[:, i], area_third)
        self._node_area_weight = np.maximum(self._node_area_weight, 1e-30)

    def _elem_to_node(self, elem_field: np.ndarray) -> np.ndarray:
        """Project element field to nodes via area-weighted averaging."""
        nodal = np.zeros(self.n_nodes, dtype=np.float64)
        weighted = elem_field * self._elem_areas / 3.0
        for i in range(3):
            np.add.at(nodal, self.elements[:, i], weighted)
        return nodal / self._node_area_weight

    @property
    def step_numbers(self) -> List[int]:
        """List of available step numbers."""
        return [int(s.split('_')[-1]) for s in self._step_names]

    def _get_step(self, step: int):
        """Get HDF5 group for a step number."""
        key = f'step_{step:04d}'
        if key not in self._steps_grp:
            raise KeyError(f"Step {step} not found. Available: {self.step_numbers}")
        return self._steps_grp[key]

    def _load_step_data(self, step: int) -> dict:
        """Load all fields for a given step."""
        grp = self._get_step(step)
        data = {}
        for key in grp.keys():
            data[key] = np.array(grp[key])
        for attr_name in ['reaction_force', 'applied_disp', 'energy_elastic',
                          'energy_fracture', 'energy_fracture_total']:
            if attr_name in grp.attrs:
                data[attr_name] = float(grp.attrs[attr_name])
        return data

    def _compute_stress(self, strain: np.ndarray) -> tuple:
        """Compute linear elastic stress from strain (undegraded).

        Dispatches on ``self.plane_stress`` so that runs performed under
        either plane assumption are plotted using the correct in-plane
        constitutive relation.

        Returns (sxx, syy, sxy) each (E,).
        """
        E = self.material.get('E', 210000.0)
        nu = self.material.get('nu', 0.3)
        if self.plane_stress:
            factor = E / (1.0 - nu ** 2)
            C00 = factor
            C01 = factor * nu
            C22 = factor * (1.0 - nu) / 2.0
        else:  # plane strain
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            C00 = factor * (1.0 - nu)
            C01 = factor * nu
            C22 = factor * (1.0 - 2.0 * nu) / 2.0

        exx, eyy, gxy = strain[:, 0], strain[:, 1], strain[:, 2]
        sxx = C00 * exx + C01 * eyy
        syy = C01 * exx + C00 * eyy
        sxy = C22 * gxy
        return sxx, syy, sxy

    def _compute_derived_field(self, field_name: str, data: dict) -> tuple:
        """Compute a derived field from step data.

        Returns (values_nodal, label, cmap).
        """
        if field_name == 'damage':
            return data['damage_nodal'], 'Damage', 'hot'

        if field_name == 'H' and 'H_nodal' in data:
            return data['H_nodal'], 'History Variable H', 'inferno'

        if field_name == 'H_elem' and 'H_elem' in data:
            return self._elem_to_node(data['H_elem']), 'H (element)', 'inferno'

        if field_name == 'psi_plus' and 'psi_plus' in data:
            return self._elem_to_node(data['psi_plus']), r'$\psi^+$', 'magma'

        if field_name == 'displacement_x' and 'displacement' in data:
            return data['displacement'][:, 0], r'$u_x$', 'coolwarm'

        if field_name == 'displacement_y' and 'displacement' in data:
            return data['displacement'][:, 1], r'$u_y$', 'coolwarm'

        if field_name == 'displacement_mag' and 'displacement' in data:
            u = data['displacement']
            return np.sqrt(u[:, 0]**2 + u[:, 1]**2), '|u|', 'coolwarm'

        # Fields requiring stress/strain
        if 'strain' not in data:
            raise ValueError(
                f"Field '{field_name}' requires strain data in H5. "
                f"Available datasets: {list(data.keys())}")

        strain = data['strain']
        exx, eyy, gxy = strain[:, 0], strain[:, 1], strain[:, 2]

        if 'stress' in data:
            stress = data['stress']
            sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
        else:
            sxx, syy, sxy = self._compute_stress(strain)

        nu = self.material.get('nu', 0.3)

        # Under plane stress σ_zz = 0 by assumption. Under plane strain
        # σ_zz = ν(σ_xx + σ_yy) from ε_zz = 0.
        if self.plane_stress:
            szz = np.zeros_like(sxx)
        else:
            szz = nu * (sxx + syy)
        exy = gxy / 2.0

        field_map = {
            'von_mises_stress': (
                np.sqrt(sxx**2 + syy**2 + szz**2 - sxx*syy - sxx*szz - syy*szz + 3*sxy**2 + 1e-30),
                'von Mises Stress', 'jet'),
            'von_mises_strain': (
                np.sqrt(2.0 / 3.0) * np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2 + 1e-30),
                'von Mises Strain', 'viridis'),
            'max_principal_stress': (
                (sxx+syy)/2 + np.sqrt(((sxx-syy)/2)**2 + sxy**2 + 1e-30),
                'Max Principal Stress', 'RdBu_r'),
            'min_principal_stress': (
                (sxx+syy)/2 - np.sqrt(((sxx-syy)/2)**2 + sxy**2 + 1e-30),
                'Min Principal Stress', 'RdBu_r'),
            'max_principal_strain': (
                (exx+eyy)/2 + np.sqrt(((exx-eyy)/2)**2 + (gxy/2)**2 + 1e-30),
                'Max Principal Strain', 'plasma'),
            'min_principal_strain': (
                (exx+eyy)/2 - np.sqrt(((exx-eyy)/2)**2 + (gxy/2)**2 + 1e-30),
                'Min Principal Strain', 'plasma'),
            'hydrostatic_stress': (
                (sxx + syy + nu * (sxx + syy)) / 3.0,
                'Hydrostatic Stress', 'coolwarm'),
            'stress_triaxiality': (
                (sxx + syy + nu * (sxx + syy)) / (3.0 * np.sqrt(
                    sxx**2 + syy**2 + (nu * (sxx + syy))**2
                    - sxx*syy - sxx * nu * (sxx + syy) - syy * nu * (sxx + syy)
                    + 3*sxy**2 + 1e-30) + 1e-30),
                'Stress Triaxiality', 'PiYG'),
            'strain_xx': (exx, r'$\varepsilon_{xx}$', 'RdBu_r'),
            'strain_yy': (eyy, r'$\varepsilon_{yy}$', 'RdBu_r'),
            'strain_xy': (gxy, r'$\gamma_{xy}$', 'RdBu_r'),
            'stress_xx': (sxx, r'$\sigma_{xx}$', 'RdBu_r'),
            'stress_yy': (syy, r'$\sigma_{yy}$', 'RdBu_r'),
            'stress_xy': (sxy, r'$\sigma_{xy}$', 'RdBu_r'),
        }

        if field_name not in field_map:
            raise ValueError(
                f"Unknown field '{field_name}'. Available: "
                f"{list(field_map.keys()) + ['damage', 'H', 'H_elem', 'psi_plus', 'displacement_x', 'displacement_y', 'displacement_mag']}")

        val, label, cmap = field_map[field_name]
        # Element fields need projection to nodes
        if len(val) == self.n_elems:
            val = self._elem_to_node(val)
        return val, label, cmap

    def info(self):
        """Print summary of the H5 file contents."""
        steps = self.step_numbers
        print(f"H5 file: {self.h5_path}")
        print(f"  Nodes: {self.n_nodes}, Elements: {self.n_elems}")
        print(f"  Material: {self.material}")
        print(f"  Energy split: {self.energy_split}")
        print(f"  Steps: {len(steps)} ({min(steps)} - {max(steps)})")

        # Check what datasets are in the first step
        first = self._get_step(steps[0])
        print(f"  Datasets per step: {list(first.keys())}")
        print(f"  Attrs per step: {list(first.attrs.keys())}")

    def _plot_field(self, ax, values, title, cmap, vmin=None, vmax=None):
        """Plot a nodal field on the triangulation."""
        tcf = ax.tricontourf(self._tri, values, levels=64, cmap=cmap,
                             vmin=vmin, vmax=vmax)
        plt.colorbar(tcf, ax=ax, shrink=0.8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (mm)', fontsize=8)
        ax.set_ylabel('y (mm)', fontsize=8)
        ax.tick_params(labelsize=7)

    def snapshot(self, step: int, fields: List[str] = None,
                 save_path: str = None, dpi: int = 150):
        """Generate a multi-panel PNG for a single step.

        Parameters
        ----------
        step : int
        fields : list of str — field names to plot
        save_path : str or None — auto-generated if None
        dpi : int
        """
        if fields is None:
            fields = ['damage', 'von_mises_stress', 'von_mises_strain']

        data = self._load_step_data(step)
        n = len(fields)
        fig_w = min(6 * n, 13)
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 5))
        if n == 1:
            axes = [axes]

        for ax, fname in zip(axes, fields):
            val, label, cmap = self._compute_derived_field(fname, data)
            vmin = 0.0 if fname == 'damage' else None
            vmax = 1.0 if fname == 'damage' else None
            self._plot_field(ax, val, f'{label} (step {step})',
                             cmap, vmin=vmin, vmax=vmax)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08,
                            wspace=0.30)
        if save_path is None:
            save_path = os.path.join(
                self.output_dir, f'snapshot_step{step:04d}.png')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Snapshot saved: {save_path}")

    def make_gif(self, fields: List[str] = None, fps: int = 10,
                 save_path: str = None, max_frames: int = 200,
                 dpi: int = 120):
        """Generate an animated GIF from all steps.

        Parameters
        ----------
        fields : list of str
        fps : int
        save_path : str or None
        max_frames : int — cap the number of frames (skip steps to fit)
        dpi : int
        """
        from PIL import Image
        import io

        if fields is None:
            fields = ['damage', 'von_mises_stress', 'von_mises_strain']

        steps = self.step_numbers
        skip = max(1, len(steps) // max_frames)
        selected = steps[::skip]

        print(f"Generating GIF: {len(selected)} frames, "
              f"{len(fields)} panels, {fps} fps...")

        frames = []
        n = len(fields)
        fig_w = min(6 * n, 13)
        # Reuse figure across frames to avoid per-frame creation overhead
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 5))
        if n == 1:
            axes = [axes]
        for i, step in enumerate(selected):
            data = self._load_step_data(step)
            for ax, fname in zip(axes, fields):
                ax.cla()
                val, label, cmap = self._compute_derived_field(fname, data)
                vmin = 0.0 if fname == 'damage' else None
                vmax = 1.0 if fname == 'damage' else None
                self._plot_field(ax, val, f'{label} (step {step})',
                                 cmap, vmin=vmin, vmax=vmax)

            fig.subplots_adjust(left=0.04, right=0.96, top=0.92,
                                bottom=0.08, wspace=0.30)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

            if (i + 1) % 20 == 0:
                print(f"  Frame {i+1}/{len(selected)}")
        plt.close(fig)

        if save_path is None:
            field_tag = '_'.join(fields[:3])
            save_path = os.path.join(self.output_dir,
                                     f'postprocess_{field_tag}.gif')

        # Uniform size
        target_size = frames[0].size
        for i in range(1, len(frames)):
            if frames[i].size != target_size:
                frames[i] = frames[i].resize(target_size, Image.LANCZOS)

        duration = int(1000 / fps)
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                       duration=duration, loop=0)
        print(f"GIF saved: {save_path} ({len(frames)} frames, {fps} fps)")

    def energy_plot(self, save_path: str = None, dpi: int = 150):
        """Plot energy evolution over all steps.

        Requires strain and damage data in the H5 file.
        """
        steps = self.step_numbers
        max_d_vals = []
        max_H_vals = []
        max_psi_vals = []

        for step in steps:
            data = self._load_step_data(step)
            max_d_vals.append(data['damage_nodal'].max())
            if 'H_elem' in data:
                max_H_vals.append(data['H_elem'].max())
            if 'psi_plus' in data:
                max_psi_vals.append(data['psi_plus'].max())

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        ax1.plot(steps, max_d_vals, 'r-', linewidth=1.5, label='max(d)')
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('max(d)', color='red', fontsize=11)
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.set_ylim(0, 1.05)

        ax2 = ax1.twinx()
        if max_H_vals:
            ax2.plot(steps, max_H_vals, 'b-', linewidth=1.5, label='max(H)')
            ax2.set_ylabel('max(H)', color='blue', fontsize=11)
            ax2.tick_params(axis='y', labelcolor='blue')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper left', fontsize=10)

        ax1.set_title('Damage & History Variable Evolution', fontsize=13)
        ax1.grid(True, alpha=0.3)

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'energy_evolution.png')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Energy plot saved: {save_path}")

    def field_evolution_plot(self, field_name: str, save_path: str = None,
                             dpi: int = 150):
        """Plot the max value of a field over all steps.

        Parameters
        ----------
        field_name : str — any field name supported by _compute_derived_field
        """
        steps = self.step_numbers
        vals = []

        for step in steps:
            data = self._load_step_data(step)
            try:
                v, label, _ = self._compute_derived_field(field_name, data)
                vals.append(np.max(v))
            except (ValueError, KeyError):
                break

        if not vals:
            print(f"Cannot compute '{field_name}' from H5 data.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps[:len(vals)], vals, 'b-', linewidth=1.5)
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(f'max({field_name})', fontsize=11)
        ax.set_title(f'{field_name} Evolution', fontsize=13)
        ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = os.path.join(self.output_dir,
                                     f'{field_name}_evolution.png')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved: {save_path}")

    def close(self):
        """Close the H5 file."""
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# PFM-Bench evaluation metrics — thin wrappers around metrics.PFMBenchMetrics.
# Prefer using PFMBenchMetrics directly for new code.
# ---------------------------------------------------------------------------

def crack_path_error(d_pred, d_true, mesh, threshold=0.9):
    """Max lateral crack deviation. See metrics.PFMBenchMetrics.hausdorff_distance."""
    from .metrics import PFMBenchMetrics
    m = PFMBenchMetrics(mesh, crack_threshold=threshold)
    return m.hausdorff_distance(d_pred, d_true)


def energy_error(E_pred, E_true):
    """Relative energy error: |E_pred - E_true| / |E_true|."""
    return abs(E_pred - E_true) / max(abs(E_true), 1e-30)


def peak_load_error(F_pred, F_true):
    """Relative peak force error: |F_pred - F_true| / |F_true|."""
    return abs(F_pred - F_true) / max(abs(F_true), 1e-30)


def damage_field_metrics(d_pred, d_true):
    """Damage comparison metrics. See metrics.PFMBenchMetrics.evaluate."""
    from .metrics import PFMBenchMetrics
    m = PFMBenchMetrics()
    return {
        'mse': m.mse(d_pred, d_true),
        'mae': (d_pred - d_true).abs().mean().item(),
        'max_error': m.linf(d_pred, d_true),
        'dice_95': m.dice_coefficient(d_pred, d_true, threshold=0.95),
    }


def main():
    """CLI entry point for offline post-processing."""
    parser = argparse.ArgumentParser(
        description='Offline post-processing for phast H5 files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available fields:
  damage, H, H_elem, psi_plus,
  displacement_x, displacement_y, displacement_mag,
  von_mises_stress, von_mises_strain,
  max_principal_stress, min_principal_stress,
  max_principal_strain, min_principal_strain,
  hydrostatic_stress, stress_triaxiality,
  strain_xx, strain_yy, strain_xy,
  stress_xx, stress_yy, stress_xy

Examples:
  python -m phast.postprocess_hdf5 data.h5 --info
  python -m phast.postprocess_hdf5 data.h5 --gif --fields damage max_principal_stress
  python -m phast.postprocess_hdf5 data.h5 --snapshot 50
  python -m phast.postprocess_hdf5 data.h5 --energy_plot
  python -m phast.postprocess_hdf5 data.h5 --evolution max_principal_stress
        """)

    parser.add_argument('h5_path', help='Path to H5 file')
    parser.add_argument('--output_dir', '-o', default=None,
                        help='Output directory (default: same as H5 dir)')
    parser.add_argument('--info', action='store_true',
                        help='Print H5 file summary')
    parser.add_argument('--gif', action='store_true',
                        help='Generate animated GIF')
    parser.add_argument('--snapshot', type=int, default=None,
                        metavar='STEP', help='Generate snapshot PNG at step')
    parser.add_argument('--energy_plot', action='store_true',
                        help='Generate damage/H evolution plot')
    parser.add_argument('--evolution', type=str, default=None,
                        metavar='FIELD',
                        help='Plot max(field) over steps')
    parser.add_argument('--fields', nargs='+',
                        default=['damage', 'von_mises_stress',
                                 'von_mises_strain'],
                        help='Fields to plot (for --gif and --snapshot)')
    parser.add_argument('--fps', type=int, default=10,
                        help='GIF frames per second (default: 10)')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Max GIF frames (default: 200)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI (default: 150)')

    args = parser.parse_args()

    with PostProcessor(args.h5_path, output_dir=args.output_dir) as pp:
        if args.info:
            pp.info()

        if args.gif:
            pp.make_gif(fields=args.fields, fps=args.fps,
                        max_frames=args.max_frames, dpi=args.dpi)

        if args.snapshot is not None:
            pp.snapshot(step=args.snapshot, fields=args.fields, dpi=args.dpi)

        if args.energy_plot:
            pp.energy_plot(dpi=args.dpi)

        if args.evolution:
            pp.field_evolution_plot(args.evolution, dpi=args.dpi)

        # Default: if no action specified, print info
        if not (args.info or args.gif or args.snapshot is not None
                or args.energy_plot or args.evolution):
            pp.info()
            print("\nUse --gif, --snapshot STEP, --energy_plot, or "
                  "--evolution FIELD to generate output.")


if __name__ == '__main__':
    main()
