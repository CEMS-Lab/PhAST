"""Standardized evaluation metrics for phase-field fracture benchmarks.

Implements the PFM-Bench protocol (Hamdi & Lejeune 2026) plus additional
metrics from the phase-field literature. All metrics operate on damage fields,
force-displacement curves, and energy histories.

Usage:
    from phast.metrics import PFMBenchMetrics
    m = PFMBenchMetrics(mesh)
    report = m.evaluate(d_pred, d_ref, fd_pred, fd_ref)
"""

import torch
import csv
import os
from typing import Optional


class PFMBenchMetrics:
    """Standardized PFM-Bench evaluation metrics.

    Parameters
    ----------
    mesh : FEMMesh or None
        Mesh for coordinate-aware metrics. If None, only array-based
        metrics are available.
    crack_threshold : float
        Damage threshold for crack identification (default 0.5).
    """

    def __init__(self, mesh=None, crack_threshold: float = 0.5):
        self.mesh = mesh
        self.crack_threshold = crack_threshold

    # ------------------------------------------------------------------ #
    # Damage field metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def mse(d_pred: torch.Tensor, d_ref: torch.Tensor) -> float:
        """Mean squared error between damage fields."""
        return ((d_pred - d_ref) ** 2).mean().item()

    @staticmethod
    def relative_l2(d_pred: torch.Tensor, d_ref: torch.Tensor) -> float:
        """Relative L2 error: ||d_pred - d_ref|| / ||d_ref||."""
        ref_norm = d_ref.norm().item()
        if ref_norm < 1e-30:
            return float('inf') if d_pred.norm().item() > 1e-30 else 0.0
        return (d_pred - d_ref).norm().item() / ref_norm

    @staticmethod
    def linf(d_pred: torch.Tensor, d_ref: torch.Tensor) -> float:
        """L-infinity error: max|d_pred - d_ref|."""
        return (d_pred - d_ref).abs().max().item()

    @staticmethod
    def dice_coefficient(d_pred: torch.Tensor, d_ref: torch.Tensor,
                         threshold: float = 0.5) -> float:
        """Dice coefficient for crack region overlap.

        Dice = 2|A ∩ B| / (|A| + |B|) where A, B are thresholded crack sets.
        Returns 1.0 for perfect overlap, 0.0 for no overlap.
        """
        pred_mask = (d_pred >= threshold).float()
        ref_mask = (d_ref >= threshold).float()
        intersection = (pred_mask * ref_mask).sum()
        union_size = pred_mask.sum() + ref_mask.sum()
        if union_size.item() < 1e-30:
            return 1.0  # both empty = perfect match
        return (2.0 * intersection / union_size).item()

    @staticmethod
    def iou(d_pred: torch.Tensor, d_ref: torch.Tensor,
            threshold: float = 0.5) -> float:
        """Intersection over Union (Jaccard index) for crack region.

        IoU = |A ∩ B| / |A ∪ B|.
        """
        pred_mask = (d_pred >= threshold).float()
        ref_mask = (d_ref >= threshold).float()
        intersection = (pred_mask * ref_mask).sum()
        union = pred_mask.sum() + ref_mask.sum() - intersection
        if union.item() < 1e-30:
            return 1.0
        return (intersection / union).item()

    def crack_path_error(self, d_pred: torch.Tensor,
                         d_ref: torch.Tensor) -> float:
        """Maximum lateral deviation of predicted crack path from reference.

        Computes Hausdorff-like distance between crack centerlines.
        Requires mesh coordinates.

        Returns
        -------
        max_deviation : float
            Maximum perpendicular distance [mm] between crack paths.
        """
        if self.mesh is None:
            raise ValueError("crack_path_error requires mesh coordinates")

        coords = self.mesh.nodes  # (N, 2)
        thr = self.crack_threshold

        pred_mask = d_pred >= thr
        ref_mask = d_ref >= thr

        if pred_mask.sum() == 0 or ref_mask.sum() == 0:
            return float('inf') if (pred_mask.sum() + ref_mask.sum()) > 0 else 0.0

        pred_pts = coords[pred_mask]  # (M1, 2)
        ref_pts = coords[ref_mask]    # (M2, 2)

        # Directed Hausdorff: max over pred of min distance to ref
        # Use chunked computation to avoid OOM on large meshes
        max_dev = 0.0
        chunk = 1000
        for i in range(0, pred_pts.shape[0], chunk):
            batch = pred_pts[i:i+chunk]  # (B, 2)
            dists = torch.cdist(batch, ref_pts)  # (B, M2)
            min_dists = dists.min(dim=1).values  # (B,)
            batch_max = min_dists.max().item()
            max_dev = max(max_dev, batch_max)

        return max_dev

    @staticmethod
    def damage_histogram_kl(d_pred: torch.Tensor, d_ref: torch.Tensor,
                            n_bins: int = 50) -> float:
        """KL divergence between damage value histograms.

        Measures whether the distribution of damage values matches.
        Lower is better; 0 = identical distributions.
        """
        eps = 1e-8

        p_hist = torch.histc(d_pred.float(), bins=n_bins, min=0, max=1) + eps
        q_hist = torch.histc(d_ref.float(), bins=n_bins, min=0, max=1) + eps

        # Normalize to probability distributions
        p = p_hist / p_hist.sum()
        q = q_hist / q_hist.sum()

        kl = (p * (p / q).log()).sum().item()
        return kl

    # ------------------------------------------------------------------ #
    # Force-displacement metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def peak_force_error(fd_pred: dict, fd_ref: dict) -> dict:
        """Compare peak reaction force between predicted and reference.

        Parameters
        ----------
        fd_pred, fd_ref : dict
            Must contain 'force' (list/array) and 'displacement' (list/array).

        Returns
        -------
        dict with 'peak_force_pred', 'peak_force_ref', 'relative_error_%',
             'peak_disp_pred', 'peak_disp_ref'.
        """
        f_pred = torch.tensor(fd_pred['force'], dtype=torch.float64)
        f_ref = torch.tensor(fd_ref['force'], dtype=torch.float64)

        peak_pred = f_pred.abs().max().item()
        peak_ref = f_ref.abs().max().item()
        peak_idx_pred = f_pred.abs().argmax().item()
        peak_idx_ref = f_ref.abs().argmax().item()

        d_pred_list = fd_pred.get('displacement', [])
        d_ref_list = fd_ref.get('displacement', [])

        result = {
            'peak_force_pred': peak_pred,
            'peak_force_ref': peak_ref,
            'relative_error_%': 100.0 * abs(peak_pred - peak_ref) / max(abs(peak_ref), 1e-30),
        }
        if len(d_pred_list) > peak_idx_pred:
            result['peak_disp_pred'] = float(d_pred_list[peak_idx_pred])
        if len(d_ref_list) > peak_idx_ref:
            result['peak_disp_ref'] = float(d_ref_list[peak_idx_ref])
        return result

    @staticmethod
    def fd_curve_error(fd_pred: dict, fd_ref: dict) -> dict:
        """Force-displacement curve comparison via interpolation.

        Interpolates both curves onto common displacement grid,
        computes relative L2 error and max deviation.

        Returns
        -------
        dict with 'relative_l2_%', 'max_deviation', 'n_points'.
        """
        f_pred = torch.tensor(fd_pred['force'], dtype=torch.float64)
        d_pred = torch.tensor(fd_pred['displacement'], dtype=torch.float64)
        f_ref = torch.tensor(fd_ref['force'], dtype=torch.float64)
        d_ref = torch.tensor(fd_ref['displacement'], dtype=torch.float64)

        # Common displacement grid
        d_min = max(d_pred.min().item(), d_ref.min().item())
        d_max = min(d_pred.max().item(), d_ref.max().item())
        if d_max <= d_min:
            return {'relative_l2_%': float('inf'), 'max_deviation': float('inf'),
                    'n_points': 0}

        n_pts = 200
        d_common = torch.linspace(d_min, d_max, n_pts, dtype=torch.float64)

        # Linear interpolation
        f_pred_interp = _interp1d(d_pred, f_pred, d_common)
        f_ref_interp = _interp1d(d_ref, f_ref, d_common)

        diff = f_pred_interp - f_ref_interp
        ref_norm = f_ref_interp.norm().item()

        return {
            'relative_l2_%': 100.0 * diff.norm().item() / max(ref_norm, 1e-30),
            'max_deviation': diff.abs().max().item(),
            'n_points': n_pts,
        }

    # ------------------------------------------------------------------ #
    # Energy metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def energy_error(energy_pred: list, energy_ref: list) -> dict:
        """Compare energy evolution over time steps.

        Returns
        -------
        dict with 'final_relative_error_%', 'max_relative_error_%',
             'monotone_pred' (bool).
        """
        e_pred = torch.tensor(energy_pred, dtype=torch.float64)
        e_ref = torch.tensor(energy_ref, dtype=torch.float64)

        n = min(len(e_pred), len(e_ref))
        e_pred = e_pred[:n]
        e_ref = e_ref[:n]

        ref_max = e_ref.abs().max().item()
        if ref_max < 1e-30:
            return {'final_relative_error_%': 0.0,
                    'max_relative_error_%': 0.0, 'monotone_pred': True}

        rel_err = (e_pred - e_ref).abs() / ref_max * 100.0
        monotone = bool((e_pred[1:] >= e_pred[:-1] - 1e-12).all().item())

        return {
            'final_relative_error_%': rel_err[-1].item(),
            'max_relative_error_%': rel_err.max().item(),
            'monotone_pred': monotone,
        }

    # ------------------------------------------------------------------ #
    # Comprehensive evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, d_pred: torch.Tensor, d_ref: torch.Tensor,
                 fd_pred: Optional[dict] = None,
                 fd_ref: Optional[dict] = None,
                 energy_pred: Optional[list] = None,
                 energy_ref: Optional[list] = None) -> dict:
        """Run all applicable metrics and return a report dict.

        Parameters
        ----------
        d_pred, d_ref : (N,) damage fields
        fd_pred, fd_ref : dicts with 'force' and 'displacement' lists
        energy_pred, energy_ref : lists of total energy per step

        Returns
        -------
        report : dict with all computed metrics
        """
        report = {}

        # Damage field metrics
        report['mse'] = self.mse(d_pred, d_ref)
        report['relative_l2'] = self.relative_l2(d_pred, d_ref)
        report['linf'] = self.linf(d_pred, d_ref)
        report['dice'] = self.dice_coefficient(
            d_pred, d_ref, threshold=self.crack_threshold)
        report['iou'] = self.iou(
            d_pred, d_ref, threshold=self.crack_threshold)
        report['histogram_kl'] = self.damage_histogram_kl(d_pred, d_ref)

        if self.mesh is not None:
            report['crack_path_error_mm'] = self.crack_path_error(
                d_pred, d_ref)

        # Force-displacement metrics
        if fd_pred is not None and fd_ref is not None:
            report['peak_force'] = self.peak_force_error(fd_pred, fd_ref)
            if ('displacement' in fd_pred and 'displacement' in fd_ref):
                report['fd_curve'] = self.fd_curve_error(fd_pred, fd_ref)

        # Energy metrics
        if energy_pred is not None and energy_ref is not None:
            report['energy'] = self.energy_error(energy_pred, energy_ref)

        return report

    def print_report(self, report: dict):
        """Pretty-print an evaluation report."""
        print("\n=== PFM-Bench Evaluation Report ===\n")

        # Damage field
        print("Damage Field Metrics:")
        print(f"  MSE:              {report['mse']:.6e}")
        print(f"  Relative L2:      {report['relative_l2']:.4f}")
        print(f"  L-inf:            {report['linf']:.6f}")
        print(f"  Dice coefficient: {report['dice']:.4f}")
        print(f"  IoU:              {report['iou']:.4f}")
        print(f"  Histogram KL:     {report['histogram_kl']:.6f}")
        if 'crack_path_error_mm' in report:
            print(f"  Crack path error: {report['crack_path_error_mm']:.6f} mm")

        # Force-displacement
        if 'peak_force' in report:
            pf = report['peak_force']
            print("\nForce-Displacement Metrics:")
            print(f"  Peak force (pred): {pf['peak_force_pred']:.6f} kN")
            print(f"  Peak force (ref):  {pf['peak_force_ref']:.6f} kN")
            print(f"  Relative error:    {pf['relative_error_%']:.2f}%")

        if 'fd_curve' in report:
            fc = report['fd_curve']
            print(f"  Curve L2 error:    {fc['relative_l2_%']:.2f}%")
            print(f"  Max deviation:     {fc['max_deviation']:.6f}")

        # Energy
        if 'energy' in report:
            en = report['energy']
            print("\nEnergy Metrics:")
            print(f"  Final rel error:   {en['final_relative_error_%']:.2f}%")
            print(f"  Max rel error:     {en['max_relative_error_%']:.2f}%")
            print(f"  Monotone:          {'Yes' if en['monotone_pred'] else 'No'}")

        print()


def _interp1d(x: torch.Tensor, y: torch.Tensor,
              x_new: torch.Tensor) -> torch.Tensor:
    """Simple piecewise-linear 1D interpolation (pure PyTorch)."""
    # Sort if x is not monotonically increasing
    if not (x[1:] >= x[:-1]).all():
        sort_idx = x.argsort()
        x, y = x[sort_idx], y[sort_idx]
    idx = torch.searchsorted(x, x_new).clamp(1, len(x) - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    t = (x_new - x0) / (x1 - x0 + 1e-30)
    return y0 + t * (y1 - y0)


def load_fd_csv(path: str) -> dict:
    """Load a force-displacement CSV file.

    Auto-detects column names containing 'force'/'reaction' and
    'disp'/'displacement'.

    Returns
    -------
    dict with 'force' and 'displacement' lists
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {'force': [], 'displacement': []}

    force_col = disp_col = None
    for col in rows[0]:
        cl = col.lower()
        if 'force' in cl or 'reaction' in cl:
            force_col = col
        if 'disp' in cl or 'displacement' in cl:
            disp_col = col

    forces = []
    disps = []
    for r in rows:
        if force_col and r[force_col].strip():
            forces.append(float(r[force_col]))
        if disp_col and r[disp_col].strip():
            disps.append(float(r[disp_col]))

    min_len = min(len(forces), len(disps))
    forces, disps = forces[:min_len], disps[:min_len]

    return {'force': forces, 'displacement': disps}
