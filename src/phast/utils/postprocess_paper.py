#!/usr/bin/env python
"""
Paper-quality post-processing for phast benchmark runs.

Reads H5 + CSV files from a completed run directory and generates all
publication-ready plots and animations.

Usage::

    # Generate all plots from a run directory:
    python -m phast.postprocess_paper path/to/run_dir

    # Override DPI and output format:
    python -m phast.postprocess_paper . --dpi 600 --format pdf

    # Only specific plot categories:
    python -m phast.postprocess_paper . --fields damage,energy

    # Compare multiple runs (mesh convergence / parametric study):
    python -m phast.postprocess_paper --compare run1/ run2/ run3/

    # Skip slow GIF generation:
    python -m phast.postprocess_paper . --skip-gif
"""

import os
import json
import time
import datetime
import argparse
import shutil
import subprocess
import warnings
from typing import List, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from PIL import Image
from PIL import ImageDraw

try:
    from scipy.signal import savgol_filter as _savgol
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _smooth(y, window_frac=0.05, polyorder=3):
    """Smooth a 1-D signal for plot quality.

    Uses Savitzky--Golay if scipy is available (preserves peaks/edges),
    otherwise a centred moving average. ``window_frac`` is the window
    length expressed as a fraction of len(y), clamped to a sensible
    minimum and forced to be odd.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 7:
        return y
    win = max(7, int(round(window_frac * n)) | 1)  # odd
    win = min(win, n if n % 2 else n - 1)
    if _HAVE_SCIPY:
        po = min(polyorder, win - 1)
        return _savgol(y, win, po)
    pad = win // 2
    ypad = np.pad(y, pad, mode='edge')
    kernel = np.ones(win) / win
    return np.convolve(ypad, kernel, mode='valid')


# Suppress tight_layout warnings for figures with shared colorbars
warnings.filterwarnings('ignore', message='.*tight_layout.*')


# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════

def log(msg, level='INFO'):
    """Print timestamped log message."""
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# COLOR THEME — change here to switch all plot colors at once
#
# Options for each field (uncomment your preference):
#   damage:   'inferno', 'hot', 'cmc.batlow', 'cmr.ember', 'magma'
#   stress:   'RdBu_r', 'coolwarm', 'cmc.vik', 'seismic'
#   stress_vm:'plasma', 'cmr.flamingo', 'YlOrRd', 'magma'
#   energy:   'inferno', 'cmr.ocean', 'viridis'
#   disp:     'coolwarm', 'RdBu_r', 'cmc.vik'
#   H_field:  'YlOrRd', 'inferno', 'cmc.lajolla'
# ═══════════════════════════════════════════════════════════════════════
THEME = {
    # --- Field colormaps ---
    'damage':           'inferno',      # sequential, dark=0, bright=1
    'stress_diverging': 'PuOr_r',       # diverging, purple=compression, orange=tension
    'stress_vm':        'coolwarm',     # blue->red for von Mises magnitude
    'strain_energy':    'inferno',
    'displacement':     'coolwarm',
    'H_field':          'coolwarm',

    # --- Line plot colors (up to 6 curves) ---
    'line_colors': [
        '#E63946',   # red
        '#457B9D',   # steel blue
        '#2A9D8F',   # teal
        '#E9C46A',   # gold
        '#F4A261',   # sandy orange
        '#264653',   # dark teal
    ],

    # --- Energy-specific line colors ---
    'elastic_color':  '#457B9D',
    'kinetic_color':  '#E63946',
    'fracture_color': '#2A9D8F',
    'total_color':    '#264653',

    # --- Figure settings ---
    # dpi=200 keeps 6.5" figures under 1800 px (6.5 × 200 = 1300 px).
    # font_size=10 matches cas-sc body text; ticks (font_size-1) and
    # labels bumped via rcParams above so axis numerics stay readable
    # when LaTeX scales subfigures to 0.3-0.5\linewidth.
    'dpi':             200,
    'figwidth':        6.5,       # single-column (inches)
    'figwidth_double': 13.0,      # double-column
    'font_size':       10,
    'title_size':      11,
    'background':      'white',
}


# ═══════════════════════════════════════════════════════════════════════
# Publication-quality matplotlib defaults
# ═══════════════════════════════════════════════════════════════════════
# Applied globally so every figure inherits the same look-and-feel.
# Designed to match Elsevier / CMAME style guidelines: serif font, modest
# axes, clear gridlines, sensible line widths, minor ticks, no titles by
# default (titles are kept in the postprocessor for diagnostic
# convenience but suppressed in publication-quality mode below).

plt.rcParams.update({
    # Font -- plot figures are often placed at 0.3-0.5\linewidth in
    # subfigure rows, which scales axis numerics down; bump ticks
    # and labels to 11-12 pt so they render near body-text size in
    # the final PDF. Full-width damage multipanels render at the
    # native size and stay readable.
    'font.family':         'serif',
    'font.serif':          ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset':    'stix',
    'font.size':           10,
    'axes.labelsize':      12,
    'axes.titlesize':      11,
    'legend.fontsize':     10,
    'xtick.labelsize':     11,
    'ytick.labelsize':     11,
    # Lines and markers
    'lines.linewidth':     1.6,
    'lines.markersize':    5,
    # Axes
    'axes.linewidth':      0.8,
    'axes.edgecolor':      '#222222',
    'axes.labelcolor':     '#222222',
    'axes.grid':           True,
    'axes.axisbelow':      True,
    'grid.linewidth':      0.4,
    'grid.alpha':          0.35,
    'grid.color':          '#888888',
    # Ticks
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size':    4,
    'ytick.major.size':    4,
    'xtick.minor.size':    2,
    'ytick.minor.size':    2,
    # Legend
    'legend.frameon':      True,
    'legend.framealpha':   0.9,
    'legend.edgecolor':    '#cccccc',
    'legend.fancybox':     False,
    # Saving
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
    'figure.dpi':          110,
    'figure.autolayout':   False,
    # Math
    'text.usetex':         False,
})


def _get_cmap(name):
    """Get colormap by name, with fallback for optional packages."""
    try:
        return plt.get_cmap(name)
    except ValueError:
        pass
    # Try cmcrameri
    if name.startswith('cmc.'):
        try:
            import cmcrameri.cm as cmc
            return getattr(cmc, name[4:])
        except (ImportError, AttributeError):
            warnings.warn(f"Colormap '{name}' requires cmcrameri. "
                          f"Install: pip install cmcrameri. Using 'inferno'.")
            return plt.get_cmap('inferno')
    # Try cmasher
    if name.startswith('cmr.'):
        try:
            import cmasher as cmr
            return getattr(cmr, name[4:])
        except (ImportError, AttributeError):
            warnings.warn(f"Colormap '{name}' requires cmasher. "
                          f"Install: pip install cmasher. Using 'plasma'.")
            return plt.get_cmap('plasma')
    return plt.get_cmap('viridis')


def _apply_theme():
    """Apply THEME settings to matplotlib rcParams.

    Axis labels and ticks are sized ABOVE body text (font_size+2/+1)
    so they stay readable when LaTeX scales figures into subfigure
    placements. Do not downgrade ticks below font_size — that
    produced axis numerics that rendered smaller than caption body
    text in the PDF.
    """
    plt.rcParams.update({
        'font.size': THEME['font_size'],
        'axes.titlesize': THEME['title_size'],
        'axes.labelsize': THEME['font_size'] + 2,
        'legend.fontsize': THEME['font_size'],
        'xtick.labelsize': THEME['font_size'] + 1,
        'ytick.labelsize': THEME['font_size'] + 1,
        'figure.facecolor': THEME['background'],
        'axes.facecolor': THEME['background'],
        'savefig.facecolor': THEME['background'],
        'savefig.dpi': THEME['dpi'],
        'figure.dpi': 100,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def _load_csv(path, sep=','):
    """Load CSV file, return dict of column arrays."""
    if not os.path.exists(path):
        return None
    try:
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            cols = {h: [] for h in reader.fieldnames}
            for row in reader:
                for h in reader.fieldnames:
                    try:
                        cols[h].append(float(row[h]) if row[h] else np.nan)
                    except (ValueError, TypeError):
                        cols[h].append(np.nan)
        return {k: np.array(v) for k, v in cols.items()}
    except Exception as e:
        log(f"Failed to load {path}: {e}", level='ERROR')
        return None


class BenchmarkPostProcessor:
    """Paper-quality post-processor for completed benchmark runs.

    Reads H5 + CSV files from a run directory and generates all plots
    needed for publication.

    Parameters
    ----------
    run_dir : str
        Directory containing training_data.h5, run_metadata.json, and CSV files.
    dpi : int or None
        Override DPI (default from THEME).
    fmt : str
        Output format: 'png', 'pdf', 'svg'.
    """

    def __init__(self, run_dir: str, dpi: int = None, fmt: str = 'png',
                 figures_dir: str = None):
        self.run_dir = os.path.abspath(run_dir)
        self.dpi = dpi or THEME['dpi']
        self.fmt = fmt
        # --figures-dir override allows writing outputs to a fresh tree
        # without touching the source run_dir.
        self.figures_dir = os.path.abspath(figures_dir) if figures_dir \
            else os.path.join(self.run_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        # When --figures-dir redirects output to a fresh tree, write
        # generated CSVs (energy.csv, crack_tip.csv, history.csv) next
        # to the figures/ dir rather than back into the source run_dir.
        # This makes the regen tree self-contained so downstream
        # --compare calls (velocity_parametric, dissipation_rate_compare,
        # normalized_energies_compare) can read CSVs from the regen
        # output without needing the original h5 scratch path.
        self.csv_dir = os.path.dirname(self.figures_dir) \
            if figures_dir else self.run_dir

        # Counters for summary
        self._plots_generated = 0
        self._plots_skipped = 0
        self._raster_cache = {}

        log(f"Processing run directory: {self.run_dir}")

        # Load metadata
        meta_path = os.path.join(run_dir, 'run_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            log(f"run_metadata.json found")
        else:
            self.metadata = {}
            log("run_metadata.json missing", level='WARN')

        solver_cfg = self.metadata.get('solver', {})
        self.is_dynamic = solver_cfg.get('solver_type', '') == 'explicit'
        self.dt = solver_cfg.get('dt', None)
        self.problem_name = self.metadata.get('problem', 'Benchmark')

        mat = self.metadata.get('material', {})
        self.E = mat.get('E', 210000.0)
        self.nu = mat.get('nu', 0.3)
        self.Gc = mat.get('Gc', 2.7)
        self.l0 = mat.get('l0', 0.1)
        self.rho = mat.get('rho', 7.8e-9)
        self.energy_split = mat.get('energy_split', 'spectral')
        self.c_R = solver_cfg.get('c_R_m_s', None)

        log(f"Problem: {self.problem_name}, Dynamic: {self.is_dynamic}, "
            f"Solver: {solver_cfg.get('solver_type', 'unknown')}")
        log(f"Material: E={self.E}, nu={self.nu}, Gc={self.Gc}, l0={self.l0}, "
            f"split={self.energy_split}")

        # Load trajectory store. Zarr is the current default; H5 remains
        # supported for historical runs.
        h5_path = os.path.join(run_dir, 'training_data.h5')
        zarr_path = os.path.join(run_dir, 'training_data.zarr')
        self._h5 = None
        self._zarr = None
        self._store_kind = None
        self.has_h5 = os.path.exists(h5_path) or os.path.exists(zarr_path)
        if os.path.exists(h5_path):
            import h5py
            log(f"H5 file found ({os.path.getsize(h5_path) / 1e6:.1f} MB)")
            self._h5 = h5py.File(h5_path, 'r')
            sim = self._h5['simulation_data']
            self._store_kind = 'h5'
        elif os.path.exists(zarr_path):
            import zarr
            log("Zarr trajectory found")
            self._zarr = zarr.open(str(zarr_path), mode='r')
            sim = self._zarr['simulation_data']
            self._store_kind = 'zarr'

        if self.has_h5:
            mesh_grp = sim['mesh']
            self.nodes = np.array(mesh_grp['node_coordinates'])
            self.elements = np.array(mesh_grp['element_connectivity'])
            self.n_nodes = len(self.nodes)
            self.n_elems = len(self.elements)
            elems_for_plot = self.elements
            if elems_for_plot.ndim == 2 and elems_for_plot.shape[1] == 4:
                elems_for_plot = np.vstack([
                    elems_for_plot[:, [0, 1, 2]],
                    elems_for_plot[:, [0, 2, 3]],
                ])
            self._plot_elements = elems_for_plot
            self._tri = mtri.Triangulation(
                self.nodes[:, 0], self.nodes[:, 1], self._plot_elements)
            self._steps_grp = sim['steps']
            self._step_names = sorted(
                self._steps_grp.keys(),
                key=lambda s: int(s.split('_')[-1]))
            self._precompute_projection()

            # Log H5 fields
            log(f"{self._store_kind.upper()}: {self.n_nodes} nodes, {self.n_elems} elements, "
                f"{len(self._step_names)} snapshots")
            if self._step_names:
                first_step = self._steps_grp[self._step_names[0]]
                fields = list(first_step.keys())
                attrs = list(first_step.attrs.keys())
                log(f"{self._store_kind.upper()} fields per step: {fields}")
                log(f"{self._store_kind.upper()} attrs per step: {attrs}")
        else:
            log("No training_data.h5 or training_data.zarr found", level='WARN')

        # Check CSV files
        csv_names = ['energy.csv', 'crack_tip.csv', 'history.csv', 'results.csv']
        found_csvs = []
        missing_csvs = []
        for name in csv_names:
            p = os.path.join(run_dir, name)
            if os.path.exists(p):
                found_csvs.append(f"{name} ({os.path.getsize(p)} bytes)")
            else:
                missing_csvs.append(name)
        if found_csvs:
            log(f"CSVs found: {', '.join(found_csvs)}")
        if missing_csvs:
            log(f"CSVs missing: {', '.join(missing_csvs)}")

        # Load CSVs
        self.energy_csv = _load_csv(os.path.join(run_dir, 'energy.csv'))
        self.history_csv = _load_csv(os.path.join(run_dir, 'history.csv'))
        self.crack_csv = _load_csv(os.path.join(run_dir, 'crack_tip.csv'))
        self.results_csv = _load_csv(os.path.join(run_dir, 'results.csv'))

        self.has_crack_csv = self.crack_csv is not None and len(next(iter(self.crack_csv.values()), [])) > 0
        self.has_energy_csv = self.energy_csv is not None

        _apply_theme()

    def _precompute_projection(self):
        """Area-weighted element-to-node projection."""
        p = self.nodes[self.elements]
        if self.elements.shape[1] == 4:
            x = p[:, :, 0]
            y = p[:, :, 1]
            self._elem_areas = 0.5 * np.abs(
                np.sum(x * np.roll(y, -1, axis=1)
                       - y * np.roll(x, -1, axis=1), axis=1))
        else:
            v1 = p[:, 1] - p[:, 0]
            v2 = p[:, 2] - p[:, 0]
            self._elem_areas = 0.5 * np.abs(
                v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        self._node_area_weight = np.zeros(self.n_nodes, dtype=np.float64)
        nloc = self.elements.shape[1]
        area_share = self._elem_areas / float(nloc)
        for i in range(nloc):
            np.add.at(self._node_area_weight, self.elements[:, i], area_share)
        self._node_area_weight = np.maximum(self._node_area_weight, 1e-30)

    def _elem_to_node(self, elem_field):
        """Project element field to nodes via area-weighted averaging."""
        elem_field = np.asarray(elem_field)
        if elem_field.ndim == 2:
            elem_field = np.nanmean(elem_field, axis=1)
        nodal = np.zeros(self.n_nodes, dtype=np.float64)
        nloc = self.elements.shape[1]
        weighted = elem_field * self._elem_areas / float(nloc)
        for i in range(nloc):
            np.add.at(nodal, self.elements[:, i], weighted)
        return nodal / self._node_area_weight

    @property
    def step_numbers(self):
        return [int(s.split('_')[-1]) for s in self._step_names]

    def _get_step(self, step):
        key = f'step_{step:04d}'
        return self._steps_grp[key]

    def _get_time_us(self, step):
        """Get simulation time in microseconds for a step."""
        grp = self._get_step(step)
        if 'time_s' in grp.attrs:
            return grp.attrs['time_s'] * 1e6
        if self.dt:
            return step * self.dt * 1e6
        return float(step)

    def _get_stress(self, grp, data):
        """Get stress from H5 data, or recompute from strain."""
        if 'stress' in data:
            s = data['stress']
            return s[:, 0], s[:, 1], s[:, 2]
        if 'strain' in data:
            strain = data['strain']
            E, nu = self.E, self.nu
            factor = E / ((1 + nu) * (1 - 2 * nu))
            C00, C01 = factor * (1 - nu), factor * nu
            C22 = factor * (1 - 2 * nu) / 2.0
            exx, eyy, gxy = strain[:, 0], strain[:, 1], strain[:, 2]
            return C00*exx + C01*eyy, C01*exx + C00*eyy, C22*gxy
        return None, None, None

    def _savefig(self, fig, name, max_px=2000):
        """Save figure to figures directory, capped at max_px on the
        larger dimension. Publication figures need >= ~200 dpi at
        column width; this cap keeps PNGs reviewable in Claude Code
        (2000-px ceiling per feedback_image_size memory) without
        dropping below that dpi on single/double-column layouts."""
        path = os.path.join(self.figures_dir, f'{name}.{self.fmt}')
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                    facecolor=THEME['background'])
        plt.close(fig)
        # Post-save downscale if needed — applies to every PNG this
        # class produces (damage_multipanel, stress_*_multipanel, etc).
        if self.fmt == 'png' and max_px:
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    if max(w, h) > max_px:
                        scale = max_px / max(w, h)
                        new_size = (int(w * scale), int(h * scale))
                        im.resize(new_size, Image.LANCZOS).save(
                            path, optimize=True)
                        log(f"Saved: {name}.{self.fmt} "
                            f"({w}x{h} -> {new_size[0]}x{new_size[1]} "
                            f"at 2000-px cap)")
                        return
            except Exception as e:
                log(f"Post-save cap failed for {name}: {e}", level='WARN')
        log(f"Saved: {name}.{self.fmt}")

    def close(self):
        if self._h5:
            self._h5.close()
        self._zarr = None

    # ═══════════════════════════════════════════════════════════════════
    # CSV GENERATION FROM H5
    # ═══════════════════════════════════════════════════════════════════

    def _generate_csvs_from_h5(self):
        """Generate missing CSV files from H5 data before plot generation."""
        log("Checking for missing CSVs to generate from H5...")
        self._generate_energy_csv()
        self._generate_crack_tip_csv()
        self._generate_history_csv()

        # Reload CSVs after generation — CSVs land in csv_dir (= run_dir
        # in normal mode, = figures_dir's parent when --figures-dir).
        log("Reloading CSVs after generation...")
        self.energy_csv = _load_csv(os.path.join(self.csv_dir, 'energy.csv'))
        self.history_csv = _load_csv(os.path.join(self.csv_dir, 'history.csv'))
        self.crack_csv = _load_csv(os.path.join(self.csv_dir, 'crack_tip.csv'))
        self.results_csv = _load_csv(os.path.join(self.run_dir, 'results.csv'))

        self.has_crack_csv = self.crack_csv is not None and len(next(iter(self.crack_csv.values()), [])) > 0
        self.has_energy_csv = self.energy_csv is not None
        log(f"After reload: energy_csv={self.has_energy_csv}, "
            f"crack_tip_csv={self.has_crack_csv}, "
            f"history_csv={self.history_csv is not None}")

    def _generate_energy_csv(self):
        """Generate energy.csv from H5 snapshot attributes."""
        energy_path = os.path.join(self.csv_dir, 'energy.csv')
        if os.path.exists(energy_path):
            # Check if file has data rows (not just header)
            n_lines = sum(1 for _ in open(energy_path))
            if n_lines > 1:
                log(f"energy.csv exists ({n_lines} rows, {os.path.getsize(energy_path)} bytes)")
                return
            log(f"energy.csv exists but is header-only ({n_lines} lines) — regenerating from H5")

        log("energy.csv missing -- generating from trajectory attributes...")
        if not self.has_h5:
            log("No trajectory store found, cannot generate energy.csv", level='WARN')
            return

        try:
            rows = []
            for key in self._step_names:
                grp = self._steps_grp[key]
                step = int(key.split('_')[-1])
                t_s = grp.attrs.get('time_s', 0.0)
                elastic = grp.attrs.get('energy_elastic', 0.0)
                kinetic = grp.attrs.get('energy_kinetic', 0.0)
                fracture = grp.attrs.get('energy_fracture', 0.0)
                total = grp.attrs.get('energy_total', 0.0)
                rows.append((step, t_s, elastic, kinetic, fracture, total))

            if rows:
                with open(energy_path, 'w') as ef:
                    ef.write('step,t_s,elastic,kinetic,fracture,total\n')
                    for r in rows:
                        ef.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n')
                log(f"Generated energy.csv ({len(rows)} rows)")
            else:
                log("No energy attributes found in trajectory snapshots", level='WARN')
        except Exception as e:
            log(f"Failed to generate energy.csv: {e}", level='ERROR')

    def _generate_crack_tip_csv(self):
        """Generate crack_tip.csv from H5 damage fields."""
        crack_path = os.path.join(self.csv_dir, 'crack_tip.csv')
        if os.path.exists(crack_path):
            n_lines = sum(1 for _ in open(crack_path))
            if n_lines > 1:
                log(f"crack_tip.csv exists ({n_lines} rows, {os.path.getsize(crack_path)} bytes)")
                return
            log(f"crack_tip.csv is header-only ({n_lines} lines) — regenerating from H5")

        log("crack_tip.csv missing -- generating from trajectory damage fields...")
        if not self.has_h5:
            log("No trajectory store found, cannot generate crack_tip.csv", level='WARN')
            return

        # Read metadata for c_R and l0 (used for branching detection).
        # ``crack_vel_mms`` is computed from tip-position deltas in
        # mm/s, so the normaliser ``c_R`` must also be in mm/s for
        # ``crack_vel_frac_cR`` to be a true fraction (issue #240).
        # We prefer ``c_R_mm_s`` and only fall back to ``c_R_m_s`` after
        # converting it to mm/s; older runs that wrote ``c_R_m_s``
        # (numerically meters/sec) used to be read as if it were mm/s,
        # producing ``vel_frac`` ~1000x too small.
        meta_path = os.path.join(self.run_dir, 'run_metadata.json')
        c_R = 0.0  # 0 == unknown; vel_frac falls back to NaN
        l0_meta = 0.25
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as mf:
                    meta = json.load(mf)
                solver_meta = meta.get('solver', {}) or {}
                if 'c_R_mm_s' in solver_meta:
                    c_R = float(solver_meta['c_R_mm_s'])
                elif 'c_R_m_s' in solver_meta:
                    # Legacy key: stored in m/s but the rest of the
                    # pipeline works in mm/s. Convert.
                    c_R = float(solver_meta['c_R_m_s']) * 1.0e3
                l0_meta = float(meta.get('material', {}).get('l0', 0.25))
            except Exception as e:
                log(f"Error reading metadata for c_R: {e}", level='WARN')
        if c_R <= 0:
            log("c_R unavailable in run_metadata.json; "
                "crack_vel_frac_cR will be NaN (issue #240).", level='WARN')

        try:
            x = self.nodes[:, 0]
            y = self.nodes[:, 1]

            # Get notch tip x (approximate from mesh -- use half-width as default)
            x_min, x_max = x.min(), x.max()
            notch_x = (x_min + x_max) / 2  # rough estimate

            # Pass 1: collect (step, t_s, tip_x, n_tips, branched) per
            # snapshot; defer velocity until we can smooth tip_x(t).
            pre_rows = []  # (step, t_s, tip_x, branched)

            for key in self._step_names:
                grp = self._steps_grp[key]
                step = int(key.split('_')[-1])
                t_s = grp.attrs.get('time_s', 0.0)

                if 'damage_nodal' not in grp:
                    continue
                d = np.array(grp['damage_nodal'])

                if d.max() < 0.5:
                    continue

                # Find crack tip (rightmost x where d > 0.5, beyond notch)
                crack_mask = (d > 0.5) & (x > notch_x + 0.1)
                if not crack_mask.any():
                    continue

                tip_x = x[crack_mask].max()

                # Branching detection: check for fully damaged (d > 0.9)
                # nodes well outside the central crack band, where
                # "well outside" scales with the regularisation length
                # so that wide-band benchmarks (e.g., SENT with l0=0.5)
                # do not trigger a false positive.
                y_mid = (y.min() + y.max()) / 2
                band_half = max(4.0 * l0_meta, 1.5)
                fully_cracked = (d > 0.9) & (x > notch_x + 0.1)
                upper = fully_cracked & (y > y_mid + band_half)
                lower = fully_cracked & (y < y_mid - band_half)
                # Require a minimum number of nodes on each side to
                # avoid false positives from a single noisy element.
                branched = 1 if (upper.sum() >= 10 and lower.sum() >= 10) else 0

                pre_rows.append((step, float(t_s), float(tip_x), int(branched)))

            # Pass 2: smooth tip_x(t) before numerical differentiation.
            rows = []
            if pre_rows:
                x_arr = np.array([r[2] for r in pre_rows], dtype=float)
                if x_arr.size >= 5:
                    try:
                        from scipy.signal import savgol_filter
                        x_sm = savgol_filter(
                            x_arr, window_length=5, polyorder=2,
                            mode='interp')
                    except ImportError:
                        kernel = np.ones(5) / 5.0
                        pad = np.concatenate(
                            [np.full(2, x_arr[0]), x_arr,
                             np.full(2, x_arr[-1])])
                        x_sm = np.convolve(pad, kernel, mode='valid')
                else:
                    x_sm = x_arr.copy()

                for i, (step, t_s, _tip_x, branched) in enumerate(pre_rows):
                    if i == 0:
                        dt_track = t_s if t_s > 0 else 1e-12
                        vel = (x_sm[i] - notch_x) / dt_track
                    else:
                        dt_track = (t_s - pre_rows[i - 1][1]) or 1e-12
                        vel = (x_sm[i] - x_sm[i - 1]) / dt_track
                    vel_frac = (vel / c_R) if c_R > 0 else float('nan')
                    n_tips = 2 if branched else 1
                    rows.append((step, t_s * 1e6, float(x_sm[i]),
                                 n_tips, float(vel), float(vel_frac),
                                 branched))

            if rows:
                with open(crack_path, 'w') as cf:
                    cf.write('step,t_us,crack_tip_x_mm,n_crack_tips,crack_vel_mms,crack_vel_frac_cR,branched\n')
                    for r in rows:
                        cf.write(f'{r[0]},{r[1]:.4f},{r[2]:.6f},{r[3]},{r[4]:.2f},{r[5]:.4f},{r[6]}\n')
                log(f"Generated crack_tip.csv ({len(rows)} rows)")
            else:
                log("No crack data found in trajectory snapshots (max_d < 0.5 everywhere)", level='WARN')
        except Exception as e:
            log(f"Failed to generate crack_tip.csv: {e}", level='ERROR')

    def _generate_history_csv(self):
        """Generate history.csv from H5 snapshots."""
        hist_path = os.path.join(self.csv_dir, 'history.csv')
        if os.path.exists(hist_path) and os.path.getsize(hist_path) > 10:
            log(f"history.csv exists ({os.path.getsize(hist_path)} bytes)")
            return

        log("history.csv missing -- generating from trajectory...")
        if not self.has_h5:
            log("No trajectory store, cannot generate history.csv", level='WARN')
            return

        try:
            rows = []
            for key in self._step_names:
                grp = self._steps_grp[key]
                step = int(key.split('_')[-1])
                d = np.array(grp['damage_nodal']) if 'damage_nodal' in grp else np.array([0.0])
                psi = np.array(grp['psi_plus']) if 'psi_plus' in grp else np.array([0.0])
                H = np.array(grp.get('H_nodal', grp.get('H_elem', [0.0])))
                rows.append((step, H.max(), psi.max(), d.max(), 0.0, 0.0))

            if rows:
                with open(hist_path, 'w') as hf:
                    hf.write('step,max_H_nodal,max_psi_plus,max_damage,delta_H,delta_damage\n')
                    for r in rows:
                        hf.write(f'{r[0]},{r[1]:.6f},{r[2]:.4e},{r[3]:.8f},{r[4]},{r[5]}\n')
                log(f"Generated history.csv ({len(rows)} rows)")
            else:
                log("No step data found in trajectory for history.csv", level='WARN')
        except Exception as e:
            log(f"Failed to generate history.csv: {e}", level='ERROR')

    # ═══════════════════════════════════════════════════════════════════
    # A. SPATIAL FIELD PLOTS
    # ═══════════════════════════════════════════════════════════════════

    def _select_key_steps(self, n=4):
        """Auto-select key timesteps.

        Prefer crack_tip.csv as the signal: span from the first step
        where the tip advances beyond the notch (crack_tip_x > notch +
        2*l0) to the last step where the tip is still advancing (= crack
        reaches the boundary). This reliably picks snapshots across the
        propagation window instead of concentrating on initiation or
        post-completion states.

        Fallback to max_d milestones if crack_tip.csv is missing: span
        from first max_d > 0.1 to the latest step where a crack tip
        is still moving (detected via the per-step damage-centroid
        x-extent). As a last resort, evenly space across the run.
        """
        steps = self.step_numbers
        if len(steps) <= n:
            return steps

        # Primary path: use crack_tip.csv if present. t_us in CSV rows
        # maps back to H5 steps by time.
        if (getattr(self, 'is_dynamic', False)
                and getattr(self, 'has_crack_csv', False)
                and getattr(self, 'has_h5', False)):
            try:
                cc = self.crack_csv  # dict of numpy arrays
                tip_x = np.asarray(cc['crack_tip_x_mm'])
                t_us = np.asarray(cc['t_us'])
                # Start: first row where the tip has advanced past the
                # notch. Without a known notch length we use the first
                # row where tip_x exceeds its initial value by >= 2 mm;
                # tightly-captured cracks move more than that by the
                # time the first crack_tip row is written.
                tip0 = float(tip_x[0]) if len(tip_x) else 0.0
                moving = np.where(tip_x > tip0 + 2.0)[0]
                if len(moving) >= 2:
                    first_t = float(t_us[moving[0]])
                    # End: last row where the tip is still advancing
                    # (frame-to-frame change > 0.01 mm).
                    dx = np.abs(np.diff(tip_x))
                    advancing_idx = np.where(dx > 0.01)[0]
                    last_t = (float(t_us[advancing_idx[-1] + 1])
                              if len(advancing_idx) > 0
                              else float(t_us[-1]))
                    # If the crack branched, the interesting window
                    # extends past primary-tip pinning at the plate
                    # boundary (branches continue to propagate while
                    # the primary tip x is frozen). Extend last_t to
                    # the last branched-row time so snapshot panels
                    # capture the Y-shape.
                    branched = cc.get('branched')
                    if branched is not None:
                        br = np.where(np.asarray(branched) > 0)[0]
                        if len(br) > 0:
                            last_t = max(last_t, float(t_us[br[-1]]))
                    if last_t > first_t:
                        t_targets_us = np.linspace(first_t, last_t, n)
                        step_times_us = np.array([
                            float(self._get_time_us(s)) for s in steps
                        ])
                        chosen = []
                        for tt in t_targets_us:
                            idx = int(np.argmin(np.abs(step_times_us - tt)))
                            chosen.append(steps[idx])
                        # Deduplicate while preserving order.
                        seen = set()
                        uniq = [s for s in chosen
                                if s not in seen and not seen.add(s)]
                        if len(uniq) == n:
                            return uniq
            except Exception:
                pass

        # Fallback: damage-milestone on the H5 itself. Better than naive
        # even spacing in T when the crack reaches the boundary early.
        if getattr(self, 'is_dynamic', False) and getattr(self, 'has_h5', False):
            try:
                max_d = np.array([
                    float(np.array(self._get_step(s)['damage_nodal']).max())
                    for s in steps
                ])
                if (max_d > 0.1).any():
                    first = int(np.argmax(max_d > 0.1))
                    # End of "action": last step where max_d is still
                    # growing OR first step where max_d has stabilised
                    # at its maximum for several consecutive frames.
                    dmax = float(max_d.max())
                    stable = np.where(max_d >= 0.99 * dmax)[0]
                    last = int(stable[0]) if len(stable) else len(steps) - 1
                    # If crack reaches boundary well before the last H5
                    # step, only span up to the first saturation.
                    if last <= first:
                        last = len(steps) - 1
                    indices = np.linspace(first, last, n, dtype=int)
                    return [steps[i] for i in indices]
            except Exception:
                pass

        # Last resort: evenly spaced in time, skip t=0.
        indices = np.linspace(0, len(steps) - 1, n + 1, dtype=int)[1:]
        return [steps[i] for i in indices]

    def _domain_aspect(self):
        """Compute domain width/height ratio for figure sizing."""
        xr = self.nodes[:, 0].max() - self.nodes[:, 0].min()
        yr = self.nodes[:, 1].max() - self.nodes[:, 1].min()
        return xr / max(yr, 1e-10)

    def _multipanel_figure(self, n_panels, title=None):
        """Create a properly sized multi-panel figure with shared colorbar space."""
        aspect = self._domain_aspect()
        # Panel height: 4 inches. Width scales with domain aspect ratio.
        panel_h = 4.0
        panel_w = max(panel_h * aspect, 2.5)  # minimum 2.5 inches wide
        # Extra space for colorbar
        fig_w = panel_w * n_panels + 1.5
        fig, axes = plt.subplots(
            1, n_panels, figsize=(fig_w, panel_h + 0.8),
            gridspec_kw={'wspace': 0.05})
        if n_panels == 1:
            axes = [axes]
        # No suptitle: the LaTeX caption carries all explanatory text.
        return fig, axes

    def _add_colorbar(self, fig, axes, mappable, label):
        """Add a properly positioned colorbar to a multi-panel figure."""
        cbar = fig.colorbar(
            mappable, ax=axes, label=label,
            shrink=0.85, aspect=25, pad=0.02)
        cbar.ax.tick_params(labelsize=THEME['font_size'] - 2)
        return cbar

    def plot_damage_multipanel(self, steps=None, n_panels=4):
        """Damage field at key timesteps — multi-panel figure.

        Fonts here are set smaller than the global theme because the
        saved PNG is cropped to ~5.3 in via bbox_inches='tight' and is
        then magnified by LaTeX to \\linewidth (~6.5 in); the net ~1.2x
        magnification was pushing axis labels and ticks above the
        caption body text. Sizes below land in the 10-11 pt effective
        range after LaTeX scaling.
        """
        log("Generating: damage_multipanel")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping damage multipanel", level='WARN')
            self._plots_skipped += 1
            return
        if steps is None:
            steps = self._select_key_steps(n_panels)

        _tick_sz   = 6
        _label_sz  = 6
        _title_sz  = 6

        cmap = _get_cmap(THEME['damage'])
        fig, axes = self._multipanel_figure(len(steps), self.problem_name)

        for i, (ax, s) in enumerate(zip(axes, steps)):
            grp = self._get_step(s)
            d = np.array(grp['damage_nodal'])
            t_us = self._get_time_us(s)
            tcf = ax.tricontourf(self._tri, d, levels=64,
                                 cmap=cmap, vmin=0, vmax=1)
            if self.is_dynamic:
                ax.set_title(f't = {t_us:.1f} $\\mu$s', fontsize=_title_sz)
            else:
                ax.set_title(f'Step {s}', fontsize=_title_sz)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_ylabel('y (mm)', fontsize=_label_sz)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('x (mm)', fontsize=_label_sz)
            ax.tick_params(axis='both', labelsize=_tick_sz)

        cbar = fig.colorbar(
            tcf, ax=axes, label='Damage $d$',
            shrink=0.85, aspect=25, pad=0.02)
        cbar.ax.tick_params(labelsize=_tick_sz)
        cbar.set_label('Damage $d$', fontsize=_label_sz)
        self._savefig(fig, 'damage_multipanel')
        self._plots_generated += 1
        log(f"Generated damage_multipanel ({time.time() - t0:.1f}s)")

    def _default_stress_type(self):
        """Choose stress type based on material behavior.

        Brittle fracture (phase-field) -> max principal stress (crack-driving).
        Ductile materials -> von Mises (yield criterion).
        """
        # All phase-field models are brittle by nature
        return 'max_principal'

    def plot_stress_fields(self, steps=None, stress_type=None):
        """Stress field at key timesteps.

        Default: max principal stress (appropriate for brittle fracture).
        Use stress_type='von_mises' for ductile materials.
        """
        if stress_type is None:
            stress_type = self._default_stress_type()
        log(f"Generating: stress_{stress_type}_multipanel "
            f"(default for brittle fracture, use stress_type='von_mises' for ductile)")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping stress fields", level='WARN')
            self._plots_skipped += 1
            return
        if steps is None:
            steps = self._select_key_steps(3)

        # Determine colormap: diverging for principal/component, sequential for VM
        is_diverging = stress_type in ('max_principal', 'hydrostatic', 'xx', 'yy', 'xy')
        cmap_key = 'stress_diverging' if is_diverging else 'stress_vm'

        # First pass: compute all fields and find global limits for consistent colorbar
        fields_list = []
        for s in steps:
            grp = self._get_step(s)
            data = {k: np.array(grp[k]) for k in grp.keys()}
            sxx, syy, sxy = self._get_stress(grp, data)
            if sxx is None:
                log(f"No stress data at step {s}, skipping stress fields", level='WARN')
                self._plots_skipped += 1
                return

            if stress_type == 'max_principal':
                avg = (sxx + syy) / 2.0
                R = np.sqrt(((sxx - syy) / 2.0)**2 + sxy**2)
                field = self._elem_to_node(avg + R)
                label = 'Max principal stress $\\sigma_1$ (MPa)'
            elif stress_type == 'von_mises':
                vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
                field = self._elem_to_node(vm)
                label = 'von Mises stress (MPa)'
            elif stress_type == 'hydrostatic':
                field = self._elem_to_node((sxx + syy) * (1 + self.nu) / 3.0)
                label = 'Hydrostatic stress (MPa)'
            else:
                comp = {'xx': sxx, 'yy': syy, 'xy': sxy}.get(stress_type, sxx)
                field = self._elem_to_node(comp)
                label = f'$\\sigma_{{{stress_type}}}$ (MPa)'
            fields_list.append(field)

        # Consistent color limits across panels
        all_vals = np.concatenate(fields_list)
        if is_diverging:
            vlim = max(abs(np.nanpercentile(all_vals, 2)),
                       abs(np.nanpercentile(all_vals, 98)))
            vmin, vmax = -vlim, vlim
        else:
            vmin = 0
            vmax = np.nanpercentile(all_vals, 98)

        cmap = _get_cmap(THEME[cmap_key])
        title = f'{self.problem_name} — {stress_type.replace("_", " ").title()}'
        fig, axes = self._multipanel_figure(len(steps), title)

        for i, (ax, s, field) in enumerate(zip(axes, steps, fields_list)):
            tcf = ax.tripcolor(self._tri, field, shading='gouraud',
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               rasterized=True)
            t_us = self._get_time_us(s)
            ax.set_title(f't = {t_us:.1f} $\\mu$s' if self.is_dynamic
                         else f'Step {s}', fontsize=THEME['font_size'])
            ax.set_aspect('equal')
            if i == 0:
                ax.set_ylabel('y (mm)')
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('x (mm)')

        self._add_colorbar(fig, axes, tcf, label)
        self._savefig(fig, f'stress_{stress_type}_multipanel')
        self._plots_generated += 1
        log(f"Generated stress_{stress_type}_multipanel ({time.time() - t0:.1f}s)")

    def plot_displacement_field(self, steps=None):
        """Displacement magnitude at key timesteps."""
        log("Generating: displacement_multipanel")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping displacement field", level='WARN')
            self._plots_skipped += 1
            return
        if steps is None:
            steps = self._select_key_steps(3)

        cmap = _get_cmap(THEME['displacement'])
        fig, axes = self._multipanel_figure(
            len(steps), f'{self.problem_name} — Displacement')

        # First pass for consistent colorbar
        vmax = 0
        for s in steps:
            grp = self._get_step(s)
            u = np.array(grp['displacement'])
            vmax = max(vmax, np.sqrt(u[:, 0]**2 + u[:, 1]**2).max())

        for i, (ax, s) in enumerate(zip(axes, steps)):
            grp = self._get_step(s)
            u = np.array(grp['displacement'])
            u_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
            tcf = ax.tricontourf(self._tri, u_mag, levels=64,
                                 cmap=cmap, vmin=0, vmax=vmax)
            t_us = self._get_time_us(s)
            ax.set_title(f't = {t_us:.1f} $\\mu$s' if self.is_dynamic
                         else f'Step {s}', fontsize=THEME['font_size'])
            ax.set_aspect('equal')
            if i == 0:
                ax.set_ylabel('y (mm)')
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('x (mm)')
        self._add_colorbar(fig, axes, tcf, '|u| (mm)')
        self._savefig(fig, 'displacement_multipanel')
        self._plots_generated += 1
        log(f"Generated displacement_multipanel ({time.time() - t0:.1f}s)")

    def plot_damage_profile(self, steps=None, axis='y', position=None):
        """Damage cross-section: d vs y/l0 at the crack plane."""
        log("Generating: damage_profile")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping damage profile", level='WARN')
            self._plots_skipped += 1
            return
        if steps is None:
            steps = self._select_key_steps(4)

        # Find crack plane position (default: domain midpoint)
        if position is None:
            if axis == 'y':
                position = (self.nodes[:, 1].max() + self.nodes[:, 1].min()) / 2
            else:
                position = (self.nodes[:, 0].max() + self.nodes[:, 0].min()) / 2

        # Select nodes near the line
        band_width = 2 * self.l0
        coord_idx = 1 if axis == 'y' else 0
        other_idx = 0 if axis == 'y' else 1
        mask = np.abs(self.nodes[:, coord_idx] - position) < band_width
        coords = self.nodes[mask, other_idx]
        sort_idx = np.argsort(coords)
        coords_sorted = coords[sort_idx]

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))
        colors = THEME['line_colors']

        for i, s in enumerate(steps):
            grp = self._get_step(s)
            d = np.array(grp['damage_nodal'])
            d_line = d[mask][sort_idx]
            t_us = self._get_time_us(s)
            label = f't = {t_us:.1f} $\\mu$s' if self.is_dynamic else f'Step {s}'
            ax.plot(coords_sorted / self.l0, d_line,
                    color=colors[i % len(colors)], label=label, linewidth=1.5)

        ax.set_xlabel(f'{axis} / $\\ell_0$')
        ax.set_ylabel('Damage $d$')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, 'damage_profile')
        self._plots_generated += 1
        log(f"Generated damage_profile ({time.time() - t0:.1f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # B. TIME-SERIES PLOTS
    # ═══════════════════════════════════════════════════════════════════

    def plot_energy_balance(self):
        """Energy balance: elastic + kinetic + fracture + total vs time."""
        log("Generating: energy_balance")
        t0 = time.time()
        if not self.has_energy_csv:
            log("No energy.csv, skipping energy balance plot", level='WARN')
            self._plots_skipped += 1
            return

        ec = self.energy_csv
        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))

        t_key = 't_s' if 't_s' in ec else 'step'
        t = ec[t_key]
        if t_key == 't_s':
            t = t * 1e6  # convert to microseconds
            xlabel = 'Time ($\\mu$s)'
        else:
            xlabel = 'Step'

        ax.plot(t, ec['elastic'], color=THEME['elastic_color'],
                label='Elastic', linewidth=1.5)
        if 'kinetic' in ec:
            ax.plot(t, ec['kinetic'], color=THEME['kinetic_color'],
                    label='Kinetic', linewidth=1.5)
        ax.plot(t, ec['fracture'], color=THEME['fracture_color'],
                label='Fracture', linewidth=1.5)
        ax.plot(t, ec['total'], color=THEME['total_color'],
                label='Total', linewidth=2, linestyle='--')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Energy (N$\\cdot$mm)')
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, 'energy_balance')
        self._plots_generated += 1
        log(f"Generated energy_balance ({time.time() - t0:.1f}s)")

    def plot_energy_normalized(self):
        """Normalized energies (each / total_0) vs time."""
        log("Generating: energy_normalized")
        t0 = time.time()
        if not self.has_energy_csv:
            log("No energy.csv, skipping normalized energy plot", level='WARN')
            self._plots_skipped += 1
            return

        ec = self.energy_csv
        if ec is None or len(ec) == 0 or 'total' not in ec or len(ec['total']) == 0:
            log("energy.csv is empty, skipping normalized energy plot", level='WARN')
            self._plots_skipped += 1
            return

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))

        t_key = 't_s' if 't_s' in ec else 'step'
        t = ec[t_key] * 1e6 if t_key == 't_s' else ec[t_key]
        total_0 = ec['total'][0] if len(ec['total']) > 0 and ec['total'][0] > 0 else 1.0

        ax.plot(t, ec['elastic'] / total_0, color=THEME['elastic_color'],
                label='Elastic / $E_0$', linewidth=1.5)
        if 'kinetic' in ec:
            ax.plot(t, ec['kinetic'] / total_0, color=THEME['kinetic_color'],
                    label='Kinetic / $E_0$', linewidth=1.5)
        ax.plot(t, ec['fracture'] / total_0, color=THEME['fracture_color'],
                label='Fracture / $E_0$', linewidth=1.5)

        ax.set_xlabel('Time ($\\mu$s)' if t_key == 't_s' else 'Step')
        ax.set_ylabel('$E / E_0$')
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, 'energy_normalized')
        self._plots_generated += 1
        log(f"Generated energy_normalized ({time.time() - t0:.1f}s)")

    def plot_force_displacement(self):
        """Force vs displacement (quasi-static)."""
        log("Generating: force_displacement")
        t0 = time.time()
        csv = self.results_csv or self.history_csv
        if csv is None:
            log("No results/history CSV, skipping force-displacement plot", level='WARN')
            self._plots_skipped += 1
            return

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))

        if 'displacement' in csv and 'reaction_kN' in csv:
            ax.plot(csv['displacement'], csv['reaction_kN'],
                    color=THEME['line_colors'][0], linewidth=1.5)
            ax.set_xlabel('Displacement (mm)')
            ax.set_ylabel('Reaction force (kN)')
        elif 'applied_disp' in csv and 'reaction_force' in csv:
            mask = ~np.isnan(csv['reaction_force'])
            ax.plot(csv['applied_disp'][mask], csv['reaction_force'][mask],
                    color=THEME['line_colors'][0], linewidth=1.5)
            ax.set_xlabel('Applied displacement (mm)')
            ax.set_ylabel('Reaction force (N)')
        else:
            log("CSV missing displacement/force columns, skipping force-displacement", level='WARN')
            plt.close(fig)
            self._plots_skipped += 1
            return

        fig.tight_layout()
        self._savefig(fig, 'force_displacement')
        self._plots_generated += 1
        log(f"Generated force_displacement ({time.time() - t0:.1f}s)")

    def plot_max_damage_vs_time(self):
        """max(d) vs time from history.csv."""
        log("Generating: max_damage_vs_time")
        t0 = time.time()
        csv = self.history_csv
        if csv is None or 'max_damage' not in csv:
            log("No history.csv or missing max_damage column, skipping", level='WARN')
            self._plots_skipped += 1
            return

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 3.5))
        steps = csv['step']
        if self.dt:
            t = steps * self.dt * 1e6
            xlabel = 'Time ($\\mu$s)'
        else:
            t = steps
            xlabel = 'Step'

        ax.plot(t, csv['max_damage'], color=THEME['line_colors'][0],
                linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('max($d$)')
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        self._savefig(fig, 'max_damage_vs_time')
        self._plots_generated += 1
        log(f"Generated max_damage_vs_time ({time.time() - t0:.1f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # C. CRACK TRACKING PLOTS
    # ═══════════════════════════════════════════════════════════════════

    def plot_crack_velocity_vs_time(self):
        """Crack tip velocity / cR vs time with branching markers."""
        log("Generating: crack_velocity_vs_time")
        t0 = time.time()
        if not self.has_crack_csv:
            log("No crack_tip.csv, skipping crack velocity plot", level='WARN')
            self._plots_skipped += 1
            return

        cc = self.crack_csv
        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))

        t = cc['t_us']
        vel_frac = cc.get('crack_vel_frac_cR', cc.get('crack_vel_mms', None))
        if vel_frac is None:
            log("No velocity column in crack_tip.csv, skipping", level='WARN')
            plt.close(fig)
            self._plots_skipped += 1
            return

        vel_smooth = _smooth(vel_frac, window_frac=0.05)
        ax.plot(t, vel_smooth, color=THEME['line_colors'][0], linewidth=1.6,
                label='Crack tip')

        # Mark branching onset with star marker (Bleyer 2017 Fig 4 convention).
        # Guard against false positives from the legacy upper/lower-half
        # detector: genuine branching fires once and persists, so >90% of
        # rows marked 'branched' is almost always the broken detector
        # tripping on the first damage band rather than a real second tip.
        if 'branched' in cc:
            branch_mask = cc['branched'] > 0
            frac_branched = float(branch_mask.mean()) if len(branch_mask) else 0.0
            spurious = frac_branched > 0.9
            if branch_mask.any() and not spurious:
                branch_idx = np.where(branch_mask)[0][0]
                branch_t = t.iloc[branch_idx] if hasattr(t, 'iloc') else t[branch_idx]
                branch_v = vel_smooth[branch_idx]
                ax.plot(branch_t, branch_v, marker='*', markersize=14,
                        color='k', zorder=5, label=f'Branching ({branch_t:.1f} µs)')
                ax.axvline(branch_t, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        # Reference lines
        if self.c_R:
            ax.axhline(0.6, color='gray', linestyle='--', linewidth=0.8,
                       label='0.6 $c_R$')
            ax.set_ylabel('$v_{\mathrm{tip}}$ / $c_R$')
        else:
            ax.set_ylabel('Crack tip velocity (mm/s)')

        ax.set_xlabel('Time ($\\mu$s)')
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, 'crack_velocity_vs_time')
        self._plots_generated += 1
        log(f"Generated crack_velocity_vs_time ({time.time() - t0:.1f}s)")

    def plot_crack_velocity_vs_position(self):
        """Crack velocity / cR vs crack tip x-position."""
        log("Generating: crack_velocity_vs_position")
        t0 = time.time()
        if not self.has_crack_csv:
            log("No crack_tip.csv, skipping crack velocity vs position", level='WARN')
            self._plots_skipped += 1
            return

        cc = self.crack_csv
        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))

        x = cc.get('crack_tip_x_mm', None)
        vel = cc.get('crack_vel_frac_cR', None)
        if x is None or vel is None:
            log("Missing crack_tip_x_mm or crack_vel_frac_cR in CSV, skipping", level='WARN')
            plt.close(fig)
            self._plots_skipped += 1
            return

        vel_smooth = _smooth(vel, window_frac=0.05)
        ax.plot(x, vel_smooth, color=THEME['line_colors'][0], linewidth=1.6,
                label='Crack tip')

        # Mark branching onset with star marker; suppress if >90% of rows
        # are marked branched (legacy-detector false positive, see above).
        if 'branched' in cc:
            branch_mask = cc['branched'] > 0
            frac_branched = float(branch_mask.mean()) if len(branch_mask) else 0.0
            spurious = frac_branched > 0.9
            if branch_mask.any() and not spurious:
                idx = np.where(branch_mask)[0][0]
                bx = x.iloc[idx] if hasattr(x, 'iloc') else x[idx]
                bv = vel.iloc[idx] if hasattr(vel, 'iloc') else vel[idx]
                ax.plot(bx, bv, marker='*', markersize=14, color='k', zorder=5,
                        label=f'Branching (x={bx:.1f} mm)')

        ax.set_xlabel('Crack tip position $x$ (mm)')
        ax.set_ylabel('$v_{\mathrm{tip}}$ / $c_R$')
        ax.axhline(0.6, color='gray', linestyle='--', linewidth=0.8,
                   label='0.6 $c_R$')
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, 'crack_velocity_vs_position')
        self._plots_generated += 1
        log(f"Generated crack_velocity_vs_position ({time.time() - t0:.1f}s)")

    def plot_dissipation_rate(self):
        """Damage dissipation rate Gamma/Gc vs crack tip position."""
        log("Generating: dissipation_rate")
        t0 = time.time()
        if not self.has_crack_csv or not self.has_energy_csv:
            log("Need both crack_tip.csv and energy.csv for dissipation rate, skipping", level='WARN')
            self._plots_skipped += 1
            return

        cc = self.crack_csv
        ec = self.energy_csv

        # Interpolate fracture energy to crack_tip timestamps
        t_crack = cc['t_us']
        t_energy = ec.get('t_s', ec.get('step', None))
        frac = ec.get('fracture', None)
        if t_energy is None or frac is None:
            log("Missing time/fracture columns in energy.csv, skipping dissipation rate", level='WARN')
            self._plots_skipped += 1
            return

        t_energy_us = t_energy * 1e6 if 't_s' in ec else t_energy
        frac_interp = np.interp(t_crack, t_energy_us, frac)

        # dGamma/dx = dE_frac/dx_tip
        x = cc.get('crack_tip_x_mm', None)
        if x is None or len(x) < 3:
            log("Not enough crack tip data for dissipation rate, skipping", level='WARN')
            self._plots_skipped += 1
            return

        # Smooth the cumulative quantities BEFORE differencing to avoid
        # amplifying numerical noise. Then take the derivative on the
        # smoothed series, then smooth the result one more time.
        frac_smooth = _smooth(frac_interp, window_frac=0.04)
        x_smooth    = _smooth(np.asarray(x),  window_frac=0.04)
        dE = np.gradient(frac_smooth)
        dx = np.gradient(x_smooth)
        dx = np.where(np.abs(dx) > 1e-10, dx, 1e-10)
        gamma_rate = _smooth(dE / dx, window_frac=0.06)

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))
        ax.plot(x_smooth, gamma_rate / self.Gc,
                color=THEME['line_colors'][0], linewidth=1.6)
        ax.set_xlabel('Crack tip position $x$ (mm)')
        ax.set_ylabel('$\\dot{\\Gamma}$ / $G_c$')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8,
                   label='$G_c$')
        ax.legend()
        fig.tight_layout()
        self._savefig(fig, 'dissipation_rate')
        self._plots_generated += 1
        log(f"Generated dissipation_rate ({time.time() - t0:.1f}s)")

    def plot_dissipation_vs_velocity(self):
        """Normalized dissipation rate Gamma/Gc vs normalized crack velocity v/cR.

        Reproduces Bleyer (2017) Fig 9 -- the universal Gamma(v) relationship.
        Points are instantaneous values during single-crack propagation
        (after initiation, before branching).

        Reference: Bleyer et al. (2017), Int J Fract 204, Fig 9, p. 94.
        """
        log("Generating: dissipation_vs_velocity (Bleyer Fig 9)")
        t0 = time.time()
        if not self.has_crack_csv or not self.has_energy_csv:
            log("Need both crack_tip.csv and energy.csv for dissipation vs velocity, skipping",
                level='WARN')
            self._plots_skipped += 1
            return

        cc = self.crack_csv
        ec = self.energy_csv

        # --- Get velocity (v / cR) from crack_tip.csv ---
        vel_frac = cc.get('crack_vel_frac_cR', None)
        x_tip = cc.get('crack_tip_x_mm', None)
        t_crack = cc.get('t_us', None)
        if vel_frac is None or x_tip is None or t_crack is None:
            log("Missing crack_vel_frac_cR, crack_tip_x_mm, or t_us in crack_tip.csv, skipping",
                level='WARN')
            self._plots_skipped += 1
            return

        # --- Compute dissipation rate Gamma = dE_fracture / dx_crack ---
        t_energy = ec.get('t_s', ec.get('step', None))
        frac = ec.get('fracture', None)
        if t_energy is None or frac is None:
            log("Missing time/fracture columns in energy.csv, skipping", level='WARN')
            self._plots_skipped += 1
            return

        t_energy_us = t_energy * 1e6 if 't_s' in ec else t_energy
        frac_interp = np.interp(t_crack, t_energy_us, frac)

        if len(x_tip) < 3:
            log("Not enough crack tip data points for dissipation vs velocity, skipping",
                level='WARN')
            self._plots_skipped += 1
            return

        # Smooth cumulative quantities BEFORE differencing
        frac_smooth = _smooth(frac_interp,            window_frac=0.04)
        x_tip_arr   = _smooth(np.asarray(x_tip),      window_frac=0.04)
        dE = np.gradient(frac_smooth)
        dx = np.gradient(x_tip_arr)
        dx = np.where(np.abs(dx) > 1e-10, dx, 1e-10)
        gamma_rate = _smooth(dE / dx, window_frac=0.06)

        # Normalize by Gc
        Gc = self.Gc if self.Gc > 0 else 1.0
        gamma_norm = gamma_rate / Gc

        # Filter to single-crack regime: after initiation (v > 0.05 cR),
        # before branching (branched == 0 if column exists)
        mask = np.abs(vel_frac) > 0.05
        if 'branched' in cc:
            mask = mask & (cc['branched'] < 0.5)
        # Also exclude negative or unreasonably high velocities
        mask = mask & (vel_frac > 0) & (vel_frac < 1.5)
        # Exclude negative dissipation (non-physical)
        mask = mask & (gamma_norm > 0)

        if mask.sum() < 2:
            log("Too few valid data points for dissipation vs velocity after filtering, skipping",
                level='WARN')
            self._plots_skipped += 1
            return

        # vel_frac and gamma_norm are time series; sort by velocity for the
        # phase-portrait plot, then smooth and bin to get a clean curve.
        v_plot = np.asarray(vel_frac[mask], dtype=float)
        g_plot = np.asarray(gamma_norm[mask], dtype=float)
        order  = np.argsort(v_plot)
        v_sorted = v_plot[order]
        g_sorted = g_plot[order]
        g_sorted = _smooth(g_sorted, window_frac=0.08)

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4.5))
        ax.plot(v_sorted, g_sorted, color=THEME['line_colors'][0],
                linewidth=1.8, label='Simulation')

        # Reference: experimental limiting velocity ~0.6 cR (Zhou 1996)
        ax.axvline(0.6, color='gray', linestyle='--', linewidth=0.8,
                   label='$0.6\\, c_R$ (Zhou 1996)')

        # Reference: Gc baseline
        ax.axhline(1.0, color=THEME['line_colors'][2], linestyle=':', linewidth=0.8,
                   label='$\\Gamma = G_c$')

        ax.set_xlabel('$v_{\mathrm{tip}}$ / $c_R$')
        ax.set_ylabel('$\\Gamma$ / $G_c$')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.legend(fontsize=THEME['font_size'] - 1)
        fig.tight_layout()
        self._savefig(fig, 'dissipation_vs_velocity')
        self._plots_generated += 1
        log(f"Generated dissipation_vs_velocity ({time.time() - t0:.1f}s)")

    def plot_damage_profiles_multi(self, x_positions=None, n_positions=5):
        """Symmetric vertical damage profiles d(y) at multiple x positions.

        Shows damage cross-sections perpendicular to crack plane at different
        positions along the crack path. Demonstrates damage zone widening
        before branching.

        Reference: Bleyer et al. (2017), Int J Fract 204, Fig 7, p. 92.
        """
        log("Generating: damage_profiles_multi (Bleyer Fig 7)")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping damage_profiles_multi", level='WARN')
            self._plots_skipped += 1
            return

        # Use last snapshot
        last_step = self.step_numbers[-1]
        grp = self._get_step(last_step)
        d = np.array(grp['damage_nodal'])
        t_us = self._get_time_us(last_step)

        x_all = self.nodes[:, 0]
        y_all = self.nodes[:, 1]
        y_mid = (y_all.max() + y_all.min()) / 2.0

        # Find crack extent: range of x where damage > 0.5 near crack plane
        crack_band = np.abs(y_all - y_mid) < 2 * self.l0
        cracked = crack_band & (d > 0.5)
        if not cracked.any():
            log("No cracked region found (max d < 0.5 near midplane), skipping",
                level='WARN')
            self._plots_skipped += 1
            return

        x_crack_min = x_all[cracked].min()
        x_crack_max = x_all[cracked].max()

        # Determine x-positions for cross-sections
        if x_positions is None:
            # Evenly spaced along crack path
            x_positions = np.linspace(x_crack_min + 0.05 * (x_crack_max - x_crack_min),
                                      x_crack_max - 0.05 * (x_crack_max - x_crack_min),
                                      n_positions)

        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        colors = THEME['line_colors']

        # Fig 8(d) inserted at 0.48\linewidth in a 2x2 layout. Figsize
        # 4.8x2.9 matches the other Fig 8 panels (build_pmma_fig8.py).
        fig, ax = plt.subplots(figsize=(4.8, 2.9))

        for i, xp in enumerate(x_positions):
            # Select nodes within a band around this x position
            band_width = 2 * self.l0
            mask = np.abs(x_all - xp) < band_width

            if mask.sum() < 5:
                log(f"  Skipping x={xp:.2f} mm: too few nodes in band", level='WARN')
                continue

            y_local = y_all[mask]
            d_local = d[mask]

            # Keep only the upper half (y >= y_mid) and normalise distance
            # from crack plane in units of l0, mirroring Bleyer Fig 7
            upper = y_local >= y_mid - 0.5 * self.l0
            y_local = y_local[upper]
            d_local = d_local[upper]

            sort_idx = np.argsort(y_local)
            y_sorted = y_local[sort_idx]
            d_sorted = d_local[sort_idx]
            y_norm = (y_sorted - y_mid) / self.l0

            label_char = labels[i] if i < len(labels) else f'x{i}'
            label = f'{label_char}: x = {xp:.1f} mm'
            ax.plot(y_norm, d_sorted,
                    color=colors[i % len(colors)], linewidth=1.6,
                    label=label)

        ax.set_xlabel('$y / \\ell_0$', fontsize=12)
        ax.set_ylabel('Damage $d$', fontsize=12)
        ax.tick_params(axis='both', labelsize=11)
        ax.set_xlim(0, 10)        # Bleyer Fig 7 range
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=10, frameon=False, borderaxespad=0.0,
                  handlelength=1.6, handletextpad=0.4, labelspacing=0.25)
        fig.tight_layout()
        self._savefig(fig, 'damage_profiles_multi')
        self._plots_generated += 1
        log(f"Generated damage_profiles_multi ({time.time() - t0:.1f}s)")

    def plot_velocity_with_holes(self, hole_positions=None):
        """Crack velocity vs position with gray bands showing hole locations.

        Shows how crack velocity oscillates when passing through an array of holes.
        Gray vertical bands indicate hole x-positions.

        Reference: Bleyer et al. (2017), Int J Fract 204, Fig 17, p. 95.
        """
        log("Generating: velocity_with_holes (Bleyer Fig 17)")
        t0 = time.time()
        if not self.has_crack_csv:
            log("No crack_tip.csv, skipping velocity_with_holes", level='WARN')
            self._plots_skipped += 1
            return

        cc = self.crack_csv
        x = cc.get('crack_tip_x_mm', None)
        vel = cc.get('crack_vel_frac_cR', None)
        if x is None or vel is None:
            log("Missing crack_tip_x_mm or crack_vel_frac_cR in CSV, skipping",
                level='WARN')
            self._plots_skipped += 1
            return

        # --- Determine hole positions ---
        if hole_positions is None:
            # Try to read from metadata
            n_holes = self.metadata.get('n_holes', None)
            hole_spacing = self.metadata.get('hole_spacing_mm', None)
            hole_diameter = self.metadata.get('hole_diameter_mm', None)
            hole_range_str = self.metadata.get('hole_range_x', None)

            if n_holes is not None and hole_spacing is not None:
                # Compute hole centers from range or from start position
                if hole_range_str is not None:
                    # Parse "[x_start, x_end]"
                    try:
                        cleaned = hole_range_str.strip('[] ')
                        parts = [float(p.strip()) for p in cleaned.split(',')]
                        x_start = parts[0]
                    except (ValueError, IndexError):
                        x_start = None
                else:
                    x_start = None

                if x_start is not None:
                    hole_positions = [x_start + i * hole_spacing
                                     for i in range(n_holes)]
                else:
                    # Estimate: evenly space along crack domain
                    x_min, x_max = x.min(), x.max()
                    x_center = (x_min + x_max) / 2.0
                    total_span = (n_holes - 1) * hole_spacing
                    x_start = x_center - total_span / 2.0
                    hole_positions = [x_start + i * hole_spacing
                                     for i in range(n_holes)]

                if hole_diameter is None:
                    hole_diameter = hole_spacing * 0.3  # rough default
            else:
                log("No hole info in metadata and no hole_positions given; "
                    "plotting velocity without hole bands", level='WARN')

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))

        # Draw gray bands for hole locations
        if hole_positions is not None:
            hole_diameter = self.metadata.get('hole_diameter_mm', None)
            if hole_diameter is None:
                # Estimate radius from spacing
                spacing = self.metadata.get('hole_spacing_mm', 1.0)
                hole_diameter = spacing * 0.3
            r = hole_diameter / 2.0
            for i, hx in enumerate(hole_positions):
                label = 'Holes' if i == 0 else None
                ax.axvspan(hx - r, hx + r, alpha=0.2, color='gray', label=label)

        vel_smooth = _smooth(np.asarray(vel, dtype=float), window_frac=0.04)
        ax.plot(x, vel_smooth, color=THEME['line_colors'][0],
                linewidth=1.6, label='Crack tip')

        # Reference lines
        ax.axhline(0.6, color='gray', linestyle='--', linewidth=0.8,
                   label='$0.6\\, c_R$')

        ax.set_xlabel('Crack tip position $x$ (mm)')
        ax.set_ylabel('$v_{\mathrm{tip}}$ / $c_R$')
        ax.legend(fontsize=THEME['font_size'] - 1)
        fig.tight_layout()
        self._savefig(fig, 'velocity_with_holes')
        self._plots_generated += 1
        log(f"Generated velocity_with_holes ({time.time() - t0:.1f}s)")

    def plot_space_time_diagram(self):
        """Crack-tip x-position vs time (line plot).

        Replaces the previous damage kymograph (space_time_diagram), which
        was visually dominated by the holes in perforated-plate benchmarks
        and hard to interpret. This simpler line plot directly shows the
        crack-tip trajectory in (x, t) space, from which crack speed,
        arrest events, and re-nucleation jumps can be read off.

        Output filename is kept as ``space_time_diagram`` for backward
        compatibility with downstream scripts.
        """
        log("Generating: space_time_diagram (crack tip x vs time)")
        t0_wall = time.time()

        if not self.has_crack_csv:
            log("No crack_tip.csv, skipping crack-tip trajectory plot",
                level='WARN')
            self._plots_skipped += 1
            return

        ct = self.crack_csv
        # Pick the time and x columns flexibly across schema variants
        t_key = 't_us' if 't_us' in ct else ('t_s' if 't_s' in ct else None)
        x_key = ('crack_tip_x_mm' if 'crack_tip_x_mm' in ct
                 else ('x_tip' if 'x_tip' in ct
                       else ('x' if 'x' in ct else None)))
        if t_key is None or x_key is None:
            log(f"crack_tip.csv missing required columns "
                f"(have: {list(ct.keys())}), skipping", level='WARN')
            self._plots_skipped += 1
            return

        t = np.asarray(ct[t_key])
        if t_key == 't_s':
            t = t * 1e6  # to µs
        x = np.asarray(ct[x_key])

        if len(t) < 2:
            log("Too few crack-tip samples, skipping", level='WARN')
            self._plots_skipped += 1
            return

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))
        ax.plot(t, x, color=THEME.get('crack_color', 'C3'), linewidth=1.6,
                label='Crack tip')
        ax.set_xlabel('Time ($\\mu$s)')
        ax.set_ylabel('Crack-tip position $x$ (mm)')
        ax.grid(True, alpha=0.3)

        # Mark branching time if recorded
        if 'branched' in ct:
            bm = np.asarray(ct['branched']) > 0
            if bm.any():
                bt = t[bm][0]
                ax.axvline(bt, color='black', linestyle='--', linewidth=1,
                           label=f'Branching ({bt:.1f} µs)')

        ax.legend(loc='best', fontsize=THEME['font_size'] - 1)
        fig.tight_layout()
        self._savefig(fig, 'space_time_diagram')
        self._plots_generated += 1
        log(f"Generated space_time_diagram ({time.time() - t0_wall:.1f}s)")

    # ═══════════════════════════════════════════════════════════════════
    # D. MULTI-RUN COMPARISON
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def plot_mesh_convergence(dirs, output_dir=None, fmt='png'):
        """Compare damage profiles from multiple mesh sizes."""
        log(f"Generating mesh convergence comparison ({len(dirs)} runs)")
        if output_dir is None:
            output_dir = os.path.join(dirs[0], 'figures')
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))
        colors = THEME['line_colors']

        for i, d in enumerate(dirs):
            bp = BenchmarkPostProcessor(d)
            if not bp.has_h5:
                continue
            steps = bp.step_numbers
            last = steps[-1]
            grp = bp._get_step(last)
            dam = np.array(grp['damage_nodal'])

            # Profile along crack plane
            mid_y = (bp.nodes[:, 1].max() + bp.nodes[:, 1].min()) / 2
            mask = np.abs(bp.nodes[:, 1] - mid_y) < 2 * bp.l0
            x = bp.nodes[mask, 0]
            d_line = dam[mask]
            sort_idx = np.argsort(x)

            n_nodes = bp.metadata.get('mesh', {}).get('n_nodes', '?')
            ax.plot(x[sort_idx] / bp.l0, d_line[sort_idx],
                    color=colors[i % len(colors)], linewidth=1.2,
                    label=f'{n_nodes} nodes')
            bp.close()

        ax.set_xlabel('$x / \\ell_0$')
        ax.set_ylabel('Damage $d$')
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, f'mesh_convergence.{fmt}')
        fig.savefig(path, dpi=THEME['dpi'], bbox_inches='tight')
        plt.close(fig)
        log(f"Saved: {path}")

    @staticmethod
    def plot_velocity_parametric(dirs, output_dir=None, fmt='png'):
        """Compare crack velocities from multiple runs (parametric study)."""
        log(f"Generating velocity parametric comparison ({len(dirs)} runs)")
        if output_dir is None:
            output_dir = os.path.join(dirs[0], 'figures')
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4))
        colors = THEME['line_colors']

        for i, d in enumerate(dirs):
            cc = _load_csv(os.path.join(d, 'crack_tip.csv'))
            if cc is None:
                continue
            meta_path = os.path.join(d, 'run_metadata.json')
            label = os.path.basename(d)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    m = json.load(f)
                delta_U = m.get('delta_U_mm', m.get('solver', {}).get('delta_U_mm', ''))
                if delta_U:
                    label = f'$\\Delta U$ = {delta_U} mm'

            t = cc.get('t_us', None)
            v = cc.get('crack_vel_frac_cR', None)
            if t is not None and v is not None:
                c = colors[i % len(colors)]
                ax.plot(t, v, color=c, linewidth=1.2, label=label)

                # Star marker at branching onset (Bleyer Fig 4 convention)
                if 'branched' in cc:
                    bm = cc['branched'] > 0
                    if bm.any():
                        idx = np.where(bm)[0][0]
                        bt = t.iloc[idx] if hasattr(t, 'iloc') else t[idx]
                        bv = v.iloc[idx] if hasattr(v, 'iloc') else v[idx]
                        ax.plot(bt, bv, marker='*', markersize=12,
                                color=c, markeredgecolor='k', zorder=5)

        ax.set_xlabel('Time ($\\mu$s)')
        ax.set_ylabel('$v_{\mathrm{tip}}$ / $c_R$')
        ax.axhline(0.6, color='gray', linestyle='--', linewidth=0.8,
                   label='$0.6\\, c_R$')
        ax.legend(fontsize=THEME['font_size'] - 1)
        fig.tight_layout()
        path = os.path.join(output_dir, f'velocity_parametric.{fmt}')
        fig.savefig(path, dpi=THEME['dpi'], bbox_inches='tight')
        plt.close(fig)
        log(f"Saved: {path}")

    @staticmethod
    def plot_dissipation_rate_compare(dirs, labels=None, output_dir=None,
                                      fmt='png'):
        """Multi-run damage dissipation rate Gamma/Gc vs crack tip position.

        Reproduces Bleyer (2017) Fig. 6: overlay of normalised dissipation
        rate against crack tip horizontal position for different loading
        levels. Each run directory must contain ``crack_tip.csv`` and
        ``energy.csv`` populated with cumulative fracture energy.
        """
        log(f"Generating dissipation_rate comparison ({len(dirs)} runs)")
        if output_dir is None:
            output_dir = os.path.join(dirs[0], 'figures')
        os.makedirs(output_dir, exist_ok=True)

        if labels is None:
            labels = [os.path.basename(d) for d in dirs]

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4.5))
        colors = THEME['line_colors']

        for i, (d, lbl) in enumerate(zip(dirs, labels)):
            cc = _load_csv(os.path.join(d, 'crack_tip.csv'))
            ec = _load_csv(os.path.join(d, 'energy.csv'))
            if cc is None or ec is None:
                log(f"  missing CSV in {d}, skipping", level='WARN')
                continue
            x = np.asarray(cc.get('crack_tip_x_mm', []), dtype=float)
            t_crack = np.asarray(cc.get('t_us', []), dtype=float)
            t_energy = ec.get('t_s', ec.get('step', None))
            frac = ec.get('fracture', None)
            if t_energy is None or frac is None or x.size < 5:
                log(f"  insufficient data in {d}, skipping", level='WARN')
                continue
            t_e = np.asarray(t_energy, dtype=float)
            if 't_s' in ec:
                t_e = t_e * 1e6
            frac_interp = np.interp(t_crack, t_e, np.asarray(frac, dtype=float))

            # Trim the steady-tip phase: once the crack reaches the right
            # boundary it stops moving but fracture energy keeps growing,
            # which makes dE/dx blow up. Keep only the strictly propagating
            # regime, defined as everything before x reaches its asymptote
            # (within 0.3 mm of x_max).
            x_max_obs = x.max()
            propagating = x < (x_max_obs - 0.3)
            if propagating.sum() < 10:
                propagating = np.ones_like(x, dtype=bool)
            x       = x[propagating]
            t_crack = t_crack[propagating]
            frac_interp = frac_interp[propagating]

            # Smooth then differentiate, then smooth again
            frac_s = _smooth(frac_interp, window_frac=0.04)
            x_s    = _smooth(x,            window_frac=0.04)
            dE = np.gradient(frac_s)
            dx = np.gradient(x_s)
            # Mask points where dx is too small (numerical noise)
            valid = np.abs(dx) > 1e-4   # 0.1 µm of crack advance per sample
            dx_safe = np.where(valid, dx, 1.0)
            gamma_rate = dE / dx_safe
            gamma_rate[~valid] = np.nan
            gamma_rate = _smooth(np.nan_to_num(gamma_rate, nan=np.nanmean(gamma_rate)),
                                 window_frac=0.06)

            # Try to read Gc from metadata
            meta = {}
            mp = os.path.join(d, 'run_metadata.json')
            if os.path.exists(mp):
                meta = json.load(open(mp))
            Gc = meta.get('material', {}).get('Gc', 0.3)
            if Gc <= 0:
                Gc = 0.3

            c = colors[i % len(colors)]
            ax.plot(x_s, gamma_rate / Gc, color=c, linewidth=1.6, label=lbl)

            # Star marker at branching position
            if 'branched' in cc:
                bm = np.asarray(cc['branched']) > 0
                if bm.any():
                    bidx = int(np.where(bm)[0][0])
                    if bidx < len(x_s):
                        ax.plot(x_s[bidx], (gamma_rate / Gc)[bidx],
                                marker='*', markersize=12, color=c,
                                markeredgecolor='k', zorder=5)

        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8,
                   label='$\\Gamma = G_c$')
        ax.axhline(2.0, color='gray', linestyle='--', linewidth=0.8,
                   label='$\\Gamma = 2 G_c$')
        ax.set_xlabel('Crack tip position $x$ (mm)')
        ax.set_ylabel('$\\Gamma / G_c$')
        ax.legend(fontsize=THEME['font_size'] - 1)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        path = os.path.join(output_dir, f'dissipation_rate_compare.{fmt}')
        fig.savefig(path, dpi=THEME['dpi'], bbox_inches='tight')
        plt.close(fig)
        log(f"Saved: {path}")

    @staticmethod
    def plot_normalized_energies_compare(dirs, labels=None, output_dir=None,
                                         fmt='png',
                                         line_styles=('-', '--', ':', '-.'),
                                         t_max_us=None):
        """Multi-configuration normalised energy overlay (Bleyer Fig 16).

        For each run directory, reads ``energy.csv`` and plots the
        elastic, kinetic, and dissipated (fracture) energies normalised
        by the initial elastic energy on a single axis. Different
        configurations are distinguished by line style; energy types
        are distinguished by colour, matching Bleyer et al. (2017)
        Fig.~16 (elastic in blue, dissipated in red, kinetic in green).

        Parameters
        ----------
        dirs : list of str
            Run directories to overlay (one per configuration).
        labels : list of str, optional
            Display labels for each configuration. Defaults to the
            basename of each directory.
        output_dir : str, optional
            Where to save the figure. Defaults to ``dirs[0]/figures``.
        fmt : {'png','pdf','svg'}
        line_styles : tuple of str
            Cycled across configurations (max 4).
        t_max_us : float, optional
            Crop the time axis to ``[0, t_max_us]`` to focus on the
            single-crack regime, as Bleyer does (their Fig.~16 stops
            at 20~µs).
        """
        log(f"Generating normalised energies comparison "
            f"({len(dirs)} configurations)")
        if output_dir is None:
            output_dir = os.path.join(dirs[0], 'figures')
        os.makedirs(output_dir, exist_ok=True)

        if labels is None:
            labels = [os.path.basename(d) for d in dirs]

        # Bleyer Fig 16 colours: elastic blue, kinetic green, dissipated red
        colour_elastic    = '#1f77b4'
        colour_kinetic    = '#2ca02c'
        colour_dissipated = '#d62728'

        fig, ax = plt.subplots(figsize=(THEME['figwidth'], 4.5))

        for i, (d, lbl) in enumerate(zip(dirs, labels)):
            ec = _load_csv(os.path.join(d, 'energy.csv'))
            if ec is None:
                log(f"  no energy.csv in {d}, skipping", level='WARN')
                continue

            t_key = 't_us' if 't_us' in ec else (
                't_s' if 't_s' in ec else 'step')
            t = np.asarray(ec[t_key], dtype=float)
            if t_key == 't_s':
                t = t * 1e6   # to µs

            elastic = np.asarray(ec.get('elastic', []), dtype=float)
            kinetic = np.asarray(ec.get('kinetic', []), dtype=float)
            frac    = np.asarray(ec.get('fracture', []), dtype=float)
            if elastic.size == 0:
                log(f"  no 'elastic' column in {d}/energy.csv, skipping",
                    level='WARN')
                continue

            E0 = elastic[0] if elastic[0] > 0 else 1.0
            ls = line_styles[i % len(line_styles)]

            # Crop to focus on single-crack regime if requested
            if t_max_us is not None:
                m = t <= t_max_us
                t = t[m]; elastic = elastic[m]
                kinetic = kinetic[m] if kinetic.size else kinetic
                frac    = frac[m] if frac.size else frac

            ax.plot(t, elastic / E0, color=colour_elastic,
                    linestyle=ls, linewidth=1.6,
                    label=f'{lbl}' if i == 0 else None)
            if kinetic.size:
                ax.plot(t, kinetic / E0, color=colour_kinetic,
                        linestyle=ls, linewidth=1.6)
            if frac.size:
                ax.plot(t, frac / E0, color=colour_dissipated,
                        linestyle=ls, linewidth=1.6)

        # Two-part legend: line styles for configurations, colours for types
        from matplotlib.lines import Line2D
        cfg_handles = [
            Line2D([0], [0], color='#222', linestyle=line_styles[i % len(line_styles)],
                   linewidth=1.6, label=lbl)
            for i, lbl in enumerate(labels)
        ]
        type_handles = [
            Line2D([0], [0], color=colour_elastic,    linewidth=1.6, label='Elastic'),
            Line2D([0], [0], color=colour_dissipated, linewidth=1.6, label='Dissipated'),
            Line2D([0], [0], color=colour_kinetic,    linewidth=1.6, label='Kinetic'),
        ]
        first_legend  = ax.legend(handles=cfg_handles,  loc='center left',
                                  fontsize=THEME['font_size'] - 1, title='Configuration')
        ax.add_artist(first_legend)
        ax.legend(handles=type_handles, loc='upper right',
                  fontsize=THEME['font_size'] - 1, title='Energy')

        ax.set_xlabel('Time ($\\mu$s)')
        ax.set_ylabel('Relative energy ($E/E_0$)')
        if t_max_us is not None:
            ax.set_xlim(0, t_max_us)
        fig.tight_layout()
        path = os.path.join(output_dir, f'normalized_energies_compare.{fmt}')
        fig.savefig(path, dpi=THEME['dpi'], bbox_inches='tight')
        plt.close(fig)
        log(f"Saved: {path}")

    # ═══════════════════════════════════════════════════════════════════
    # E. ANIMATION
    # ═══════════════════════════════════════════════════════════════════

    def _rasterize_frame(self, fig, palette=True):
        """Render a matplotlib figure to a PIL Image.

        Palette conversion (P-mode, 128 colors) is what keeps the final
        animated GIF in the 1–5 MB target band; adaptive palette avoids
        dithering artefacts on smooth scalar fields.
        """
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        rgb = Image.frombytes('RGBA', (w, h), buf).convert('RGB')
        if palette:
            return rgb.convert('P', palette=Image.ADAPTIVE, colors=128)
        return rgb

    def _save_animation(self, frames, stem, fps, animation_format):
        """Save frames as GIF/APNG/MP4, falling back cleanly when needed."""
        animation_format = (animation_format or 'gif').lower().lstrip('.')
        if animation_format not in {'gif', 'apng', 'mp4'}:
            raise ValueError(
                f"Unsupported animation format '{animation_format}'. "
                "Choose gif, apng, or mp4.")

        if animation_format == 'mp4' and shutil.which('ffmpeg') is None:
            log("ffmpeg not found; falling back to GIF", level='WARN')
            animation_format = 'gif'

        ext = 'png' if animation_format == 'apng' else animation_format
        path = os.path.join(self.figures_dir, f'{stem}.{ext}')

        if animation_format == 'mp4':
            rgb_frames = [f.convert('RGB') for f in frames]
            w, h = rgb_frames[0].size
            if w % 2 or h % 2:
                w += w % 2
                h += h % 2
                rgb_frames = [f.resize((w, h), Image.Resampling.LANCZOS)
                              for f in rgb_frames]
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{w}x{h}',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-vcodec', 'libx264',
                '-preset', 'veryfast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                path,
            ]
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            try:
                for frame in rgb_frames:
                    proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                proc.stdin.close()
                stderr = proc.stderr.read().decode('utf-8', errors='replace')
                proc.wait()
            except Exception:
                proc.kill()
                raise
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed while writing {path}: {stderr}")
        elif animation_format == 'apng':
            rgb_frames = [f.convert('RGB') for f in frames]
            rgb_frames[0].save(
                path, format='PNG', save_all=True,
                append_images=rgb_frames[1:],
                duration=1000 // fps, loop=0)
        else:
            palette_frames = [
                f if f.mode == 'P'
                else f.convert('P', palette=Image.ADAPTIVE, colors=128)
                for f in frames
            ]
            palette_frames[0].save(
                path, save_all=True, append_images=palette_frames[1:],
                duration=1000 // fps, loop=0, optimize=True, disposal=2)

        return path, animation_format.upper()

    def _raster_plan(self, width=960):
        """Precompute pixel -> triangle barycentric weights for fast videos.

        Matplotlib ``tripcolor`` is publication-friendly but expensive for
        repeated unstructured-mesh frames. For animations we can keep the
        mesh fixed, rasterise to a regular image grid once, and then update
        only nodal values per frame.
        """
        key = int(width)
        if key in self._raster_cache:
            return self._raster_cache[key]

        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        aspect = (ymax - ymin) / max(xmax - xmin, 1e-12)
        height = max(2, int(round(width * aspect)))

        gx = np.linspace(xmin, xmax, width)
        gy = np.linspace(ymax, ymin, height)
        xx, yy = np.meshgrid(gx, gy)
        tri_idx = self._tri.get_trifinder()(xx.ravel(), yy.ravel())
        inside = tri_idx >= 0

        elem = self._plot_elements[tri_idx[inside]]
        x1 = x[elem[:, 0]]
        y1 = y[elem[:, 0]]
        x2 = x[elem[:, 1]]
        y2 = y[elem[:, 1]]
        x3 = x[elem[:, 2]]
        y3 = y[elem[:, 2]]
        px = xx.ravel()[inside]
        py = yy.ravel()[inside]
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
        w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
        w3 = 1.0 - w1 - w2

        plan = {
            'shape': (height, width),
            'inside': inside,
            'elem': elem,
            'weights': np.column_stack((w1, w2, w3)),
        }
        self._raster_cache[key] = plan
        return plan

    def _raster_frame(self, nodal_field, cmap_name, vmin, vmax, label,
                      width=960):
        plan = self._raster_plan(width)
        img = np.full((*plan['shape'], 4), 255, dtype=np.uint8)
        values = np.asarray(nodal_field)[plan['elem']]
        values = np.sum(values * plan['weights'], axis=1)
        values = np.clip((values - vmin) / max(vmax - vmin, 1e-30), 0, 1)
        rgba = (plt.get_cmap(cmap_name)(values) * 255).astype(np.uint8)
        flat = img.reshape((-1, 4))
        flat[plan['inside']] = rgba
        pil = Image.fromarray(img, mode='RGBA').convert('RGB')
        draw = ImageDraw.Draw(pil)
        draw.rectangle((8, 8, 310, 34), fill=(255, 255, 255))
        draw.text((14, 14), label, fill=(0, 0, 0))
        return pil

    def make_damage_animation(self, fps=12, max_frames=80,
                              animation_format='mp4', renderer='raster',
                              raster_width=960):
        """Generate damage evolution animation from H5 data."""
        log(f"Generating: damage_evolution.{animation_format}")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping damage animation", level='WARN')
            self._plots_skipped += 1
            return

        steps = self.step_numbers
        skip = max(1, len(steps) // max_frames)
        selected = steps[::skip]
        cmap = _get_cmap(THEME['damage'])
        renderer = (renderer or 'matplotlib').lower()
        palette = (animation_format or 'gif').lower() == 'gif'
        log(f"Rendering {len(selected)} frames for damage animation...")

        frames = []
        for s in selected:
            grp = self._get_step(s)
            d = np.array(grp['damage_nodal'])
            t_us = self._get_time_us(s)

            if renderer == 'raster':
                frames.append(self._raster_frame(
                    d, THEME['damage'], 0.0, 1.0,
                    f't = {t_us:.1f} us  |  max(d) = {d.max():.3f}',
                    width=raster_width))
                continue

            # figsize*dpi = 700x450 px per frame — readable on retina
            # screens and in Overleaf previews; total GIF stays in the
            # 1-5 MB band after 128-color palette + optimize.
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
            ax.tripcolor(self._tri, d, shading='gouraud',
                         cmap=cmap, vmin=0, vmax=1)
            ax.set_aspect('equal')
            ax.set_title(f't = {t_us:.1f} $\\mu$s  |  max(d) = {d.max():.3f}')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            fig.tight_layout()
            frames.append(self._rasterize_frame(fig, palette=palette))
            plt.close(fig)

        if frames:
            path, fmt = self._save_animation(
                frames, 'damage_evolution', fps, animation_format)
            self._plots_generated += 1
            size_mb = os.path.getsize(path) / 1e6
            log(f"Saved: {os.path.basename(path)} [{fmt}] "
                f"({len(frames)} frames, {size_mb:.1f} MB, "
                f"{time.time() - t0:.1f}s)")

    def make_damage_gif(self, fps=12, max_frames=80):
        """Backward-compatible wrapper for callers expecting GIF output."""
        self.make_damage_animation(fps=fps, max_frames=max_frames,
                                   animation_format='gif')

    def make_stress_animation(self, stress_type='von_mises', fps=12,
                              max_frames=80, animation_format='mp4',
                              renderer='raster', raster_width=960):
        """Generate stress evolution animation from H5 data."""
        log(f"Generating: stress_evolution.{animation_format}")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping stress animation", level='WARN')
            self._plots_skipped += 1
            return

        steps = self.step_numbers
        skip = max(1, len(steps) // max_frames)
        selected = steps[::skip]
        cmap = _get_cmap(THEME['stress_vm'])

        # First pass: find global vmax (sample a subset; full-pass is wasteful)
        vmax = 0
        for s in selected[::max(1, len(selected)//10)]:
            grp = self._get_step(s)
            data = {k: np.array(grp[k]) for k in grp.keys()}
            sxx, syy, sxy = self._get_stress(grp, data)
            if sxx is not None:
                vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
                vmax = max(vmax, self._elem_to_node(vm).max())
        if vmax == 0:
            log("No stress data for animation, skipping", level='WARN')
            self._plots_skipped += 1
            return

        renderer = (renderer or 'matplotlib').lower()
        palette = (animation_format or 'gif').lower() == 'gif'
        log(f"Rendering {len(selected)} frames for stress animation...")
        frames = []
        for s in selected:
            grp = self._get_step(s)
            data = {k: np.array(grp[k]) for k in grp.keys()}
            sxx, syy, sxy = self._get_stress(grp, data)
            if sxx is None:
                continue
            vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
            vm_nodal = self._elem_to_node(vm)
            t_us = self._get_time_us(s)

            if renderer == 'raster':
                frames.append(self._raster_frame(
                    vm_nodal, cmap, 0.0, vmax,
                    f't = {t_us:.1f} us  |  VM stress',
                    width=raster_width))
                continue

            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
            ax.tripcolor(self._tri, vm_nodal, shading='gouraud',
                         cmap=cmap, vmin=0, vmax=vmax)
            ax.set_aspect('equal')
            ax.set_title(f't = {t_us:.1f} $\\mu$s  |  VM stress')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            fig.tight_layout()
            frames.append(self._rasterize_frame(fig, palette=palette))
            plt.close(fig)

        if frames:
            path, fmt = self._save_animation(
                frames, 'stress_evolution', fps, animation_format)
            self._plots_generated += 1
            size_mb = os.path.getsize(path) / 1e6
            log(f"Saved: {os.path.basename(path)} [{fmt}] "
                f"({len(frames)} frames, {size_mb:.1f} MB, "
                f"{time.time() - t0:.1f}s)")

    def make_stress_gif(self, stress_type='von_mises', fps=12, max_frames=80):
        """Backward-compatible wrapper for callers expecting GIF output."""
        self.make_stress_animation(stress_type=stress_type, fps=fps,
                                   max_frames=max_frames,
                                   animation_format='gif')

    def make_displacement_animation(self, fps=12, max_frames=80,
                                    animation_format='mp4',
                                    renderer='raster', raster_width=960):
        """Generate displacement-magnitude evolution animation from H5 data."""
        log(f"Generating: displacement_evolution.{animation_format}")
        t0 = time.time()
        if not self.has_h5:
            log("No H5 data, skipping displacement animation", level='WARN')
            self._plots_skipped += 1
            return

        steps = self.step_numbers
        skip = max(1, len(steps) // max_frames)
        selected = steps[::skip]
        cmap = _get_cmap(THEME['displacement'])

        vmax = 0.0
        for s in selected[::max(1, len(selected)//10)]:
            grp = self._get_step(s)
            if 'displacement' not in grp:
                continue
            u = np.array(grp['displacement'])
            u_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
            vmax = max(vmax, float(u_mag.max()))
        if vmax <= 0.0:
            log("No displacement data for animation, skipping", level='WARN')
            self._plots_skipped += 1
            return

        renderer = (renderer or 'matplotlib').lower()
        palette = (animation_format or 'gif').lower() == 'gif'
        log(f"Rendering {len(selected)} frames for displacement animation...")
        frames = []
        for s in selected:
            grp = self._get_step(s)
            if 'displacement' not in grp:
                continue
            u = np.array(grp['displacement'])
            u_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
            t_us = self._get_time_us(s)

            if renderer == 'raster':
                frames.append(self._raster_frame(
                    u_mag, THEME['displacement'], 0.0, vmax,
                    f't = {t_us:.1f} us  |  max(|u|) = {u_mag.max():.3e} mm',
                    width=raster_width))
                continue

            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
            ax.tripcolor(self._tri, u_mag, shading='gouraud',
                         cmap=cmap, vmin=0, vmax=vmax)
            ax.set_aspect('equal')
            ax.set_title(
                f't = {t_us:.1f} $\\mu$s  |  max(|u|) = {u_mag.max():.3e} mm')
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            fig.tight_layout()
            frames.append(self._rasterize_frame(fig, palette=palette))
            plt.close(fig)

        if frames:
            path, fmt = self._save_animation(
                frames, 'displacement_evolution', fps, animation_format)
            self._plots_generated += 1
            size_mb = os.path.getsize(path) / 1e6
            log(f"Saved: {os.path.basename(path)} [{fmt}] "
                f"({len(frames)} frames, {size_mb:.1f} MB, "
                f"{time.time() - t0:.1f}s)")

    def make_displacement_gif(self, fps=12, max_frames=80):
        """Backward-compatible wrapper for callers expecting GIF output."""
        self.make_displacement_animation(fps=fps, max_frames=max_frames,
                                         animation_format='gif')

    # ═══════════════════════════════════════════════════════════════════
    # MAIN: Generate all plots
    # ═══════════════════════════════════════════════════════════════════

    def plot_solver_telemetry(self):
        """Four-panel solver-telemetry diagnostic (issue #300).

        Reads ``solver_telemetry.csv`` (per-step Newton/PCG iteration counts
        + staggered residual + dt) emitted by ``run_config.py`` and
        ``timing_per_step.csv`` (per-step wall time) and renders:

          (1) iters per step  : newton, pcg_mech, pcg_pf overlaid
          (2) residual decay  : log-y stagger residual vs step
          (3) dt history      : dt vs step (flat for non-adaptive runs)
          (4) wall-time CDF   : cumulative wall time across the run

        Produced filename: ``solver_telemetry.png`` under the run's
        figures dir. Skipped silently with a warning if the CSV is
        missing (e.g. legacy runs predating #300).
        """
        log("Generating: solver_telemetry")
        t0_wall = time.time()

        tele_path = os.path.join(self.csv_dir, 'solver_telemetry.csv')
        if not os.path.exists(tele_path):
            log("solver_telemetry.csv missing -- skipping (legacy run?)",
                level='WARN')
            self._plots_skipped += 1
            return

        tele = _load_csv(tele_path)
        if tele is None or 'step' not in tele:
            log("solver_telemetry.csv unreadable -- skipping", level='WARN')
            self._plots_skipped += 1
            return

        step = np.asarray(tele['step'])
        newton = np.asarray(tele.get('newton_iters', np.zeros_like(step)))
        pcg_m = np.asarray(tele.get('pcg_iters_mech', np.zeros_like(step)))
        pcg_p = np.asarray(tele.get('pcg_iters_pf', np.zeros_like(step)))
        resid = np.asarray(tele.get('residual', np.full_like(step, np.nan, dtype=float)))
        dts = np.asarray(tele.get('dt', np.zeros_like(step, dtype=float)))

        # Wall time from timing_per_step.csv (Total Step Time, seconds).
        timing_path = os.path.join(self.csv_dir, 'timing_per_step.csv')
        wall = None
        if os.path.exists(timing_path):
            tcsv = _load_csv(timing_path)
            if tcsv is not None and 'Total Step Time' in tcsv:
                wall = np.asarray(tcsv['Total Step Time'])

        fig, axes = plt.subplots(2, 2, figsize=(THEME['figwidth_double'] * 0.6, 7))
        ax_it, ax_r, ax_dt, ax_w = axes.ravel()

        ax_it.plot(step, newton, label='Newton/stagger', linewidth=1.4)
        ax_it.plot(step, pcg_m, label='PCG mech', linewidth=1.0, alpha=0.8)
        ax_it.plot(step, pcg_p, label='PCG PF', linewidth=1.0, alpha=0.8)
        ax_it.set_xlabel('Step'); ax_it.set_ylabel('Iterations')
        ax_it.set_title('Iters / step'); ax_it.grid(True, alpha=0.3)
        ax_it.legend(fontsize=THEME['font_size'] - 1)

        # Residual: positive only on log axis; mask NaN (explicit dynamics).
        finite = np.isfinite(resid) & (resid > 0)
        if finite.any():
            ax_r.semilogy(step[finite], resid[finite], '.-', markersize=3, linewidth=1)
            ax_r.set_ylabel('Stagger residual')
        else:
            ax_r.text(0.5, 0.5, 'No stagger residual\n(explicit dynamics)',
                      ha='center', va='center', transform=ax_r.transAxes)
        ax_r.set_xlabel('Step'); ax_r.set_title('Residual at convergence')
        ax_r.grid(True, alpha=0.3, which='both')

        ax_dt.plot(step, dts, linewidth=1.2)
        ax_dt.set_xlabel('Step'); ax_dt.set_ylabel('dt')
        ax_dt.set_title('Timestep history'); ax_dt.grid(True, alpha=0.3)

        if wall is not None and len(wall) == len(step):
            ax_w.plot(step, np.cumsum(wall), linewidth=1.4)
            ax_w.set_xlabel('Step'); ax_w.set_ylabel('Cumulative wall (s)')
            ax_w.set_title('Wall-time CDF')
        else:
            ax_w.text(0.5, 0.5, 'timing_per_step.csv\nunavailable',
                      ha='center', va='center', transform=ax_w.transAxes)
            ax_w.set_title('Wall-time CDF')
        ax_w.grid(True, alpha=0.3)

        fig.tight_layout()
        self._savefig(fig, 'solver_telemetry')
        self._plots_generated += 1
        log(f"Generated solver_telemetry ({time.time() - t0_wall:.1f}s)")

    def generate_all(self, skip_gif=False, fields=None, animation_format='mp4',
                     animation_fields='damage', max_frames=80,
                     animation_renderer='raster', raster_width=960):
        """Generate all applicable plots for this run."""
        overall_t0 = time.time()
        self._plots_generated = 0
        self._plots_skipped = 0

        categories = (fields or 'all').split(',')
        do_all = 'all' in categories

        log(f"{'=' * 60}")
        log(f"Generating plots for: {self.problem_name}")
        log(f"Output: {self.figures_dir}")
        log(f"Categories: {categories}")
        log(f"{'=' * 60}")

        # Generate missing CSVs from H5
        if self.has_h5:
            self._generate_csvs_from_h5()

        # A. Spatial fields
        if do_all or 'damage' in categories:
            try:
                self.plot_damage_multipanel()
            except Exception as e:
                log(f"Error in plot_damage_multipanel: {e}", level='ERROR')
                self._plots_skipped += 1
            try:
                self.plot_damage_profile()
            except Exception as e:
                log(f"Error in plot_damage_profile: {e}", level='ERROR')
                self._plots_skipped += 1

        if do_all or 'stress' in categories:
            try:
                self.plot_stress_fields()  # auto-selects max_principal for brittle
            except Exception as e:
                log(f"Error in plot_stress_fields: {e}", level='ERROR')
                self._plots_skipped += 1
            try:
                self.plot_displacement_field()
            except Exception as e:
                log(f"Error in plot_displacement_field: {e}", level='ERROR')
                self._plots_skipped += 1

        # B. Time series
        if do_all or 'energy' in categories:
            try:
                self.plot_energy_balance()
            except Exception as e:
                log(f"Error in plot_energy_balance: {e}", level='ERROR')
                self._plots_skipped += 1
            try:
                self.plot_energy_normalized()
            except Exception as e:
                log(f"Error in plot_energy_normalized: {e}", level='ERROR')
                self._plots_skipped += 1
            try:
                self.plot_max_damage_vs_time()
            except Exception as e:
                log(f"Error in plot_max_damage_vs_time: {e}", level='ERROR')
                self._plots_skipped += 1
            if not self.is_dynamic:
                try:
                    self.plot_force_displacement()
                except Exception as e:
                    log(f"Error in plot_force_displacement: {e}", level='ERROR')
                    self._plots_skipped += 1

        # C. Crack tracking
        if do_all or 'crack' in categories:
            if self.is_dynamic and self.has_crack_csv:
                try:
                    self.plot_crack_velocity_vs_time()
                except Exception as e:
                    log(f"Error in plot_crack_velocity_vs_time: {e}", level='ERROR')
                    self._plots_skipped += 1
                try:
                    self.plot_crack_velocity_vs_position()
                except Exception as e:
                    log(f"Error in plot_crack_velocity_vs_position: {e}", level='ERROR')
                    self._plots_skipped += 1
                try:
                    self.plot_dissipation_rate()
                except Exception as e:
                    log(f"Error in plot_dissipation_rate: {e}", level='ERROR')
                    self._plots_skipped += 1
                try:
                    self.plot_dissipation_vs_velocity()
                except Exception as e:
                    log(f"Error in plot_dissipation_vs_velocity: {e}", level='ERROR')
                    self._plots_skipped += 1
                try:
                    self.plot_damage_profiles_multi()
                except Exception as e:
                    log(f"Error in plot_damage_profiles_multi: {e}", level='ERROR')
                    self._plots_skipped += 1
                try:
                    self.plot_velocity_with_holes()
                except Exception as e:
                    log(f"Error in plot_velocity_with_holes: {e}", level='ERROR')
                    self._plots_skipped += 1
                try:
                    self.plot_space_time_diagram()
                except Exception as e:
                    log(f"Error in plot_space_time_diagram: {e}", level='ERROR')
                    self._plots_skipped += 1
            elif self.is_dynamic and not self.has_crack_csv:
                log("Dynamic run but no crack_tip.csv, skipping crack plots", level='WARN')

        # D. Solver telemetry (issue #300). Always cheap: it's just CSV
        # plotting, no H5 reads. Categorised under 'telemetry' for users
        # who want to opt out.
        if do_all or 'telemetry' in categories:
            try:
                self.plot_solver_telemetry()
            except Exception as e:
                log(f"Error in plot_solver_telemetry: {e}", level='ERROR')
                self._plots_skipped += 1

        # E. Animation
        if not skip_gif and (do_all or 'gif' in categories):
            anim_fields = {
                item.strip().lower()
                for item in (animation_fields or 'damage,stress').split(',')
                if item.strip()
            }
            if 'damage' in anim_fields:
                try:
                    self.make_damage_animation(
                        max_frames=max_frames,
                        animation_format=animation_format,
                        renderer=animation_renderer,
                        raster_width=raster_width)
                except Exception as e:
                    log(f"Error in make_damage_animation: {e}", level='ERROR')
                    self._plots_skipped += 1
            if 'stress' in anim_fields or 'max_principal_stress' in anim_fields:
                try:
                    self.make_stress_animation(
                        max_frames=max_frames,
                        animation_format=animation_format,
                        renderer=animation_renderer,
                        raster_width=raster_width)
                except Exception as e:
                    log(f"Error in make_stress_animation: {e}", level='ERROR')
                    self._plots_skipped += 1
            if 'displacement' in anim_fields or 'u' in anim_fields:
                try:
                    self.make_displacement_animation(
                        max_frames=max_frames,
                        animation_format=animation_format,
                        renderer=animation_renderer,
                        raster_width=raster_width)
                except Exception as e:
                    log(f"Error in make_displacement_animation: {e}",
                        level='ERROR')
                    self._plots_skipped += 1

        total_time = time.time() - overall_t0
        log(f"{'=' * 60}")
        log(f"SUMMARY: {self._plots_generated} plots generated, "
            f"{self._plots_skipped} skipped, total time {total_time:.1f}s")
        log(f"All figures in: {self.figures_dir}")
        log(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Paper-quality post-processing for phast benchmarks')
    parser.add_argument('run_dir', nargs='?', default='.',
                        help='Run directory (default: current dir)')
    parser.add_argument('--dpi', type=int, default=None,
                        help='Override DPI (default: 300)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'],
                        default='png', help='Output format')
    parser.add_argument('--skip-gif', action='store_true',
                        help='Skip animation generation (slow)')
    parser.add_argument('--only-gifs', action='store_true',
                        help='Only regenerate animations (skip all PNG plots)')
    parser.add_argument('--animation-format',
                        choices=['gif', 'apng', 'mp4'],
                        default=os.environ.get('PF_ANIMATION_FORMAT', 'mp4'),
                        help='Animation container. MP4 needs ffmpeg and is '
                             'usually much smaller than GIF.')
    parser.add_argument('--animation-fields',
                        default=os.environ.get(
                            'PF_ANIMATION_FIELDS', 'damage'),
                        help='Comma-separated animation fields: '
                             'damage,stress,displacement')
    parser.add_argument('--animation-frames', type=int,
                        default=int(os.environ.get(
                            'PF_ANIMATION_FRAMES', '80')),
                        help='Maximum animation frames')
    parser.add_argument('--animation-renderer',
                        choices=['matplotlib', 'raster'],
                        default=os.environ.get(
                            'PF_ANIMATION_RENDERER', 'raster'),
                        help='Animation renderer. raster precomputes a fixed '
                             'pixel grid and is much faster for large meshes.')
    parser.add_argument('--raster-width', type=int,
                        default=int(os.environ.get(
                            'PF_ANIMATION_RASTER_WIDTH', '960')),
                        help='Pixel width for --animation-renderer=raster')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='Write outputs here instead of <run_dir>/figures/ '
                             '(useful when regenerating figures from archived runs)')
    parser.add_argument('--fields', type=str, default=None,
                        help='Comma-separated: damage,stress,energy,crack,gif,all')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='Multiple run dirs for comparison plots')

    args = parser.parse_args()

    if args.compare:
        BenchmarkPostProcessor.plot_mesh_convergence(
            args.compare, fmt=args.format)
        BenchmarkPostProcessor.plot_velocity_parametric(
            args.compare, fmt=args.format)
        return

    bp = BenchmarkPostProcessor(args.run_dir, dpi=args.dpi, fmt=args.format,
                                figures_dir=args.figures_dir)
    if args.only_gifs:
        bp.generate_all(skip_gif=False, fields='gif',
                        animation_format=args.animation_format,
                        animation_fields=args.animation_fields,
                        max_frames=args.animation_frames,
                        animation_renderer=args.animation_renderer,
                        raster_width=args.raster_width)
    else:
        bp.generate_all(skip_gif=args.skip_gif, fields=args.fields,
                        animation_format=args.animation_format,
                        animation_fields=args.animation_fields,
                        max_frames=args.animation_frames,
                        animation_renderer=args.animation_renderer,
                        raster_width=args.raster_width)
    bp.close()


if __name__ == '__main__':
    main()
