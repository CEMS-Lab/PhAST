"""Shared matplotlib rcParams for paper figures.

Matches the serif style used by ``postprocess_paper.py`` so that
figures produced by demo/helper scripts visually match those produced by
the benchmark post-processor (and the surrounding cas-sc / STIX LaTeX
body text).

Usage::

    import matplotlib
    matplotlib.use('Agg')
    from paper_figure_style import apply_style
    apply_style()

    import matplotlib.pyplot as plt
    # ... now every figure will inherit the paper style ...
"""

import matplotlib.pyplot as plt


PAPER_STYLE = {
    'font.family':         'serif',
    'font.serif':          ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset':    'stix',
    'font.size':           10,
    'axes.labelsize':      11,
    'axes.titlesize':      11,
    'legend.fontsize':     9,
    'xtick.labelsize':     9,
    'ytick.labelsize':     9,
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
    'axes.unicode_minus':  False,
}


def apply_style():
    """Apply the paper style to matplotlib rcParams (global)."""
    plt.rcParams.update(PAPER_STYLE)
