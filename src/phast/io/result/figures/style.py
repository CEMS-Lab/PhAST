"""Figure style, colour maps and the save rule shared by every standard figure."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure

from ....utils.paper_figure_style import PAPER_STYLE

MAX_PX = 1990            # every PNG stays below 2000 px on both axes
DEFAULT_DPI = 150

STYLE = dict(PAPER_STYLE)
STYLE.update({
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "savefig.bbox": "tight",
})

DAMAGE_CMAP = "magma"          # damage: one sequential map, always 0..1
SEQUENTIAL_CMAP = "viridis"    # non-negative quantities (magnitudes, H)
DIVERGING_CMAP = "RdBu_r"      # signed quantities, symmetric limits
PHASE_COLOURS = ["#2F5DA8", "#D9B382", "#7FC4CF", "#B9C0C8", "#A3223F", "#5D8ACE", "#C98B3B",
                 "#4E8A3E", "#8E6BB0", "#E3A6B4"]
SET_COLOURS = ["#A3223F", "#215CAF", "#B7791F", "#2E7D32", "#6A3D9A", "#00838F", "#5D4037"]
LINE_COLOURS = ["#215CAF", "#A3223F", "#006C84", "#B7791F", "#16191D", "#6A3D9A", "#2E7D32", "#8C564B"]

def clean(text) -> str:
    """Text taken from a result file, as a string; every label drawn from file data passes through here."""
    return str(text)


def phase_colour(index: int) -> str:
    return PHASE_COLOURS[index % len(PHASE_COLOURS)]


@contextmanager
def figure(width: float, height: float, layout: str | None = None):
    """A new Figure (no pyplot state) built under the shared style."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(width, height), layout=layout)
        FigureCanvasAgg(fig)
        yield fig


def content_bottom(fig: Figure) -> float:
    """Lowest drawn point of the figure, in figure coordinates (after layout)."""
    fig.draw_without_rendering()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return bbox.y0 / fig.get_figheight()


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def save(fig: Figure, path: Path, dpi: int) -> tuple[int, int]:
    """Save ``fig`` as PNG below 2000 px on both axes; returns (width, height)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(STYLE):
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        w, h = png_size(path)
        tries = 0
        while max(w, h) > MAX_PX and tries < 4:
            dpi = max(20, int(dpi * MAX_PX / max(w, h) * 0.98))
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            w, h = png_size(path)
            tries += 1
    assert max(w, h) < 2000, f"{path.name} is {w} x {h} px"
    return w, h
