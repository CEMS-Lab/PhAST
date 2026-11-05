"""Replot the cross-mesh recovery outputs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.current_plotting import main


if __name__ == "__main__":
    main()
