"""Compatibility wrapper for the public neo-Hookean plate example."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("neohookean_plate") / "run.py"), run_name="__main__")
