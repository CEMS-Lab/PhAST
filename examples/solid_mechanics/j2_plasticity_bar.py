"""Compatibility wrapper for the public J2 bar example."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("j2_bar") / "run.py"), run_name="__main__")
