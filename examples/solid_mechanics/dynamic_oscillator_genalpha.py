"""Compatibility wrapper for the public generalized-alpha oscillator example."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("generalized_alpha_oscillator") / "run.py"), run_name="__main__")
