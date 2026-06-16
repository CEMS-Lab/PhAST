"""Compatibility wrapper for the public mixed-precision CG example."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("mixed_precision_cg") / "run.py"), run_name="__main__")
