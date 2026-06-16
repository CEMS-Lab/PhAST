"""Compatibility wrapper for the public linear plate example."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("linear_plate") / "run.py"), run_name="__main__")
