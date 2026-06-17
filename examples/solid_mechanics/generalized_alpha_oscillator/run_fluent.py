"""Public Python/manual runner for the generalized-alpha oscillator diagnostic."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import run_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Run the diagnostic into --output-dir.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--output-dir", default="runs/generalized_alpha_oscillator")
    args = parser.parse_args()

    print("OK: diagnostic config is available for manual Python execution.")
    print("Canonical deck: examples/solid_mechanics/generalized_alpha_oscillator/config.yaml")
    if args.run:
        os.environ["PHAST_SOLID_MECH_OUTPUT_DIR"] = args.output_dir
        run_example(args.config)


if __name__ == "__main__":
    main()
