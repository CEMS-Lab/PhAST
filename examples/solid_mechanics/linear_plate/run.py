"""Compatibility wrapper for the YAML-first linear plate example."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from phast.solid_mechanics_runners.linear_plate import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the linear elastic solid-mechanics FEA example.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
