"""Run the selected cross-mesh synthetic toughness recovery."""

from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
RESULT_SHA256 = "972b12661ca1bd2bcdd37d915b0aba73566c3e59266f1949428052f57ef5a987"


def verify_recorded_results() -> dict:
    """Check the fixed historical numerical excerpt without importing the solver."""
    data = (BUNDLE / "recorded_results.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != RESULT_SHA256:
        raise ValueError("The recorded paper-result excerpt has changed")
    return json.loads(data)


def validate_selected_config(config: dict) -> None:
    """Keep this worked example on the mesh pair of both recorded recoveries."""
    if config.get("example") != "cross_mesh_toughness":
        raise ValueError("this entry point requires cross_mesh_toughness")
    profiles = config.get("profiles", {})
    if set(profiles) != {"full"}:
        raise ValueError("this worked example includes only the full profile")
    record = verify_recorded_results()
    profile = profiles["full"]
    pair = (profile.get("h_target_mm"), profile.get("h_trial_mm"))
    selected = (record["original_mesh_size_mm"], record["recovery_mesh_size_mm"])
    if pair != selected:
        raise ValueError("this worked example requires the 0.125 mm original and 0.1875 mm recovery mesh")


if __name__ == "__main__":
    verify_recorded_results()
    if sys.argv[1:] == ["--verify-only"]:
        print("Verified the recorded AD and Brent results; no calculation executed.")
    else:
        import argparse
        import yaml

        selection = argparse.ArgumentParser(add_help=False)
        selection.add_argument("--config", type=Path, default=BUNDLE / "config.yaml")
        selection.add_argument("--profile", choices=("full",), default="full")
        selected, _ = selection.parse_known_args()
        try:
            validate_selected_config(yaml.safe_load(selected.config.read_text()))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            selection.error(str(error))
        sys.path.insert(0, str(BUNDLE.parent))
        from _shared.current_recovery import main

        raise SystemExit(main(BUNDLE))
