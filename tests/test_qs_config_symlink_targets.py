from pathlib import Path

import yaml

from phast.config_validation import validate_config_file, format_errors


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

QS_PROBLEM_CONFIGS = (
    "QS_lshaped_concrete.yaml",
    "QS_notched_holed_plate.yaml",
    "QS_notched_holed_plate_comsol_strict.yaml",
)

QS_MANIFEST_CONFIGS = (
    "QS_sens_tpb_rescue_visuals.yaml",
    "QS_sens_tpb_peak_window_corrected.yaml",
    "QS_mesh_convergence_arc_length.yaml",
)


def test_qs_root_config_symlinks_resolve():
    for name in QS_PROBLEM_CONFIGS + QS_MANIFEST_CONFIGS:
        path = CONFIG_DIR / name
        assert path.exists(), f"{path} is missing or points to a missing target"


def test_qs_problem_configs_validate():
    for name in QS_PROBLEM_CONFIGS:
        path = CONFIG_DIR / name
        _, errors = validate_config_file(path)
        assert errors == [], format_errors(errors, path)


def test_qs_manifest_configs_are_parseable_case_lists():
    for name in QS_MANIFEST_CONFIGS:
        path = CONFIG_DIR / name
        data = yaml.safe_load(path.read_text())
        assert data["schema_version"] == 1
        assert data["common_args"]
        assert data["cases"]
        for case in data["cases"]:
            assert case["label"]
            assert case["module"]
            assert case["compare"]
            assert case["args"]
