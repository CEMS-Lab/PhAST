"""Filesystem checks for the public example artifact contract."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "examples" / "PUBLIC_EXAMPLES_CONTRACT.yaml"


def test_all_required_public_example_artifacts_exist() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    missing: list[str] = []

    for family in contract["families"]:
        for example in family["examples"]:
            example_path = ROOT / example["path"]
            for artifact in example.get("required_artifacts", []):
                artifact_path = example_path / artifact
                if not artifact_path.is_file():
                    missing.append(
                        f"{family['id']}/{example['id']}: "
                        f"{artifact_path.relative_to(ROOT)}"
                    )

    assert not missing, "Missing required public example artifacts:\n" + "\n".join(missing)
