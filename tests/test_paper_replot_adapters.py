"""Bounded adapter contracts using synthetic files, without scientific dependencies."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


ADAPTERS = Path(__file__).resolve().parents[1] / "reproduction" / "paper1"
NAMES = ("q2", "inverse", "forward")


def load_adapter(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ADAPTERS / f"replot_{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=NAMES)
def adapter(request: pytest.FixtureRequest) -> ModuleType:
    return load_adapter(request.param)


def entry(path: str = "data/field.bin", legacy: str = "results/field.bin") -> dict[str, Any]:
    return {"path": path, "legacy_path": legacy, "kind": "copy", "groups": ["q2"],
            "bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest()}


def make_archive(root: Path, rows: list[dict[str, Any]]) -> Path:
    root.mkdir()
    for row in rows:
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    (root / "manifest.json").write_text(json.dumps({"files": rows, "payload_bytes": 4 * len(rows)}))
    return root


def check_input(adapter: ModuleType, root: Path, row: dict[str, Any]) -> Path:
    if adapter.__name__ == "q2":
        return adapter.verified_path(root, adapter.relative_path(row["path"]), row)
    if adapter.__name__ == "inverse":
        return adapter.verify(root, row["path"], row)
    return adapter.checked_path(root, row)


def run_adapter(name: str, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-B", str(ADAPTERS / f"replot_{name}.py"), *args],
                          cwd=cwd, capture_output=True, text=True, check=False, timeout=15)


def test_exact_input_hash(adapter: ModuleType, tmp_path: Path) -> None:
    row = entry()
    root = make_archive(tmp_path / "archive", [row])
    path = check_input(adapter, root, row)
    assert path == root / row["path"]
    assert adapter.digest(path) == row["sha256"]


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "data/../../outside",
                                  "/absolute", "C:/absolute", "C:relative", "data\\field", None, 3])
def test_unsafe_input_paths(adapter: ModuleType, tmp_path: Path, name: Any) -> None:
    with pytest.raises(ValueError):
        check_input(adapter, tmp_path, {**entry(), "path": name})


@pytest.mark.parametrize("change", ["missing", "resized", "hash"])
def test_changed_input_rejected(adapter: ModuleType, tmp_path: Path, change: str) -> None:
    row = entry()
    root = make_archive(tmp_path / "archive", [row])
    path = root / row["path"]
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"longer" if change == "resized" else b"edit")
    with pytest.raises(ValueError):
        check_input(adapter, root, row)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_symlink_escape_rejected(adapter: ModuleType, tmp_path: Path, kind: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "field.bin").write_bytes(b"data")
    root = tmp_path / "archive"
    root.mkdir()
    try:
        if kind == "directory":
            (root / "data").symlink_to(outside, target_is_directory=True)
        else:
            (root / "data").mkdir()
            (root / "data/field.bin").symlink_to(outside / "field.bin")
    except OSError as error:
        pytest.skip(f"Symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="escapes|symlink"):
        check_input(adapter, root, entry())


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("relation", ["existing", "inside", "ancestor", "same", "symlink"])
def test_output_must_be_fresh_and_disjoint(tmp_path: Path, name: str, relation: str) -> None:
    archive = tmp_path / "archive"
    if relation == "ancestor":
        output = tmp_path / "not-created"
        archive = output / "archive"
    else:
        archive.mkdir()
        output = tmp_path / "output"
        if relation == "existing":
            output.mkdir()
            (output / "sentinel").write_text("unchanged")
        elif relation == "inside":
            output = archive / "output"
        elif relation == "same":
            output = archive
        elif relation == "symlink":
            link = tmp_path / "archive-link"
            try:
                link.symlink_to(archive, target_is_directory=True)
            except OSError as error:
                pytest.skip(f"Symlinks unavailable: {error}")
            output = link / "output"
    result = run_adapter(name, tmp_path, "--archive", str(archive), "--output", str(output))
    assert result.returncode != 0
    assert "fresh output directory" in result.stderr
    if relation == "existing":
        assert (output / "sentinel").read_text() == "unchanged"
    elif relation != "same":
        assert not output.exists()


@pytest.mark.parametrize("name", NAMES)
def test_help_without_plotting_stack(tmp_path: Path, name: str) -> None:
    result = subprocess.run([sys.executable, "-B", "-S", str(ADAPTERS / f"replot_{name}.py"),
                             "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--archive" in result.stdout and "--output" in result.stdout
    assert "--remote-availability" not in result.stdout


def test_forward_private_option_removed(tmp_path: Path) -> None:
    result = run_adapter("forward", tmp_path, "--archive", "archive", "--output", "output",
                         "--remote-availability", "unused.json")
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["path", "legacy_path"])
def test_q2_duplicate_mapping_rejected(field: str) -> None:
    module = load_adapter("q2")
    first, second = entry(), entry("data/other.bin", "results/other.bin")
    second[field] = "./" + first[field]
    with pytest.raises(ValueError, match="Duplicate"):
        module.selected_entries({"files": [first, second]})


@pytest.mark.parametrize(("field", "value"), [("bytes", -1), ("bytes", True), ("bytes", 4.0),
                                              ("sha256", "a" * 63), ("sha256", "g" * 64),
                                              ("sha256", "A" * 64), ("sha256", None)])
def test_q2_invalid_integrity_metadata(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        load_adapter("q2").selected_entries({"files": [{**entry(), field: value}]})


def test_q2_selection_is_bounded() -> None:
    module = load_adapter("q2")
    row = entry()
    assert module.selected_entries({"files": [row, {"groups": ["other"]}]}) == [row]
    with pytest.raises(ValueError, match="no q2"):
        module.selected_entries({"files": []})


def test_inverse_exact_lock_and_identical_aliases(tmp_path: Path) -> None:
    module = load_adapter("inverse")
    first = entry()
    preferred = {**entry("code/field.bin"), "groups": ["renderers_and_checks"]}
    root = make_archive(tmp_path / "archive", [first, preferred])
    inputs = module.Inputs(root, tmp_path / "output")
    selected = inputs.take(first["legacy_path"], first["sha256"])
    assert selected.read_bytes() == b"data"
    assert inputs.used[first["legacy_path"]]["path"] == preferred["path"]
    assert inputs.adaptations[0]["kind"] == "identical_manifest_aliases"
    with pytest.raises(ValueError, match="Recorded scientific input hash"):
        inputs.take(first["legacy_path"], "0" * 64)
    old_path = str(tmp_path / "historical" / first["legacy_path"])
    assert inputs.locked_record({"path": old_path, "sha256": first["sha256"]}) == selected
    assert inputs.finish()["archive_and_disposable_inputs_unchanged"] is True


def test_inverse_simulated_windows_mapping_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_adapter("inverse")
    row = entry()
    root = make_archive(tmp_path / "archive", [row])
    safe_relative = module.safe_relative

    def windows_relative(value: str) -> PureWindowsPath:
        return PureWindowsPath(safe_relative(value).as_posix())

    # Exercise Windows stringification without changing the host filesystem.
    with monkeypatch.context() as patch:
        patch.setattr(module, "safe_relative", windows_relative)
        inputs = module.Inputs(root, tmp_path / "output")
        assert set(inputs.by_legacy) == {row["legacy_path"]}
        assert inputs.row(row["legacy_path"]) == row
    copied = inputs.take("./" + row["legacy_path"], row["sha256"])
    assert copied.read_bytes() == b"data"
    assert set(inputs.used) == {row["legacy_path"]}
    assert inputs.finish()["verified_files"] == 1
    mapping = json.loads((inputs.output / "verified_inputs.json").read_text())
    assert mapping == [{key: row[key] for key in ("path", "legacy_path", "bytes", "sha256")}]


@pytest.mark.parametrize("old_path", [r"C:\retained\results\field.bin",
                                      "C:/retained/results/field.bin", r"results\field.bin"])
def test_inverse_windows_record_separators(tmp_path: Path, old_path: str) -> None:
    module = load_adapter("inverse")
    row = entry()
    inputs = module.Inputs(make_archive(tmp_path / "archive", [row]), tmp_path / "output")
    assert inputs.old_to_legacy(old_path) == row["legacy_path"]
    copied = inputs.locked_record({"path": old_path, "sha256": row["sha256"]})
    assert copied.read_bytes() == b"data"
    with pytest.raises(ValueError, match="Recorded scientific input hash"):
        inputs.locked_record({"path": old_path, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="Invalid relative path"):
        inputs.take(r"results\field.bin")
    assert inputs.finish()["archive_and_disposable_inputs_unchanged"] is True


def test_inverse_windows_record_ambiguous_suffix_rejected(tmp_path: Path) -> None:
    module = load_adapter("inverse")
    rows = [entry(), entry("other/field.bin", "nested/results/field.bin")]
    inputs = module.Inputs(make_archive(tmp_path / "archive", rows), tmp_path / "output")
    with pytest.raises(ValueError, match="unambiguous"):
        inputs.old_to_legacy(r"C:\retained\nested\results\field.bin")


def test_inverse_renderer_simulated_windows_receipt_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_adapter("inverse")

    class WindowsRelativePath(type(Path())):
        def relative_to(self, *other: Any) -> PureWindowsPath:
            return PureWindowsPath(super().relative_to(*other).as_posix())

    root = tmp_path / "archive"
    root.mkdir()
    rows = []
    for name in ("fixture.py", "journal_figure_style.py"):
        row = entry(f"code/{name}", f"{module.SCRIPTS}/{name}")
        path = root / row["path"]
        path.parent.mkdir(exist_ok=True)
        path.write_text("VALUE = 7\n")
        row.update(bytes=path.stat().st_size, sha256=module.digest(path))
        rows.append(row)
    (root / "manifest.json").write_text(json.dumps({"files": rows}))
    inputs = module.Inputs(root, WindowsRelativePath(tmp_path / "output"))
    monkeypatch.setattr(sys, "path", sys.path.copy())

    def unchanged(tree: ast.Module) -> ast.Module:
        return tree

    assert inputs.renderer("fixture.py", unchanged)["VALUE"] == 7
    adaptation = inputs.adaptations[0]
    assert adaptation["source"] == f"{module.SCRIPTS}/fixture.py"
    assert adaptation["adapted_source"] == "adapted/fixture.py"
    assert inputs.finish()["verified_files"] == 2


def test_cross_relocation_simulated_windows_keys(tmp_path: Path) -> None:
    module = load_adapter("inverse")
    tree = ast.parse(Path(module.__file__).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cross_mesh")
    first = next(i for i, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "relocations")
    last = next(i for i, node in enumerate(function.body)
                if isinstance(node, ast.FunctionDef) and node.name == "relocate")
    # Run only the actual path-translation block, not the scientific renderer.
    block = ast.Module(body=function.body[first:last + 1], type_ignores=[])
    layout = tmp_path / "legacy"
    namespace = {"inputs": SimpleNamespace(used={"runs/results/field.bin": entry()}, layout=layout),
                 "Path": PureWindowsPath, "PurePosixPath": PurePosixPath,
                 "PureWindowsPath": PureWindowsPath, "Any": Any}
    exec(compile(block, module.__file__, "exec"), namespace)
    relocate = namespace["relocate"]
    for prefix in ("/retained/", "C:\\retained\\"):
        separator = "\\" if prefix.startswith("C:") else "/"
        file = prefix + separator.join(("runs", "results", "field.bin"))
        directory = prefix + separator.join(("runs", "results"))
        assert relocate({"file": file, "directories": [directory]}) == {
            "file": str(layout / "runs/results/field.bin"), "directories": [str(layout / "runs/results")],
        }
        assert namespace["relocations"][file] == str(layout / "runs/results/field.bin")
    assert relocate(r"C:\unverified\field.bin") == r"C:\unverified\field.bin"


def test_inverse_conflicting_aliases_rejected(tmp_path: Path) -> None:
    module = load_adapter("inverse")
    first = entry()
    conflicting = {**entry("other/field.bin"), "sha256": "0" * 64}
    inputs = module.Inputs(make_archive(tmp_path / "archive", [first, conflicting]), tmp_path / "output")
    with pytest.raises(ValueError, match="conflicting"):
        inputs.take(first["legacy_path"])


@pytest.mark.parametrize("field", ["path", "legacy_path"])
@pytest.mark.parametrize("value", ["../escape", r"results\field.bin"])
def test_inverse_unsafe_manifest_mapping_rejected(tmp_path: Path, field: str, value: str) -> None:
    module = load_adapter("inverse")
    root = make_archive(tmp_path / "archive", [])
    (root / "manifest.json").write_text(json.dumps({"files": [{**entry(), field: value}]}))
    with pytest.raises(ValueError, match="Unsafe|Invalid"):
        module.Inputs(root, tmp_path / "output")


def test_inverse_duplicate_archive_path_rejected(tmp_path: Path) -> None:
    module = load_adapter("inverse")
    root = make_archive(tmp_path / "archive", [entry(), entry()])
    with pytest.raises(ValueError, match="Duplicate archive path"):
        module.Inputs(root, tmp_path / "output")


def test_inverse_pinned_variant_not_a_copy(tmp_path: Path) -> None:
    module = load_adapter("inverse")
    row = {**entry(), "kind": "source_variant"}
    inputs = module.Inputs(make_archive(tmp_path / "archive", [row]), tmp_path / "output")
    with pytest.raises(ValueError, match="Missing or conflicting"):
        inputs.take(row["legacy_path"])


@pytest.mark.parametrize("changed", ["archive", "replay", "manifest"])
def test_inverse_final_recheck(tmp_path: Path, changed: str) -> None:
    module = load_adapter("inverse")
    row = entry()
    root = make_archive(tmp_path / "archive", [row])
    inputs = module.Inputs(root, tmp_path / "output")
    copied = inputs.take(row["legacy_path"])
    if changed == "manifest":
        (root / "manifest.json").write_text(json.dumps({"files": [{**row, "sha256": "0" * 64}]}))
    else:
        (root / row["path"] if changed == "archive" else copied).write_bytes(b"edit")
    with pytest.raises(ValueError, match="SHA-256|manifest entry changed"):
        inputs.finish()


@pytest.mark.parametrize("name", ["torch", "torch.linalg", "phast", "torch_pf_solver", "gmsh"])
def test_inverse_solver_import_guard(name: str) -> None:
    with pytest.raises(ImportError, match="prohibited"):
        load_adapter("inverse").NoSolverImports().find_spec(name)


def test_cross_renderer_adapter_only_wraps_expected_calls() -> None:
    module = load_adapter("inverse")
    tree = ast.parse("spec.loader.exec_module(c)\n"
                     "def read(path, expected):\n"
                     "    return json.loads(register(path, expected).read_text())\n"
                     "def scientific(x):\n"
                     "    return x * x + 1\n")
    scientific = ast.dump(tree.body[-1])
    adapter = module.CrossRendererAdapter()
    transformed = adapter.visit(tree)
    assert (adapter.loader_count, adapter.read_count) == (1, 1)
    assert ast.dump(transformed.body[-1]) == scientific
    assert "portable_load_contracts(c, contracts)" in ast.unparse(transformed)
    assert "portable_relocate(json.loads(register(path, expected).read_text()))" in ast.unparse(transformed)
