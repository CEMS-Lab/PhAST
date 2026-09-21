#!/usr/bin/env python3
"""Replot retained inverse data with exact-input checks; no simulation or new gradients."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.abc
from importlib.machinery import ModuleSpec
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import runpy
import shutil
import sys
import traceback
from types import ModuleType
from typing import Any


# Historical relative paths are replay identifiers, not public workflow names.
EVIDENCE = "papers/paper/reviewer_evidence"
STUDIES = EVIDENCE + "/studies"
SCRIPTS = "scripts/cmame_revision"
CASES = ("at2_memory", "alumina_fd", "joint_selected", "scalar", "cross_mesh")
GAPS = {
    "at2_memory": "One interior AT2 update only; no trajectory-memory or runtime claim.",
    "alumina_fd": "The historical sweep is distinct from both baseline inversion and later "
                  "backward-rule replay. Raw historical perturbation fields remain absent.",
    "joint_selected": "Evaluation 26 is the selected minimum-loss state, not terminal "
                      "evaluation 200. Full terminal fields remain absent. The selected "
                      "93.005% AD-FD disagreement and unverified relative projected-KKT "
                      "criterion are not resolved by replotting.",
    "scalar": "Baseline fields, meshes and execution-source/environment locks remain "
              "missing. Glass closure losses are pre-update states 0-9; alumina losses "
              "are accepted states 1-10. The six-start 4x failure remains included.",
    "cross_mesh": "All six outcomes are retained, including coarse AD budget exhaustion "
                  "and fine AD abnormal line-search termination. Recovery, gradient "
                  "verification and comparative efficiency remain separate questions.",
}


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*path.parts)


def verify(root: Path, name: str, row: dict[str, Any]) -> Path:
    path = root / safe_relative(name)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Input escapes root or is a symlink: {name}")
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        raise ValueError(f"Missing or resized input: {name}")
    if digest(path) != row["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {name}")
    return path


class NoSolverImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None = None,
                  target: ModuleType | None = None) -> ModuleSpec | None:
        if fullname.split(".")[0] in {"torch", "phast", "torch_pf_solver", "gmsh"}:
            raise ImportError(f"Numerical-solver import prohibited in replot: {fullname}")
        return None


class Inputs:
    def __init__(self, archive: Path, output: Path) -> None:
        self.archive = archive
        self.output = output
        self.layout = output / "legacy"
        self.layout.mkdir(parents=True)
        self.manifest = archive / "manifest.json"
        self.manifest_hash = digest(self.manifest)
        self.rows = json.loads(self.manifest.read_text())["files"]
        self.by_legacy: dict[str, list[dict[str, Any]]] = {}
        names = set()
        for row in self.rows:
            name = safe_relative(row["path"]).as_posix()
            if name in names:
                raise ValueError(f"Duplicate archive path: {name}")
            names.add(name)
            # Pinned source variants must never collide with the plotting layout.
            if row.get("kind") != "copy":
                continue
            legacy = safe_relative(row["legacy_path"]).as_posix()
            self.by_legacy.setdefault(legacy, []).append(row)
        self.used: dict[str, dict[str, Any]] = {}
        self.adaptations: list[dict[str, Any]] = []

    def row(self, legacy: str) -> dict[str, Any]:
        legacy = safe_relative(legacy).as_posix()
        matches = self.by_legacy.get(legacy, [])
        if not matches or len({(r["bytes"], r["sha256"]) for r in matches}) != 1:
            raise ValueError(f"Missing or conflicting copy mapping for {legacy}")
        chosen = sorted(matches, key=lambda r: (
            "renderers_and_checks" not in r.get("groups", []), r["path"]))[0]
        if len(matches) > 1:
            alias = {"kind": "identical_manifest_aliases", "legacy_path": legacy,
                     "chosen": chosen["path"], "sha256": chosen["sha256"],
                     "aliases": sorted(r["path"] for r in matches)}
            if alias not in self.adaptations:
                self.adaptations.append(alias)
        return chosen

    def take(self, legacy: str, expected: str | None = None) -> Path:
        legacy = safe_relative(legacy).as_posix()
        row = self.row(legacy)
        if expected is not None and row["sha256"] != expected:
            raise ValueError(f"Recorded scientific input hash differs: {legacy}")
        target = self.layout / safe_relative(legacy)
        if legacy not in self.used:
            source = verify(self.archive, row["path"], row)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            verify(self.layout, legacy, row)
            self.used[legacy] = row
        return target

    def group(self, name: str) -> None:
        matches = [r for r in self.rows if r.get("kind") == "copy"
                   and name in r.get("groups", [])]
        if not matches:
            raise ValueError(f"Missing copy group: {name}")
        for row in matches:
            self.take(row["legacy_path"])

    def read(self, legacy: str) -> dict[str, Any]:
        return json.loads(self.take(legacy).read_text())

    def old_to_legacy(self, value: str) -> str:
        # Historical records may contain native separators; manifests may not.
        value = value.replace("\\", "/")
        if value in self.by_legacy:
            return value
        matches = [name for name in self.by_legacy if value.endswith("/" + name)]
        if len(matches) != 1:
            raise ValueError(f"Old path has no unambiguous manifest mapping: {value}")
        return matches[0]

    def locked_record(self, record: dict[str, Any]) -> Path:
        return self.take(self.old_to_legacy(record["path"]), record["sha256"])

    def renderer(self, name: str, transform: Callable[[ast.Module], ast.Module] | None = None,
                 additions: dict[str, Any] | None = None) -> dict[str, Any]:
        source = self.take(SCRIPTS + "/" + name)
        self.take(SCRIPTS + "/journal_figure_style.py")
        directory = str(source.parent)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        if transform is None:
            return runpy.run_path(str(source), run_name="_retained_inverse_plotter")
        tree = ast.parse(source.read_text(), filename=str(source))
        tree = ast.fix_missing_locations(transform(tree))
        adapted = self.output / "adapted" / name
        adapted.parent.mkdir(parents=True, exist_ok=True)
        adapted.write_text(ast.unparse(tree) + "\n")
        self.adaptations.append({"source": source.relative_to(self.layout).as_posix(),
                                 "source_sha256": digest(source),
                                 "adapted_source": adapted.relative_to(self.output).as_posix(),
                                 "adapted_sha256": digest(adapted)})
        namespace = {"__file__": str(source), "__name__": "_portable_inverse_plotter",
                     **(additions or {})}
        exec(compile(tree, str(source), "exec"), namespace)
        return namespace

    def finish(self) -> dict[str, Any]:
        for legacy, row in self.used.items():
            verify(self.archive, row["path"], row)
            verify(self.layout, legacy, row)
        current = {r["path"]: r for r in json.loads(self.manifest.read_text())["files"]}
        for row in self.used.values():
            if current.get(row["path"]) != row:
                raise ValueError(f"Used public manifest entry changed: {row['path']}")
        mapping = [{k: row[k] for k in ("path", "legacy_path", "bytes", "sha256")}
                   for row in sorted(self.used.values(), key=lambda x: x["path"])]
        (self.output / "verified_inputs.json").write_text(json.dumps(mapping, indent=2) + "\n")
        return {"verified_files": len(mapping), "verified_bytes": sum(r["bytes"] for r in mapping),
                "archive_and_disposable_inputs_unchanged": True,
                "manifest_sha256_at_start": self.manifest_hash,
                "manifest_sha256_at_end": digest(self.manifest),
                "used_manifest_entries_unchanged": True,
                "mapping_file": "verified_inputs.json"}


def call_main(namespace: dict[str, Any], arguments: Sequence[str | Path]) -> None:
    saved = sys.argv
    try:
        sys.argv = [namespace["__file__"], *map(str, arguments)]
        namespace["main"]()
    finally:
        sys.argv = saved


def memory(inputs: Inputs, figures: Path) -> dict[str, Any]:
    inputs.group("implicit_memory")
    inputs.take(SCRIPTS + "/at2_backward_memory_aggregate.py")
    summary = inputs.take(EVIDENCE + "/results/r2_2_at2_isolated_20260826_job117684/summary.json")
    ns = inputs.renderer("render_at2_isolated_memory_audit.py")
    call_main(ns, ["--summary", summary, "--output", figures / "at2_memory.png", "--journal",
                   "--receipt", figures / "at2_memory_receipt.json"])
    rows = json.loads(summary.read_text())["mesh_levels"]
    if len(rows) != 4:
        raise ValueError("Expected four memory mesh levels")
    return {"mesh_levels": 4, "matched_pairs": 12, "renderer_source_unchanged": True}


def alumina(inputs: Inputs, figures: Path) -> dict[str, Any]:
    inputs.group("alumina_fd_sweep")
    csv = inputs.take(EVIDENCE + "/results/alumina_fd_active_set_job111131/alumina_fd_active_set_audit.csv")
    ns = inputs.renderer("render_alumina_fd_active_set_audit.py")
    selected = ns["select_rows"](csv, 1e-8)
    if len(selected) != 5:
        raise ValueError("Expected exactly five rows at damage tolerance 1e-8")
    call_main(ns, ["--csv", csv, "--damage-tol", "1e-8", "--journal",
                   "--output", figures / "alumina_fd.png"])
    return {"damage_tolerance": 1e-8, "plotted_rows": 5,
            "plotted_relative_steps": [float(r["relative_step"]) for r in selected],
            "renderer_source_unchanged": True}


def joint(inputs: Inputs, figures: Path) -> dict[str, Any]:
    inputs.group("joint_selected_fields")
    inputs.group("joint_four_start")
    summary = inputs.read(STUDIES + "/r1_5_joint_gc_l0_multistart_20260904/"
                          "collection_20260905T123100Z/raw/job125734_2/summary.json")
    if summary["selected_evaluation"] != 26 or summary["objective_evaluations"] != 201:
        raise ValueError("Joint selected-state identity differs")
    ns = inputs.renderer("render_v2_joint_figure.py")
    fn = ns["render_joint_recovery"]
    fn.__globals__["OUTPUT"] = figures / "joint_selected.png"
    fn()
    return {"selected_evaluation": 26, "terminal_evaluation": 200,
            "selected_loss": summary["selected_loss"],
            "selected_parameters": summary["selected"],
            "selected_gradient_check_pass": summary["selected_gradient_verification_pass"],
            "renderer_adaptation": "Output constant redirected only; numerical plotting unchanged."}


def scalar(inputs: Inputs, figures: Path) -> dict[str, Any]:
    import numpy as np
    receipt_name = EVIDENCE + "/figure_improvements_20260906/scalar_recovery_composite/receipt.json"
    receipt = inputs.read(receipt_name)
    for record in receipt["inputs"] + receipt["sources"] + receipt["preserved_standalone_outputs"]:
        inputs.locked_record(record)
    baseline_path = inputs.locked_record(receipt["scalar_input_hash_lock"])
    baseline = json.loads(baseline_path.read_text())
    for record in baseline["inputs"]:
        inputs.locked_record(record)
    paths = [inputs.locked_record(r) for r in receipt["inputs"][:2]]
    folder = inputs.layout / EVIDENCE / "results/r1_m4_752_reparam_v4_20260826_job117725"
    for name in ("target_damage_clean.npy", "target_damage_observed.npy"):
        values = np.load(folder / name, allow_pickle=False)
        if not np.isfinite(values).all() or values.shape != (2842,):
            raise ValueError(f"Invalid retained scalar target: {name}")
    checksums = {}
    for line in (folder / "sha256sums_relative.txt").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        checksums[name.removeprefix("./")] = sha
    for name in ("optimisation_history.csv", "optimisation_summary.csv", "protocol.json",
                 "config.json", "target_damage_clean.npy", "target_damage_observed.npy"):
        if digest(folder / name) != checksums[name]:
            raise ValueError(f"Original scalar checksum fails: {name}")
    ns = inputs.renderer("render_inverse_journal.py")
    glass, al = [ns["scalar_data"](path) for path in paths]
    multi = ns["composite_multistart_data"](folder)
    ns["apply_style"]()
    plotted = ns["scalar_protocol_composite"](glass, al, multi, figures)
    adaptation = {"kind": "retained_pure_plotting_functions",
                  "original_receipt": receipt_name,
                  "omitted_preservation_archive": receipt["preservation_archive"],
                  "replacement": "Verified all original exact inputs, plot sources, scalar "
                                 "input hash lock, original six file checksums, both target "
                                 "arrays and all eight standalone outputs against recorded "
                                 "SHA-256 plus public manifest; only the unrelated broad "
                                 "preservation tarball predicate is omitted.",
                  "numeric_plotting_functions_unchanged": True}
    inputs.adaptations.append(adaptation)
    return {"adaptation": adaptation, "plotted_values": plotted}


class CrossRendererAdapter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.loader_count = 0
        self.read_count = 0

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        if ast.unparse(node) == "spec.loader.exec_module(c)":
            self.loader_count += 1
            return ast.copy_location(ast.parse("portable_load_contracts(c, contracts)").body[0], node)
        return self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if node.value and ast.unparse(node.value) == "json.loads(register(path, expected).read_text())":
            self.read_count += 1
            node.value = ast.Call(func=ast.Name(id="portable_relocate", ctx=ast.Load()),
                                  args=[node.value], keywords=[])
        return self.generic_visit(node)


def cross_mesh(inputs: Inputs, figures: Path) -> dict[str, Any]:
    receipt_name = EVIDENCE + "/version_2_20260906/inverse_figures_receipt.json"
    receipt = inputs.read(receipt_name)
    omitted = STUDIES + "/r1_4_clean_sent_20260904/source_125692.tar.gz"
    if omitted not in receipt["inputs"] or len(receipt["inputs"]) != 316:
        raise ValueError("Unrecognised cross-mesh exact-input contract")
    for name, record in receipt["inputs"].items():
        if name != omitted:
            inputs.take(name, record["sha256"])
    omitted_hash = receipt["inputs"][omitted]["sha256"]
    script = inputs.take(SCRIPTS + "/render_v2_inverse_figures.py")
    if digest(script) != receipt["script_sha256"]:
        raise ValueError("Cross renderer differs from archived scientific figure receipt")

    relocations = {}
    relocation_candidates = set(inputs.used)
    relocation_candidates.update(PurePosixPath(name).parent.as_posix() for name in inputs.used)
    relocation_candidates.discard(".")
    def relocate(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and (value.startswith("/") or PureWindowsPath(value).is_absolute()):
            normalized = value.replace("\\", "/")
            matches = [name for name in relocation_candidates if normalized.endswith("/" + name)]
            if len(matches) == 1:
                mapped = str(inputs.layout / matches[0])
                relocations[value] = mapped
                return mapped
        return value

    def load_contracts(module: ModuleType, source: str | Path) -> None:
        tree = ast.parse(Path(source).read_text())
        replaced = 0
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ANCHORS"
                                                    for t in node.targets):
                if not isinstance(node.value, ast.Dict):
                    raise ValueError("Unexpected source anchor declaration")
                for index, key in enumerate(node.value.keys):
                    if isinstance(key, ast.Constant) and key.value == "source_125692.tar.gz":
                        if ast.literal_eval(node.value.values[index]) != omitted_hash:
                            raise ValueError("Original source-archive anchor does not match receipt")
                        del node.value.keys[index]
                        del node.value.values[index]
                        replaced += 1
                        break
        if replaced != 1:
            raise ValueError("Expected one excluded broad-source archive anchor")
        adapted = inputs.output / "adapted/contracts.py"
        adapted.parent.mkdir(parents=True, exist_ok=True)
        adapted.write_text(ast.unparse(tree) + "\n")
        inputs.adaptations.append({"source": Path(source).relative_to(inputs.layout).as_posix(),
                                  "source_sha256": digest(source),
                                  "adapted_source": adapted.relative_to(inputs.output).as_posix(),
                                  "adapted_sha256": digest(adapted),
                                  "omitted_archive_sha256": omitted_hash,
                                  "equivalent_recorded_input_hashes_checked": 315,
                                  "all_other_source_and_numerical_contract_checks_preserved": True})
        exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)

    adapter = CrossRendererAdapter()
    def transform(tree: ast.Module) -> ast.Module:
        result = adapter.visit(tree)
        if (adapter.loader_count, adapter.read_count) != (1, 1):
            raise ValueError("Cross renderer adaptation did not match exactly twice")
        return result
    ns = inputs.renderer("render_v2_inverse_figures.py", transform,
                         {"portable_load_contracts": load_contracts, "portable_relocate": relocate})
    matplotlib, plt = ns["configure_plotting"]()
    target, items, registered = ns["load_evidence"]()
    outcomes = []
    for method in ("AD", "Brent"):
        for mesh in ns["MESHES"]:
            row = items[method, mesh]["row"]
            outcomes.append({key: row[key] for key in ("method", "mesh", "Gc", "Gc_error_percent",
                                                        "holdout_relative_l2", "criteria_pass",
                                                        "optimizer_success", "status")})
    exported = {}
    for stem, function in (("inverse_observations_meshes", ns["observations_figure"]),
                           ("cross_mesh_outcomes", ns["outcomes_figure"])):
        fig, axes = function(plt, target, items)
        exported[stem] = ns["export_and_check"](fig, axes, stem)
        plt.close(fig)
        for ext in ("pdf", "png"):
            shutil.copyfile(inputs.layout / exported[stem][ext], figures / f"{stem}.{ext}")
    for name, record in registered.items():
        if digest(inputs.layout / name) != record["sha256"]:
            raise ValueError(f"Cross plotting input changed: {name}")
    inputs.adaptations.append({"kind": "in_memory_absolute_path_translation",
                              "mapping": relocations,
                              "scope": "Verified exact files and their immediate parent directories only",
                              "original_metadata_bytes_unchanged": True})
    return {"exact_recorded_inputs_verified": 315, "omitted_broad_archive": omitted,
            "outcomes": outcomes, "export_checks": exported,
            "gradient_verified": False, "numerical_checks": "Original mesh/sampler/field "
            "identity, eight-time observation reconstruction, losses, forward receipt, "
            "source equality, outcome checks and PDF layout checks all executed."}


def visual_checks(figures: Path, output: Path) -> list[dict[str, Any]]:
    import fitz
    import numpy as np
    from PIL import Image, ImageDraw
    previews = output / "pdf_previews"
    previews.mkdir()
    records, tiles = [], []
    for pdf in sorted(figures.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            if len(doc) != 1:
                raise ValueError(f"Unexpected PDF page count: {pdf}")
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pixels.std() < 10 or not page.get_text().strip():
                raise ValueError(f"Blank PDF or absent labels: {pdf}")
            png = previews / (pdf.stem + ".png")
            pix.save(png)
            records.append({"pdf": pdf.relative_to(output).as_posix(), "pages": 1,
                            "pixel_standard_deviation": float(pixels.std()),
                            "width_inches": page.rect.width / 72,
                            "height_inches": page.rect.height / 72,
                            "preview": png.relative_to(output).as_posix()})
            with Image.open(png) as original:
                tile = Image.new("RGB", (760, 650), "white")
                view = original.convert("RGB")
                view.thumbnail((740, 610))
                tile.paste(view, ((760-view.width)//2, 30))
                ImageDraw.Draw(tile).text((12, 8), pdf.stem, fill="black")
                tiles.append(tile)
    if tiles:
        sheet = Image.new("RGB", (1520, 650*((len(tiles)+1)//2)), "#dddddd")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 2)*760, (i//2)*650))
        sheet.save(output / "contact_sheet.png")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError("Use a fresh output directory disjoint from the archive")
    output.mkdir(parents=True)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    for var, folder in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache"), ("TMPDIR", "tmp")):
        target = output / "runtime" / folder
        target.mkdir(parents=True)
        os.environ[var] = str(target)
    sys.meta_path.insert(0, NoSolverImports())
    figures = output / "figures"
    figures.mkdir()
    inputs = Inputs(archive, output)
    reports = {}
    functions = dict(zip(CASES, (memory, alumina, joint, scalar, cross_mesh)))
    for name in dict.fromkeys(args.cases):
        print(f"Replotting {name}", flush=True)
        try:
            reports[name] = {"status": "passed", **functions[name](inputs, figures), "scientific_gap": GAPS[name]}
        except Exception as error:
            reports[name] = {"status": "failed", "error": str(error),
                             "traceback": traceback.format_exc(), "scientific_gap": GAPS[name]}
            print(f"FAILED {name}: {error}", flush=True)
    integrity = inputs.finish()
    visual = visual_checks(figures, output)
    forbidden = [name for name in sys.modules if name.split(".")[0] in
                 {"torch", "phast", "torch_pf_solver", "gmsh"}]
    if forbidden:
        raise ValueError(f"Unexpected solver modules: {forbidden}")
    success = all(row["status"] == "passed" for row in reports.values())
    receipt = {"status": "completed" if success else "partial", "cases": reports,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "wrapper_sha256": digest(Path(__file__)), "integrity": integrity,
               "adaptations": inputs.adaptations, "pdf_checks": visual,
               "visual_review": "PDF previews generated; human-visible inspection required.",
               "raw_numerical_data_modified": False, "simulations_run": 0,
               "solver_modules_imported": forbidden,
               "runtime": {"python": platform.python_version(), **{
                   name: importlib.metadata.version(name) for name in
                   ("numpy", "matplotlib", "h5py", "meshio", "PyMuPDF", "Pillow")}},
               "products": [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                             "sha256": digest(path)} for path in sorted(figures.iterdir()) if path.is_file()]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "cases": {key: value["status"] for key, value in reports.items()},
                      "figures": len(visual), "receipt": str(output / "receipt.json")}), flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
