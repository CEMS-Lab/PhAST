"""``phast figures <run>``: draw the standard figures of a stored run and list them in ``manifest.json``."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from ....result import Result, ResultLoadError, load_result
from .data import RunData
from .sheets import SCALAR_ROLES, compute_interface_norms, compute_norms, component_sheet, damage_steps_sheet, \
    evolution_sheet, field_pages, field_sheet, interface_sheet, setup_sheet
from .style import DEFAULT_DPI

MANIFEST = "manifest.json"
KINDS = ("setup", "fields", "evolution")


def sha256_path(path: Path) -> str:
    """SHA-256 of a file; for a directory store, of its relative paths and file contents in sorted order."""
    h = hashlib.sha256()
    path = Path(path)
    if path.is_file():
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        return h.hexdigest()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode())
        with child.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
    return h.hexdigest()


def _relative(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def draw_all(result: Result | str | Path, out_dir: str | Path | None = None, *, steps: str = "first,peak,last",
             dpi: int = DEFAULT_DPI, fields: list[str] | None = None, deform_scale: float | str | None = None,
             kinds: tuple[str, ...] = KINDS) -> list[dict[str, Any]]:
    """Draw the standard figures of ``result`` into ``out_dir`` (default ``<run>/figures``).

    ``steps`` lists ``first``, ``peak``, ``middle``, ``last`` or step values
    as :meth:`phast.Result.field` takes them (negative values count from the
    end). ``fields`` restricts the maps to these field names. ``deform_scale``
    draws 2D maps on the deformed shape (a factor, or ``"auto"`` for a largest
    displacement of 5 % of the domain). Returns the manifest rows, which are
    also written to ``<out_dir>/manifest.json``.
    """
    if not isinstance(result, Result):
        result = load_result(result)
    run = RunData(result)
    out = Path(out_dir) if out_dir is not None else result.path / "figures"
    out.mkdir(parents=True, exist_ok=True)
    previous = _previous_files(out)
    rows: list[dict[str, Any]] = []
    if "setup" in kinds:
        rows.append(setup_sheet(run, out / "setup.png", dpi))
    chosen = run.choose_steps(steps) if run.n_steps else []
    if chosen and "fields" in kinds:
        infos = run.ordered_fields(SCALAR_ROLES | {"components"}, fields)
        norms = compute_norms(run, infos, [s for s, _ in chosen])
        pages = field_pages(run, infos)
        for step, reason in chosen:
            for i, (group, members) in enumerate(pages):
                prefix = "fields" if group in ("fields", "state") else "fields_derived"
                same = [g for g, _ in pages[:i] if g == group]
                suffix = f"_{len(same) + 1}" if same else ""
                row = field_sheet(run, step, reason, members, norms, out / f"{prefix}_step{step}{suffix}.png", dpi,
                                  deform_scale, group=group)
                if row is not None:
                    rows.append(row)
            row = component_sheet(run, step, reason, infos, norms, out / f"components_step{step}.png", dpi,
                                  deform_scale)
            if row is not None:
                rows.append(row)
        if run.interfaces:
            inorms = compute_interface_norms(run, [s for s, _ in chosen])
            for step, reason in chosen:
                row = interface_sheet(run, step, reason, inorms, out / f"interfaces_step{step}.png", dpi)
                if row is not None:
                    rows.append(row)
        if any(i.name == "damage" for i in infos):
            row = damage_steps_sheet(run, chosen, norms, out / "damage_steps.png", dpi, deform_scale)
            if row is not None:
                rows.append(row)
    if run.n_steps and "evolution" in kinds:
        row = evolution_sheet(run, chosen, run.damage_measures(), out / "evolution.png", dpi)
        if row is not None:
            rows.append(row)
    sources = [{"file": _relative(p, result.path), "sha256": sha256_path(p)} for p in result.source_files()]
    for row in rows:
        path = out / row["file"]
        row.update(artifact_type="image", size_bytes=path.stat().st_size,
                   review_dimension_passed=bool(max(row["width_px"], row["height_px"]) < 2000),
                   visual_scope="standard_figures", generator="phast figures", layouts=run.layouts,
                   source_files=sources)
    # figures of the kinds drawn now replace those of an earlier call; other kinds are kept
    drawn = {row["file"] for row in rows}
    kept = [r for r in previous if _kind_of(r) not in kinds and r["file"] not in drawn and (out / r["file"]).exists()]
    for r in previous:
        if _kind_of(r) in kinds and r["file"] not in drawn:
            (out / r["file"]).unlink(missing_ok=True)
    (out / MANIFEST).write_text(json.dumps(kept + rows, indent=2) + "\n", encoding="utf-8")
    return rows


def _kind_of(row: dict) -> str:
    """Figure kind of a manifest row: setup, fields or evolution (from the file name)."""
    name = Path(row["file"]).name
    if name.startswith("setup"):
        return "setup"
    if name.startswith("evolution"):
        return "evolution"
    return "fields"


def _previous_files(out: Path) -> list[dict]:
    """PNG rows listed by an earlier manifest in ``out`` (only those files are ever removed)."""
    try:
        rows = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        return [dict(r, file=Path(r["file"]).name) for r in rows if str(r.get("file", "")).endswith(".png")]
    except (OSError, ValueError, TypeError, KeyError):
        return []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phast figures",
        description="Draw the standard figures (setup, fields, evolution) of a stored run. Reads a layout-2.x "
                    "result file through phast.load_result; writes PNG files below 2000 px and manifest.json.")
    p.add_argument("run", help="run directory containing result.vtkhdf, or the result file")
    p.add_argument("-o", "--out", help="output directory (default: <run directory>/figures)")
    p.add_argument("--steps", default="first,peak,last",
                   help="comma-separated: first, peak, middle, last, or step values as Result.field takes them "
                        "(negative values count from the end), e.g. 0,5,-1")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"resolution (default {DEFAULT_DPI}; reduced "
                                                                "automatically to keep images below 2000 px)")
    p.add_argument("--fields", help="comma-separated field names to map (default: every stored field)")
    p.add_argument("--deform-scale", help="draw 2D maps on the deformed shape: a factor, or 'auto'")
    p.add_argument("--kinds", default=",".join(KINDS), help="comma-separated subset of setup,fields,evolution; "
                   "figures listed by an earlier manifest in the output directory and not drawn again are removed")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else None
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    try:
        deform = args.deform_scale
        if deform not in (None, "auto"):
            deform = float(deform)
        rows = draw_all(args.run, args.out, steps=args.steps, dpi=args.dpi, fields=fields, deform_scale=deform,
                        kinds=kinds)
    except (ResultLoadError, FileNotFoundError, ValueError) as exc:
        print(f"phast figures: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else load_result(args.run).path / "figures"
    for row in rows:
        print(f"{row['kind']:<13} {row['file']}  {row['width_px']} x {row['height_px']} px")
    print(f"manifest: {out / MANIFEST}")
    return 0
