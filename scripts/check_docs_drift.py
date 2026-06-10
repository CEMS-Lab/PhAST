"""Fail on known-stale documentation/config wording.

This is intentionally narrow. It catches phrases that have already caused
user-facing drift: old enum names, deprecated solver-status claims, and stale
benchmark comments. Add patterns only when there is a clear replacement.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PATHS = (
    "README.md",
    "DOCUMENTATION.md",
    "docs",
    "configs",
)


@dataclass(frozen=True)
class StalePattern:
    name: str
    pattern: re.Pattern[str]
    message: str


PATTERNS = (
    StalePattern(
        name="stagger-energy-enum",
        pattern=re.compile(
            r"(stagger_criterion|--stagger_criterion)\s*[:= ]\s*"
            r"['\"]?energy['\"]?",
            re.IGNORECASE,
        ),
        message="Use 'am_energy' for the stagger energy criterion.",
    ),
    StalePattern(
        name="secant-primary-claim",
        pattern=re.compile(
            r"(SecantCGSolver[^\n]{0,120}\bprimary\b|"
            r"\bprimary\b[^\n]{0,120}SecantCGSolver)",
            re.IGNORECASE,
        ),
        message=(
            "Do not claim SecantCGSolver is the primary implicit path; "
            "document current solver selection instead."
        ),
    ),
    StalePattern(
        name="quasistatic-unused-claim",
        pattern=re.compile(
            r"(QuasiStaticSolver[^\n]{0,120}\bunused\b|"
            r"\bunused\b[^\n]{0,120}QuasiStaticSolver)",
            re.IGNORECASE,
        ),
        message="Do not claim QuasiStaticSolver is unused.",
    ),
    StalePattern(
        name="rigid-connector-ignored-claim",
        pattern=re.compile(
            r"(silently\s+ignores?[^\n]{0,80}rigid[_ -]?connector|"
            r"rigid[_ -]?connector[^\n]{0,80}silently\s+ignores?)",
            re.IGNORECASE,
        ),
        message=(
            "Rigid connector support has changed; do not claim it is "
            "silently ignored without a current code audit."
        ),
    ),
    StalePattern(
        name="monolithic-production-claim",
        pattern=re.compile(
            r"MonolithicSolver[^\n]{0,120}"
            r"(bypassing\s+the\s+stagger\s+loop\s+entirely|"
            r"\bcustomer[- ]facing\b|\bproduction\b)",
            re.IGNORECASE,
        ),
        message=(
            "MonolithicSolver is experimental until its bound-constrained "
            "irreversibility work closes; document staggered implicit solves "
            "as the customer-facing path."
        ),
    ),
    StalePattern(
        name="legacy-run-yaml-command",
        pattern=re.compile(
            r"python\s+-m\s+phast\.run_yaml\b|"
            r"^example:\s*phast\.run_yaml\b",
            re.IGNORECASE,
        ),
        message=(
            "Use the canonical YAML CLI: "
            "'python -m phast run configs/<name>.yaml'."
        ),
    ),
    StalePattern(
        name="example-key-required-claim",
        pattern=re.compile(
            r"Every\s+YAML\s+must\s+have\s+a\s+top-level\s+`?example`?",
            re.IGNORECASE,
        ),
        message=(
            "The legacy example field is optional/provenance-only; it is "
            "not required by the canonical YAML run path."
        ),
    ),
    StalePattern(
        name="at1-post-clamp-claim",
        pattern=re.compile(
            r"AT1[^\n]{0,160}projected[_ -]?cg[^\n]{0,80}"
            r"\bor\s+post[- ]?clamp\b",
            re.IGNORECASE,
        ),
        message=(
            "AT1 requires projected bound enforcement; do not present "
            "post-clamp as an equivalent AT1 damage solve."
        ),
    ),
    StalePattern(
        name="qs-preconditioner-default-auto-claim",
        pattern=re.compile(
            r"\|\s*`--preconditioner`\s*\|\s*str\s*\|\s*auto\s*\|",
            re.IGNORECASE,
        ),
        message=(
            "Quasi-static benchmark CLIs default to Jacobi; do not document "
            "'auto' as the customer-facing --preconditioner default."
        ),
    ),
    StalePattern(
        name="legacy-timing-comparison-path",
        pattern=re.compile(
            r"examples/timing_comparisons|timing_comparisons/sent_clean",
            re.IGNORECASE,
        ),
        message=(
            "Timing comparison artifacts live under "
            "examples/dynamic/timing_comparisons/{sent,kalthoff}."
        ),
    ),
    StalePattern(
        name="legacy-doc-link",
        pattern=re.compile(
            r"tutorial/(minimal-example|inversion-demo)\.md|api/index\.rst",
            re.IGNORECASE,
        ),
        message="Link to the current docs/tutorial or docs/api Markdown pages.",
    ),
    StalePattern(
        name="hdf5-primary-output-claim",
        pattern=re.compile(
            r"HDF5\s+snapshots[^\n]{0,120}Main\s+reusable\s+simulation\s+output",
            re.IGNORECASE,
        ),
        message=(
            "Document HDF5 as legacy compatibility and prefer current "
            "trajectory/output conventions."
        ),
    ),
    StalePattern(
        name="hdf5-packager-claim",
        pattern=re.compile(
            r"\bHDF5\s+packager\b|"
            r"\bHDF5\s+fallback\b|"
            r"\bH5\s+fallback\b",
            re.IGNORECASE,
        ),
        message=(
            "Use a Zarr-only packager for new dataset-generation workflows; "
            "HDF5 is legacy solver/post-processing compatibility."
        ),
    ),
    StalePattern(
        name="zarr-guaranteed-smaller-claim",
        pattern=re.compile(
            r"Zarr\s+is\s+(guaranteed|universally)\s+smaller|"
            r"Zarr\s+(always|guaranteedly)\s+"
            r"(writes|produces|creates|yields)\s+smaller",
            re.IGNORECASE,
        ),
        message=(
            "Do not make blanket size claims; Zarr size depends on dtype, "
            "chunking, compressor, data smoothness, metadata, and archive layout."
        ),
    ),
    StalePattern(
        name="b7-stale-half-plate-height",
        pattern=re.compile(
            r"100\s*x\s*40\s*mm\s+half[- ]plate|"
            r"100\s*×\s*40\s*mm\s+half[- ]plate|"
            r"half[- ]plate[^\n]{0,80}H:\s*40\.0",
            re.IGNORECASE,
        ),
        message=(
            "COMSOL B7 full sample height is 40 mm, but the symmetry "
            "computational half-plate is height/2 = 20 mm."
        ),
    ),
    StalePattern(
        name="b7-stale-full-plate-height",
        pattern=re.compile(
            r"Mirroring\s+about\s+y=0\s*->\s*100\s*x\s*80\s*mm|"
            r"100\s*x\s*80\s*full[- ]plate|"
            r"100\s*×\s*80\s*full[- ]plate",
            re.IGNORECASE,
        ),
        message=(
            "The COMSOL B7 full-plate equivalent is 100 x 40 mm; "
            "100 x 80 mm was a stale doubled-height interpretation."
        ),
    ),
    StalePattern(
        name="b7-stale-comsol-verlet-parity",
        pattern=re.compile(
            r"COMSOL[^\n]{0,120}(explicit\s+Verlet|Verlet\s+velocity|"
            r"Velocity[- ]Verlet)|"
            r"explicit[- ]dynamics\s+version\s*\(Verlet",
            re.IGNORECASE,
        ),
        message=(
            "Strict COMSOL B7 parity uses the Time Dependent generalized-alpha "
            "setup (rho_inf/high-frequency amplification 0.5); do not document "
            "Verlet as the COMSOL parity integrator."
        ),
    ),
)


def _iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() in {
                ".md",
                ".rst",
                ".txt",
                ".yaml",
                ".yml",
            }:
                files.append(child)
    return sorted(set(files))


def check_paths(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in _iter_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for stale in PATTERNS:
            for match in stale.pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                snippet = text[match.start():match.end()].replace("\n", " ")
                errors.append(
                    f"{path}:{line_no}: {stale.name}: {stale.message} "
                    f"(matched {snippet!r})"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Files/directories to scan. Defaults to public docs/configs.",
    )
    args = parser.parse_args(argv)

    errors = check_paths([Path(p) for p in args.paths])
    if errors:
        print("Known-stale documentation/config wording found:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
