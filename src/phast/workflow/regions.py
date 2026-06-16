"""Internal region-reference validation for workflow specs."""
from __future__ import annotations

from dataclasses import dataclass

from .specs import ProblemSpec


@dataclass(frozen=True)
class RegionReferenceIssue:
    owner: str
    region: str
    message: str


def validate_problem_spec_regions(spec: ProblemSpec) -> list[RegionReferenceIssue]:
    """Return advisory issues for references to undeclared regions."""
    declared = {region.name for region in spec.regions}
    issues: list[RegionReferenceIssue] = []

    for material in spec.materials:
        if material.region:
            _append_missing(issues, declared, "material", material.region)
    for initial_condition in spec.initial_conditions:
        if initial_condition.region:
            _append_missing(
                issues, declared, "initial_condition", initial_condition.region
            )
    for boundary_condition in spec.boundary_conditions:
        _append_missing(
            issues, declared, "boundary_condition", boundary_condition.region
        )
    for history in spec.outputs.history:
        if history.region:
            _append_missing(issues, declared, "history_output", history.region)
    return issues


def _append_missing(
    issues: list[RegionReferenceIssue],
    declared: set[str],
    owner: str,
    region: str,
) -> None:
    if region not in declared:
        issues.append(
            RegionReferenceIssue(
                owner=owner,
                region=region,
                message=f"{owner} region {region!r} is not declared",
            )
        )
