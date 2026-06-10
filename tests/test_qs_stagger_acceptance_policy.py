import argparse
import math

from examples.quasistatic._run_utils import (
    add_stagger_acceptance_args,
    apply_stagger_acceptance_policy,
)


def _parse(*extra):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutback_stagger_fraction", type=float, default=0.85)
    parser.add_argument("--residual_cutback_limit", type=float, default=1.0e-4)
    add_stagger_acceptance_args(parser)
    return parser.parse_args(list(extra))


def test_stagger_policy_fail_preserves_strict_default():
    args = _parse()

    fail = apply_stagger_acceptance_policy(args)

    assert fail is True
    assert args.cutback_stagger_fraction == 0.85
    assert args.residual_cutback_limit == 1.0e-4


def test_stagger_policy_warn_accepts_without_changing_cutbacks():
    args = _parse("--stagger_nonconvergence_policy", "warn")

    fail = apply_stagger_acceptance_policy(args)

    assert fail is False
    assert args.cutback_stagger_fraction == 0.85
    assert args.residual_cutback_limit == 1.0e-4


def test_stagger_policy_phasefieldx_accepts_and_disables_cap_cutbacks():
    args = _parse("--stagger_nonconvergence_policy", "phasefieldx")

    fail = apply_stagger_acceptance_policy(args)

    assert fail is False
    assert args.cutback_stagger_fraction > 1.0
    assert math.isinf(args.residual_cutback_limit)
