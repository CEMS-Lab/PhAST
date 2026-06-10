"""Pytest hooks for the configured ``tests/`` collection root."""

from __future__ import annotations

import pytest

from ._tier_markers import auto_tier_markers, auto_timeout_seconds


def pytest_collection_modifyitems(config, items):
    """Apply coarse test tiers so ``pytest -m <tier>`` is useful."""
    for item in items:
        existing = {mark.name for mark in item.iter_markers()}
        inferred = auto_tier_markers(item.nodeid, existing)
        for marker in sorted(inferred):
            item.add_marker(getattr(pytest.mark, marker))
        timeout = auto_timeout_seconds(existing | inferred)
        if timeout is not None:
            item.add_marker(pytest.mark.timeout(timeout))
