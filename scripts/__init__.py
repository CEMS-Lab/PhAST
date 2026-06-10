"""Top-level ``scripts`` namespace.

Marker file added in #251 so the diagnostics dispatcher is reachable as a
runnable module:

    python -m scripts.diagnostics.dispatch <category> <name> [args...]

The pyproject ``[tool.setuptools.packages.find]`` filter is restricted to
``phast*``, so this package is NOT shipped on install --- it is
purely for in-tree CLI use.
"""
