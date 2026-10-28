"""Machine-readable schema of the PhAST result file.

Writers store :func:`schema_text` in ``meta/schema_yaml`` so every file
carries the schema it was written against.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

SCHEMA_NAME = "phast.result"
SCHEMA_VERSION = "2.0.0"          # what the writer writes unless a 2.1 feature is used
SCHEMA_VERSION_21 = "2.1.0"       # written when a file uses an item added in layout 2.1
KNOWN_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_21)
LATEST_VERSION = KNOWN_VERSIONS[-1]


def schema_version_for(file_version: str) -> str:
    """The known schema a file written against ``file_version`` is checked with.

    ``2.0.x`` is checked with 2.0.0, ``2.1.x`` with 2.1.0, and a later minor
    version with the newest schema this build knows (its added items are not
    checked). Raises ``ValueError`` for a version that is not ``2.MINOR.PATCH``.
    """
    parts = str(file_version).split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts) or parts[0] != "2":
        raise ValueError(f"{file_version!r} is not a 2.MINOR.PATCH version")
    minor = int(parts[1])
    known_minors = [int(v.split(".")[1]) for v in KNOWN_VERSIONS]
    pick = max((m for m in known_minors if m <= minor), default=known_minors[0])
    return KNOWN_VERSIONS[known_minors.index(pick)]


def schema_text(version: str = SCHEMA_VERSION) -> str:
    """Return the schema YAML text for ``version``."""
    path = resources.files(__package__).joinpath("schema", f"phast_result_{version}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"no PhAST result schema for version {version!r}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_schema(version: str = SCHEMA_VERSION) -> dict[str, Any]:
    """Return the parsed schema for ``version``."""
    schema = yaml.safe_load(schema_text(version))
    if schema.get("schema") != SCHEMA_NAME or schema.get("version") != version:
        raise ValueError(f"schema file for {version!r} declares {schema.get('schema')!r} {schema.get('version')!r}")
    return schema
