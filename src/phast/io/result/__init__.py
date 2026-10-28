"""PhAST result file standard (layout 2.0, with the 2.1 additions of the specification).

The layout is specified in ``docs/user_guide/result_file_layout.md`` and its use
is described in ``docs/user_guide/result_file.md``. This package holds the
machine-readable schema, the writer, the reader of one result file (used by
``phast.load_result``), the validator and the recovery and archive commands.
"""
from .schema import SCHEMA_NAME, SCHEMA_VERSION, SCHEMA_VERSION_21, load_schema, schema_text, schema_version_for
from .cohesive import interface_connectivity
from .crack_tips import fit_speed, locate_tip
from .finalize import finalize
from .reader import ResultFile, is_result_file
from .recover import Recovery, recover
from .validate import Finding, Report, validate
from .writer import (Field, ImageGrid, ResultWriter, ResumeRefused, SoftStop, TIME_LIMIT_EXIT_CODE, UnstructuredMesh,
                     estimate_size, isotropic_wave_speeds, refinement_maps, sha256_file)

__all__ = ["SCHEMA_NAME", "SCHEMA_VERSION", "SCHEMA_VERSION_21", "load_schema", "schema_text", "schema_version_for",
           "fit_speed", "locate_tip", "interface_connectivity", "isotropic_wave_speeds", "refinement_maps",
           "Field", "ImageGrid", "ResultWriter", "ResumeRefused", "SoftStop", "TIME_LIMIT_EXIT_CODE", "UnstructuredMesh",
           "estimate_size", "finalize", "sha256_file",
           "ResultFile", "is_result_file",
           "Finding", "Report", "Recovery", "recover", "validate"]
