"""Public read-only access to existing PhAST result directories."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


class ResultLoadError(ValueError):
    """Raised when a path is not a supported PhAST result directory."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _coerce_csv_value(value: str) -> Any:
    if value == "":
        return value
    try:
        number = int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
    return number


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif"}:
        return "image"
    if suffix in {".mp4", ".mov", ".webm"}:
        return "video"
    return "artifact"


_FIELD_ALIASES = {
    "d": "damage",
    "damage": "damage",
    "damage_nodal": "damage",
    "u": "displacement",
    "displacement": "displacement",
    "H": "history_field",
    "H_elem": "history_field",
    "H_nodal": "history_field_nodal",
    "history": "history_field",
    "history_field": "history_field",
    "history_field_nodal": "history_field_nodal",
    "psi_plus": "psi_plus",
    "strain": "strain",
    "stress": "stress",
    "velocity": "velocity",
    "v": "velocity",
    "acceleration": "acceleration",
    "a": "acceleration",
}


_DERIVED_FIELD_NAMES = {
    "displacement_mag",
    "displacement_magnitude",
    "max_principal_stress",
    "principal_stress",
    "von_mises",
    "von_mises_stress",
    "von_mises_strain",
}


def _reference_field_name(name: str) -> str:
    return _FIELD_ALIASES.get(name, name)


def _step_key_sort(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except (IndexError, ValueError):
        return -1


class Result:
    """Read-only handle for a PhAST run/output directory.

    The initial public surface is intentionally conservative: it reads existing
    manifests, metadata, CSV histories, and visual artifacts without changing
    output formats or invoking postprocessors.
    """

    _CSV_FILES = (
        "results.csv",
        "response.csv",
        "history.csv",
        "energy.csv",
        "solver_telemetry.csv",
        "timing_per_step.csv",
    )

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"Result directory does not exist: {self.path}")

        self._run_manifest_path = self.path / "run_manifest.json"
        self._run_metadata_path = self.path / "run_metadata.json"
        self._visual_manifest_path = self.path / "visual_manifest.json"
        if not self._run_manifest_path.exists() and not self._run_metadata_path.exists():
            raise ResultLoadError(
                f"{self.path} is missing run_manifest.json or run_metadata.json"
            )

        self._manifest_cache: dict[str, Any] | None = None
        self._metadata_cache: dict[str, Any] | None = None
        self._history_cache: dict[str, list[dict[str, Any]]] | None = None
        self._field_cache: dict[str, set[str]] | None = None
        self._field_source_cache: dict[str, tuple[str, set[str]]] | None = None
        self._visual_cache: list[dict[str, Any]] | None = None

    def manifest(self) -> dict[str, Any]:
        """Return the run manifest, falling back to legacy run metadata."""
        if self._manifest_cache is None:
            if self._run_manifest_path.exists():
                self._manifest_cache = _read_json(self._run_manifest_path)
            else:
                self._manifest_cache = dict(self.metadata())
        return dict(self._manifest_cache)

    def metadata(self) -> dict[str, Any]:
        """Return run metadata where available, otherwise manifest metadata."""
        if self._metadata_cache is None:
            if self._run_metadata_path.exists():
                self._metadata_cache = _read_json(self._run_metadata_path)
            else:
                self._metadata_cache = dict(self.manifest())
        return dict(self._metadata_cache)

    def history_names(self) -> list[str]:
        """List available CSV history names and supported aliases."""
        return sorted(self._histories().keys())

    def history(self, name: str) -> list[dict[str, Any]]:
        """Return a CSV-backed history by name.

        Current standard names include ``response`` for solid-mechanics
        response tables, ``energy``, ``history``, ``solver_telemetry``,
        ``timing_per_step``, and ``reaction_force`` when a history CSV exposes
        that column.
        """
        histories = self._histories()
        try:
            rows = histories[name]
        except KeyError as exc:
            available = ", ".join(sorted(histories))
            raise KeyError(
                f"Unknown history {name!r}. Available histories: {available}"
            ) from exc
        return [dict(row) for row in rows]

    def mesh(self) -> dict[str, Any]:
        """Return read-only mesh metadata or provenance for this result."""
        metadata_mesh = self.metadata().get("mesh")
        if isinstance(metadata_mesh, dict) and metadata_mesh:
            return dict(metadata_mesh)

        manifest_config = self.manifest().get("config")
        if isinstance(manifest_config, dict):
            for key in ("mesh", "geometry"):
                value = manifest_config.get(key)
                if isinstance(value, dict) and value:
                    return dict(value)

        trajectory_mesh = self._zarr_mesh_metadata() or self._h5_mesh_metadata()
        if trajectory_mesh:
            return trajectory_mesh

        raise ResultLoadError(f"No mesh metadata found in result directory: {self.path}")

    def field_names(self) -> list[str]:
        """List reference trajectory field names discovered read-only."""
        return sorted(self._fields().keys())

    def has_field(self, name: str) -> bool:
        """Return True if a stored field or supported alias is present."""
        return _reference_field_name(name) in self._fields()

    def field(self, name: str, step: int = -1):
        """Load a directly stored trajectory field as a NumPy array."""
        reference = _reference_field_name(name)
        sources = self._field_sources()
        if reference not in sources:
            if name in _DERIVED_FIELD_NAMES or reference in _DERIVED_FIELD_NAMES:
                available = ", ".join(self.field_names())
                raise ResultLoadError(
                    f"Field {name!r} is not stored directly in this result. "
                    f"Available stored fields: {available}"
                )
            available = ", ".join(self.field_names())
            raise ResultLoadError(
                f"Unknown field {name!r}. Available fields: {available}"
            )
        source_kind, raw_names = sources[reference]
        raw_name = name if name in raw_names else sorted(raw_names)[0]
        if source_kind == "zarr":
            return self._load_zarr_field(raw_name, step=step)
        if source_kind == "h5":
            return self._load_h5_field(raw_name, step=step)
        raise ResultLoadError(f"Unsupported field source {source_kind!r}")

    def visuals(self) -> list[dict[str, Any]]:
        """Return visual manifest rows or discovered media artifacts."""
        if self._visual_cache is None:
            if self._visual_manifest_path.exists():
                visual_rows = _read_json(self._visual_manifest_path)
                self._visual_cache = [dict(row) for row in visual_rows]
            else:
                rows = []
                for child in sorted(self.path.iterdir()):
                    if child.suffix.lower() not in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".gif",
                        ".mp4",
                        ".mov",
                        ".webm",
                    }:
                        continue
                    rows.append(
                        {
                            "file": child.name,
                            "artifact_type": _media_type(child),
                            "size_bytes": child.stat().st_size,
                        }
                    )
                self._visual_cache = rows
        return [dict(row) for row in self._visual_cache]

    def postprocess(
        self,
        *,
        fields: list[str] | tuple[str, ...] | str | None = None,
        dpi: int | None = None,
        format: str = "png",
        skip_gif: bool = False,
        only_gifs: bool = False,
        animation_format: str | None = None,
        animation_fields: list[str] | tuple[str, ...] | str | None = None,
        animation_frames: int | None = None,
    ) -> int:
        """Run the existing explicit postprocess CLI for this result directory."""
        cmd = [
            sys.executable,
            "-m",
            "phast",
            "postprocess",
            str(self.path),
            "--format",
            format,
        ]
        if dpi is not None:
            cmd.extend(["--dpi", str(int(dpi))])
        if fields is not None:
            cmd.extend(["--fields", _csv_arg(fields)])
        if skip_gif:
            cmd.append("--skip-gif")
        if only_gifs:
            cmd.append("--only-gifs")
        if animation_format is not None:
            cmd.extend(["--animation-format", animation_format])
        if animation_fields is not None:
            cmd.extend(["--animation-fields", _csv_arg(animation_fields)])
        if animation_frames is not None:
            cmd.extend(["--animation-frames", str(int(animation_frames))])
        completed = subprocess.run(cmd, check=False)
        return int(completed.returncode)

    def plot(self, field: str, *, step: int = -1, **options: Any):
        """Report the current explicit postprocessing boundary."""
        raise ResultLoadError(
            "Result.plot() does not generate new plots yet. Use visuals() to "
            "inspect existing visual artifacts, or run python -m phast "
            "postprocess <run_dir> explicitly."
        )

    def animate(self, field: str, **options: Any):
        """Report the current explicit animation boundary."""
        raise ResultLoadError(
            "Result.animate() does not generate new animations yet. Use "
            "visuals() to inspect existing animation artifacts, or run "
            "python -m phast postprocess <run_dir> explicitly."
        )

    def export(self, format: str, **options: Any):
        """Report the current explicit export boundary."""
        raise ResultLoadError(
            "Result.export() is deferred to the explicit postprocess/export "
            "pipeline and does not write new artifacts from a read-only Result."
        )

    def _histories(self) -> dict[str, list[dict[str, Any]]]:
        if self._history_cache is None:
            histories: dict[str, list[dict[str, Any]]] = {}
            for filename in self._CSV_FILES:
                path = self.path / filename
                if not path.exists():
                    continue
                rows = _read_csv(path)
                stem = path.stem
                histories[stem] = rows
                if filename == "response.csv":
                    histories["response"] = rows
                columns = set(rows[0]) if rows else set()
                if {"displacement", "reaction_kN"} <= columns:
                    histories["load_displacement"] = rows
                if {"applied_disp", "reaction_force"} <= columns:
                    histories["load_displacement"] = rows
                if any("reaction_force" in row for row in rows):
                    histories["reaction_force"] = rows
                if any("max_damage" in row for row in rows):
                    histories["max_damage"] = rows
                elif any("max_d" in row for row in rows):
                    histories["max_damage"] = [
                        {
                            **row,
                            "max_damage": row["max_d"],
                        }
                        for row in rows
                    ]
                if filename == "results.csv" and any("reaction_kN" in row for row in rows):
                    histories.setdefault("reaction", rows)
            self._history_cache = histories
        return self._history_cache

    def _fields(self) -> dict[str, set[str]]:
        if self._field_cache is None:
            self._field_cache = {
                name: set(raw_names)
                for name, (_, raw_names) in self._field_sources().items()
            }
        return self._field_cache

    def _field_sources(self) -> dict[str, tuple[str, set[str]]]:
        if self._field_source_cache is None:
            sources: dict[str, tuple[str, set[str]]] = {}
            for source_kind, names in (
                ("zarr", self._discover_zarr_fields()),
                ("h5", self._discover_h5_fields()),
            ):
                for raw_name in names:
                    reference = _reference_field_name(raw_name)
                    if reference not in sources:
                        sources[reference] = (source_kind, {raw_name})
                    elif sources[reference][0] == source_kind:
                        sources[reference][1].add(raw_name)
            self._field_source_cache = sources
        return self._field_source_cache

    def _discover_zarr_fields(self) -> set[str]:
        zarr_path = self.path / "training_data.zarr"
        if not zarr_path.is_dir():
            return set()
        try:
            import zarr
        except ImportError as exc:
            raise ResultLoadError(
                "Zarr trajectory discovery requires the optional zarr package"
            ) from exc

        root = zarr.open(str(zarr_path), mode="r")
        try:
            sim = root["simulation_data"]
        except KeyError:
            return set()

        names: set[str] = set()
        if "fields" in sim:
            names.update(sim["fields"].keys())
        if "trajectory" in sim:
            names.update({
                key
                for key in sim["trajectory"].keys()
                if key not in {"step", "time_s", "applied_disp", "reaction_force"}
            })
            return names
        if "steps" not in sim:
            return names
        steps = sim["steps"]
        step_keys = sorted(
            [key for key in steps.keys() if key.startswith("step_")],
            key=_step_key_sort,
        )
        if not step_keys:
            return set()
        for key in step_keys:
            names.update(steps[key].keys())
        return names

    def _load_zarr_field(self, raw_name: str, *, step: int):
        import numpy as np
        import zarr

        zarr_path = self.path / "training_data.zarr"
        root = zarr.open(str(zarr_path), mode="r")
        sim = root["simulation_data"]
        if "fields" in sim and raw_name in sim["fields"]:
            field_array = sim["fields"][raw_name]
            if getattr(field_array, "ndim", 0) == 0:
                return np.asarray(field_array[()])
            if len(field_array) <= 0:
                raise ResultLoadError(f"Zarr store {zarr_path} contains no snapshots")
            index = self._compact_zarr_field_index(sim, raw_name, step, len(field_array), zarr_path)
            return np.asarray(field_array[index])
        if "trajectory" in sim and raw_name in sim["trajectory"]:
            traj = sim["trajectory"]
            field_count = len(traj[raw_name])
            count = min(int(traj.attrs.get("count", field_count)), field_count)
            if field_count <= 0 or count <= 0:
                raise ResultLoadError(f"Zarr store {zarr_path} contains no snapshots")
            index = count - 1 if step == -1 else self._dense_zarr_index(
                traj, raw_name, step, field_count, zarr_path
            )
            return np.asarray(traj[raw_name][index])

        if "steps" not in sim:
            raise ResultLoadError(f"Zarr store {zarr_path} contains no step groups")
        step_group = self._select_step_group_with_field(
            sim["steps"], raw_name, step, zarr_path, "Zarr"
        )
        return np.asarray(step_group[raw_name])

    def _dense_zarr_index(
        self,
        traj,
        raw_name: str,
        step: int,
        field_count: int,
        zarr_path: Path,
    ) -> int:
        import numpy as np

        if "step" not in traj:
            raise ResultLoadError(
                f"Zarr dense trajectory {zarr_path} cannot select step {step}: "
                "missing step index"
            )
        steps = np.asarray(traj["step"][:], dtype=np.int64)
        matches = np.where(steps == int(step))[0]
        if matches.size == 0:
            raise ResultLoadError(f"Step {step} not found in Zarr store {zarr_path}")
        index = int(matches[-1])
        if index >= field_count:
            raise ResultLoadError(
                f"Field {raw_name!r} is not stored for step {step} in {zarr_path}"
            )
        return index

    def _compact_zarr_field_index(
        self,
        sim,
        raw_name: str,
        step: int,
        field_count: int,
        zarr_path: Path,
    ) -> int:
        if step == -1:
            return field_count - 1
        if "trajectory" in sim and "step" in sim["trajectory"]:
            return self._dense_zarr_index(
                sim["trajectory"], raw_name, step, field_count, zarr_path
            )
        index = int(step)
        if index < 0 or index >= field_count:
            raise ResultLoadError(f"Step {step} not found in Zarr store {zarr_path}")
        return index

    def _discover_h5_fields(self) -> set[str]:
        h5_path = self.path / "training_data.h5"
        if not h5_path.is_file():
            return set()
        try:
            import h5py
        except ImportError as exc:
            raise ResultLoadError(
                "H5 trajectory discovery requires the optional h5py package"
            ) from exc

        with h5py.File(h5_path, "r") as h5:
            if "simulation_data/steps" not in h5:
                return set()
            steps = h5["simulation_data/steps"]
            step_keys = sorted(
                [key for key in steps.keys() if key.startswith("step_")],
                key=_step_key_sort,
            )
            if not step_keys:
                return set()
            names: set[str] = set()
            for key in step_keys:
                names.update(steps[key].keys())
            return names

    def _load_h5_field(self, raw_name: str, *, step: int):
        import h5py
        import numpy as np

        h5_path = self.path / "training_data.h5"
        with h5py.File(h5_path, "r") as h5:
            if "simulation_data/steps" not in h5:
                raise ResultLoadError(f"H5 store {h5_path} contains no step groups")
            step_group = self._select_step_group_with_field(
                h5["simulation_data/steps"], raw_name, step, h5_path, "H5"
            )
            return np.asarray(step_group[raw_name])

    def _select_step_group(self, steps, step: int, store_path: Path, store_kind: str):
        step_keys = sorted(
            [key for key in steps.keys() if key.startswith("step_")],
            key=_step_key_sort,
        )
        if not step_keys:
            raise ResultLoadError(f"{store_kind} store {store_path} contains no step groups")
        if step == -1:
            return steps[step_keys[-1]]
        matches = [key for key in step_keys if _step_key_sort(key) == int(step)]
        if not matches:
            raise ResultLoadError(
                f"Step {step} not found in {store_kind} store {store_path}"
            )
        return steps[matches[-1]]

    def _select_step_group_with_field(
        self,
        steps,
        raw_name: str,
        step: int,
        store_path: Path,
        store_kind: str,
    ):
        step_keys = sorted(
            [key for key in steps.keys() if key.startswith("step_")],
            key=_step_key_sort,
        )
        if not step_keys:
            raise ResultLoadError(f"{store_kind} store {store_path} contains no step groups")
        if step == -1:
            for key in reversed(step_keys):
                if raw_name in steps[key]:
                    return steps[key]
            raise ResultLoadError(
                f"Field {raw_name!r} is not stored in {store_kind} store {store_path}"
            )
        matches = [key for key in step_keys if _step_key_sort(key) == int(step)]
        if not matches:
            raise ResultLoadError(
                f"Step {step} not found in {store_kind} store {store_path}"
            )
        group = steps[matches[-1]]
        if raw_name not in group:
            raise ResultLoadError(
                f"Field {raw_name!r} is not stored for step {step} in {store_path}"
            )
        return group

    def _zarr_mesh_metadata(self) -> dict[str, Any]:
        zarr_path = self.path / "training_data.zarr"
        if not zarr_path.is_dir():
            return {}
        try:
            import zarr
        except ImportError as exc:
            raise ResultLoadError(
                "Zarr mesh discovery requires the optional zarr package"
            ) from exc

        root = zarr.open(str(zarr_path), mode="r")
        try:
            mesh = root["simulation_data"]["mesh"]
        except KeyError:
            return {}
        return self._mesh_group_metadata(mesh, source=zarr_path.name)

    def _h5_mesh_metadata(self) -> dict[str, Any]:
        h5_path = self.path / "training_data.h5"
        if not h5_path.is_file():
            return {}
        try:
            import h5py
        except ImportError as exc:
            raise ResultLoadError(
                "H5 mesh discovery requires the optional h5py package"
            ) from exc

        with h5py.File(h5_path, "r") as h5:
            if "simulation_data/mesh" not in h5:
                return {}
            return self._mesh_group_metadata(
                h5["simulation_data/mesh"], source=h5_path.name
            )

    def _mesh_group_metadata(self, mesh_group, *, source: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in ("n_nodes", "n_elements"):
            if key in mesh_group.attrs:
                metadata[key] = int(mesh_group.attrs[key])
        if "n_nodes" not in metadata and "node_coordinates" in mesh_group:
            metadata["n_nodes"] = int(mesh_group["node_coordinates"].shape[0])
        if "n_elements" not in metadata and "element_connectivity" in mesh_group:
            metadata["n_elements"] = int(mesh_group["element_connectivity"].shape[0])
        if metadata:
            metadata["source"] = source
        return metadata

    def __repr__(self) -> str:
        return f"Result(path={str(self.path)!r})"


def load_result(path: str | Path) -> Result:
    """Load a read-only :class:`Result` from an existing run directory."""
    return Result(path)


def _csv_arg(value: list[str] | tuple[str, ...] | str) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)
