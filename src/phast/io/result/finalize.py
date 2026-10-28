"""Archive copy of a closed result file: gzip 4, shuffle and fletcher32.

The archive is rewritten dataset by dataset with the same shapes, chunks,
dtypes, attributes and soft links, compared value by value with the source, and
only then replaces it. Filters are built into every HDF5 library, so the
archive opens on every platform and in ParaView.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from .writer import LIBVER, sha256_file

ARCHIVE_FILTERS = dict(compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)


def _copy(src: h5py.Group, dst: h5py.Group):
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for name in src:
        link = src.get(name, getlink=True)
        if isinstance(link, h5py.SoftLink):
            dst[name] = h5py.SoftLink(link.path)
            continue
        obj = src[name]
        if isinstance(obj, h5py.Group):
            _copy(obj, dst.create_group(name, track_order=True))
            continue
        if obj.shape == () or obj.dtype.kind in "OSU" or obj.size == 0 and obj.maxshape == obj.shape:
            ds = dst.create_dataset(name, data=obj[()], dtype=obj.dtype, track_times=False)
        else:
            ds = dst.create_dataset(name, shape=obj.shape, maxshape=obj.maxshape, dtype=obj.dtype,
                                    chunks=obj.chunks or True, track_times=False, **ARCHIVE_FILTERS)
            if obj.size:
                ds[...] = obj[...]
        for k, v in obj.attrs.items():
            ds.attrs[k] = v


def _equal(a: h5py.File, b: h5py.File) -> list[str]:
    bad = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            x, y = obj[()], b[name][()]
            same = (np.array_equal(x, y, equal_nan=np.asarray(x).dtype.kind == "f")
                    if np.asarray(x).dtype.kind not in "OSU"
                    else np.shape(x) == np.shape(y) and bool(np.all(np.asarray(x) == np.asarray(y))))   # text, also arrays of it
            if not same:
                bad.append(name)
    a.visititems(visit)
    return bad


def finalize(path: str | os.PathLike, archive: bool = True) -> Path:
    """Rewrite a closed result file with the archive filters, check it and replace the source."""
    path = Path(path)
    if not archive:
        return path
    with h5py.File(path, "r") as src:
        if src.attrs.get("status") == "running":
            raise RuntimeError(f"{path.name} is still running; recover or close it first")
    tmp = path.with_name(path.name + ".archive.partial")
    with h5py.File(path, "r") as src, h5py.File(tmp, "w", libver=LIBVER, track_order=True) as dst:
        _copy(src, dst)
    with h5py.File(path, "r") as src, h5py.File(tmp, "r") as dst:
        bad = _equal(src, dst)
    if bad:
        tmp.unlink()
        raise RuntimeError(f"archive copy differs from the source in {bad[:5]}")
    os.replace(tmp, path)
    path.with_name(path.name + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return path
