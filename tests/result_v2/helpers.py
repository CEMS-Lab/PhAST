"""Comparison of two result files, shared by the writer tests."""
from __future__ import annotations

import json

import h5py
import numpy as np

_ROOT_SKIP = {"created", "finished"}


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OSU" or b.dtype.kind in "OSU":
        return str(a.astype(str)) == str(b.astype(str)) if a.shape == b.shape else False
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def _attr(v):
    v = v.decode() if isinstance(v, bytes) else v
    if isinstance(v, str) and v.startswith("{"):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v.tolist() if isinstance(v, np.ndarray) else v


def compare(ref_path, new_path):
    """Every dataset, link and dataset attribute of the reference appears equal in the new file."""
    problems = []
    with h5py.File(ref_path, "r") as ref, h5py.File(new_path, "r") as new:
        for k, v in ref.attrs.items():
            if k not in _ROOT_SKIP and _attr(new.attrs.get(k)) != _attr(v):
                problems.append(f"root attr {k}: {new.attrs.get(k)!r} != {v!r}")

        def visit(name, link):
            if name.startswith("meta"):
                return
            if isinstance(link, h5py.SoftLink):
                nl = new.get(name, getlink=True)
                if not isinstance(nl, h5py.SoftLink) or nl.path != link.path:
                    problems.append(f"link {name}")
                return
            obj = ref[name]
            if isinstance(obj, h5py.Group):
                for k, v in obj.attrs.items():
                    if name in new and _attr(new[name].attrs.get(k)) != _attr(v):
                        problems.append(f"{name} attr {k}: {new[name].attrs.get(k)!r} != {v!r}")
                return
            if name not in new:
                problems.append(f"missing {name}")
                return
            d = new[name]
            if not _same(obj[...], d[...]):
                problems.append(f"values {name}")
            if obj.dtype.kind != d.dtype.kind or (obj.dtype.kind == "f" and obj.dtype != d.dtype):
                problems.append(f"dtype {name}: {d.dtype} != {obj.dtype}")
            for k, v in obj.attrs.items():
                if _attr(d.attrs.get(k)) != _attr(v):
                    problems.append(f"{name} attr {k}: {d.attrs.get(k)!r} != {v!r}")
        ref.visititems_links(visit)
    return problems
