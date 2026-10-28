"""Open every conformance fixture in ParaView and compare against expected.json.

Run with ParaView's Python, not pytest:

    pvpython tests/result_v2/paraview_check.py <dir written by fixtures.py>

Uses only json and paraview.simple (no numpy: ParaView's bundled numpy does not
load on every macOS install). Prints a JSON report and exits 1 on any mismatch.
"""
import json
import os
import sys

from paraview import servermanager
from paraview.simple import OpenDataFile, GetParaViewVersion


def timesteps(reader):
    ts = reader.TimestepValues
    if ts is None:
        return []
    try:
        return [float(t) for t in ts]
    except TypeError:
        return [float(ts)]


def check(out_dir):
    expected = json.load(open(os.path.join(out_dir, "expected.json")))
    report = {"paraview": str(GetParaViewVersion()), "fixtures": {}}
    failures = 0
    for name, exp in expected.items():
        if "file" not in exp:
            continue
        res = {"errors": []}
        try:
            reader = OpenDataFile(os.path.join(out_dir, exp["file"]))
            if reader is None:
                raise RuntimeError("no reader found")
            ts = timesteps(reader)
            res["nsteps"] = len(ts)
            if len(ts) != exp["nsteps"]:
                res["errors"].append("steps %d, expected %d" % (len(ts), exp["nsteps"]))
            if "times" in exp:
                for a, b in zip(ts, exp["times"]):
                    if abs(a - b) > 1e-12 * max(1.0, abs(b)):
                        res["errors"].append("time %r, expected %r" % (a, b))
            reader.UpdatePipeline(ts[0] if ts else None) if ts else reader.UpdatePipeline()
            data = servermanager.Fetch(reader)
            if data.GetNumberOfPoints() != exp["n_points"] or data.GetNumberOfCells() != exp["n_cells"]:
                res["errors"].append("size %d points %d cells, expected %d %d" % (
                    data.GetNumberOfPoints(), data.GetNumberOfCells(), exp["n_points"], exp["n_cells"]))
            checked = 0
            for p in exp["probes"]:
                if p["step"] is not None:
                    if p["step"] >= len(ts):
                        res["errors"].append("probe step %d missing" % p["step"])
                        continue
                    reader.UpdatePipeline(ts[p["step"]])
                    data = servermanager.Fetch(reader)
                attrs = data.GetPointData() if p["where"] == "point" else data.GetCellData()
                arr = attrs.GetArray(p["array"])
                if arr is None:
                    res["errors"].append("array %s missing at step %r" % (p["array"], p["step"]))
                    continue
                got = arr.GetComponent(p["index"], p["component"])
                if abs(got - p["value"]) > 1e-6 * max(1.0, abs(p["value"])):
                    res["errors"].append("%s[%d,%d] step %r = %r, expected %r" % (
                        p["array"], p["index"], p["component"], p["step"], got, p["value"]))
                checked += 1
            res["probes_checked"] = checked
        except Exception as exc:  # report and continue with the next fixture
            res["errors"].append(repr(exc))
        failures += bool(res["errors"])
        report["fixtures"][name] = res
    report["failures"] = failures
    return report


if __name__ == "__main__":
    rep = check(sys.argv[1])
    print(json.dumps(rep, indent=1))
    sys.exit(1 if rep["failures"] else 0)
