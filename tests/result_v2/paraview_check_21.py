"""Open the layout-2.1 changing-mesh fixtures in ParaView and compare against expected_21.json.

Run with ParaView's Python, not pytest:

    pvpython tests/result_v2/paraview_check_21.py <dir written by fixtures_21.py>

For every stored step it checks the number of points and cells, the point coordinates of the last
point, the connectivity of the last cell and the probe values of the arrays, so that a wrong offset
for the second mesh of a part (points, cells, connectivity or array rows) cannot pass. Uses only json
and paraview.simple (no numpy). Prints a JSON report and exits 1 on any mismatch.
"""
import json
import os
import sys

from paraview import servermanager
from paraview.simple import GetParaViewVersion, OpenDataFile


def timesteps(reader):
    ts = reader.TimestepValues
    if ts is None:
        return []
    try:
        return [float(t) for t in ts]
    except TypeError:
        return [float(ts)]


def check(out_dir):
    expected = json.load(open(os.path.join(out_dir, "expected_21.json")))
    report = {"paraview": str(GetParaViewVersion()), "fixtures": {}}
    failures = 0
    for name, exp in expected.items():
        res = {"errors": [], "steps_checked": 0}
        try:
            reader = OpenDataFile(os.path.join(out_dir, exp["file"]))
            if reader is None:
                raise RuntimeError("no reader found")
            ts = timesteps(reader)
            res["nsteps"] = len(ts)
            if len(ts) != exp["nsteps"]:
                res["errors"].append("steps %d, expected %d" % (len(ts), exp["nsteps"]))
            for a, b in zip(ts, exp["times"]):
                if abs(a - b) > 1e-12 * max(1.0, abs(b)):
                    res["errors"].append("time %r, expected %r" % (a, b))
            for s, size in enumerate(exp["steps"]):
                if s >= len(ts):
                    break
                reader.UpdatePipeline(ts[s])
                data = servermanager.Fetch(reader)
                if data.GetNumberOfPoints() != size["n_points"] or data.GetNumberOfCells() != size["n_cells"]:
                    res["errors"].append("step %d: %d points %d cells, expected %d %d" % (
                        s, data.GetNumberOfPoints(), data.GetNumberOfCells(), size["n_points"], size["n_cells"]))
                    continue
                p = exp["points"][s]
                got = data.GetPoint(p["point"])
                if max(abs(got[i] - p["xyz"][i]) for i in range(3)) > 1e-12:
                    res["errors"].append("step %d: point %d at %r, expected %r" % (s, p["point"], got, p["xyz"]))
                c = exp["cells"][s]
                cell = data.GetCell(c["cell"])
                ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                if ids != c["points"]:
                    res["errors"].append("step %d: cell %d has points %r, expected %r" % (s, c["cell"], ids, c["points"]))
                for pr in exp["probes"]:
                    if pr["step"] != s:
                        continue
                    attrs = data.GetPointData() if pr["where"] == "point" else data.GetCellData()
                    arr = attrs.GetArray(pr["array"])
                    if arr is None:
                        res["errors"].append("step %d: array %s missing" % (s, pr["array"]))
                        continue
                    value = arr.GetComponent(pr["index"], pr["component"])
                    if abs(value - pr["value"]) > 1e-6 * max(1.0, abs(pr["value"])):
                        res["errors"].append("step %d: %s[%d,%d] = %r, expected %r" % (
                            s, pr["array"], pr["index"], pr["component"], value, pr["value"]))
                res["steps_checked"] += 1
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
