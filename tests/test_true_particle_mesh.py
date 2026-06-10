import math

from phast.mesh import FEMMesh
from phast.mesh_generator import rectangular_sent_circular_inclusions


def test_rectangular_sent_circular_inclusions_is_not_double_counted(tmp_path):
    msh = tmp_path / "sent_particles.msh"
    rectangular_sent_circular_inclusions(
        str(msh),
        W=32.0,
        H=16.0,
        a=4.0,
        inclusions=[(16.0, 9.5, 1.6)],
        l0=0.1,
        h_crack=0.8,
        h_coarse=3.0,
        verbose=False,
    )
    mesh = FEMMesh(str(msh), device="cpu")
    total_area = float(mesh.areas.sum())
    notch_eps = min(0.01 * min(32.0, 16.0), 0.01)
    expected_area = 32.0 * 16.0 - 0.5 * 4.0 * (2.0 * notch_eps)
    assert math.isclose(total_area, expected_area, rel_tol=1e-6, abs_tol=1e-6)
