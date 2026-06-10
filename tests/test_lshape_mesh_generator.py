from pathlib import Path

from phast import mesh_generator


def test_lshape_load_point_is_next_to_reentrant_corner(tmp_path, monkeypatch):
    """Ambati L-panel load point is 30 mm from the inner corner."""

    monkeypatch.setattr(mesh_generator, "_run_gmsh", lambda *args, **kwargs: None)
    msh = tmp_path / "lshape.msh"

    mesh_generator.l_shaped_panel(
        str(msh),
        L=250.0,
        l0=1.1875,
        h_crack=0.3,
        h_coarse=25.0,
        verbose=False,
    )

    geo = Path(str(msh).replace(".msh", ".geo")).read_text()
    assert "Point(7) = {L + 30.0, L, 0, h_crack}" in geo
    assert "Line(3) = {3, 7};   // cutout horizontal left part" in geo
    assert 'Physical Point("load_point") = {7};' in geo
    assert 'Physical Point("load_segment") = {7};' in geo
    assert 'Physical Curve("load_segment")' not in geo
