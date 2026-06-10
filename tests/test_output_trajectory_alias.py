"""Output trajectory alias tests."""

from phast.config import load_config, resolve_config


def test_output_trajectory_alias_enables_zarr_snapshots(tmp_path):
    yaml_path = tmp_path / "trajectory_alias.yaml"
    yaml_path.write_text("""
name: trajectory-alias
output:
  trajectory: true
  h5_every: 7
""")
    cfg = load_config(str(yaml_path))

    assert cfg.output.trajectory is True
    assert cfg.output.h5 is True
    assert resolve_config(cfg)["solver_config"].h5_every == 7


def test_output_h5_alias_sets_public_trajectory_flag(tmp_path):
    yaml_path = tmp_path / "h5_alias.yaml"
    yaml_path.write_text("""
name: h5-alias
output:
  h5: true
""")
    cfg = load_config(str(yaml_path))

    assert cfg.output.h5 is True
    assert cfg.output.trajectory is True
