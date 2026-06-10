"""CLI/API consistency regressions for run_config."""

import sys

import pytest


def test_gen_alpha_missing_config_fails_before_dry_run(monkeypatch, tmp_path,
                                                       capsys):
    from phast import run_config

    missing = tmp_path / 'does-not-exist.yaml'
    monkeypatch.setattr(
        sys, 'argv',
        ['run_config', str(missing), '--time_integrator', 'gen_alpha'],
    )

    with pytest.raises(SystemExit) as exc:
        run_config.main()

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert 'cannot read config' in captured.err
    assert 'gen_alpha' not in captured.out


def test_profile_override_is_registered(monkeypatch, capsys):
    from phast import run_config

    monkeypatch.setattr(sys, 'argv', ['run_config', '--help'])

    with pytest.raises(SystemExit) as exc:
        run_config.main()

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert '--profile' in captured.out


def test_validate_only_prints_nonfatal_advisories(monkeypatch, tmp_path,
                                                  capsys):
    from phast import run_config

    cfg = tmp_path / 'coarse.yaml'
    cfg.write_text(
        """
geometry:
  parameters:
    h_crack: 0.75
material:
  l0: 1.0
solver:
  solver_type: explicit
""",
        encoding='utf-8',
    )
    monkeypatch.setattr(
        sys, 'argv',
        ['run_config', str(cfg), '--validate-only'],
    )

    with pytest.raises(SystemExit) as exc:
        run_config.main()

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert 'passes schema validation' in captured.out
    assert 'missing schema_version' in captured.err
    assert 'h/l0=0.75' in captured.err
