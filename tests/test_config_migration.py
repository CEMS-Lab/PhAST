"""Lock-in tests for the configs/*.yaml migration to inline material +
new BC vocabulary (issue #147, phase 3.1 of epic #136).

Each migrated config is checked against the *legacy* (preset+overrides)
specification it replaced. Two equivalences are pinned:

* The resolved :class:`Material` dataclass is ``==`` to the legacy one
  (bit-exact: same E, nu, Gc, l0, rho, energy_split, pf_model,
  plane_stress, eta_residual).
* For BC migrations (``neumann``/``fix component=<axis>`` -> ``traction``/
  ``symmetry``), the resolved :class:`BoundaryConditions` produce the
  same nodal force / mask / value tensors as their legacy counterparts.

In addition:

* Every migrated config still passes ``validate_config_file``
  (delegated to :class:`tests.test_config_validation.TestShippedConfigsValid`).
* :func:`test_unit_suffix_string_form` locks in the unit-suffix
  showcase on ``configs/B1_branching_glass.yaml`` so that CI catches
  any regression in :func:`units.parse_quantity` of the SI suffix
  paths flagged for ergonomic preset overrides.
"""

from pathlib import Path

import pytest
import torch

from phast.config import load_config
from phast.config_validation import (
    validate_config_file, format_errors,
)
from phast.material import create_material
from phast.boundary_conditions import BoundaryConditions

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / 'configs'


# ---------------------------------------------------------------------------
# Migrated configs and their legacy (preset+overrides) twin.
# ---------------------------------------------------------------------------

# (yaml_basename, legacy preset, legacy overrides dict)
MIGRATED_MATERIALS = [
    ('B1_branching_glass.yaml',
     'glass_borden', {'l0': 0.25, 'energy_split': 'spectral'}),
    ('B2_kalthoff_winkler.yaml',
     'maraging_steel_kw', {'l0': 0.5, 'energy_split': 'spectral'}),
    ('B3_dynamic_sent.yaml',
     'glass_borden', {'l0': 0.5, 'energy_split': 'spectral'}),
    ('B5_pmma_branching.yaml',
     'pmma_bleyer', {}),
    ('B6_perforated_10holes.yaml',
     'pmma_bleyer', {}),
    ('B6_perforated_1hole_far.yaml',
     'pmma_bleyer', {}),
    ('B6_perforated_1hole_near.yaml',
     'pmma_bleyer', {}),
    ('B6_perforated_30holes.yaml',
     'pmma_bleyer', {}),
    ('B7_dynamic_crack_branching_comsol.yaml',
     'glass_borden',
     {'l0': 0.5, 'pf_model': 'AT1', 'energy_split': 'amor'}),
    ('QS_lshaped_concrete.yaml',
     'l_shaped_concrete', {'l0': 1.1875, 'energy_split': 'spectral'}),
    ('QS_notched_holed_plate.yaml',
     'cement_mortar_ambati', {}),
]


def _resolve_material(cfg):
    """Mimic ``resolve_config``'s material resolution path."""
    inline = {
        k: getattr(cfg.material, k)
        for k in ('E', 'nu', 'Gc', 'l0', 'rho', 'eta_residual',
                  'energy_split', 'pf_model', 'plane_stress', 'kinematics')
        if getattr(cfg.material, k) is not None
    }
    args = dict(cfg.material.overrides)
    args.update(inline)
    return create_material(preset=cfg.material.preset, **args)


# ---------------------------------------------------------------------------
# Material bit-exactness across the migration
# ---------------------------------------------------------------------------

class TestMigratedMaterialsBitExact:
    """Resolved Material dataclass must match the legacy spec exactly."""

    @pytest.mark.parametrize('basename,preset,overrides', MIGRATED_MATERIALS)
    def test_material_dataclass_equal(self, basename, preset, overrides):
        cfg = load_config(_CONFIG_DIR / basename)
        m_new = _resolve_material(cfg)
        m_old = create_material(preset=preset, **overrides)
        assert m_new == m_old, (
            f"Migration of {basename} drifted: legacy material is\n"
            f"  {m_old}\n"
            f"but config now resolves to\n  {m_new}"
        )


# ---------------------------------------------------------------------------
# BC vocabulary bit-exactness
# ---------------------------------------------------------------------------

def _unit_square_mesh():
    class _M:
        pass
    m = _M()
    m.nodes = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=torch.float64)
    m.elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    m.n_nodes = 4
    return m


class TestNewBCVocabularyBitExact:
    """`traction ramp_type=constant` and `symmetry axis=<a>` are
    drop-in replacements for `neumann` and `fix component=<a>`."""

    def test_traction_constant_matches_neumann(self):
        mesh = _unit_square_mesh()
        top = torch.tensor([2, 3], dtype=torch.long)

        a = BoundaryConditions(4, device='cpu', dtype=torch.float64)
        a.add_neumann(top, [0.0, 1.0])
        b = BoundaryConditions(4, device='cpu', dtype=torch.float64)
        b.add_traction(top, [0.0, 1.0], ramp_type='constant')

        fa = a.get_neumann_forces(mesh)
        fb = b.get_neumann_forces(mesh)
        assert torch.equal(fa, fb)

    def test_symmetry_y_matches_fix_component_1(self):
        idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        a = BoundaryConditions(n_nodes=10, device='cpu',
                               dtype=torch.float64)
        a.fix(idx, component=1)
        b = BoundaryConditions(n_nodes=10, device='cpu',
                               dtype=torch.float64)
        b.add_symmetry(idx, axis='y')
        ma, va = a.get_masks_and_values()
        mb, vb = b.get_masks_and_values()
        assert torch.equal(ma, mb)
        assert torch.equal(va, vb)


# ---------------------------------------------------------------------------
# Schema validator still happy with all migrated configs.
# ---------------------------------------------------------------------------

class TestMigratedConfigsValidate:
    """Each migrated config validates cleanly under the schema."""

    @pytest.mark.parametrize(
        'basename,_p,_o', MIGRATED_MATERIALS,
        ids=[t[0] for t in MIGRATED_MATERIALS],
    )
    def test_validates(self, basename, _p, _o):
        path = _CONFIG_DIR / basename
        raw, errs = validate_config_file(path)
        assert errs == [], format_errors(errs, path)


# ---------------------------------------------------------------------------
# Unit-suffix lock-in: at least one shipped config exercises the SI
# suffix string form on the inline material block (issue #141).
# ---------------------------------------------------------------------------

def test_unit_suffix_string_form_in_b1():
    """B1's inline material uses string suffix forms ("32 GPa", "3 J/m^2",
    "0.5 mm", "2450 kg/m^3"). The resolved Material must still equal the
    legacy preset+overrides specification bit-exactly."""
    path = _CONFIG_DIR / 'B1_branching_glass.yaml'
    with open(path) as fh:
        text = fh.read()
    # Spot-check the suffix strings actually appear in the config
    # (catches accidental "fix" of the showcase by future editors).
    for needle in ('"32 GPa"', '"3 J/m^2"', '"0.25 mm"', '"2450 kg/m^3"'):
        assert needle in text, (
            f"Expected unit-suffix form {needle!r} in B1 config; the "
            f"showcase is the only place we lock in this feature."
        )

    cfg = load_config(path)
    m_new = _resolve_material(cfg)
    m_old = create_material(preset='glass_borden',
                            l0=0.25, energy_split='spectral')
    assert m_new == m_old
