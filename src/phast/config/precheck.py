#!/usr/bin/env python3
"""
Pre-simulation diagnostic calculator for explicit dynamics phase-field fracture.

Computes wave speeds, CFL timestep, mesh resolution checks, mass scaling
advisory, time/cost estimates, and energy scales --- all BEFORE running
the solver.  Designed for planning mesh refinement, choosing dt_safety,
and sanity-checking YAML configurations.

Usage
-----
    # From a material preset:
    python -m phast precheck \
        --preset glass_borden --h_min 0.5 --l0 0.5 --t_total 8e-5

    # From explicit material parameters:
    python -m phast precheck \
        --E 32000 --nu 0.2 --rho 2.45e-9 --Gc 3e-3 --l0 0.5 \
        --h_min 0.5 --t_total 8e-5

    # From a YAML config:
    python -m phast precheck \
        --config configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml

    # Self-test with reference values:
    python -m phast precheck --test

References
----------
  - Miehe et al. (2010), CMAME 199:2765 (AT2, mesh size h <= l0/2)
  - Bourdin et al. (2000), JMPS 48:797 (gamma-convergence, h << l0)
  - Borden et al. (2012), CMAME 217:77 (dynamic branching, AT2)
  - Viktorov (1967), Rayleigh and Lamb Waves (c_R approximation)
  - Fineberg & Marder (1999), Phys. Rep. 313:1 (branching at ~0.6*c_R)
  - Amor et al. (2009), JMPS 57:1209 (vol-dev split)
  - Wu (2017), JMPS 103:72 (AT1 compact support)
"""

import math
import sys
import os
from dataclasses import dataclass
from typing import Optional


# ======================================================================
# Core diagnostic dataclass
# ======================================================================

@dataclass
class DiagnosticResult:
    """All pre-simulation diagnostic quantities."""

    # Material identifiers
    material_name: str = ""
    pf_model: str = "AT2"
    plane_assumption: str = "plane strain"

    # Elastic constants (MPa)
    E: float = 0.0
    nu: float = 0.0
    rho: float = 0.0      # tonne/mm^3
    lam: float = 0.0      # First Lame (MPa)
    mu: float = 0.0       # Shear modulus (MPa)
    kappa: float = 0.0    # Bulk modulus (MPa)

    # Fracture parameters
    Gc: float = 0.0       # N/mm
    l0: float = 0.0       # mm

    # Wave speeds (mm/s)
    c_p_plane_strain: float = 0.0
    c_p_plane_stress: float = 0.0
    c_p: float = 0.0      # active (based on plane_assumption)
    c_s: float = 0.0
    c_R: float = 0.0
    c_branch: float = 0.0   # 0.6 * c_R

    # CFL / time stepping
    h_min: float = 0.0    # mm (incircle diameter)
    dt_CFL: float = 0.0   # s
    dt_safety: float = 0.8
    dt: float = 0.0       # s (dt_CFL * dt_safety)
    t_total: float = 0.0  # s
    n_steps: int = 0

    # Phase-field resolution
    l0_over_h: float = 0.0
    resolution_rating: str = ""
    elements_across_band_AT2: float = 0.0
    elements_across_band_AT1: float = 0.0

    # Subcycling
    c_p_over_c_R: float = 0.0
    max_safe_damage_every: int = 1

    # Mesh (optional)
    n_nodes: int = 0
    n_elems: int = 0

    # Mass scaling (optional)
    dt_target: float = 0.0
    mass_scale_factor: float = 1.0


# ======================================================================
# Compute functions
# ======================================================================

def compute_wave_speeds(E: float, nu: float, rho: float):
    """Compute all wave speeds (mm/s) for both plane assumptions.

    Parameters
    ----------
    E : float   Young's modulus (MPa)
    nu : float  Poisson's ratio
    rho : float Density (tonne/mm^3)

    Returns
    -------
    dict with keys: c_p_pe, c_p_ps, c_s, c_R, c_branch, mu, lam_pe, lam_ps, kappa
    """
    mu = E / (2.0 * (1.0 + nu))
    kappa = E / (3.0 * (1.0 - 2.0 * nu))

    # First Lame parameter
    lam_pe = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))   # plane strain
    lam_ps = E * nu / (1.0 - nu**2)                       # plane stress (effective 2D)

    # P-wave speeds
    c_p_pe = math.sqrt((lam_pe + 2.0 * mu) / rho)  # plane strain
    c_p_ps = math.sqrt((lam_ps + 2.0 * mu) / rho)  # plane stress
    # Equivalent closed forms:
    #   plane strain: c_p = sqrt(E*(1-nu) / (rho*(1+nu)*(1-2*nu)))
    #   plane stress: c_p = sqrt(E / (rho*(1-nu^2)))

    # S-wave (shear) -- same for plane strain and plane stress
    c_s = math.sqrt(mu / rho)

    # Rayleigh wave -- Viktorov (1967) approximation, <0.5% error
    c_R = c_s * (0.862 + 1.14 * nu) / (1.0 + nu)

    # Empirical branching onset (Fineberg & Marder 1999)
    c_branch = 0.6 * c_R

    return {
        'c_p_pe': c_p_pe, 'c_p_ps': c_p_ps,
        'c_s': c_s, 'c_R': c_R, 'c_branch': c_branch,
        'mu': mu, 'lam_pe': lam_pe, 'lam_ps': lam_ps, 'kappa': kappa,
    }


def compute_CFL(h_min: float, c_p: float, safety: float = 0.8):
    """CFL condition for explicit dynamics.

    The standard CFL bound for lumped-mass central-difference with
    linear triangles is:
        dt <= h_min / c_p

    where h_min is the minimum incircle diameter (= 4*Area/perimeter),
    which is the standard choice in Abaqus, LS-DYNA, and Akantu.

    The safety factor (0.5-0.9 typical) absorbs 2D geometry effects
    and the approximation of using the minimum element size globally.

    Parameters
    ----------
    h_min : float   Minimum element incircle diameter (mm)
    c_p : float     P-wave speed (mm/s)
    safety : float  Safety factor (default 0.8)

    Returns
    -------
    dt_CFL, dt : float (s)
    """
    dt_CFL = h_min / c_p
    dt = dt_CFL * safety
    return dt_CFL, dt


def compute_resolution(l0: float, h_min: float, pf_model: str = 'AT2'):
    """Phase-field mesh resolution diagnostics.

    AT2 (Ambrosio-Tortorelli 2nd order):
        - Damage profile: d(x) = exp(-|x|/l0), support ~ 4*l0 (to 2% level)
        - Minimum: h <= l0/2 (Miehe 2010, Bourdin 2000)
        - Recommended: h <= l0/4 for quantitative convergence studies
        - Elements across full band (8*l0 width): 8*l0 / h

    AT1 (Pham et al. 2011, linear damage):
        - Damage profile has COMPACT SUPPORT at distance l0 from crack center
        - Much steeper gradient at the band boundary than AT2
        - Minimum: h <= l0/2
        - Recommended: h <= l0/4 (fewer elements across band, so each matters more)
        - Elements across full band (2*l0 width): 2*l0 / h

    Parameters
    ----------
    l0 : float     Regularization length (mm)
    h_min : float  Minimum element size (mm)
    pf_model : str 'AT2' or 'AT1'

    Returns
    -------
    dict with l0_over_h, rating, n_elems_AT2, n_elems_AT1
    """
    l0_over_h = l0 / h_min if h_min > 0 else float('inf')

    if l0_over_h < 2.0:
        rating = "UNDER-RESOLVED (l0/h < 2) -- results unreliable"
    elif l0_over_h < 4.0:
        rating = "ADEQUATE (2 <= l0/h < 4) -- sufficient for engineering accuracy"
    else:
        rating = "WELL-RESOLVED (l0/h >= 4) -- suitable for convergence studies"

    # Elements across the full damage band
    n_AT2 = 8.0 * l0 / h_min if h_min > 0 else float('inf')  # band ~ 8*l0
    n_AT1 = 2.0 * l0 / h_min if h_min > 0 else float('inf')  # band ~ 2*l0

    return {
        'l0_over_h': l0_over_h,
        'rating': rating,
        'n_elems_AT2': n_AT2,
        'n_elems_AT1': n_AT1,
    }


def compute_mass_scaling(dt_CFL: float, dt_target: float):
    """Selective mass scaling factor.

    Increases element mass to raise the stable timestep:
        m_scaled = m * (dt_target / dt_CFL)^2

    Only safe in the quasi-static regime where inertial effects are
    negligible. Abaqus uses this in its 'fixed mass scaling' option.
    COMSOL applies it via the 'mass scaling factor' in explicit dynamics.

    Parameters
    ----------
    dt_CFL : float   Current CFL timestep (s)
    dt_target : float  Desired timestep (s)

    Returns
    -------
    mass_scale_factor : float (>= 1.0)
    """
    if dt_target <= dt_CFL:
        return 1.0
    return (dt_target / dt_CFL) ** 2


def compute_energy_scales(Gc: float, l0: float, E: float, rho: float,
                          crack_length: float = 0.0,
                          domain_area: float = 0.0,
                          typical_stress: float = 0.0,
                          typical_velocity: float = 0.0):
    """Characteristic energy scales for sanity checks.

    Parameters
    ----------
    Gc : float          Fracture toughness (N/mm)
    l0 : float          Regularization length (mm)
    E : float           Young's modulus (MPa)
    rho : float         Density (tonne/mm^3)
    crack_length : float  Expected crack length (mm)
    domain_area : float   Domain area (mm^2)
    typical_stress : float  Characteristic stress (MPa)
    typical_velocity : float  Characteristic velocity (mm/s)

    Returns
    -------
    dict with Griffith energy, elastic energy, kinetic energy, AT2 critical stress
    """
    result = {}

    # Griffith energy: energy to create a crack of given length
    if crack_length > 0:
        result['E_griffith'] = Gc * crack_length  # N*mm

    # AT2 critical stress (1D homogeneous nucleation):
    # sigma_c = (27/256 * E * Gc / l0)^(1/2)  for AT2
    # sigma_c = (3/8 * E * Gc / l0)^(1/2)     for AT1  [Pham et al. 2011]
    result['sigma_c_AT2'] = math.sqrt(27.0 / 256.0 * E * Gc / l0)
    result['sigma_c_AT1'] = math.sqrt(3.0 / 8.0 * E * Gc / l0)

    # Elastic energy density * domain area
    if typical_stress > 0 and domain_area > 0:
        eps = typical_stress / E
        psi = 0.5 * typical_stress * eps  # MPa (= N/mm^2)
        result['E_elastic'] = psi * domain_area  # N*mm

    # Kinetic energy density * domain area
    if typical_velocity > 0 and domain_area > 0:
        result['E_kinetic'] = 0.5 * rho * typical_velocity**2 * domain_area  # N*mm

    # Internal length energy scale: Gc * l0 (regularization stiffness)
    result['Gc_times_l0'] = Gc * l0
    result['Gc_over_l0'] = Gc / l0

    return result


def compute_subcycling_ratio(c_p: float, c_R: float):
    """Validate damage subcycling frequency.

    The damage wave propagates at ~0.6*c_R (Rayleigh speed limit for
    crack tips). The CFL timestep is based on c_p. The ratio c_p/c_R
    determines how many mechanical steps can be taken per damage solve
    without missing damage front propagation.

    max_safe_damage_every = floor(c_p / (0.6 * c_R))

    Parameters
    ----------
    c_p : float  P-wave speed (mm/s)
    c_R : float  Rayleigh wave speed (mm/s)

    Returns
    -------
    ratio, max_safe_damage_every : float, int
    """
    ratio = c_p / c_R
    # Damage propagates at ~0.6*c_R; CFL resolves c_p
    # So each CFL step advances the damage front by c_p/(0.6*c_R) * h
    # relative to one element size
    max_safe = int(math.floor(c_p / (0.6 * c_R)))
    return ratio, max(1, max_safe)


# ======================================================================
# Full diagnostic report
# ======================================================================

def run_diagnostics(
    E: float, nu: float, rho: float, Gc: float, l0: float,
    h_min: float = 0.0,
    t_total: float = 0.0,
    dt_safety: float = 0.8,
    pf_model: str = 'AT2',
    plane_stress: bool = False,
    n_nodes: int = 0,
    n_elems: int = 0,
    dt_target: float = 0.0,
    crack_length: float = 0.0,
    domain_area: float = 0.0,
    typical_stress: float = 0.0,
    typical_velocity: float = 0.0,
    material_name: str = '',
) -> DiagnosticResult:
    """Run all pre-simulation diagnostics.

    Can be called without a mesh (just h_min) for pre-meshing planning,
    or with full mesh statistics for a complete report.

    Returns
    -------
    DiagnosticResult with all computed quantities.
    """
    r = DiagnosticResult()
    r.material_name = material_name
    r.pf_model = pf_model
    r.plane_assumption = "plane stress" if plane_stress else "plane strain"
    r.E = E
    r.nu = nu
    r.rho = rho
    r.Gc = Gc
    r.l0 = l0
    r.h_min = h_min
    r.dt_safety = dt_safety
    r.t_total = t_total
    r.n_nodes = n_nodes
    r.n_elems = n_elems
    r.dt_target = dt_target

    # --- Wave speeds ---
    ws = compute_wave_speeds(E, nu, rho)
    r.lam = ws['lam_pe'] if not plane_stress else ws['lam_ps']
    r.mu = ws['mu']
    r.kappa = ws['kappa']
    r.c_p_plane_strain = ws['c_p_pe']
    r.c_p_plane_stress = ws['c_p_ps']
    r.c_p = ws['c_p_ps'] if plane_stress else ws['c_p_pe']
    r.c_s = ws['c_s']
    r.c_R = ws['c_R']
    r.c_branch = ws['c_branch']

    # --- CFL ---
    if h_min > 0:
        r.dt_CFL, r.dt = compute_CFL(h_min, r.c_p, dt_safety)
        if t_total > 0:
            r.n_steps = int(math.ceil(t_total / r.dt))

    # --- Phase-field resolution ---
    if h_min > 0 and l0 > 0:
        res = compute_resolution(l0, h_min, pf_model)
        r.l0_over_h = res['l0_over_h']
        r.resolution_rating = res['rating']
        r.elements_across_band_AT2 = res['n_elems_AT2']
        r.elements_across_band_AT1 = res['n_elems_AT1']

    # --- Subcycling ---
    ratio, max_de = compute_subcycling_ratio(r.c_p, r.c_R)
    r.c_p_over_c_R = ratio
    r.max_safe_damage_every = max_de

    # --- Mass scaling ---
    if dt_target > 0 and r.dt_CFL > 0:
        r.mass_scale_factor = compute_mass_scaling(r.dt_CFL, dt_target)

    return r


def format_report(r: DiagnosticResult,
                  crack_length: float = 0.0,
                  domain_area: float = 0.0,
                  typical_stress: float = 0.0,
                  typical_velocity: float = 0.0) -> str:
    """Format a DiagnosticResult as a human-readable plain-text report."""
    lines = []

    def add(s=''):
        lines.append(s)

    def sec(title):
        add()
        add(title)
        add("-" * 60)

    # Header
    add("=" * 68)
    add("  PRE-SIMULATION DIAGNOSTIC REPORT")
    if r.material_name:
        add(f"  Material: {r.material_name}")
    add(f"  Model: {r.pf_model}, {r.plane_assumption}")
    add("=" * 68)

    # --- 1. Material ---
    sec("1. MATERIAL PARAMETERS")
    add(f"  Young's modulus    E     = {r.E:,.1f} MPa  ({r.E/1000:.1f} GPa)")
    add(f"  Poisson's ratio    nu    = {r.nu}")
    add(f"  Density            rho   = {r.rho:.3e} tonne/mm^3  ({r.rho*1e12:.1f} kg/m^3)")
    add(f"  Fracture toughness Gc    = {r.Gc} N/mm  ({r.Gc*1000:.1f} J/m^2)")
    add(f"  Regularization     l0    = {r.l0} mm")
    add()
    add("  Derived elastic constants:")
    add(f"    First Lame (lambda) = {r.lam:,.2f} MPa")
    add(f"    Shear modulus (mu)  = {r.mu:,.2f} MPa")
    add(f"    Bulk modulus (K)    = {r.kappa:,.2f} MPa")
    add(f"    Gc / l0             = {r.Gc/r.l0 if r.l0 > 0 else 0:.4f}")
    add(f"    Gc * l0             = {r.Gc*r.l0:.6f}")

    # --- 2. Wave Speeds ---
    sec("2. WAVE SPEEDS")
    add()
    add("  P-wave (dilatational):")
    add(f"    Plane strain:  c_p = sqrt((lam+2*mu)/rho)     = {r.c_p_plane_strain:,.0f} mm/s  ({r.c_p_plane_strain/1e3:,.1f} m/s)")
    add(f"    Plane stress:  c_p = sqrt(E/(rho*(1-nu^2)))   = {r.c_p_plane_stress:,.0f} mm/s  ({r.c_p_plane_stress/1e3:,.1f} m/s)")
    add(f"    Active ({r.plane_assumption}):  c_p = {r.c_p:,.0f} mm/s")
    add()
    add("  S-wave (shear):")
    add(f"    c_s = sqrt(mu/rho)                            = {r.c_s:,.0f} mm/s  ({r.c_s/1e3:,.1f} m/s)")
    add()
    add("  Rayleigh wave (Viktorov 1967 approx, <0.5% error):")
    add(f"    c_R = c_s*(0.862+1.14*nu)/(1+nu)             = {r.c_R:,.0f} mm/s  ({r.c_R/1e3:,.1f} m/s)")
    add()
    add("  Crack tip velocity limit:")
    add(f"    Theoretical max:     c_R                      = {r.c_R:,.0f} mm/s")
    add(f"    Branching onset:     0.6*c_R                  = {r.c_branch:,.0f} mm/s  ({r.c_branch/1e3:,.1f} m/s)")
    add()
    add("  Speed ratios:")
    add(f"    c_p / c_s = {r.c_p/r.c_s if r.c_s > 0 else 0:.3f}")
    add(f"    c_p / c_R = {r.c_p_over_c_R:.3f}")
    add(f"    c_s / c_R = {r.c_s/r.c_R if r.c_R > 0 else 0:.3f}")

    # --- 3. CFL Condition ---
    sec("3. CFL CONDITION")
    if r.h_min > 0:
        add()
        add("  Formula: dt_CFL = h_min / c_p")
        add(f"    h_min (incircle diameter) = {r.h_min:.6f} mm")
        add(f"    c_p                       = {r.c_p:,.0f} mm/s")
        add(f"    dt_CFL                    = {r.dt_CFL:.6e} s")
        add()
        add(f"  With safety factor {r.dt_safety}:")
        add(f"    dt = {r.dt_safety} * dt_CFL     = {r.dt:.6e} s")
        add()
        add("  Safety factor guidance:")
        add("    0.5 -- conservative, good for debugging")
        add("    0.8 -- standard (used in phast, Borden 2012)")
        add("    0.9 -- aggressive, test carefully")
        add("    1.0 -- theoretical limit (Akantu default, no safety margin)")
        if r.t_total > 0:
            add()
            add(f"  Time stepping:")
            add(f"    t_total = {r.t_total:.6e} s")
            add(f"    n_steps = ceil(t_total / dt) = {r.n_steps:,}")
    else:
        add()
        add("  (Provide --h_min to compute CFL timestep)")

    # --- 4. Phase-Field Resolution ---
    sec("4. PHASE-FIELD MESH RESOLUTION")
    if r.h_min > 0 and r.l0 > 0:
        add()
        add(f"  l0 / h_min = {r.l0} / {r.h_min:.6f} = {r.l0_over_h:.2f}")
        add(f"  Rating: {r.resolution_rating}")
        add()
        add("  AT2 (Ambrosio-Tortorelli quadratic):")
        add("    Damage profile: d(x) = exp(-|x|/l0), support ~ 4*l0 each side")
        add(f"    Full band width: ~8*l0 = {8*r.l0:.2f} mm")
        add(f"    Elements across band: {r.elements_across_band_AT2:.1f}")
        add("    Minimum requirement: h <= l0/2 (Miehe 2010)")
        add("    Recommended:         h <= l0/4 (convergence studies)")
        add()
        add("  AT1 (linear damage, compact support):")
        add("    Damage profile: compact support at distance l0 from crack center")
        add(f"    Full band width: 2*l0 = {2*r.l0:.2f} mm")
        add(f"    Elements across band: {r.elements_across_band_AT1:.1f}")
        add("    Minimum requirement: h <= l0/2")
        add("    Recommended:         h <= l0/4 (steep gradient at boundary)")
        add()
        add("  Resolution quality bands:")
        add("    l0/h < 2:  UNDER-RESOLVED -- results unreliable")
        add("    l0/h = 2:  minimum acceptable (Miehe 2010)")
        add("    l0/h = 4:  recommended for quantitative accuracy")
        add("    l0/h > 6:  well-resolved, diminishing returns")
        add()
        # Gamma-convergence advisory
        if r.l0_over_h < 3.0:
            add("  ** Advisory: consider gamma_correction=True in material config")
            add("     to compensate for over-prediction of Gc on coarse meshes")
            add("     (Bourdin et al. 2000). Factor: 1 / (1 + h/(c_w*l0)).")
    else:
        add()
        add("  (Provide --h_min and --l0 to compute resolution metrics)")

    # --- 5. Mass Matrix ---
    sec("5. MASS MATRIX")
    add()
    add("  Lumped mass (row-sum) for T3 elements:")
    add("    M_i = rho * sum_e (A_e / 3)  for node i in element e")
    add("    Vector mass: M_vec = [m0,m0, m1,m1, ...] (2 DOFs/node)")
    add()
    add("  Properties:")
    add("    + Diagonal -> O(N) inversion in explicit dynamics")
    add("    + Preserves total mass exactly (row-sum lumping)")
    add("    + No iterative solve needed for acceleration")
    add("    - Introduces dispersion error in wave propagation")
    add("    - Phase velocity ~2% too low for 6 elements/wavelength")
    add("    - Acceptable for explicit dynamics (always used)")
    add()
    add("  Consistent mass (for damage field in CG solver):")
    add("    M_e = rho*A_e/12 * [2,1,1; 1,2,1; 1,1,2]")
    add("    Used by the AT2 damage CG solver (accuracy-critical)")
    if r.n_nodes > 0:
        add()
        add(f"  Mesh statistics:")
        add(f"    Nodes: {r.n_nodes:,}")
        if r.n_elems > 0:
            add(f"    Elements: {r.n_elems:,}")

    # --- 6. Mass Scaling ---
    sec("6. MASS SCALING ADVISORY")
    if r.dt_target > 0 and r.dt_CFL > 0:
        add()
        add(f"  Target timestep:  dt_target = {r.dt_target:.6e} s")
        add(f"  CFL timestep:     dt_CFL    = {r.dt_CFL:.6e} s")
        add(f"  Required factor:  (dt_target/dt_CFL)^2 = {r.mass_scale_factor:.2f}")
        add()
        if r.mass_scale_factor > 100:
            add("  ** WARNING: mass scale factor > 100 -- severe inertia distortion!")
            add("     Consider mesh refinement instead of mass scaling.")
        elif r.mass_scale_factor > 10:
            add("  ** CAUTION: mass scale factor > 10 -- verify quasi-static regime")
        elif r.mass_scale_factor > 1.0:
            add("  Mass scaling is moderate -- safe for quasi-static loading")
        else:
            add("  No mass scaling needed (dt_target <= dt_CFL)")
        add()
        add("  Formula: m_scaled = m * (dt_target / dt_CFL)^2")
        add("  Only safe when:")
        add("    - Loading is quasi-static (kinetic energy << strain energy)")
        add("    - No wave propagation physics to resolve")
        add("    - Abaqus: *FIXED MASS SCALING, DT=dt_target")
        add("    - COMSOL: explicit dynamics 'mass scaling factor' parameter")
    else:
        add()
        add("  (Provide --dt_target to compute mass scaling requirements)")
        add()
        add("  Mass scaling increases element mass to raise the stable timestep:")
        add("    m_scaled = m * (dt_target / dt_CFL)^2")
        add("  Only valid for quasi-static problems (no inertial effects).")

    # --- 7. Subcycling ---
    sec("7. DAMAGE SUBCYCLING")
    add()
    add(f"  c_p / c_R = {r.c_p_over_c_R:.3f}")
    add(f"  c_p / (0.6*c_R) = {r.c_p/(0.6*r.c_R) if r.c_R > 0 else 0:.3f}")
    add(f"  Max safe damage_every = floor(c_p / (0.6*c_R)) = {r.max_safe_damage_every}")
    add()
    add("  The damage front propagates at ~0.6*c_R (crack tip speed).")
    add("  The CFL timestep resolves c_p. So each CFL step, the damage")
    add("  front advances by h * c_p/(0.6*c_R) relative to the mesh.")
    add("  Solving damage every N steps is safe if N <= c_p/(0.6*c_R).")
    add()
    add(f"  For nu={r.nu}:")
    add(f"    damage_every=1:  conservative (every step)")
    add(f"    damage_every=3:  {'SAFE' if r.max_safe_damage_every >= 3 else 'UNSAFE -- reduce to ' + str(r.max_safe_damage_every)} (throughput setting after validation)")
    add(f"    damage_every=5:  {'SAFE' if r.max_safe_damage_every >= 5 else 'UNSAFE -- reduce to ' + str(r.max_safe_damage_every)}")

    # --- 8. Time/Cost Estimates ---
    sec("8. TIME / COST ESTIMATES")
    if r.n_steps > 0:
        add()
        add(f"  Total steps:  {r.n_steps:,}")
        add()
        add("  Estimated cost per step (explicit dynamics):")
        add("    - Internal force (scatter-based): O(E) ~fast")
        add("    - Predictor/corrector: O(N)")
        add("    - Damage CG (every damage_every steps): O(N * CG_iters)")
        add()
        if r.n_nodes > 0:
            # Rough wall time estimates (CPU-based, single thread)
            us_per_step_cpu = r.n_nodes * 0.001  # ~1 us per node per step (empirical)
            us_per_step_gpu = r.n_nodes * 0.0001  # ~0.1 us per node per step
            wall_cpu = r.n_steps * us_per_step_cpu / 1e6  # seconds
            wall_gpu = r.n_steps * us_per_step_gpu / 1e6
            add(f"  Rough wall time estimates ({r.n_nodes:,} nodes, {r.n_steps:,} steps):")
            add(f"    CPU (1 core):  ~{_fmt_time(wall_cpu)}")
            add(f"    GPU (CUDA):    ~{_fmt_time(wall_gpu)}")
            add("    (Very rough -- actual time depends on CG iterations, IO, etc.)")
    else:
        add()
        add("  (Provide --t_total to compute step count and wall time estimates)")

    # --- 9. Energy Scales ---
    sec("9. ENERGY SCALES")
    energies = compute_energy_scales(
        r.Gc, r.l0, r.E, r.rho,
        crack_length=crack_length,
        domain_area=domain_area,
        typical_stress=typical_stress,
        typical_velocity=typical_velocity,
    )
    add()
    add("  Fracture parameter scales:")
    add(f"    Gc * l0  = {energies['Gc_times_l0']:.6f}  (stiffness coefficient)")
    add(f"    Gc / l0  = {energies['Gc_over_l0']:.4f}   (reaction coefficient)")
    add(f"    Ratio Gc/l0 : Gc*l0 = {energies['Gc_over_l0']/energies['Gc_times_l0']:.0f}:1")
    add()
    add("  Critical stress for homogeneous nucleation (1D):")
    add(f"    AT2: sigma_c = sqrt(27*E*Gc/(256*l0)) = {energies['sigma_c_AT2']:.2f} MPa")
    add(f"    AT1: sigma_c = sqrt(3*E*Gc/(8*l0))    = {energies['sigma_c_AT1']:.2f} MPa")
    if crack_length > 0:
        add()
        add(f"  Griffith energy (crack length = {crack_length:.2f} mm):")
        add(f"    E_griffith = Gc * L = {energies['E_griffith']:.6f} N*mm")
    if 'E_elastic' in energies:
        add()
        add(f"  Elastic energy (sigma={typical_stress:.1f} MPa, area={domain_area:.1f} mm^2):")
        add(f"    E_elastic = 0.5*sigma*eps*A = {energies['E_elastic']:.6f} N*mm")
    if 'E_kinetic' in energies:
        add()
        add(f"  Kinetic energy (v={typical_velocity:.1f} mm/s, area={domain_area:.1f} mm^2):")
        add(f"    E_kinetic = 0.5*rho*v^2*A = {energies['E_kinetic']:.6e} N*mm")

    add()
    add("=" * 68)
    add("  END OF DIAGNOSTIC REPORT")
    add("=" * 68)

    return '\n'.join(lines)


def _fmt_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.1f} hr"


# ======================================================================
# CLI
# ======================================================================

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m phast precheck",
        description="Pre-simulation diagnostic calculator for explicit dynamics phase-field fracture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Glass (Borden 2012):
  %(prog)s --preset glass_borden --h_min 0.5 --t_total 8e-5

  # Steel (Kalthoff-Winkler):
  %(prog)s --preset maraging_steel_kw --h_min 0.25 --t_total 9e-5

  # Explicit parameters:
  %(prog)s --E 32000 --nu 0.2 --rho 2.45e-9 --Gc 3e-3 --l0 0.5 --h_min 0.5

  # From YAML config:
  %(prog)s --config configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml

  # Self-test:
  %(prog)s --test
""")
    # Material source (mutually-exclusive-ish)
    p.add_argument('--preset', type=str, default=None,
                   help='Material preset name (e.g. glass_borden, maraging_steel_kw)')
    p.add_argument('--config', type=str, default=None,
                   help='Path to YAML config file')

    # Explicit material parameters (override preset)
    p.add_argument('--E', type=float, default=None, help='Young modulus (MPa)')
    p.add_argument('--nu', type=float, default=None, help='Poisson ratio')
    p.add_argument('--rho', type=float, default=None, help='Density (tonne/mm^3)')
    p.add_argument('--Gc', type=float, default=None, help='Fracture toughness (N/mm)')
    p.add_argument('--l0', type=float, default=None, help='Regularization length (mm)')
    p.add_argument('--pf_model', type=str, default='AT2', choices=['AT1', 'AT2'])
    p.add_argument('--plane_stress', action='store_true')

    # Mesh / geometry
    p.add_argument('--h_min', type=float, default=0.0,
                   help='Minimum element incircle diameter (mm)')
    p.add_argument('--n_nodes', type=int, default=0)
    p.add_argument('--n_elems', type=int, default=0)

    # Time stepping
    p.add_argument('--t_total', type=float, default=0.0, help='Total simulation time (s)')
    p.add_argument('--dt_safety', type=float, default=0.8, help='CFL safety factor')
    p.add_argument('--dt_target', type=float, default=0.0,
                   help='Target timestep for mass scaling calculation (s)')

    # Energy scales
    p.add_argument('--crack_length', type=float, default=0.0,
                   help='Expected crack length (mm)')
    p.add_argument('--domain_area', type=float, default=0.0,
                   help='Domain area (mm^2)')
    p.add_argument('--typical_stress', type=float, default=0.0,
                   help='Characteristic stress (MPa)')
    p.add_argument('--typical_velocity', type=float, default=0.0,
                   help='Characteristic velocity (mm/s)')

    # Output
    p.add_argument('--output', type=str, default=None,
                   help='Save report to file (default: print to stdout)')

    # Test
    p.add_argument('--test', action='store_true', help='Run self-test with reference values')

    # Compare all presets
    p.add_argument('--compare', action='store_true',
                   help='Print wave speed table for all material presets')

    return p.parse_args()


def _run_from_preset(preset_name: str, overrides: dict):
    """Load material from preset, returning (E, nu, rho, Gc, l0, pf_model, plane_stress, name)."""
    # Import from the package
    try:
        from phast.material import create_material
    except ImportError:
        # Running from within the package directory
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.dirname(parent))
        from phast.material import create_material

    mat = create_material(preset_name, **overrides)
    return {
        'E': mat.E, 'nu': mat.nu, 'rho': mat.rho,
        'Gc': mat.Gc, 'l0': mat.l0,
        'pf_model': mat.pf_model,
        'plane_stress': mat.plane_stress,
        'material_name': preset_name,
    }


def _run_from_config(yaml_path: str):
    """Load material + geometry from YAML config.

    Also builds the mesh via ``resolve_config`` so ``h_min``, ``n_nodes``,
    and ``n_elems`` are measured from the actual mesh. Falls back to
    material-only if mesh construction fails.
    """
    try:
        from phast.config import (
            load_config, resolve_config, _resolve_material)
    except ImportError:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.dirname(parent))
        from phast.config import (
            load_config, resolve_config, _resolve_material)

    cfg = load_config(yaml_path)
    mat = _resolve_material(cfg)

    params = {
        'E': mat.E, 'nu': mat.nu, 'rho': mat.rho,
        'Gc': mat.Gc, 'l0': mat.l0,
        'pf_model': mat.pf_model,
        'plane_stress': mat.plane_stress,
        'material_name': f"{cfg.material.preset} ({cfg.name})",
        't_total': cfg.loading.t_total if cfg.loading else 0.0,
        'dt_safety': cfg.solver.dt_safety if cfg.solver else 0.8,
    }

    try:
        objs = resolve_config(cfg)
        mesh = objs['mesh']
        params['h_min'] = mesh.elem_h.min().item()
        params['n_nodes'] = mesh.n_nodes
        params['n_elems'] = mesh.n_elems
    except Exception as e:
        print(f"[precheck] Could not build mesh from YAML ({e}); "
              f"pass --h_min manually for CFL/resolution metrics.")

    return params


def _run_self_test():
    """Verify wave speed computations against reference values."""
    print("Running self-test...")
    print()
    passed = True

    tests = [
        {
            'name': 'Glass (Borden 2012)',
            'E': 32000.0, 'nu': 0.2, 'rho': 2.45e-9,
            # Expected (mm/s)
            'c_p_pe': 3_809_444, 'c_s': 2_332_810, 'c_R': 2_119_164,
            'tol': 5000,  # tolerance in mm/s (~1 m/s)
        },
        {
            'name': 'Maraging Steel (Kalthoff-Winkler)',
            'E': 190000.0, 'nu': 0.3, 'rho': 8.0e-9,
            'c_p_pe': 5_654_000, 'c_s': 3_022_000, 'c_R': 2_799_000,
            'tol': 5000,
        },
    ]

    for t in tests:
        ws = compute_wave_speeds(t['E'], t['nu'], t['rho'])
        print(f"  {t['name']}:")

        for key, label in [('c_p_pe', 'c_p'), ('c_s', 'c_s'), ('c_R', 'c_R')]:
            computed = ws[key]
            expected = t[key]
            err = abs(computed - expected)
            ok = err < t['tol']
            status = "PASS" if ok else "FAIL"
            if not ok:
                passed = False
            print(f"    {label}: {computed:,.0f} mm/s  (expected ~{expected:,}, err={err:.0f})  [{status}]")
        print()

    # Check subcycling ratio for known cases
    print("  Subcycling ratios:")
    for name, nu_val in [("Glass (nu=0.2)", 0.2), ("Steel (nu=0.3)", 0.3)]:
        # Use glass params for computation
        E = 32000 if "Glass" in name else 190000
        rho = 2.45e-9 if "Glass" in name else 8.0e-9
        ws = compute_wave_speeds(E, nu_val, rho)
        ratio, max_de = compute_subcycling_ratio(ws['c_p_pe'], ws['c_R'])
        print(f"    {name}: c_p/c_R = {ratio:.3f}, max_damage_every = {max_de}")
    print()

    if passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    return passed


def _run_compare():
    """Print a comparison table for all material presets."""
    try:
        from phast.material import create_material
    except ImportError:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.dirname(parent))
        from phast.material import create_material

    presets = [
        'glass_borden', 'maraging_steel_kw', 'miehe_tension', 'miehe_shear',
        'three_point_bending', 'l_shaped_glass', 'l_shaped_concrete',
        'alumina_kumar', 'brittle_ceramic', 'pmma', 'pmma_bleyer',
        'soda_lime_glass',
    ]

    print(f"{'Preset':<22} {'E(GPa)':>8} {'nu':>5} {'rho(kg/m3)':>10} "
          f"{'c_p(m/s)':>9} {'c_s(m/s)':>9} {'c_R(m/s)':>9} "
          f"{'0.6cR(m/s)':>10} {'c_p/c_R':>7} {'l0(mm)':>7} {'Gc(N/mm)':>8}")
    print("-" * 130)

    for name in presets:
        try:
            mat = create_material(name)
            ws = compute_wave_speeds(mat.E, mat.nu, mat.rho)
            c_p = ws['c_p_ps'] if mat.plane_stress else ws['c_p_pe']
            ratio = c_p / ws['c_R']
            print(f"{name:<22} {mat.E/1000:>8.1f} {mat.nu:>5.2f} {mat.rho*1e12:>10.1f} "
                  f"{c_p/1e3:>9.0f} {ws['c_s']/1e3:>9.0f} {ws['c_R']/1e3:>9.0f} "
                  f"{0.6*ws['c_R']/1e3:>10.0f} {ratio:>7.3f} {mat.l0:>7.3f} {mat.Gc:>8.4f}")
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")


def main():
    args = _parse_args()

    if args.test:
        ok = _run_self_test()
        sys.exit(0 if ok else 1)

    if args.compare:
        _run_compare()
        sys.exit(0)

    # Resolve material parameters
    params = {}
    if args.config:
        params = _run_from_config(args.config)
    elif args.preset:
        overrides = {}
        if args.l0 is not None:
            overrides['l0'] = args.l0
        params = _run_from_preset(args.preset, overrides)
    elif args.E is not None:
        # All explicit
        params = {
            'E': args.E, 'nu': args.nu or 0.3, 'rho': args.rho or 7.8e-9,
            'Gc': args.Gc or 2.7, 'l0': args.l0 or 0.01,
            'pf_model': args.pf_model,
            'plane_stress': args.plane_stress,
            'material_name': 'custom',
        }
    else:
        print("Error: provide --preset, --config, or --E (with other params)")
        sys.exit(1)

    # CLI overrides
    if args.E is not None and args.preset:
        params['E'] = args.E
    if args.nu is not None and args.preset:
        params['nu'] = args.nu
    if args.rho is not None and args.preset:
        params['rho'] = args.rho
    if args.Gc is not None and args.preset:
        params['Gc'] = args.Gc
    if args.l0 is not None and 'l0' not in params:
        params['l0'] = args.l0
    if args.pf_model != 'AT2':
        params['pf_model'] = args.pf_model
    if args.plane_stress:
        params['plane_stress'] = True

    # Merge geometry / time from config or CLI (CLI wins)
    h_min = args.h_min or params.get('h_min', 0.0)
    n_nodes = args.n_nodes or params.get('n_nodes', 0)
    n_elems = args.n_elems or params.get('n_elems', 0)
    t_total = args.t_total or params.get('t_total', 0.0)
    dt_safety = args.dt_safety if args.dt_safety != 0.8 else params.get('dt_safety', 0.8)

    result = run_diagnostics(
        E=params['E'],
        nu=params['nu'],
        rho=params['rho'],
        Gc=params['Gc'],
        l0=params['l0'],
        h_min=h_min,
        t_total=t_total,
        dt_safety=dt_safety,
        pf_model=params.get('pf_model', 'AT2'),
        plane_stress=params.get('plane_stress', False),
        n_nodes=n_nodes,
        n_elems=n_elems,
        dt_target=args.dt_target,
        material_name=params.get('material_name', ''),
    )

    report = format_report(
        result,
        crack_length=args.crack_length,
        domain_area=args.domain_area,
        typical_stress=args.typical_stress,
        typical_velocity=args.typical_velocity,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
