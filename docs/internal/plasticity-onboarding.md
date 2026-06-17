# Onboarding Guide: J2 Plasticity & Cohesive Interfaces

Welcome to the `phast` plasticity and interfaces onboarding tutorial. This guide is designed for developers, researchers, and engineers who are using or contributing to the implicit elastoplasticity, ductile phase-field fracture, and cohesive zone modeling capability of this solver.

---

## 1. Physical and Numerical Scope

The solver contains three core layers under validation for nonlinear mechanics and interface fracture:

### A. J2 (von Mises) Elastoplasticity
* **Formulation:** Small-strain associative J2 plasticity with isotropic hardening.
* **Algorithm:** Material-point radial return mapping, per-element history tracking, state commit/rollback API (for line searches and Newton failures), and numerical algorithmic tangent.
* **Files:**
  * `src/phast/plasticity/j2_vonmises.py` - Return mapping kernels.
  * `src/phast/plasticity/mesh_j2.py` - Element-state management and integration loops.

### B. Coupled Ductile Phase-Field (PF) Fracture
* **Formulation:** Staggered integration of J2 plasticity with AT2 phase-field damage.
* **Coupling:** Elastic tensile strain energy plus accumulated plastic work drive the damage progression.
* **Energy Ledger:** Tracks elastic strain energy, plastic work, fracture surface energy, external work, and checking monotonicity.
* **Files:**
  * `src/phast/staggered_solver.py` - Staggered damage-mechanics loops.

### C. Zero-Thickness Cohesive Zone Models (CZM)
* **Formulation:** Bilinear traction-separation relation with scalar damage history.
* **Contact Penalty:** Built-in normal compression contact penalty to prevent interpenetration.
* **Global Coupling:** Residuals and consistent tangents are integrated directly into the global quasi-static Newton system.
* **Files:**
  * `src/phast/cohesive_elements/cohesive_elements.py` - Constitutive laws and separation state.
  * `src/phast/cohesive_elements/operator.py` - Residual and tangent assembly.

---

## 2. Onboarding Verification: Running the Benchmarks

To verify your environment is correctly set up, run the validation scripts in the repository. These script files output CSV datasets and post-processing plots.

### Verification Tier 1: Environment Check
Before launching run scripts, verify the public environment:
```bash
python -m phast doctor
```

### Verification Tier 2: J2 Plasticity Benchmarks
Run a uniaxial tension validation run that builds stress-strain curves and checks return-mapping convergence:
```bash
python examples/plasticity_interface_beta/run_j2_validation.py
```

To run the J2 backend promotion (profiling sparse mechanics solvers):
```bash
python examples/plasticity_interface_beta/run_sparse_j2_backend_promotion.py
```

### Verification Tier 3: Ductile Fracture & Sensitivity
Run the coupled ductile phase-field solver and check the energy ledger outputs:
```bash
python examples/plasticity_interface_beta/run_ductile_pf_plasticity_validation.py
python examples/plasticity_interface_beta/run_ductile_pf_sensitivity_study.py
```

### Verification Tier 4: Cohesive Benchmarks
Run the suite of cohesive interface checks:
1. **Mode-I Displacement Jump:**
   ```bash
   python examples/plasticity_interface_beta/run_cohesive_displacement_jump_benchmark.py
   ```
2. **Mixed-Mode Separation Tangent:**
   ```bash
   python examples/plasticity_interface_beta/run_cohesive_mixed_mode_benchmark.py
   ```
3. **Contact Compression Penalty:**
   ```bash
   python examples/plasticity_interface_beta/run_cohesive_contact_compression_benchmark.py
   ```
4. **Delamination Patch:**
   ```bash
   python examples/plasticity_interface_beta/run_cohesive_delamination_patch_benchmark.py
   ```
5. **Structural Double Cantilever Beam (DCB):**
   ```bash
   python examples/plasticity_interface_beta/run_structural_dcb_cohesive_benchmark.py
   ```

---

## 3. Product Claims & Limits

When designing your research or client models, keep these boundaries in mind:

### 👍 What is Supported
* 2D T3 (Triangular) and Q4 (Quadrilateral) elements.
* Isotropic linear hardening J2 plasticity.
* Bilinear traction-separation with normal contact penalty.
* Staggered explicit/implicit steps for J2 + AT2.

### 👎 What is NOT Yet Supported (Roadmap)
* Richer cohesive laws (PPR, Camanho).
* Full coupled CZM + Plasticity + PF-damage model combinations.
* 3D elements (hexahedrals/tetrahedrals).
* Arbitrary node duplication workflows on external mesh formats without manual set definition.

---

## 4. Contributing Your Code & Benchmarks
When adding new features or validating parameters:
1. Run `python -m phast doctor`.
2. Run the relevant validation script and inspect its retained artifacts.
3. Run the repository artifact-hygiene checks before committing heavy results or cached meshes.
