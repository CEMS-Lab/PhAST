"""
Mesh generation for standard phase-field fracture benchmarks.

Uses Gmsh .geo files + command-line gmsh for reliability.
Falls back to Python API only if CLI is unavailable.

Benchmarks
----------
Static / quasi-static:
- miehe_tension      : Single-edge-notch tension (Miehe et al. 2010)
- miehe_shear        : Single-edge-notch shear   (Miehe et al. 2010)
- square_plate       : Plain square (no notch, for nucleation studies)
- three_point_bending: Three-point bending with bottom-center notch (Miehe et al. 2010)
- l_shaped_panel     : L-shaped panel (Winkler 2001, Ambati 2015, Rudshaug 2024)

Dynamic (Borden 2012 / Bleyer 2017):
- rectangular_sent   : Rectangular SENT plate — dynamic SENT, crack branching (B1, B3, B4)
- rectangular_sent_comsol_structured : Half-plate SENT mapped-grid diagnostic for COMSOL B7
- kalthoff_winkler   : Two-notch plate with impact velocity BC (B2)
- crack_coalescence  : Two offset notches, cracks coalesce under tension (B5)
- brazilian_disc     : Circular disc, diametric compression / split-Hopkinson (B6)
- perforated_sent    : SENT plate with circular holes on mid-plane (B7, Bleyer 2017 Sec 4.2)
- glass_impact_vnotch: V-notch glass plate, pressure pulse on notch faces (arXiv:2411.16393 Sec 5.2)
"""

import os
from typing import Optional, Sequence, Tuple


def _run_gmsh(geo_path, msh_path, verbose=True):
    """Mesh a .geo file using gmsh Python API (opens .geo, no embed issues).

    Initialises gmsh at most once per interpreter and registers a single
    ``atexit`` finaliser. Each invocation then calls ``gmsh.clear()``
    rather than ``gmsh.finalize()`` to release the model memory without
    tearing down and re-initialising the library -- repeated
    initialise/finalise cycles are a documented source of memory growth
    in the gmsh Python API and are problematic in dataset-generation
    loops.
    """
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
        import atexit
        atexit.register(_safe_gmsh_finalize)
    try:
        gmsh.option.setNumber("General.Verbosity", 2 if verbose else 0)
        if verbose:
            print(f"[mesh_generator] Opening {geo_path} via Python API...", flush=True)
        gmsh.open(geo_path)
        gmsh.model.mesh.generate(2)

        # Stats
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
        n_nodes = len(node_tags)
        n_elems = sum(len(t) for t in elem_tags)

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh_path)
    finally:
        gmsh.clear()  # release model memory; keep library initialised
    if verbose:
        print(f"[mesh_generator] Mesh: {n_nodes} nodes, {n_elems} triangles", flush=True)
    return msh_path


def _safe_gmsh_finalize():
    """atexit hook: finalise gmsh if and only if still initialised."""
    try:
        import gmsh
        if gmsh.isInitialized():
            gmsh.finalize()
    except Exception:
        pass  # interpreter teardown; best-effort


def miehe_tension(
    output_path: str = 'miehe_tension.msh',
    L: float = 1.0,
    a: float = 0.5,
    l0: float = 0.0075,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    crack_band_width: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate mesh for Miehe single-edge-notch tension test.

    Geometry: L x L square plate with horizontal edge notch from
    left boundary to center at y = L/2.

    Parameters
    ----------
    output_path : str
        Path for output .msh file.
    L : float
        Side length (mm). Default 1.0 (Miehe benchmark).
    a : float
        Notch length from left edge (mm). Default 0.5 (half-width).
    l0 : float
        Phase-field regularization length (mm).
    h_crack : float or None
        Element size in crack zone. Default l0/2.
    h_coarse : float or None
        Element size far from crack. Default L/10.
    crack_band_width : float or None
        Half-width of refinement band. Default 3*l0.
    order : int
        Element order (1=linear).
    verbose : bool

    Returns
    -------
    output_path : str
    """
    if h_crack is None:
        h_crack = l0 / 2.0
    if h_coarse is None:
        h_coarse = L / 10.0
    if crack_band_width is None:
        crack_band_width = 3.0 * l0

    if verbose:
        print(f"[mesh_generator] Miehe SENT: L={L}, a={a}, l0={l0}", flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.6f}, h_coarse={h_coarse:.4f}, "
              f"band={crack_band_width:.4f}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    # V-notch half-opening (matches PhaseFieldX: 0.001mm)
    notch_eps = 0.001

    geo_content = f"""// Miehe SENT benchmark - auto-generated
// L={L}, a={a}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}
// V-notch opening: +/-{notch_eps} mm (matching PhaseFieldX)

h_crack = {h_crack};
h_coarse = {h_coarse};
L = {L};
a = {a};
band = {crack_band_width};

// Points — V-shaped notch with separate upper/lower lips
//   P4 -------- P3
//   |            |
//   P5 \\        |
//        > P7   P8
//   P6 //        |
//   |            |
//   P1 -------- P2
Point(1) = {{0, 0, 0, h_coarse}};               // bottom-left
Point(2) = {{L, 0, 0, h_coarse}};               // bottom-right
Point(3) = {{L, L, 0, h_coarse}};               // top-right
Point(4) = {{0, L, 0, h_coarse}};               // top-left
Point(5) = {{0, L/2 + {notch_eps}, 0, h_crack}};   // notch mouth upper
Point(6) = {{0, L/2 - {notch_eps}, 0, h_crack}};   // notch mouth lower
Point(7) = {{a, L/2, 0, h_crack}};              // notch tip
Point(8) = {{L, L/2, 0, h_crack}};              // crack path end

// Boundary + notch lines
Line(1) = {{1, 2}};   // bottom
Line(2) = {{2, 8}};   // right-lower
Line(3) = {{8, 3}};   // right-upper
Line(4) = {{3, 4}};   // top
Line(5) = {{4, 5}};   // left-upper (above notch)
Line(6) = {{5, 7}};   // notch upper lip
Line(7) = {{7, 6}};   // notch lower lip
Line(8) = {{6, 1}};   // left-lower (below notch)
Line(9) = {{7, 8}};   // crack path (expected propagation)

// Two surfaces — separated by notch, joined at crack path
Curve Loop(1) = {{1, 2, -9, 7, 8}};    // lower half
Plane Surface(1) = {{1}};
Curve Loop(2) = {{9, 3, 4, 5, 6}};     // upper half
Plane Surface(2) = {{2}};

// Physical groups
Physical Curve("bottom") = {{1}};
Physical Curve("right") = {{2, 3}};
Physical Curve("top") = {{4}};
Physical Curve("left") = {{5, 8}};
Physical Curve("notch_upper") = {{6}};
Physical Curve("notch_lower") = {{7}};
Physical Surface("plate") = {{1, 2}};

// Refinement fields
Field[1] = Distance;
Field[1].CurvesList = {{6, 7, 9}};
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0;
Field[2].DistMax = band;

// Extra refinement at notch tip
Field[3] = Distance;
Field[3].PointsList = {{7}};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack * 0.5;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = 5 * {l0};

Field[5] = Min;
Field[5].FieldsList = {{2, 4}};
Background Field = 5;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;  // Frontal-Delaunay
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def miehe_shear(
    output_path: str = 'miehe_shear.msh',
    L: float = 1.0,
    a: float = 0.5,
    l0: float = 0.0075,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate mesh for Miehe single-edge-notch shear test.

    Matches PhaseFieldX Example 1712 geometry: single surface with ±0.001mm
    notch mouth, box-field refinement covering the crack propagation region
    (notch tip toward lower-right diagonal).
    """
    if h_crack is None:
        h_crack = l0 / 2.0
    if h_coarse is None:
        h_coarse = L / 10.0

    band = 6 * l0  # wider band for diagonal crack path

    if verbose:
        print(f"[mesh_generator] Miehe SENS: L={L}, a={a}, l0={l0}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    # Notch mouth half-opening (PhaseFieldX .geo: ±0.001 mm)
    notch_eps = 0.001

    geo_content = f"""// Miehe SENS benchmark - auto-generated
// Matches PhaseFieldX Example 1712 geometry
h_crack = {h_crack};
h_coarse = {h_coarse};
L = {L};
a = {a};

// Points (PhaseFieldX convention)
//   P4 -------- P3
//   |            |
//   P5 -- P7  P8 |
//   P6           |
//   |            |
//   P1 -------- P2
//
// P5/P6 = notch mouth upper/lower (0, L/2 ± eps)
// P7 = notch tip (a, L/2)
// P8 = right midpoint (L, L/2) [for mesh structure]

Point(1) = {{0, 0, 0, h_coarse}};
Point(2) = {{L, 0, 0, h_coarse}};
Point(3) = {{L, L, 0, h_coarse}};
Point(4) = {{0, L, 0, h_coarse}};
Point(5) = {{0, L/2 + {notch_eps}, 0, h_crack}};   // notch mouth upper
Point(6) = {{0, L/2 - {notch_eps}, 0, h_crack}};   // notch mouth lower
Point(7) = {{a, L/2, 0, h_crack}};                  // notch tip

// Boundary lines + notch (single surface, no mid-line split)
Line(1) = {{1, 2}};   // bottom
Line(2) = {{2, 3}};   // right
Line(3) = {{3, 4}};   // top
Line(4) = {{4, 5}};   // left upper
Line(5) = {{5, 7}};   // notch upper lip
Line(6) = {{7, 6}};   // notch lower lip
Line(7) = {{6, 1}};   // left lower

// Single surface (PhaseFieldX style — no pre-biased crack path)
Curve Loop(1) = {{1, 2, 3, 4, 5, 6, 7}};
Plane Surface(1) = {{1}};

Physical Curve("bottom") = {{1}};
Physical Curve("right") = {{2}};
Physical Curve("top") = {{3}};
Physical Curve("left") = {{4, 7}};
Physical Curve("notch") = {{5, 6}};
Physical Surface("plate") = {{1}};

// ---- Refinement fields (PhaseFieldX style: box fields) ----

// Field 1: core refinement box (where crack propagates: notch tip → lower-right)
// PhaseFieldX: X in [-0.05, 0.5], Y in [-0.5, 0.05] (centered coords)
// Our coords: X in [a - 0.05, L], Y in [0, L/2 + 0.05]
Field[1] = Box;
Field[1].VIn = h_crack;
Field[1].VOut = h_coarse;
Field[1].XMin = a - 0.05;
Field[1].XMax = L;
Field[1].YMin = 0;
Field[1].YMax = L/2 + 0.05;

// Field 2: wider refinement box
// PhaseFieldX: X in [-0.1, 0.5], Y in [-0.5, 0.1] (centered coords)
// Our coords: X in [a - 0.1, L], Y in [0, L/2 + 0.1]
Field[2] = Box;
Field[2].VIn = h_crack * 10;
Field[2].VOut = h_coarse;
Field[2].XMin = a - 0.1;
Field[2].XMax = L;
Field[2].YMin = 0;
Field[2].YMax = L/2 + 0.1;

// Field 3: extra refinement at notch tip
Field[3] = Distance;
Field[3].PointsList = {{7}};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack * 0.5;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = 5 * {l0};

// Combine all refinement fields
Field[5] = Min;
Field[5].FieldsList = {{1, 2, 4}};
Background Field = 5;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def square_plate(
    output_path: str = 'square_plate.msh',
    L: float = 1.0,
    h: float = 0.05,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate a uniform square plate mesh (no notch)."""
    if verbose:
        print(f"[mesh_generator] Square plate: L={L}, h={h}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    geo_content = f"""// Square plate - auto-generated
h = {h};
L = {L};

Point(1) = {{0, 0, 0, h}};
Point(2) = {{L, 0, 0, h}};
Point(3) = {{L, L, 0, h}};
Point(4) = {{0, L, 0, h}};

Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};

Curve Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};

Physical Curve("bottom") = {{1}};
Physical Curve("right") = {{2}};
Physical Curve("top") = {{3}};
Physical Curve("left") = {{4}};
Physical Surface("plate") = {{1}};

Mesh.Algorithm = 6;
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def three_point_bending(
    output_path: str = 'three_point_bending.msh',
    W: float = 8.0,
    H: float = 2.0,
    a: float = 0.5,
    l0: float = 0.0075,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate mesh for three-point bending test with bottom-center notch.

    Reference: Miehe et al. (2010), PhaseFieldX Example 1714.

    Geometry: W x H rectangular beam with vertical notch of length *a*
    rising from the bottom center. The notch has a V-shaped opening
    of +/-0.2 mm (matching PhaseFieldX .geo). Single surface (no
    pre-defined crack-path line), with box-field mesh refinement.

    Physical groups
    ---------------
    bottom        : bottom boundary (full span, excluding notch opening)
    top           : top boundary
    left          : left boundary
    right         : right boundary
    notch         : both lips of the V-notch
    plate         : entire 2-D domain
    load_point    : top-center point  (W/2, H)
    support_left  : bottom-left corner point  (0, 0)
    support_right : bottom-right corner point (W, 0)

    Parameters
    ----------
    output_path : str
        Path for output .msh file.
    W : float
        Beam width (mm). Default 8.0.
    H : float
        Beam height (mm). Default 2.0.
    a : float
        Notch length from bottom edge upward (mm). Default 0.5.
    l0 : float
        Phase-field regularization length (mm).
    h_crack : float or None
        Element size in crack zone. Default l0/2.
    h_coarse : float or None
        Element size far from crack. Default H/10.
    verbose : bool

    Returns
    -------
    output_path : str
    """
    if h_crack is None:
        h_crack = l0 / 2.0
    if h_coarse is None:
        h_coarse = H / 10.0

    band = 3.0 * l0  # refinement half-width around crack path

    if verbose:
        print(f"[mesh_generator] Three-point bending: W={W}, H={H}, a={a}, l0={l0}",
              flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.6f}, h_coarse={h_coarse:.4f}, "
              f"band={band:.4f}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    # V-notch half-opening: ±0.2 mm (matches PhaseFieldX .geo)
    notch_eps = 0.2

    geo_content = f"""// Three-point bending benchmark - auto-generated
// Matches PhaseFieldX Example 1714 geometry
// W={W}, H={H}, a={a}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}
// V-notch opening: +/-{notch_eps} mm (PhaseFieldX convention)

h_crack = {h_crack};
h_coarse = {h_coarse};
W = {W};
H = {H};
a = {a};
band = {band};

// Points  (PhaseFieldX convention: single surface, no crack-path line)
//
//   P7 -------- P8 -------- P6
//   |                        |
//   |                        |
//   |                        |
//   |          P3            |
//   |          / \\           |
//   |        P2   P4        |
//   P1 ------       ------ P5
//
// P1 = bottom-left (support_left)
// P2 = notch mouth left   (W/2 - 0.2, 0)
// P3 = notch tip           (W/2, a)
// P4 = notch mouth right  (W/2 + 0.2, 0)
// P5 = bottom-right (support_right)
// P6 = top-right
// P7 = top-left
// P8 = top-center          (W/2, H)  — load_point

Point(1) = {{0, 0, 0, h_coarse}};                   // bottom-left (support_left)
Point(2) = {{W/2 - {notch_eps}, 0, 0, h_crack}};    // notch mouth left
Point(3) = {{W/2, a, 0, h_crack}};                  // notch tip
Point(4) = {{W/2 + {notch_eps}, 0, 0, h_crack}};    // notch mouth right
Point(5) = {{W, 0, 0, h_coarse}};                   // bottom-right (support_right)
Point(6) = {{W, H, 0, h_coarse}};                   // top-right
Point(7) = {{0, H, 0, h_coarse}};                   // top-left
Point(8) = {{W/2, H, 0, h_crack}};                  // top-center (load_point)

// Boundary lines + notch (single surface, no crack-path line)
Line(1)  = {{1, 2}};   // bottom-left segment
Line(2)  = {{2, 3}};   // notch left lip
Line(3)  = {{3, 4}};   // notch right lip
Line(4)  = {{4, 5}};   // bottom-right segment
Line(5)  = {{5, 6}};   // right boundary
Line(6)  = {{6, 8}};   // top-right segment
Line(7)  = {{8, 7}};   // top-left segment
Line(8)  = {{7, 1}};   // left boundary

// Single surface (PhaseFieldX style — no pre-biased crack path)
Curve Loop(1) = {{1, 2, 3, 4, 5, 6, 7, 8}};
Plane Surface(1) = {{1}};

// Physical groups
Physical Curve("bottom") = {{1, 4}};
Physical Curve("top") = {{6, 7}};
Physical Curve("left") = {{8}};
Physical Curve("right") = {{5}};
Physical Curve("notch") = {{2, 3}};
Physical Surface("plate") = {{1}};
Physical Point("load_point") = {{8}};
Physical Point("support_left") = {{1}};
Physical Point("support_right") = {{5}};

// ---- Refinement fields (PhaseFieldX style: box fields) ----

// Field 1-2: core refinement strip (±0.2 mm around crack path, full height)
Field[1] = Box;
Field[1].VIn = h_crack;
Field[1].VOut = h_coarse;
Field[1].XMin = W/2 - {notch_eps};
Field[1].XMax = W/2 + {notch_eps};
Field[1].YMin = 0;
Field[1].YMax = H;

// Field 3: wider refinement strip (±0.4 mm)
Field[3] = Box;
Field[3].VIn = h_crack * 10;
Field[3].VOut = h_coarse;
Field[3].XMin = W/2 - 2*{notch_eps};
Field[3].XMax = W/2 + 2*{notch_eps};
Field[3].YMin = 0;
Field[3].YMax = H;

// Field 4: left support refinement
Field[4] = Box;
Field[4].VIn = h_crack;
Field[4].VOut = h_coarse;
Field[4].XMin = 0;
Field[4].XMax = 0.1;
Field[4].YMin = 0;
Field[4].YMax = 0.1;

// Field 5: right support refinement
Field[5] = Box;
Field[5].VIn = h_crack;
Field[5].VOut = h_coarse;
Field[5].XMin = W - 0.1;
Field[5].XMax = W;
Field[5].YMin = 0;
Field[5].YMax = 0.1;

// Combine all refinement fields (matches PhaseFieldX — box fields only,
// no extra distance-based notch tip refinement that over-resolves the
// stress singularity and causes premature damage nucleation)
Field[10] = Min;
Field[10].FieldsList = {{1, 3, 4, 5}};
Background Field = 10;

// PhaseFieldX uses Gmsh defaults (MeshSizeFromPoints=1, ExtendFromBoundary=1).
// Respecting point-prescribed sizes prevents Gmsh from creating elements
// smaller than h_crack at the V-notch tip.
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeMin = {h_crack * 0.5};
Mesh.Algorithm = 6;  // Frontal-Delaunay
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def l_shaped_panel(
    output_path: str = 'l_shaped_panel.msh',
    L: float = 250.0,
    l0: float = 0.4,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate mesh for L-shaped panel benchmark.

    Reference: Ambati, Gerasimov & De Lorenzis (2015), Fig. 16a.
               Winkler (2001) experimental crack path.

    Geometry matches Ambati (2015) Fig. 16a — cutout in lower-right::

      P6(0,2L)----------P5(2L,2L)
        |                    |
        |                    | L
        |                    |
        |         P3--P7------P4(2L,L)
        |          ^  ^
        |          |  displacement point: 30 mm from re-entrant corner
        |          re-entrant corner
        |
      P1(0,0)-----P2(L,0)            <- bottom clamped (width L)

    BCs (Ambati 2015):
      - Bottom (P1-P2, y=0): u_i = 0 (clamped)
      - Load point on the cutout horizontal edge, 30 mm to the right of
        the re-entrant corner: u_2 = u_2^app
      - Cyclic loading protocol (Fig. 17): 0 -> +0.3U -> -0.2U -> +U

    Parameters
    ----------
    output_path : str
        Path for output .msh file.
    L : float
        Half-side length (mm). Default 250 (total 500x500, cutout 250x250).
    l0 : float
        Phase-field regularization length (mm).
    h_crack : float or None
        Element size near re-entrant corner. Default l0/4.
    h_coarse : float or None
        Element size far from crack. Default L/10.
    verbose : bool

    Returns
    -------
    output_path : str
    """
    if h_crack is None:
        h_crack = l0 / 4.0
    if h_coarse is None:
        h_coarse = L / 10.0

    band = 10.0 * l0

    if verbose:
        print(f"[mesh_generator] L-shaped panel: L={L}, l0={l0}", flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.4f}, h_coarse={h_coarse:.2f}, "
              f"band={band:.2f}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    # Ambati Fig. 16a marks the displacement application point 30 mm
    # from the re-entrant corner. Keep the physical group name
    # ``load_segment`` for compatibility with existing configs, but tag
    # the point itself rather than the full 30 mm edge.
    load_offset = 0.12 * L

    geo_content = f"""// L-shaped panel benchmark — Ambati, Gerasimov & De Lorenzis (2015) Fig. 16a
// L={L}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}
//
// Geometry: 2L x 2L square with L x L cutout in LOWER-RIGHT
//
//   P6(0,2L)-------------------P5(2L,2L)
//     |                            |
//     |                            | L
//     |                            |
//     |      P3(L,L)---P7-----------------P4(2L,L)
//     |         ^       ^
//     |         |       displacement point, 30mm from corner
//     |         |
//   P1(0,0)--P2(L,0)   <- bottom clamped
//
// Re-entrant corner at P3 = (L, L).
// Load "u" applied at P7, 30mm right of P3 on the cutout horizontal edge.
// BCs: bottom clamped, load_segment/load_point u_y prescribed (downward).

h_crack = {h_crack};
h_coarse = {h_coarse};
L = {L};
band = {band};

// Points (counter-clockwise from bottom-left)
Point(1) = {{0, 0, 0, h_coarse}};                     // bottom-left
Point(2) = {{L, 0, 0, h_coarse}};                     // bottom-right
Point(3) = {{L, L, 0, h_crack}};                      // re-entrant corner
Point(4) = {{2*L, L, 0, h_coarse}};                   // cutout horizontal meets right edge
Point(5) = {{2*L, 2*L, 0, h_coarse}};                 // top-right
Point(6) = {{0, 2*L, 0, h_coarse}};                   // top-left
Point(7) = {{L + {load_offset}, L, 0, h_crack}};      // displacement point (30mm from corner)

// Boundary lines (counter-clockwise)
Line(1) = {{1, 2}};   // bottom (clamped)
Line(2) = {{2, 3}};   // inner vertical (x=L, y=0 to L)
Line(3) = {{3, 7}};   // cutout horizontal left part
Line(4) = {{7, 4}};   // remaining cutout horizontal edge
Line(5) = {{4, 5}};   // right (x=2L)
Line(6) = {{5, 6}};   // top (y=2L)
Line(7) = {{6, 1}};   // left (x=0)

Curve Loop(1) = {{1, 2, 3, 4, 5, 6, 7}};
Plane Surface(1) = {{1}};

// Physical groups
Physical Curve("bottom") = {{1}};
Physical Curve("inner_vertical") = {{2}};
Physical Curve("cutout_horizontal") = {{3, 4}};
Physical Curve("cutout_horizontal_left") = {{3}};
Physical Curve("right") = {{5}};
Physical Curve("top") = {{6}};
Physical Curve("left") = {{7}};
Physical Surface("plate") = {{1}};
Physical Point("corner") = {{3}};
Physical Point("load_point") = {{7}};
Physical Point("load_segment") = {{7}}; // legacy BC name

// ---- Refinement ----

// Around re-entrant corner (crack initiation zone)
Field[1] = Distance;
Field[1].PointsList = {{3}};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0;
Field[2].DistMax = 30 * {l0};

// Along inner edges and expected crack path region
Field[3] = Distance;
Field[3].CurvesList = {{2, 3, 4}};
Field[3].Sampling = 200;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = band;

Field[5] = Min;
Field[5].FieldsList = {{2, 4}};
Background Field = 5;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def plate_with_holes(
    output_path: str = 'plate_with_holes.msh',
    L: float = 1.0,
    holes: Optional[list] = None,
    n_holes: int = 4,
    r_hole: float = 0.07,
    seed: int = 42,
    l0: float = 0.015,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate mesh for a plate with circular holes (multi-crack coalescence).

    Circular holes act as stress concentrators. Under tension, cracks nucleate
    at hole boundaries and coalesce through ligaments, producing complex crack
    networks unlike the single-crack benchmarks.

    Parameters
    ----------
    output_path : str
        Path for output .msh file.
    L : float
        Plate side length (mm). Default 1.0.
    holes : list of (cx, cy, r) or None
        Explicit hole list. If None, holes are placed on a grid with random
        offsets controlled by ``seed``.
    n_holes : int
        Number of holes (used only when ``holes`` is None). Default 4.
    r_hole : float
        Hole radius (mm), used as default when ``holes`` is None.
    seed : int
        Random seed for hole placement (used only when ``holes`` is None).
    l0 : float
        Phase-field regularization length (mm).
    h_crack : float or None
        Element size near hole boundaries. Default l0/2.
    h_coarse : float or None
        Coarse element size. Default L/8.
    order : int
        Element order (1=linear).
    verbose : bool

    Returns
    -------
    output_path : str

    Notes
    -----
    Default hole layout (n_holes=4, seed=42): 2×2 grid with ±10% random offset.
    For reproducible datagen use a fixed seed per simulation index.
    """
    import random as _rng

    if h_crack is None:
        h_crack = l0 / 2.0
    if h_coarse is None:
        h_coarse = L / 8.0

    # ---- Generate hole list if not provided ----
    if holes is None:
        rng = _rng.Random(seed)
        margin = r_hole + 3.0 * l0  # keep holes away from boundary
        inner = L - 2 * margin

        # Place holes on a near-uniform grid with small random offsets
        import math
        cols = max(1, round(math.sqrt(n_holes)))
        rows = math.ceil(n_holes / cols)
        dx = inner / max(1, cols - 1) if cols > 1 else 0.0
        dy = inner / max(1, rows - 1) if rows > 1 else 0.0
        jitter = 0.10 * min(dx if cols > 1 else inner, dy if rows > 1 else inner)

        holes = []
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n_holes:
                    break
                cx = margin + col * dx + rng.uniform(-jitter, jitter)
                cy = margin + row * dy + rng.uniform(-jitter, jitter)
                # clamp
                cx = max(margin, min(L - margin, cx))
                cy = max(margin, min(L - margin, cy))
                holes.append((cx, cy, r_hole))
                count += 1

    n = len(holes)

    if verbose:
        print(f"[mesh_generator] Plate with {n} holes: L={L}, l0={l0}", flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.6f}, h_coarse={h_coarse:.4f}", flush=True)
        for i, (cx, cy, r) in enumerate(holes):
            print(f"[mesh_generator]   Hole {i}: center=({cx:.4f},{cy:.4f}), r={r:.4f}",
                  flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    # ---- Build .geo content ----
    # Points 1-4: outer corners
    # For hole i (0-indexed):
    #   Center: 100 + 5*i
    #   East:   101 + 5*i
    #   North:  102 + 5*i
    #   West:   103 + 5*i
    #   South:  104 + 5*i
    # Circle arcs 100+4*i .. 103+4*i  (E→N→W→S→E)
    # Curve Loops: outer=1, hole i = 10+i

    lines = [
        f"// Plate with {n} holes — auto-generated",
        f"// L={L}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}",
        f"// Holes: {holes}",
        "",
        f"h_crack = {h_crack};",
        f"h_coarse = {h_coarse};",
        f"L = {L};",
        "",
        "// Outer boundary",
        f"Point(1) = {{0, 0, 0, h_coarse}};",
        f"Point(2) = {{L, 0, 0, h_coarse}};",
        f"Point(3) = {{L, L, 0, h_coarse}};",
        f"Point(4) = {{0, L, 0, h_coarse}};",
        "",
        "Line(1) = {1, 2};  // bottom",
        "Line(2) = {2, 3};  // right",
        "Line(3) = {3, 4};  // top",
        "Line(4) = {4, 1};  // left",
        "",
        "Curve Loop(1) = {1, 2, 3, 4};  // outer",
        "",
    ]

    hole_arc_lists = []  # collect curve tag lists for Distance fields
    for i, (cx, cy, r) in enumerate(holes):
        pc = 100 + 5 * i   # center
        pe = 101 + 5 * i   # east
        pn = 102 + 5 * i   # north
        pw = 103 + 5 * i   # west
        ps = 104 + 5 * i   # south
        ca = 100 + 4 * i   # arc E→N
        cb = 101 + 4 * i   # arc N→W
        cc = 102 + 4 * i   # arc W→S
        cd = 103 + 4 * i   # arc S→E
        cl = 10 + i         # curve loop

        lines += [
            f"// Hole {i}: center=({cx:.5f},{cy:.5f}), r={r:.5f}",
            f"Point({pc}) = {{{cx}, {cy}, 0, h_crack}};   // center",
            f"Point({pe}) = {{{cx+r}, {cy}, 0, h_crack}};  // east",
            f"Point({pn}) = {{{cx}, {cy+r}, 0, h_crack}};  // north",
            f"Point({pw}) = {{{cx-r}, {cy}, 0, h_crack}};  // west",
            f"Point({ps}) = {{{cx}, {cy-r}, 0, h_crack}};  // south",
            f"Circle({ca}) = {{{pe}, {pc}, {pn}}};",
            f"Circle({cb}) = {{{pn}, {pc}, {pw}}};",
            f"Circle({cc}) = {{{pw}, {pc}, {ps}}};",
            f"Circle({cd}) = {{{ps}, {pc}, {pe}}};",
            f"Curve Loop({cl}) = {{{ca}, {cb}, {cc}, {cd}}};",
            "",
        ]
        hole_arc_lists.extend([ca, cb, cc, cd])

    # Surface: outer loop + hole loops (Gmsh handles orientation automatically)
    hole_loop_tags = ", ".join(str(10 + i) for i in range(n))
    lines.append(f"Plane Surface(1) = {{1, {hole_loop_tags}}};")
    lines.append("")

    # Physical groups
    lines += [
        'Physical Curve("bottom") = {1};',
        'Physical Curve("right") = {2};',
        'Physical Curve("top") = {3};',
        'Physical Curve("left") = {4};',
    ]
    for i in range(n):
        ca = 100 + 4 * i
        cb = 101 + 4 * i
        cc = 102 + 4 * i
        cd = 103 + 4 * i
        lines.append(f'Physical Curve("hole_{i}") = {{{ca}, {cb}, {cc}, {cd}}};')
    lines += [
        'Physical Surface("plate") = {1};',
        "",
    ]

    # Refinement: Distance from all hole arcs
    arc_list_str = ", ".join(str(t) for t in hole_arc_lists)
    band = 4.0 * l0
    lines += [
        "// ---- Refinement near holes ----",
        "Field[1] = Distance;",
        f"Field[1].CurvesList = {{{arc_list_str}}};",
        "Field[1].Sampling = 60;",
        "",
        "Field[2] = Threshold;",
        "Field[2].InField = 1;",
        f"Field[2].SizeMin = {h_crack};",
        f"Field[2].SizeMax = {h_coarse};",
        "Field[2].DistMin = 0;",
        f"Field[2].DistMax = {band};",
        "",
        "Background Field = 2;",
        "",
        "Mesh.MeshSizeExtendFromBoundary = 0;",
        "Mesh.MeshSizeFromPoints = 0;",
        "Mesh.MeshSizeFromCurvature = 0;",
        "Mesh.Algorithm = 6;  // Frontal-Delaunay",
        f"Mesh.ElementOrder = {order};",
    ]

    geo_content = "\n".join(lines) + "\n"
    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def bazant_gap_test(
    output_path: str = 'bazant_gap_test.msh',
    L: float = 1.0,
    W: float = 0.5,
    a: float = 0.3,
    gap: float = 0.1,
    l0: float = 0.01,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Bazant gap test: two parallel horizontal notches.

    Tests phase-field behavior under parallel stress — validates
    that star_convex split handles crack interaction correctly.

    Geometry:
        Rectangle L x W with two horizontal notches:
        - Notch 1: from left edge, length a, at y = W/2 + gap/2
        - Notch 2: from left edge, length a, at y = W/2 - gap/2

    Parameters
    ----------
    output_path : str
        Path for output .msh file.
    L : float — domain length
    W : float — domain width
    a : float — notch length
    gap : float — vertical distance between notch tips
    l0 : float — regularization length
    h_crack : float or None — element size near notches (default: l0/3)
    h_coarse : float or None — coarse element size (default: 10*h_crack)
    verbose : bool

    Returns
    -------
    output_path : str
    """
    if h_crack is None:
        h_crack = l0 / 3.0
    if h_coarse is None:
        h_coarse = 10.0 * h_crack

    band = 3.0 * l0

    if verbose:
        print(f"[mesh_generator] Bazant gap test: L={L}, W={W}, a={a}, "
              f"gap={gap}, l0={l0}", flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.6f}, h_coarse={h_coarse:.4f}, "
              f"band={band:.4f}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    # V-notch half-opening
    notch_eps = 0.001

    y_upper = W / 2.0 + gap / 2.0
    y_lower = W / 2.0 - gap / 2.0

    geo_content = f"""// Bazant gap test benchmark - auto-generated
// L={L}, W={W}, a={a}, gap={gap}, l0={l0}
// Two parallel horizontal notches from left edge
// V-notch opening: +/-{notch_eps} mm

h_crack = {h_crack};
h_coarse = {h_coarse};
L = {L};
W = {W};
a = {a};
band = {band};

// Points
//   P4 ----------------------------------- P3
//   |                                       |
//   P8 \\                                    |
//        > P10 (notch 1 tip)               |
//   P7 //                                    |
//   |                                       |
//   P6 \\                                    |
//        > P9  (notch 2 tip)               |
//   P5 //                                    |
//   |                                       |
//   P1 ----------------------------------- P2

Point(1) = {{0, 0, 0, h_coarse}};                             // bottom-left
Point(2) = {{L, 0, 0, h_coarse}};                             // bottom-right
Point(3) = {{L, W, 0, h_coarse}};                             // top-right
Point(4) = {{0, W, 0, h_coarse}};                             // top-left
Point(5) = {{0, {y_lower} - {notch_eps}, 0, h_crack}};        // lower notch mouth bottom
Point(6) = {{0, {y_lower} + {notch_eps}, 0, h_crack}};        // lower notch mouth top
Point(7) = {{0, {y_upper} - {notch_eps}, 0, h_crack}};        // upper notch mouth bottom
Point(8) = {{0, {y_upper} + {notch_eps}, 0, h_crack}};        // upper notch mouth top
Point(9) = {{a, {y_lower}, 0, h_crack}};                      // lower notch tip
Point(10) = {{a, {y_upper}, 0, h_crack}};                     // upper notch tip

// Boundary lines + notch lips
Line(1)  = {{1, 5}};   // left: bottom to lower notch mouth bottom
Line(2)  = {{5, 9}};   // lower notch bottom lip
Line(3)  = {{9, 6}};   // lower notch top lip
Line(4)  = {{6, 7}};   // left: between notches
Line(5)  = {{7, 10}};  // upper notch bottom lip
Line(6)  = {{10, 8}};  // upper notch top lip
Line(7)  = {{8, 4}};   // left: above upper notch
Line(8)  = {{4, 3}};   // top
Line(9)  = {{3, 2}};   // right
Line(10) = {{2, 1}};   // bottom

// Single surface (notch lips are embedded)
Curve Loop(1) = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
Plane Surface(1) = {{1}};

// Physical groups
Physical Curve("bottom") = {{10}};
Physical Curve("right") = {{9}};
Physical Curve("top") = {{8}};
Physical Curve("left") = {{1, 4, 7}};
Physical Curve("notch_lower") = {{2, 3}};
Physical Curve("notch_upper") = {{5, 6}};
Physical Surface("plate") = {{1}};

// ---- Refinement fields ----

// Field 1-2: refine along notch lips
Field[1] = Distance;
Field[1].CurvesList = {{2, 3, 5, 6}};
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0;
Field[2].DistMax = band;

// Field 3-4: extra refinement at notch tips
Field[3] = Distance;
Field[3].PointsList = {{9, 10}};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack * 0.5;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = 5 * {l0};

// Field 5-6: refine the gap region between notch tips
Field[5] = Distance;
Field[5].CurvesList = {{4}};
Field[5].Sampling = 100;

Field[6] = Threshold;
Field[6].InField = 5;
Field[6].SizeMin = h_crack;
Field[6].SizeMax = h_coarse;
Field[6].DistMin = 0;
Field[6].DistMax = band;

Field[7] = Min;
Field[7].FieldsList = {{2, 4, 6}};
Background Field = 7;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;  // Frontal-Delaunay
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def rectangular_sent(
    output_path: str = 'rectangular_sent.msh',
    W: float = 100.0,
    H: float = 40.0,
    a: float = 50.0,
    l0: float = 0.5,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    crack_band_width: Optional[float] = None,
    branching: bool = False,
    half_plate: bool = False,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate mesh for a rectangular plate with horizontal edge notch.

    Geometry: W x H rectangle with horizontal notch from left edge
    to x = a at y = H/2 (mid-height). Suitable for dynamic SENT and
    crack branching benchmarks (Borden et al. 2012).

    Parameters
    ----------
    output_path : str
        Path for output .msh file.
    W : float
        Plate width (mm). Default 100.0.
    H : float
        Plate height (mm). Default 40.0.
    a : float
        Notch length from left edge (mm). Default 50.0 (= W/2).
    l0 : float
        Phase-field regularization length (mm).
    h_crack : float or None
        Element size in crack zone. Default l0/2.
    h_coarse : float or None
        Element size far from crack. Default max(W, H)/20.
    crack_band_width : float or None
        Half-width of refinement band. Default 5*l0.
    branching : bool
        If True, use uniform fine mesh in the entire right half of the
        plate (x > a-5) to properly resolve crack branching patterns.
        Required for Borden et al. (2012) Section 4.2 reproduction.
    order : int
        Element order (1=linear).
    verbose : bool

    Returns
    -------
    output_path : str
    """
    if h_crack is None:
        h_crack = l0 / 2.0
    if h_coarse is None:
        h_coarse = max(W, H) / 20.0
    if crack_band_width is None:
        crack_band_width = 5.0 * l0

    if verbose:
        print(f"[mesh_generator] Rectangular SENT: W={W}, H={H}, a={a}, l0={l0}",
              flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.6f}, h_coarse={h_coarse:.4f}, "
              f"band={crack_band_width:.4f}, branching={branching}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    notch_eps = min(0.01 * min(W, H), 0.01)

    # Half-plate variant: precrack runs along y=0 (symmetry plane).
    # COMSOL B7 convention -- W x H half-plate with symmetry BC at y=0,
    # equivalent to a full W x 2H plate by mirroring. The bottom edge
    # is split into 'crack' (x in [0, a], pf_dirichlet=1) and
    # 'bottom_sym' (x in [a, W], u_y=0). No physical slit -- the crack
    # lip is enforced via the phase field, mirroring glass_impact_vnotch.
    if half_plate:
        if branching:
            branch_box = f"""
// Branching zone: uniform h_crack mesh in entire right half (half-plate)
Field[6] = Box;
Field[6].VIn  = h_crack;
Field[6].VOut = h_coarse;
Field[6].XMin = {a - 5.0};
Field[6].XMax = {W};
Field[6].YMin = 0;
Field[6].YMax = {H};

Field[7] = Min;
Field[7].FieldsList = {{2, 4, 6}};
Background Field = 7;
"""
        else:
            branch_box = f"""
Field[5] = Min;
Field[5].FieldsList = {{2, 4}};
Background Field = 5;
"""

        geo_content = f"""// Rectangular SENT half-plate (symmetry at y=0) - auto-generated
// W={W}, H={H}, a={a}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}
// Precrack along y=0 from x=0 to x=a (enforced via pf_dirichlet on
// the 'crack' physical curve). Symmetry BC u_y=0 on 'bottom_sym'
// (x in [a, W]).
// branching={branching}, half_plate=True

h_crack = {h_crack};
h_coarse = {h_coarse};
W = {W};
H = {H};
a = {a};
band = {crack_band_width};

Point(1) = {{0, 0, 0, h_crack}};       // bottom-left (crack mouth)
Point(2) = {{a, 0, 0, h_crack}};       // crack tip
Point(3) = {{W, 0, 0, h_coarse}};      // bottom-right
Point(4) = {{W, H, 0, h_coarse}};      // top-right
Point(5) = {{0, H, 0, h_coarse}};      // top-left

Line(1) = {{1, 2}};   // crack (along symmetry plane y=0, x in [0, a])
Line(2) = {{2, 3}};   // bottom_sym (symmetry, x in [a, W])
Line(3) = {{3, 4}};   // right
Line(4) = {{4, 5}};   // top
Line(5) = {{5, 1}};   // left

Curve Loop(1) = {{1, 2, 3, 4, 5}};
Plane Surface(1) = {{1}};

Physical Curve("crack")      = {{1}};
Physical Curve("bottom_sym") = {{2}};
Physical Curve("right")      = {{3}};
Physical Curve("top")        = {{4}};
Physical Curve("left")       = {{5}};
Physical Surface("plate")    = {{1}};

// Refinement along the crack line and the expected propagation path
Field[1] = Distance;
Field[1].CurvesList = {{1}};
Field[1].Sampling = 300;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0;
Field[2].DistMax = band;

// Extra refinement at the crack tip
Field[3] = Distance;
Field[3].PointsList = {{2}};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack * 0.5;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = 5 * {l0};
{branch_box}
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

        with open(geo_path, 'w') as f:
            f.write(geo_content)

        if verbose:
            print(f"[mesh_generator] Wrote {geo_path} (half-plate), meshing...",
                  flush=True)

        _run_gmsh(geo_path, output_path, verbose=verbose)

        if verbose:
            print(f"[mesh_generator] Saved: {output_path}", flush=True)
        return output_path

    # For branching: uniform fine mesh in the entire right half
    # Branches diverge at ~30deg, so over 50mm they need ~25mm vertical extent
    if branching:
        branch_box = f"""
// Branching zone: uniform h_crack mesh in entire right half
// Crack branches diverge at ~30deg from horizontal — need fine mesh
// covering the full plate height beyond the notch tip.
Field[6] = Box;
Field[6].VIn  = h_crack;
Field[6].VOut = h_coarse;
Field[6].XMin = {a - 5.0};
Field[6].XMax = {W};
Field[6].YMin = 0;
Field[6].YMax = {H};

Field[7] = Min;
Field[7].FieldsList = {{2, 4, 6}};
Background Field = 7;
"""
    else:
        branch_box = f"""
Field[5] = Min;
Field[5].FieldsList = {{2, 4}};
Background Field = 5;
"""

    geo_content = f"""// Rectangular SENT benchmark - auto-generated
// W={W}, H={H}, a={a}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}
// Horizontal notch from left to x=a at y=H/2
// branching={branching}

h_crack = {h_crack};
h_coarse = {h_coarse};
W = {W};
H = {H};
a = {a};
band = {crack_band_width};

Point(1) = {{0, 0, 0, h_coarse}};
Point(2) = {{W, 0, 0, h_coarse}};
Point(3) = {{W, H, 0, h_coarse}};
Point(4) = {{0, H, 0, h_coarse}};
Point(5) = {{0, H/2 + {notch_eps}, 0, h_crack}};
Point(6) = {{0, H/2 - {notch_eps}, 0, h_crack}};
Point(7) = {{a, H/2, 0, h_crack}};
Point(8) = {{W, H/2, 0, h_crack}};

Line(1) = {{1, 2}};
Line(2) = {{2, 8}};
Line(3) = {{8, 3}};
Line(4) = {{3, 4}};
Line(5) = {{4, 5}};
Line(6) = {{5, 7}};
Line(7) = {{7, 6}};
Line(8) = {{6, 1}};
Line(9) = {{7, 8}};

Curve Loop(1) = {{1, 2, -9, 7, 8}};
Plane Surface(1) = {{1}};
Curve Loop(2) = {{9, 3, 4, 5, 6}};
Plane Surface(2) = {{2}};

Physical Curve("bottom") = {{1}};
Physical Curve("right") = {{2, 3}};
Physical Curve("top") = {{4}};
Physical Curve("left") = {{5, 8}};
Physical Curve("notch_upper") = {{6}};
Physical Curve("notch_lower") = {{7}};
Physical Surface("plate") = {{1, 2}};

// Refinement along notch and crack path
Field[1] = Distance;
Field[1].CurvesList = {{6, 7, 9}};
Field[1].Sampling = 300;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0;
Field[2].DistMax = band;

// Extra refinement at notch tip
Field[3] = Distance;
Field[3].PointsList = {{7}};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack * 0.5;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = 5 * {l0};
{branch_box}
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def rectangular_sent_liu_structured(
    output_path: str = 'rectangular_sent_liu_structured.msh',
    W: float = 100.0,
    H: float = 40.0,
    a: float = 50.0,
    l0: float = 0.25,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """Structured split-quad SENT mesh for Liu-style B1 branching.

    Liu et al. (2025) show a quad-dominant computation mesh with a very
    regular, symmetric refinement pattern for the Borden dynamic branching
    plate. The current solver kernels are P1-triangle based, so this
    generator builds a structured quadrilateral grid and splits every quad
    into two triangles with a mirrored checkerboard diagonal pattern. This
    preserves the symmetry and regularity benefits of a quad mesh while
    remaining compatible with the existing matrix-free triangle operators.

    The initial notch is represented as an internal ``crack`` line node set
    along ``0 <= x <= a, y = H/2``. Use it with ``pf_dirichlet`` and
    ``initial_conditions.preseed_notch_nodesets`` to match the common
    Borden/Liu initial-crack convention without cutting a geometric slit.
    """
    import numpy as np
    import meshio

    if h_crack is None:
        h_crack = l0 / 2.0
    # ``h_coarse`` is accepted for YAML/API symmetry with rectangular_sent.
    # This structured reference mesh intentionally uses one uniform spacing:
    # the regular grid is the diagnostic variable.
    _ = h_coarse

    nx = int(round(W / h_crack))
    ny = int(round(H / h_crack))
    if nx < 2 or ny < 2:
        raise ValueError("structured SENT mesh needs at least 2 cells in x/y")
    hx = W / nx
    hy = H / ny
    if abs(hx - h_crack) / max(h_crack, 1e-30) > 1e-8:
        raise ValueError(f"W={W} must be divisible by h_crack={h_crack}")
    if abs(hy - h_crack) / max(h_crack, 1e-30) > 1e-8:
        raise ValueError(f"H={H} must be divisible by h_crack={h_crack}")

    x = np.linspace(0.0, W, nx + 1)
    y = np.linspace(0.0, H, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.zeros((nx + 1) * (ny + 1)),
    ])

    def node(i: int, j: int) -> int:
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        # Mirror the checkerboard pattern about the mid-height so the
        # diagonal choice itself does not bias the upper/lower branch.
        j_sym = min(j, ny - 1 - j)
        for i in range(nx):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n01 = node(i, j + 1)
            n11 = node(i + 1, j + 1)
            if (i + j_sym) % 2 == 0:
                tris.append([n00, n10, n11])
                tris.append([n00, n11, n01])
            else:
                tris.append([n00, n10, n01])
                tris.append([n10, n11, n01])

    line_cells = []
    line_sets = {
        'bottom': [],
        'top': [],
        'left': [],
        'right': [],
        'crack': [],
    }

    def add_line(name: str, n0: int, n1: int) -> None:
        line_sets[name].append(len(line_cells))
        line_cells.append([n0, n1])

    for i in range(nx):
        add_line('bottom', node(i, 0), node(i + 1, 0))
        add_line('top', node(i, ny), node(i + 1, ny))
    for j in range(ny):
        add_line('left', node(0, j), node(0, j + 1))
        add_line('right', node(nx, j), node(nx, j + 1))

    j_mid = int(round((H / 2.0) / hy))
    i_crack = int(round(a / hx))
    if abs(y[j_mid] - H / 2.0) > 1e-10:
        raise ValueError("H/2 must lie on the structured grid")
    if abs(x[i_crack] - a) > 1e-10:
        raise ValueError("a must lie on the structured grid")
    for i in range(i_crack):
        add_line('crack', node(i, j_mid), node(i + 1, j_mid))

    cells = [
        ('line', np.asarray(line_cells, dtype=np.int64)),
        ('triangle', np.asarray(tris, dtype=np.int64)),
    ]
    field_data = {
        'bottom': np.asarray([1, 1], dtype=np.int32),
        'top': np.asarray([2, 1], dtype=np.int32),
        'left': np.asarray([3, 1], dtype=np.int32),
        'right': np.asarray([4, 1], dtype=np.int32),
        'crack': np.asarray([5, 1], dtype=np.int32),
        'plate': np.asarray([6, 2], dtype=np.int32),
    }
    line_tags = np.zeros(len(line_cells), dtype=np.int32)
    for name, ids in line_sets.items():
        line_tags[np.asarray(ids, dtype=np.int64)] = field_data[name][0]
    tri_tags = np.full(len(tris), field_data['plate'][0], dtype=np.int32)
    cell_data = {
        'gmsh:physical': [line_tags, tri_tags],
        'gmsh:geometrical': [line_tags.copy(), tri_tags.copy()],
    }
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=cell_data,
        field_data=field_data,
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    meshio.write(output_path, mesh, file_format='gmsh22', binary=False)

    if verbose:
        print("[mesh_generator] Liu structured SENT split-quad mesh:",
              flush=True)
        print(f"[mesh_generator]   W={W}, H={H}, a={a}, h={h_crack}, "
              f"nodes={len(points)}, triangles={len(tris)}", flush=True)
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def rectangular_sent_comsol_structured(
    output_path: str = 'rectangular_sent_comsol_structured.msh',
    W: float = 100.0,
    H: float = 20.0,
    a: float = 50.0,
    l0: float = 0.5,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """Structured split-quad half-plate SENT mesh for COMSOL B7 parity.

    COMSOL's Dynamic Crack Branching application uses symmetry about the
    X-axis: the full sample height is 40 mm, but the modeled rectangle is
    ``height/2 = 20 mm``. The mesh is mapped quadrilateral with ``h = l0/4``
    and a pre-existing bottom-edge crack.
    The production mechanics kernels in this repository are currently P1
    triangle based, so this diagnostic keeps the mapped-grid node placement
    and symmetry boundary groups while splitting each quad into two triangles
    with an alternating diagonal pattern.

    Boundary/node sets:
      - ``crack``: bottom edge from x=0 to x=a for phase-field Dirichlet
      - ``bottom_sym``: bottom edge from x=a to x=W for vertical symmetry
      - ``top``, ``left``, ``right``: outer boundaries
    """
    import numpy as np
    import meshio

    if h_crack is None:
        h_crack = l0 / 4.0
    # Accepted for API/YAML symmetry with rectangular_sent; this mesh is
    # intentionally uniform because regularity is the diagnostic variable.
    _ = h_coarse

    nx = int(round(W / h_crack))
    ny = int(round(H / h_crack))
    if nx < 2 or ny < 2:
        raise ValueError("COMSOL structured SENT mesh needs at least 2 cells in x/y")
    hx = W / nx
    hy = H / ny
    if abs(hx - h_crack) / max(abs(h_crack), 1e-30) > 1e-8:
        raise ValueError(f"W={W} must be divisible by h_crack={h_crack}")
    if abs(hy - h_crack) / max(abs(h_crack), 1e-30) > 1e-8:
        raise ValueError(f"H={H} must be divisible by h_crack={h_crack}")

    x = np.linspace(0.0, W, nx + 1)
    y = np.linspace(0.0, H, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.zeros((nx + 1) * (ny + 1)),
    ])

    def node(i: int, j: int) -> int:
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n01 = node(i, j + 1)
            n11 = node(i + 1, j + 1)
            if (i + j) % 2 == 0:
                tris.append([n00, n10, n11])
                tris.append([n00, n11, n01])
            else:
                tris.append([n00, n10, n01])
                tris.append([n10, n11, n01])

    line_cells = []
    line_sets = {
        'bottom_sym': [],
        'top': [],
        'left': [],
        'right': [],
        'crack': [],
    }

    def add_line(name: str, n0: int, n1: int) -> None:
        line_sets[name].append(len(line_cells))
        line_cells.append([n0, n1])

    i_crack = int(round(a / hx))
    if i_crack < 1 or i_crack > nx:
        raise ValueError("a must satisfy 0 < a <= W for the COMSOL SENT crack")
    if abs(x[i_crack] - a) > 1e-10:
        raise ValueError("a must lie on the structured grid")

    for i in range(nx):
        if i < i_crack:
            add_line('crack', node(i, 0), node(i + 1, 0))
        else:
            add_line('bottom_sym', node(i, 0), node(i + 1, 0))
        add_line('top', node(i, ny), node(i + 1, ny))
    for j in range(ny):
        add_line('left', node(0, j), node(0, j + 1))
        add_line('right', node(nx, j), node(nx, j + 1))

    cells = [
        ('line', np.asarray(line_cells, dtype=np.int64)),
        ('triangle', np.asarray(tris, dtype=np.int64)),
    ]
    field_data = {
        'bottom_sym': np.asarray([1, 1], dtype=np.int32),
        'top': np.asarray([2, 1], dtype=np.int32),
        'left': np.asarray([3, 1], dtype=np.int32),
        'right': np.asarray([4, 1], dtype=np.int32),
        'crack': np.asarray([5, 1], dtype=np.int32),
        'plate': np.asarray([6, 2], dtype=np.int32),
    }
    line_tags = np.zeros(len(line_cells), dtype=np.int32)
    for name, ids in line_sets.items():
        line_tags[np.asarray(ids, dtype=np.int64)] = field_data[name][0]
    tri_tags = np.full(len(tris), field_data['plate'][0], dtype=np.int32)
    cell_data = {
        'gmsh:physical': [line_tags, tri_tags],
        'gmsh:geometrical': [line_tags.copy(), tri_tags.copy()],
    }
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=cell_data,
        field_data=field_data,
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    meshio.write(output_path, mesh, file_format='gmsh22', binary=False)

    if verbose:
        print("[mesh_generator] COMSOL structured half-plate SENT split-quad mesh:",
              flush=True)
        print(f"[mesh_generator]   W={W}, H={H}, a={a}, h={h_crack}, "
              f"nodes={len(points)}, triangles={len(tris)}", flush=True)
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def rectangular_sent_circular_inclusions(
    output_path: str = 'rectangular_sent_particles.msh',
    W: float = 32.0,
    H: float = 16.0,
    a: float = 4.0,
    inclusions: Sequence[Tuple[float, float, float]] = ((16.0, 9.5, 1.6),),
    l0: float = 0.1,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    crack_band_width: Optional[float] = None,
    h_inclusion: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate a SENT plate with true conforming circular inclusions.

    Unlike :func:`rectangular_sent`, this geometry does not model the
    inclusion as a post-mesh Gaussian material field. Each inclusion is a
    circular Gmsh disk surface, and the matrix surface uses the same circle
    loops as holes. The mesh therefore has element edges on the
    matrix-particle interface.

    The function intentionally keeps the inclusion layout fixed: it is for
    forward/reference simulations and visual checks. Gradient-based recovery
    of moving particle centres should use a fixed-mesh smooth indicator or a
    shape-derivative/remeshing workflow, not this conforming geometry directly.
    """
    if h_crack is None:
        h_crack = l0 / 2.0
    if h_coarse is None:
        h_coarse = max(W, H) / 20.0
    if crack_band_width is None:
        crack_band_width = 5.0 * l0
    if h_inclusion is None:
        h_inclusion = h_crack

    inclusions = tuple((float(x), float(y), float(r))
                       for x, y, r in inclusions)
    for i, (x, y, r) in enumerate(inclusions, start=1):
        if r <= 0.0:
            raise ValueError(f"inclusion {i} has non-positive radius {r}")
        if x - r <= 0.0 or x + r >= W or y - r <= 0.0 or y + r >= H:
            raise ValueError(
                f"inclusion {i} at ({x}, {y}) radius {r} is not strictly "
                f"inside the {W} x {H} mm plate")
    for i, (xi, yi, ri) in enumerate(inclusions):
        for j, (xj, yj, rj) in enumerate(inclusions[i + 1:], start=i + 2):
            dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if dist <= ri + rj:
                raise ValueError(
                    f"inclusions {i + 1} and {j} overlap or touch "
                    f"(distance={dist}, radii={ri}+{rj})")

    if verbose:
        print(f"[mesh_generator] Rectangular SENT with circular inclusions: "
              f"W={W}, H={H}, a={a}, n={len(inclusions)}", flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.6f}, "
              f"h_inclusion={h_inclusion:.6f}, h_coarse={h_coarse:.4f}",
              flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')
    notch_eps = min(0.01 * min(W, H), 0.01)

    point_lines = [
        f"Point(1) = {{0, 0, 0, h_coarse}};",
        f"Point(2) = {{W, 0, 0, h_coarse}};",
        f"Point(3) = {{W, H, 0, h_coarse}};",
        f"Point(4) = {{0, H, 0, h_coarse}};",
        f"Point(5) = {{0, H/2 + {notch_eps}, 0, h_crack}};",
        f"Point(6) = {{a, H/2, 0, h_crack}};",
        f"Point(7) = {{0, H/2 - {notch_eps}, 0, h_crack}};",
    ]
    line_lines = [
        "Line(1) = {1, 2};",
        "Line(2) = {2, 3};",
        "Line(3) = {3, 4};",
        "Line(4) = {4, 5};",
        "Line(5) = {5, 6};",
        "Line(6) = {6, 7};",
        "Line(7) = {7, 1};",
        "Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};",
    ]

    matrix_hole_loop_ids = []
    physical_particle_surfaces = []
    circle_curve_ids = []
    next_point = 20
    next_curve = 20
    next_loop = 20
    next_surface = 20
    for idx, (x, y, r) in enumerate(inclusions, start=1):
        c = next_point
        p_r = next_point + 1
        p_t = next_point + 2
        p_l = next_point + 3
        p_b = next_point + 4
        point_lines.extend([
            f"Point({c}) = {{{x}, {y}, 0, h_inclusion}};",
            f"Point({p_r}) = {{{x + r}, {y}, 0, h_inclusion}};",
            f"Point({p_t}) = {{{x}, {y + r}, 0, h_inclusion}};",
            f"Point({p_l}) = {{{x - r}, {y}, 0, h_inclusion}};",
            f"Point({p_b}) = {{{x}, {y - r}, 0, h_inclusion}};",
        ])
        c1, c2, c3, c4 = next_curve, next_curve + 1, next_curve + 2, next_curve + 3
        matrix_loop = next_loop + 1000
        line_lines.extend([
            f"Circle({c1}) = {{{p_r}, {c}, {p_t}}};",
            f"Circle({c2}) = {{{p_t}, {c}, {p_l}}};",
            f"Circle({c3}) = {{{p_l}, {c}, {p_b}}};",
            f"Circle({c4}) = {{{p_b}, {c}, {p_r}}};",
            f"Curve Loop({next_loop}) = {{{c1}, {c2}, {c3}, {c4}}};",
            f"Curve Loop({matrix_loop}) = {{-{c4}, -{c3}, -{c2}, -{c1}}};",
            f"Plane Surface({next_surface}) = {{{next_loop}}};",
            f"Physical Surface(\"particle_{idx}\") = {{{next_surface}}};",
        ])
        matrix_hole_loop_ids.append(matrix_loop)
        physical_particle_surfaces.append(next_surface)
        circle_curve_ids.extend([c1, c2, c3, c4])
        next_point += 5
        next_curve += 4
        next_loop += 1
        next_surface += 1

    holes = ", ".join(str(lid) for lid in matrix_hole_loop_ids)
    matrix_loops = "1" + (", " + holes if holes else "")
    all_particle_surfaces = ", ".join(str(sid) for sid in physical_particle_surfaces)
    field_curves = "5, 6" + (", " + ", ".join(str(cid) for cid in circle_curve_ids)
                             if circle_curve_ids else "")

    geo_content = f"""// Rectangular SENT with conforming circular inclusions - auto-generated
// W={W}, H={H}, a={a}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}
// Includes true particle disk surfaces and a matrix surface with circular holes.

h_crack = {h_crack};
h_coarse = {h_coarse};
h_inclusion = {h_inclusion};
W = {W};
H = {H};
a = {a};
band = {crack_band_width};

{chr(10).join(point_lines)}

{chr(10).join(line_lines)}

Plane Surface(1) = {{{matrix_loops}}};

Physical Curve("bottom") = {{1}};
Physical Curve("right") = {{2}};
Physical Curve("top") = {{3}};
Physical Curve("left") = {{4, 7}};
Physical Curve("notch_upper") = {{5}};
Physical Curve("notch_lower") = {{6}};
Physical Curve("particle_interfaces") = {{{", ".join(str(c) for c in circle_curve_ids)}}};
Physical Surface("matrix") = {{1}};
Physical Surface("particles") = {{{all_particle_surfaces}}};

// Refinement along notch lips and particle interfaces.
Field[1] = Distance;
Field[1].CurvesList = {{{field_curves}}};
Field[1].Sampling = 400;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0;
Field[2].DistMax = band;

// Extra refinement near notch tip.
Field[3] = Distance;
Field[3].PointsList = {{6}};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack * 0.5;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = 5 * {l0};

// Fine mesh in the crack-particle interaction region.
Field[5] = Box;
Field[5].VIn  = h_crack;
Field[5].VOut = h_coarse;
Field[5].XMin = {a - 5.0};
Field[5].XMax = {W};
Field[5].YMin = 0;
Field[5].YMax = {H};

Field[6] = Min;
Field[6].FieldsList = {{2, 4, 5}};
Background Field = 6;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing with OCC fragments...",
              flush=True)

    # Generate the actual mesh through the Gmsh OCC API. The built-in
    # kernel is brittle for a matrix surface with shared disk surfaces:
    # depending on loop orientation it can mesh the particle disk on top of
    # an uncut matrix surface. OCC BooleanFragments gives disjoint matrix
    # and particle surfaces while keeping a conforming interface.
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
        import atexit
        atexit.register(_safe_gmsh_finalize)
    try:
        gmsh.clear()
        gmsh.option.setNumber("General.Verbosity", 2 if verbose else 0)
        gmsh.model.add("rectangular_sent_circular_inclusions")
        occ = gmsh.model.occ

        p1 = occ.addPoint(0, 0, 0)
        p2 = occ.addPoint(W, 0, 0)
        p3 = occ.addPoint(W, H, 0)
        p4 = occ.addPoint(0, H, 0)
        p5 = occ.addPoint(0, H / 2.0 + notch_eps, 0)
        p6 = occ.addPoint(a, H / 2.0, 0)
        p7 = occ.addPoint(0, H / 2.0 - notch_eps, 0)
        lines = [
            occ.addLine(p1, p2),
            occ.addLine(p2, p3),
            occ.addLine(p3, p4),
            occ.addLine(p4, p5),
            occ.addLine(p5, p6),
            occ.addLine(p6, p7),
            occ.addLine(p7, p1),
        ]
        outer_loop = occ.addCurveLoop(lines)
        plate = occ.addPlaneSurface([outer_loop])
        disks = [occ.addDisk(x, y, 0, r, r) for x, y, r in inclusions]
        if disks:
            occ.fragment([(2, plate)], [(2, disk) for disk in disks])
        occ.synchronize()

        surf_tags = [tag for _, tag in gmsh.model.getEntities(2)]
        particle_surfs = []
        matrix_surfs = []
        for tag in surf_tags:
            cx, cy, _ = gmsh.model.occ.getCenterOfMass(2, tag)
            if any((cx - x) ** 2 + (cy - y) ** 2 <= (0.35 * r) ** 2
                   for x, y, r in inclusions):
                particle_surfs.append(tag)
            else:
                matrix_surfs.append(tag)
        if matrix_surfs:
            pg = gmsh.model.addPhysicalGroup(2, matrix_surfs)
            gmsh.model.setPhysicalName(2, pg, "matrix")
        if particle_surfs:
            pg = gmsh.model.addPhysicalGroup(2, particle_surfs)
            gmsh.model.setPhysicalName(2, pg, "particles")

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box, "VIn", h_crack)
        gmsh.model.mesh.field.setNumber(box, "VOut", h_coarse)
        gmsh.model.mesh.field.setNumber(box, "XMin", a - 5.0)
        gmsh.model.mesh.field.setNumber(box, "XMax", W)
        gmsh.model.mesh.field.setNumber(box, "YMin", 0.0)
        gmsh.model.mesh.field.setNumber(box, "YMax", H)
        gmsh.model.mesh.field.setAsBackgroundMesh(box)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(output_path)
        if verbose:
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
            n_elems = sum(len(t) for t in elem_tags)
            print(f"[mesh_generator] Mesh: {len(node_tags)} nodes, "
                  f"{n_elems} triangles", flush=True)
    finally:
        gmsh.clear()
    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic benchmark meshes (B2, B5, B6)
# ─────────────────────────────────────────────────────────────────────────────

def kalthoff_winkler(
    output_path: str = 'kalthoff_winkler.msh',
    W: float = 100.0,
    H: float = 200.0,
    a: float = 50.0,
    notch_y1: float = 125.0,
    notch_y2: float = 75.0,
    l0: float = 0.195,
    h_crack: float = 0.25,
    h_coarse: float = 5.0,
    crack_band: float = 10.0,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Kalthoff-Winkler geometry: W×H plate with two parallel horizontal edge notches.

    Reference: Borden et al. (2012) Section 4.3.

    Geometry
    --------
    - Rectangle: W × H (default 100 × 200 mm, full specimen)
    - Notch 1 (upper): from left edge to x = a, at y = notch_y1 (H/2 + 25)
    - Notch 2 (lower): from left edge to x = a, at y = notch_y2 (H/2 - 25)
    - Impact zone: left edge between notch_y2 < y < notch_y1

    Named physical groups:
    - 'left_impact'  : left edge between notches (impactor region)
    - 'left_top'     : left edge above upper notch
    - 'left_bottom'  : left edge below lower notch
    - 'right', 'top', 'bottom' : remaining outer edges
    """
    eps = min(0.01 * min(W, H), 0.02)
    h_notch = h_crack / 2.0

    geo_path = output_path.replace('.msh', '.geo')
    geo_content = f"""\
// Kalthoff-Winkler impact benchmark (Borden 2012)
// W={W} mm, H={H} mm, notch depth a={a} mm
// Notch 1 (upper) at y={notch_y1}, Notch 2 (lower) at y={notch_y2}

W = {W};  H = {H};  a = {a};
y1 = {notch_y1};  y2 = {notch_y2};
eps = {eps};

Point(1)  = {{0, 0, 0}};
Point(2)  = {{{W}, 0, 0}};
Point(3)  = {{{W}, {H}, 0}};
Point(4)  = {{0, {H}, 0}};

// Notch 2 (lower)
Point(5)  = {{0, y2 + eps/2, 0}};
Point(6)  = {{0, y2 - eps/2, 0}};
Point(7)  = {{a, y2, 0}};

// Notch 1 (upper)
Point(8)  = {{0, y1 + eps/2, 0}};
Point(9)  = {{0, y1 - eps/2, 0}};
Point(10) = {{a, y1, 0}};

Line(1)  = {{1, 6}};       // left bottom
Line(2)  = {{5, 9}};       // left impact zone
Line(3)  = {{8, 4}};       // left top
Line(4)  = {{4, 3}};       // top
Line(5)  = {{3, 2}};       // right
Line(6)  = {{2, 1}};       // bottom
Line(7)  = {{6, 7}};       // notch2 lower wall
Line(8)  = {{7, 5}};       // notch2 upper wall
Line(9)  = {{9, 10}};      // notch1 lower wall
Line(10) = {{10, 8}};      // notch1 upper wall

Curve Loop(1) = {{6, 1, 7, 8, 2, 9, 10, 3, 4, 5}};
Plane Surface(1) = {{1}};

Physical Curve("bottom")       = {{6}};
Physical Curve("right")        = {{5}};
Physical Curve("top")          = {{4}};
Physical Curve("left_top")     = {{3}};
Physical Curve("left_impact")  = {{2}};
Physical Curve("left_bottom")  = {{1}};
Physical Curve("notch1")       = {{9, 10}};
Physical Curve("notch2")       = {{7, 8}};
Physical Surface("plate")      = {{1}};

// Refinement around notch tips (fine at tip, transition to coarse)
Field[1] = Distance;
Field[1].PointsList = {{7, 10}};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = {h_notch};
Field[2].SizeMax = {h_coarse};
Field[2].DistMin = {crack_band};
Field[2].DistMax = {crack_band * 4};

// Inter-notch zone: fine mesh between the two notches (impact region)
Field[3] = Box;
Field[3].VIn  = {h_crack};
Field[3].VOut = {h_coarse};
Field[3].XMin = 0;  Field[3].XMax = {a + crack_band};
Field[3].YMin = {notch_y2 - crack_band};
Field[3].YMax = {notch_y1 + crack_band};

// Crack propagation zones: cracks travel at ~68 deg from each notch tip.
// Upper crack: (a, notch_y1) -> up-right toward (W, H)
// Lower crack: (a, notch_y2) -> down-right toward (W, 0)
// Use a wide band from notch tips to right edge, expanding vertically
// to encompass the full crack path with margin.
// Upper crack zone
Field[4] = Box;
Field[4].VIn  = {h_crack};
Field[4].VOut = {h_coarse};
Field[4].XMin = {a - crack_band};  Field[4].XMax = {W};
Field[4].YMin = {notch_y1 - crack_band};
Field[4].YMax = {H};

// Lower crack zone
Field[5] = Box;
Field[5].VIn  = {h_crack};
Field[5].VOut = {h_coarse};
Field[5].XMin = {a - crack_band};  Field[5].XMax = {W};
Field[5].YMin = 0;
Field[5].YMax = {notch_y2 + crack_band};

Field[6] = Min;
Field[6].FieldsList = {{2, 3, 4, 5}};
Background Field = 6;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def crack_coalescence(
    output_path: str = 'crack_coalescence.msh',
    W: float = 50.0,
    H: float = 50.0,
    a1: float = 10.0,
    y1: float = 16.67,
    a2: float = 10.0,
    y2: float = 33.33,
    l0: float = 0.5,
    h_crack: float = 0.25,
    h_coarse: float = 2.5,
    crack_band: float = 3.0,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Two-notch crack coalescence geometry (B5).

    Geometry
    --------
    - Rectangle: W x H (default 50 x 50 mm)
    - Notch 1 (lower): from left edge (x=0) to x = a1, at y = y1 (H/2 - H/6)
    - Notch 2 (upper): from right edge (x=W) to x = W - a2, at y = y2 (H/2 + H/6)
    - Default notch length: 10 mm each
    - Notch positions: +/-H/3 from centre (y1~16.67, y2~33.33 mm)

    Named physical groups:
    - 'left', 'right', 'top', 'bottom' : outer boundary segments
    - 'notch1', 'notch2'               : notch walls
    """
    eps = min(0.01 * min(W, H), 0.02)
    h_notch = h_crack / 2.0

    geo_path = output_path.replace('.msh', '.geo')
    geo_content = f"""\
// Crack coalescence benchmark (B5)
// Notch1 (lower-left): x=0 to x={a1} at y={y1}
// Notch2 (upper-right): x={W} to x={W-a2} at y={y2}

W = {W};  H = {H};
a1 = {a1};  y1 = {y1};
a2 = {a2};  y2 = {y2};
eps = {eps};

Point(1)  = {{0, 0, 0}};
Point(2)  = {{{W}, 0, 0}};
Point(3)  = {{{W}, {H}, 0}};
Point(4)  = {{0, {H}, 0}};

// Notch 1 (lower-left)
Point(5)  = {{0, y1 + eps/2, 0}};
Point(6)  = {{0, y1 - eps/2, 0}};
Point(7)  = {{a1, y1, 0}};

// Notch 2 (upper-right)
Point(8)  = {{{W}, y2 + eps/2, 0}};
Point(9)  = {{{W}, y2 - eps/2, 0}};
Point(10) = {{W-a2, y2, 0}};

Line(1)  = {{1, 6}};
Line(2)  = {{5, 4}};
Line(3)  = {{4, 3}};
Line(4)  = {{3, 8}};
Line(5)  = {{9, 2}};
Line(6)  = {{2, 1}};
Line(7)  = {{6, 7}};
Line(8)  = {{7, 5}};
Line(9)  = {{8, 10}};
Line(10) = {{10, 9}};

Curve Loop(1) = {{6, 1, 7, 8, 2, 3, 4, 9, 10, 5}};
Plane Surface(1) = {{1}};

Physical Curve("left")    = {{1, 2}};
Physical Curve("top")     = {{3}};
Physical Curve("right")   = {{4, 5}};
Physical Curve("bottom")  = {{6}};
Physical Curve("notch1")  = {{7, 8}};
Physical Curve("notch2")  = {{9, 10}};
Physical Surface("plate") = {{1}};

Field[1] = Distance;
Field[1].PointsList = {{7, 10}};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = {h_notch};
Field[2].SizeMax = {h_coarse};
Field[2].DistMin = {crack_band};
Field[2].DistMax = {crack_band * 4};

Field[3] = Box;
Field[3].VIn  = {h_crack};
Field[3].VOut = {h_coarse};
Field[3].XMin = {a1 - 2.0};  Field[3].XMax = {W - a2 + 2.0};
Field[3].YMin = {y1 - crack_band};  Field[3].YMax = {y2 + crack_band};

Field[4] = Min;
Field[4].FieldsList = {{2, 3}};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def brazilian_disc(
    output_path: str = 'brazilian_disc.msh',
    R: float = 25.0,
    notch_len: float = 5.0,
    contact_half_width: float = 1.5,
    l0: float = 0.5,
    h_crack: float = 0.25,
    h_coarse: float = 2.5,
    crack_band: float = 3.0,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Brazilian disc geometry: circular disc with optional central vertical pre-notch.

    Geometry
    --------
    - Circle of radius R centred at origin (0, 0)
    - Optional vertical pre-notch: thin rectangle from (0, -notch_len) to (0, +notch_len)
    - Loading arcs: top (y ≈ +R) and bottom (y ≈ -R); identified by coordinate in run.py

    Uses OpenCASCADE factory for BooleanDifference (notch subtraction).
    Node sets for loading BCs are selected by coordinate search in run.py.
    """
    import math as _math

    eps = max(0.01 * R, 0.02)

    geo_path = output_path.replace('.msh', '.geo')

    if notch_len > 0:
        geo_content = f"""\
// Brazilian disc with central vertical pre-notch
SetFactory("OpenCASCADE");

Disk(1) = {{0, 0, 0, {R}}};
Rectangle(2) = {{-{eps/2:.6f}, -{notch_len}, 0, {eps:.6f}, {2*notch_len}}};
BooleanDifference{{ Surface{{1}}; Delete; }}{{ Surface{{2}}; Delete; }};

Field[1] = Box;
Field[1].VIn  = {h_crack};
Field[1].VOut = {h_coarse};
Field[1].XMin = {-crack_band};  Field[1].XMax = {crack_band};
Field[1].YMin = {-R};  Field[1].YMax = {R};

Background Field = 1;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""
    else:
        geo_content = f"""\
// Brazilian disc (plain, no notch)
SetFactory("OpenCASCADE");

Disk(1) = {{0, 0, 0, {R}}};

Field[1] = Box;
Field[1].VIn  = {h_crack};
Field[1].VOut = {h_coarse};
Field[1].XMin = {-crack_band};  Field[1].XMax = {crack_band};
Field[1].YMin = {-R};  Field[1].YMax = {R};

Background Field = 1;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


def glass_impact_vnotch(
    output_path: str,
    W: float = 100.0,
    H_half: float = 37.5,
    notch_depth: float = 19.0,
    crack_len: float = 14.0,
    notch_mouth_half: float = 5.0,
    l0: float = 0.25,
    h_crack: Optional[float] = None,
    h_coarse: float = 1.0,
    crack_band: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """V-notch glass plate for impact experiments (upper half with symmetry).

    Reference: arXiv:2411.16393, Section 5.2.

    Geometry (upper half, symmetry at y=0)
    --------
    - Domain: W x H_half (default 100 x 37.5 mm)
    - V-notch on left edge: mouth at (0, notch_mouth_half), tip at (notch_depth, 0)
    - Initial crack: from notch tip rightward along y=0 for crack_len mm
    - Symmetry plane: y = 0 (bottom edge)

    Named physical groups:
    - 'symmetry'     : bottom edge (y=0), right of crack tip
    - 'crack'        : bottom edge along initial crack (y=0, x < notch_depth + crack_len)
    - 'notch'        : V-notch upper lip (left edge to notch tip)
    - 'left'         : left edge above notch mouth
    - 'top', 'right' : remaining outer edges
    - 'plate'        : surface
    """
    if h_crack is None:
        h_crack = l0 / 4.0
    if crack_band is None:
        crack_band = 5 * l0

    crack_tip_x = notch_depth + crack_len
    h_notch = h_crack / 2.0

    geo_path = output_path.replace('.msh', '.geo')
    geo_content = f"""\
// Glass impact V-notch (upper half, symmetry at y=0)
// arXiv:2411.16393, Section 5.2
// W={W} mm, H_half={H_half} mm
// V-notch depth={notch_depth} mm, crack={crack_len} mm

// --- Points ---
Point(1)  = {{{W}, 0, 0}};                // bottom-right
Point(2)  = {{{W}, {H_half}, 0}};         // top-right
Point(3)  = {{0, {H_half}, 0}};           // top-left
Point(4)  = {{0, {notch_mouth_half}, 0}}; // notch mouth
Point(5)  = {{{notch_depth}, 0, 0}};      // notch tip
Point(6)  = {{{crack_tip_x}, 0, 0}};      // crack tip

// --- Lines ---
Line(1)  = {{6, 1}};       // symmetry (right of crack tip)
Line(2)  = {{1, 2}};       // right edge
Line(3)  = {{2, 3}};       // top edge
Line(4)  = {{3, 4}};       // left edge (above notch)
Line(5)  = {{4, 5}};       // V-notch upper lip
Line(6)  = {{5, 6}};       // initial crack (along symmetry plane)

Curve Loop(1) = {{1, 2, 3, 4, 5, 6}};
Plane Surface(1) = {{1}};

Physical Curve("symmetry")  = {{1}};
Physical Curve("right")     = {{2}};
Physical Curve("top")       = {{3}};
Physical Curve("left")      = {{4}};
Physical Curve("notch")     = {{5}};
Physical Curve("crack")     = {{6}};
Physical Surface("plate")   = {{1}};

// --- Refinement ---
Field[1] = Distance;
Field[1].PointsList = {{5, 6}};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = {h_notch};
Field[2].SizeMax = {h_coarse};
Field[2].DistMin = {crack_band};
Field[2].DistMax = {crack_band * 4};

Field[3] = Box;
Field[3].VIn  = {h_crack};
Field[3].VOut = {h_coarse};
Field[3].XMin = {notch_depth - crack_band};
Field[3].XMax = {W};
Field[3].YMin = 0;
Field[3].YMax = {crack_band * 2};

Field[4] = Distance;
Field[4].CurvesList = {{5, 6}};

Field[5] = Threshold;
Field[5].InField = 4;
Field[5].SizeMin = {h_crack};
Field[5].SizeMax = {h_coarse};
Field[5].DistMin = {crack_band};
Field[5].DistMax = {crack_band * 3};

Field[6] = Min;
Field[6].FieldsList = {{2, 3, 5}};
Background Field = 6;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Perforated SENT plate (B7 — Bleyer et al. 2017 Section 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def perforated_sent(
    output_path: str = 'perforated_sent.msh',
    W: float = 32.0,
    H: float = 16.0,
    a: float = 4.0,
    notch_y: float = 8.0,
    n_holes: int = 30,
    hole_diameter: float = 0.4,
    hole_spacing: float = 0.9,
    hole_start_x: Optional[float] = None,
    hole_offset_y: float = 0.0,
    l0: float = 0.1,
    h_crack: Optional[float] = None,
    h_coarse: Optional[float] = None,
    h_hole: Optional[float] = None,
    crack_band_width: Optional[float] = None,
    order: int = 1,
    verbose: bool = True,
) -> str:
    """Generate mesh for a perforated SENT plate (Bleyer et al. 2017, Sec 4.2).

    Geometry: W x H rectangle with horizontal edge notch from left to x = a
    at y = notch_y, plus n_holes circular holes of diameter hole_diameter
    placed on the mid-plane (y = notch_y) ahead of the crack tip.

    The holes are centered at:
        x_i = hole_start_x + i * hole_spacing,  y_i = notch_y
    for i = 0, 1, ..., n_holes-1.

    Parameters
    ----------
    output_path : str
        Path for output .msh file.
    W : float
        Plate width (mm). Default 100.0.
    H : float
        Plate height (mm). Default 40.0.
    a : float
        Notch length from left edge (mm). Default 50.0.
    notch_y : float
        Notch y-coordinate (mm). Default 20.0 (= H/2).
    n_holes : int
        Number of circular holes. Default 30.
    hole_diameter : float
        Diameter of each hole (mm). Default 0.4.
    hole_spacing : float
        Center-to-center spacing between holes (mm). Default 0.9.
    hole_start_x : float or None
        x-coordinate of first hole center. Default: a + 1.0 (1 mm ahead
        of notch tip).
    l0 : float
        Phase-field regularization length (mm).
    h_crack : float or None
        Element size in crack/hole zone. Default l0/2.
    h_coarse : float or None
        Element size far from crack. Default max(W,H)/20.
    h_hole : float or None
        Element size on hole boundaries. Default h_crack.
    crack_band_width : float or None
        Half-width of refinement band around the crack path. Default 5*l0.
    order : int
        Element order (1=linear).
    verbose : bool

    Returns
    -------
    output_path : str
    """
    if h_crack is None:
        h_crack = l0 / 2.0
    if h_coarse is None:
        h_coarse = max(W, H) / 20.0
    if h_hole is None:
        h_hole = h_crack
    if crack_band_width is None:
        crack_band_width = 5.0 * l0
    if hole_start_x is None:
        hole_start_x = a + 1.0

    hole_r = hole_diameter / 2.0

    if verbose:
        print(f"[mesh_generator] Perforated SENT: W={W}, H={H}, a={a}, "
              f"l0={l0}", flush=True)
        print(f"[mesh_generator]   {n_holes} holes: D={hole_diameter} mm, "
              f"spacing={hole_spacing} mm, start_x={hole_start_x} mm",
              flush=True)
        print(f"[mesh_generator]   h_crack={h_crack:.6f}, "
              f"h_coarse={h_coarse:.4f}, h_hole={h_hole:.6f}, "
              f"band={crack_band_width:.4f}", flush=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    geo_path = output_path.replace('.msh', '.geo')

    notch_eps = min(0.01 * min(W, H), 0.01)

    # Use OpenCASCADE geometry kernel for clean boolean operations.
    # Build a rectangle, cut notch and holes with BooleanDifference.
    # This avoids the edge-recovery problems that arise when circle
    # arcs intersect the crack-path line in the built-in kernel.

    # Build hole Disk commands (OpenCASCADE)
    hole_disk_lines = []
    for i in range(n_holes):
        cx = hole_start_x + i * hole_spacing
        # Disk(tag) = {cx, cy, cz, rx, ry};
        # Tag starts at 2 (surface 1 is the plate)
        hole_disk_lines.append(
            f"Disk({i + 10}) = {{{cx}, {notch_y + hole_offset_y}, 0, {hole_r}}};"
        )

    # List of hole surface tags for BooleanDifference
    # In OCC .geo syntax, multiple surfaces are separated by '; '
    hole_tags_list = '; '.join(f'Surface{{{i + 10}}}'
                               for i in range(n_holes))

    # After BooleanDifference, all hole curves and the plate boundary
    # are re-tagged by OCC. We use Physical groups with bounding-box
    # queries via Gmsh built-in field approach instead.

    geo_content = f"""// Perforated SENT plate (Bleyer et al. 2017 Sec 4.2) - auto-generated
// W={W}, H={H}, a={a}, l0={l0}, h_crack={h_crack}, h_coarse={h_coarse}
// {n_holes} holes: D={hole_diameter}, spacing={hole_spacing}, start_x={hole_start_x}
// Uses OpenCASCADE kernel for boolean difference (holes on crack path)

SetFactory("OpenCASCADE");

// ---- Plate with V-notch ----
// Build the plate as a rectangle, then cut the notch as a thin rectangle
Rectangle(1) = {{0, 0, 0, {W}, {H}}};

// V-notch: thin rectangle from left edge to x=a at y=notch_y
// Height = 2*notch_eps (very thin slit)
Rectangle(2) = {{0, {notch_y - notch_eps}, 0, {a}, {2 * notch_eps}}};

// Cut notch from plate
BooleanDifference(3) = {{ Surface{{1}}; Delete; }}{{ Surface{{2}}; Delete; }};

// ---- Holes ----
{chr(10).join(hole_disk_lines)}

// Cut all holes from the notched plate
BooleanDifference(100) = {{ Surface{{3}}; Delete; }}{{ {hole_tags_list}; Delete; }};

// ---- Physical groups (use bounding-box selection) ----
// After OCC boolean ops, entity tags change. We select boundaries
// by geometric location.

// We use the resulting surface (tag 100) as the plate
Physical Surface("plate") = {{100}};

// Boundary physical groups are assigned after meshing via
// transfinite or by the solver's identify_boundaries() which
// uses coordinate-based node set detection.

// ---- Refinement fields ----
// Field 1-2: band refinement along the notch plane y=notch_y
Field[1] = Box;
Field[1].VIn  = {h_crack};
Field[1].VOut = {h_coarse};
Field[1].XMin = 0;
Field[1].XMax = {W};
Field[1].YMin = {notch_y - crack_band_width};
Field[1].YMax = {notch_y + crack_band_width};

// Field 2: extra refinement near notch tip
Field[2] = Ball;
Field[2].VIn  = {h_crack * 0.5};
Field[2].VOut = {h_coarse};
Field[2].XCenter = {a};
Field[2].YCenter = {notch_y};
Field[2].Radius = {5 * l0};

// Field 3: box refinement covering the hole zone
Field[3] = Box;
Field[3].VIn  = {h_hole};
Field[3].VOut = {h_coarse};
Field[3].XMin = {hole_start_x - 1.0};
Field[3].XMax = {hole_start_x + (n_holes - 1) * hole_spacing + 1.0};
Field[3].YMin = {notch_y - 2.0};
Field[3].YMax = {notch_y + 2.0};

Field[4] = Min;
Field[4].FieldsList = {{1, 2, 3}};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.CharacteristicLengthMin = {h_crack * 0.3};
Mesh.CharacteristicLengthMax = {h_coarse};
Mesh.Algorithm = 6;
Mesh.ElementOrder = {order};
"""

    with open(geo_path, 'w') as f:
        f.write(geo_content)

    if verbose:
        print(f"[mesh_generator] Wrote {geo_path}, meshing...", flush=True)

    _run_gmsh(geo_path, output_path, verbose=verbose)

    if verbose:
        print(f"[mesh_generator] Saved: {output_path}", flush=True)
    return output_path
