// =====================================================================
// Notched holed plate -- COMSOL Application Library 6.4
//   "Brittle Fracture of a Holed Plate" (Geomechanics Module)
// Original phase-field formulation: Ambati, Gerasimov & De Lorenzis
// (2015), Comput. Mech. 55, 383-405.
//
// Geometry (mm):
//   Plate           : 65 W x 120 H, plane stress, thickness 1 mm
//   Notch           : 0.5 mm tall x 10 mm long, left edge, y_centre = 65
//   Large hole      : r = 10 mm, centre (36.5, 51)
//   Upper pin hole  : r = 5  mm, centre (20, 100)   (Physical "upper_pin")
//   Lower pin hole  : r = 5  mm, centre (20,  20)   (Physical "lower_pin")
//
// Loading kinematic (rigid-pin approximation, see README):
//   prescribed +u_y on upper_pin, -u_y on lower_pin.
//
// Refinement: a horizontal band (y in [45, 70]) at h_fine; bulk h_coarse.
// =====================================================================

h_fine    = 0.30;   // ~ 1.2 * l0 in the crack band
h_pin     = 1.0;    // around the loaded pin holes
h_coarse  = 4.0;    // bulk

// ---- Outer boundary (CCW, with rectangular notch slit on the left) ----
Point(1) = {0.0,  0.0,   0, h_coarse};
Point(2) = {65.0, 0.0,   0, h_coarse};
Point(3) = {65.0, 120.0, 0, h_coarse};
Point(4) = {0.0,  120.0, 0, h_coarse};
Point(5) = {0.0,  65.25, 0, h_fine};   // top of notch on left edge
Point(6) = {10.0, 65.25, 0, h_fine};   // top corner at notch tip
Point(7) = {10.0, 64.75, 0, h_fine};   // bottom corner at notch tip
Point(8) = {0.0,  64.75, 0, h_fine};   // bottom of notch on left edge

Line(1) = {1, 2};   // bottom
Line(2) = {2, 3};   // right
Line(3) = {3, 4};   // top
Line(4) = {4, 5};   // left-upper (above notch)
Line(5) = {5, 6};   // notch top wall
Line(6) = {6, 7};   // notch tip (vertical)
Line(7) = {7, 8};   // notch bottom wall
Line(8) = {8, 1};   // left-lower (below notch)

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};

// ---- Large hole (centre 36.5, 51, r=10) ----
Point(100) = {36.5, 51.0, 0, h_fine};        // centre
Point(101) = {46.5, 51.0, 0, h_fine};        // east
Point(102) = {36.5, 61.0, 0, h_fine};        // north
Point(103) = {26.5, 51.0, 0, h_fine};        // west
Point(104) = {36.5, 41.0, 0, h_fine};        // south
Circle(100) = {101, 100, 102};
Circle(101) = {102, 100, 103};
Circle(102) = {103, 100, 104};
Circle(103) = {104, 100, 101};
Curve Loop(10) = {100, 101, 102, 103};

// ---- Upper pin hole (20, 100, r=5) ----
Point(200) = {20.0, 100.0, 0, h_pin};
Point(201) = {25.0, 100.0, 0, h_pin};
Point(202) = {20.0, 105.0, 0, h_pin};
Point(203) = {15.0, 100.0, 0, h_pin};
Point(204) = {20.0,  95.0, 0, h_pin};
Circle(110) = {201, 200, 202};
Circle(111) = {202, 200, 203};
Circle(112) = {203, 200, 204};
Circle(113) = {204, 200, 201};
Curve Loop(11) = {110, 111, 112, 113};

// ---- Lower pin hole (20, 20, r=5) ----
Point(300) = {20.0, 20.0, 0, h_pin};
Point(301) = {25.0, 20.0, 0, h_pin};
Point(302) = {20.0, 25.0, 0, h_pin};
Point(303) = {15.0, 20.0, 0, h_pin};
Point(304) = {20.0, 15.0, 0, h_pin};
Circle(120) = {301, 300, 302};
Circle(121) = {302, 300, 303};
Circle(122) = {303, 300, 304};
Circle(123) = {304, 300, 301};
Curve Loop(12) = {120, 121, 122, 123};

Plane Surface(1) = {1, 10, 11, 12};

// ---- Physical groups (become node sets in FEMMesh) ----
Physical Curve("bottom")     = {1};
Physical Curve("right")      = {2};
Physical Curve("top")        = {3};
Physical Curve("notch")      = {5, 6, 7};
Physical Curve("big_hole")   = {100, 101, 102, 103};
Physical Curve("upper_pin")  = {110, 111, 112, 113};
Physical Curve("lower_pin")  = {120, 121, 122, 123};
Physical Surface("plate")    = {1};

// ---- Pin centres ------------------------------------------
// Isolated Physical Points at each pin-hole centre. They are *not*
// embedded in Plane Surface(1) (they fall inside the excluded hole loops),
// but gmsh still emits them as `vertex` cells which meshio reads and
// FEMMesh exposes as named node sets. These nodes carry DOFs but have
// no element connectivity, so they contribute zero stiffness to the
// global system -- exactly what a kinematic control point should do.
// They are the master nodes for the pin-hole rigid_connector BCs:
// COMSOL's rigid connector locks (u_x, u_y) at the pin centre, not at
// an arbitrary boundary node, and reproducing that geometry is what
// makes the load-displacement curve match the reference.
Physical Point("upper_pin_centre") = {200};
Physical Point("lower_pin_centre") = {300};

// ---- Refinement: band around expected crack path + the big hole ----
Field[1] = Box;
Field[1].VIn  = h_fine;
Field[1].VOut = h_coarse;
Field[1].XMin = 0;
Field[1].XMax = 65.0;
Field[1].YMin = 45.0;
Field[1].YMax = 70.0;

Field[2] = Distance;
Field[2].CurvesList = {100, 101, 102, 103};
Field[2].Sampling = 80;

Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = h_fine;
Field[3].SizeMax = h_coarse;
Field[3].DistMin = 0;
Field[3].DistMax = 8.0;

// Pin-hole refinement: ensure enough discrete nodes around each pin so
// the (kinematic) rigid-pin BC is well-sampled. Circumference 2*pi*5 ~ 31 mm,
// so SizeMin=1.0 gives ~31 nodes/pin.
Field[5] = Distance;
Field[5].CurvesList = {110, 111, 112, 113, 120, 121, 122, 123};
Field[5].Sampling = 60;

Field[6] = Threshold;
Field[6].InField = 5;
Field[6].SizeMin = h_pin;
Field[6].SizeMax = h_coarse;
Field[6].DistMin = 0;
Field[6].DistMax = 6.0;

Field[4] = Min;
Field[4].FieldsList = {1, 3, 6};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;        // Frontal-Delaunay
Mesh.ElementOrder = 1;
