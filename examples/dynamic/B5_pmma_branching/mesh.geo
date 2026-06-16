// Rectangular SENT benchmark - auto-generated
// W=32.0, H=16.0, a=4.0, l0=0.1, h_crack=0.025, h_coarse=1.0
// Horizontal notch from left to x=a at y=H/2
// branching=True

h_crack = 0.025;
h_coarse = 1.0;
W = 32.0;
H = 16.0;
a = 4.0;
band = 0.5;

Point(1) = {0, 0, 0, h_coarse};
Point(2) = {W, 0, 0, h_coarse};
Point(3) = {W, H, 0, h_coarse};
Point(4) = {0, H, 0, h_coarse};
Point(5) = {0, H/2 + 0.01, 0, h_crack};
Point(6) = {0, H/2 - 0.01, 0, h_crack};
Point(7) = {a, H/2, 0, h_crack};
Point(8) = {W, H/2, 0, h_crack};

Line(1) = {1, 2};
Line(2) = {2, 8};
Line(3) = {8, 3};
Line(4) = {3, 4};
Line(5) = {4, 5};
Line(6) = {5, 7};
Line(7) = {7, 6};
Line(8) = {6, 1};
Line(9) = {7, 8};

Curve Loop(1) = {1, 2, -9, 7, 8};
Plane Surface(1) = {1};
Curve Loop(2) = {9, 3, 4, 5, 6};
Plane Surface(2) = {2};

Physical Curve("bottom") = {1};
Physical Curve("right") = {2, 3};
Physical Curve("top") = {4};
Physical Curve("left") = {5, 8};
Physical Curve("notch_upper") = {6};
Physical Curve("notch_lower") = {7};
Physical Surface("plate") = {1, 2};

// Refinement along notch and crack path
Field[1] = Distance;
Field[1].CurvesList = {6, 7, 9};
Field[1].Sampling = 300;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0;
Field[2].DistMax = band;

// Extra refinement at notch tip
Field[3] = Distance;
Field[3].PointsList = {7};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_crack * 0.5;
Field[4].SizeMax = h_coarse;
Field[4].DistMin = 0;
Field[4].DistMax = 5 * 0.1;

// Branching zone: uniform h_crack mesh in entire right half
// Crack branches diverge at ~30deg from horizontal — need fine mesh
// covering the full plate height beyond the notch tip.
Field[6] = Box;
Field[6].VIn  = h_crack;
Field[6].VOut = h_coarse;
Field[6].XMin = -1.0;
Field[6].XMax = 32.0;
Field[6].YMin = 0;
Field[6].YMax = 16.0;

Field[7] = Min;
Field[7].FieldsList = {2, 4, 6};
Background Field = 7;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
