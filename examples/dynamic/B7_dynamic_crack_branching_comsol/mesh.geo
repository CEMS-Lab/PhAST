// B7 dynamic crack branching, COMSOL full-plate equivalent.
// Public Gmsh recipe matching config.yaml geometry.
// Units: mm.

h_crack  = 0.125;
h_coarse = 1.0;
W = 100.0;
H = 40.0;
a = 50.0;
eps = 0.01;

Point(1) = {0, 0, 0, h_coarse};
Point(2) = {W, 0, 0, h_coarse};
Point(3) = {W, H, 0, h_coarse};
Point(4) = {0, H, 0, h_coarse};

Point(5) = {0, H/2 - eps, 0, h_crack};
Point(6) = {a, H/2, 0, h_crack};
Point(7) = {0, H/2 + eps, 0, h_crack};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 7};
Line(5) = {7, 6};
Line(6) = {6, 5};
Line(7) = {5, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

Field[1] = Distance;
Field[1].CurvesList = {5, 6};
Field[1].Sampling = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_crack;
Field[2].SizeMax = h_coarse;
Field[2].DistMin = 0.0;
Field[2].DistMax = 2.5;

Field[3] = Box;
Field[3].VIn = h_crack;
Field[3].VOut = h_coarse;
Field[3].XMin = 45.0;
Field[3].XMax = W;
Field[3].YMin = 0.0;
Field[3].YMax = H;
Field[3].Thickness = 1.0;

Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;

Physical Surface("plate") = {1};
Physical Curve("bottom") = {1};
Physical Curve("right") = {2};
Physical Curve("top") = {3};
Physical Curve("left") = {4, 7};
Physical Curve("notch.boundary") = {5, 6};

Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
