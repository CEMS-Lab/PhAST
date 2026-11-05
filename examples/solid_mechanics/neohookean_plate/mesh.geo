// Neo-Hookean cantilever plate.
// Public Gmsh recipe matching config.yaml mesh dimensions.
// Units: m.

L = 1.0;
H = 0.2;
h = L / 20.0;

Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, H, 0, h};
Point(4) = {0, H, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Surface("body") = {1};
Physical Curve("bottom") = {1};
Physical Curve("right") = {2};
Physical Curve("top") = {3};
Physical Curve("left") = {4};

Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeExtendFromBoundary = 1;
