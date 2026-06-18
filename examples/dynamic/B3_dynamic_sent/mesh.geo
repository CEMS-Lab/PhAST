// B3 dynamic SENT benchmark geometry
// L=40.0, a=20.0, l0=0.5, h_crack=0.25, h_coarse=2.0
// V-notch opening: +/-0.001 mm

h_crack = 0.25;
h_coarse = 2.0;
L = 40.0;
a = 20.0;
band = 1.5;

// Points — V-shaped notch with separate upper/lower lips
//   P4 -------- P3
//   |            |
//   P5 \        |
//        > P7   P8
//   P6 //        |
//   |            |
//   P1 -------- P2
Point(1) = {0, 0, 0, h_coarse};               // bottom-left
Point(2) = {L, 0, 0, h_coarse};               // bottom-right
Point(3) = {L, L, 0, h_coarse};               // top-right
Point(4) = {0, L, 0, h_coarse};               // top-left
Point(5) = {0, L/2 + 0.001, 0, h_crack};   // notch mouth upper
Point(6) = {0, L/2 - 0.001, 0, h_crack};   // notch mouth lower
Point(7) = {a, L/2, 0, h_crack};              // notch tip
Point(8) = {L, L/2, 0, h_crack};              // crack path end

// Boundary + notch lines
Line(1) = {1, 2};   // bottom
Line(2) = {2, 8};   // right-lower
Line(3) = {8, 3};   // right-upper
Line(4) = {3, 4};   // top
Line(5) = {4, 5};   // left-upper (above notch)
Line(6) = {5, 7};   // notch upper lip
Line(7) = {7, 6};   // notch lower lip
Line(8) = {6, 1};   // left-lower (below notch)
Line(9) = {7, 8};   // crack path (expected propagation)

// Two surfaces — separated by notch, joined at crack path
Curve Loop(1) = {1, 2, -9, 7, 8};    // lower half
Plane Surface(1) = {1};
Curve Loop(2) = {9, 3, 4, 5, 6};     // upper half
Plane Surface(2) = {2};

// Physical groups
Physical Curve("bottom") = {1};
Physical Curve("right") = {2, 3};
Physical Curve("top") = {4};
Physical Curve("left") = {5, 8};
Physical Curve("notch_upper") = {6};
Physical Curve("notch_lower") = {7};
Physical Surface("plate") = {1, 2};

// Refinement fields
Field[1] = Distance;
Field[1].CurvesList = {6, 7, 9};
Field[1].Sampling = 100;

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
Field[4].DistMax = 5 * 0.5;

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;  // Frontal-Delaunay
Mesh.ElementOrder = 1;
