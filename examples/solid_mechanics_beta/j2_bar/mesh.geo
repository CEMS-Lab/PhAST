// J2 plasticity waisted bar.
// Public Gmsh recipe matching the config.yaml dimensions and the Python mesh
// generator's Gaussian waist profile. Units: m.

L = 1.0;
H = 0.25;
nx = 18;
h = L / nx;
waist_depth = 0.35;
waist_width_fraction = 0.18;
waist_width = waist_width_fraction * L;
profile_points = 72;

bottom_curves[] = {};
top_curves[] = {};

For i In {0:profile_points}
  x = L * i / profile_points;
  local_height = H * (1.0 - waist_depth * Exp(-((x - 0.5 * L) / waist_width)^2));
  Point(1000 + i) = {x, -0.5 * local_height, 0, h};
  Point(2000 + i) = {x,  0.5 * local_height, 0, h};
EndFor

For i In {0:profile_points - 1}
  Line(1000 + i) = {1000 + i, 1000 + i + 1};
  bottom_curves[] += {1000 + i};

  Line(2000 + i) = {2000 + i + 1, 2000 + i};
  top_curves[] += {2000 + i};
EndFor

right_curve = 3000;
left_curve = 3001;
Line(right_curve) = {1000 + profile_points, 2000 + profile_points};
Line(left_curve) = {2000, 1000};

Curve Loop(1) = {bottom_curves[], right_curve, top_curves[], left_curve};
Plane Surface(1) = {1};

Physical Surface("body") = {1};
Physical Curve("bottom") = {bottom_curves[]};
Physical Curve("right") = {right_curve};
Physical Curve("top") = {top_curves[]};
Physical Curve("left") = {left_curve};

Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeExtendFromBoundary = 1;
