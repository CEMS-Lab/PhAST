// Perforated SENT plate (Bleyer et al. 2017 Sec 4.2) - auto-generated
// W=32.0, H=16.0, a=4.0, l0=0.1, h_crack=0.025, h_coarse=1.0
// 30 holes: D=0.4, spacing=0.9, start_x=5.0
// Uses OpenCASCADE kernel for boolean difference (holes on crack path)

SetFactory("OpenCASCADE");

// ---- Plate with V-notch ----
// Build the plate as a rectangle, then cut the notch as a thin rectangle
Rectangle(1) = {0, 0, 0, 32.0, 16.0};

// V-notch: thin rectangle from left edge to x=a at y=notch_y
// Height = 2*notch_eps (very thin slit)
Rectangle(2) = {0, 7.99, 0, 4.0, 0.02};

// Cut notch from plate
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };

// ---- Holes ----
Disk(10) = {5.0, 8.0, 0, 0.2};
Disk(11) = {5.9, 8.0, 0, 0.2};
Disk(12) = {6.8, 8.0, 0, 0.2};
Disk(13) = {7.7, 8.0, 0, 0.2};
Disk(14) = {8.6, 8.0, 0, 0.2};
Disk(15) = {9.5, 8.0, 0, 0.2};
Disk(16) = {10.4, 8.0, 0, 0.2};
Disk(17) = {11.3, 8.0, 0, 0.2};
Disk(18) = {12.2, 8.0, 0, 0.2};
Disk(19) = {13.1, 8.0, 0, 0.2};
Disk(20) = {14.0, 8.0, 0, 0.2};
Disk(21) = {14.9, 8.0, 0, 0.2};
Disk(22) = {15.8, 8.0, 0, 0.2};
Disk(23) = {16.700000000000003, 8.0, 0, 0.2};
Disk(24) = {17.6, 8.0, 0, 0.2};
Disk(25) = {18.5, 8.0, 0, 0.2};
Disk(26) = {19.4, 8.0, 0, 0.2};
Disk(27) = {20.3, 8.0, 0, 0.2};
Disk(28) = {21.2, 8.0, 0, 0.2};
Disk(29) = {22.1, 8.0, 0, 0.2};
Disk(30) = {23.0, 8.0, 0, 0.2};
Disk(31) = {23.900000000000002, 8.0, 0, 0.2};
Disk(32) = {24.8, 8.0, 0, 0.2};
Disk(33) = {25.7, 8.0, 0, 0.2};
Disk(34) = {26.6, 8.0, 0, 0.2};
Disk(35) = {27.5, 8.0, 0, 0.2};
Disk(36) = {28.400000000000002, 8.0, 0, 0.2};
Disk(37) = {29.3, 8.0, 0, 0.2};
Disk(38) = {30.2, 8.0, 0, 0.2};
Disk(39) = {31.1, 8.0, 0, 0.2};

// Cut all holes from the notched plate
BooleanDifference(100) = { Surface{3}; Delete; }{ Surface{10}; Surface{11}; Surface{12}; Surface{13}; Surface{14}; Surface{15}; Surface{16}; Surface{17}; Surface{18}; Surface{19}; Surface{20}; Surface{21}; Surface{22}; Surface{23}; Surface{24}; Surface{25}; Surface{26}; Surface{27}; Surface{28}; Surface{29}; Surface{30}; Surface{31}; Surface{32}; Surface{33}; Surface{34}; Surface{35}; Surface{36}; Surface{37}; Surface{38}; Surface{39}; Delete; };

// ---- Physical groups (use bounding-box selection) ----
// After OCC boolean ops, entity tags change. We select boundaries
// by geometric location.

// We use the resulting surface (tag 100) as the plate
Physical Surface("plate") = {100};

// Boundary physical groups are assigned after meshing via
// transfinite or by the solver's identify_boundaries() which
// uses coordinate-based node set detection.

// ---- Refinement fields ----
// Field 1-2: band refinement along the notch plane y=notch_y
Field[1] = Box;
Field[1].VIn  = 0.025;
Field[1].VOut = 1.0;
Field[1].XMin = 0;
Field[1].XMax = 32.0;
Field[1].YMin = 7.5;
Field[1].YMax = 8.5;

// Field 2: extra refinement near notch tip
Field[2] = Ball;
Field[2].VIn  = 0.0125;
Field[2].VOut = 1.0;
Field[2].XCenter = 4.0;
Field[2].YCenter = 8.0;
Field[2].Radius = 0.5;

// Field 3: box refinement covering the hole zone
Field[3] = Box;
Field[3].VIn  = 0.025;
Field[3].VOut = 1.0;
Field[3].XMin = 4.0;
Field[3].XMax = 32.1;
Field[3].YMin = 6.0;
Field[3].YMax = 10.0;

Field[4] = Min;
Field[4].FieldsList = {1, 2, 3};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.CharacteristicLengthMin = 0.0075;
Mesh.CharacteristicLengthMax = 1.0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
