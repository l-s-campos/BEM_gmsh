// Parameters
lc = 0.4;  // Element characteristic length (adjust for finer/coarser mesh)
SetFactory("OpenCASCADE");
// Cube geometry: 8 vertices
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};
Point(5) = {0, 0, 1, lc};
Point(6) = {1, 0, 1, lc};
Point(7) = {1, 1, 1, lc};
Point(8) = {0, 1, 1, lc};

// Edges (12 lines)
Line(1) = {1, 2}; // bottom face
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6}; // top face
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5}; // vertical edges
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Faces (6 curve loops)
Curve Loop(1) = {1, 2, 3, 4};    // bottom face
Curve Loop(2) = {5, 6, 7, 8};    // top face
Curve Loop(3) = {1, 10, -5, -9}; // front face
Curve Loop(4) = {2, 11, -6, -10}; // right face
Curve Loop(5) = {3, 12, -7, -11}; // back face
Curve Loop(6) = {4, 9, -8, -12};  // left face

// Surfaces
Plane Surface(1) = {1}; // bottom
Plane Surface(2) = {2}; // top
Plane Surface(3) = {3}; // front
Plane Surface(4) = {4}; // right
Plane Surface(5) = {5}; // back
Plane Surface(6) = {6}; // left

// Volume
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// Transfinite curves and surfaces for structured mesh
Transfinite Curve {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} = 11;
Transfinite Surface {1, 2, 3, 4, 5, 6};
Transfinite Volume {1};

Recombine Surface {1, 2, 3, 4, 5, 6};

// Physical groups
Physical Surface("0;0") = {1};
Physical Surface("0;1") = {2};
Physical Surface("1;0") = {3,4,5,6};
Physical Volume("Domain") = {1};

// Mesh generation
Mesh 3;
ReorientMesh Volume {1};
