// Parameters
lc = 0.1;  // Element characteristic length (adjust for finer/coarser mesh)

// Unit square geometry from image
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};


// Boundary lines
Line(1) = {1, 2}; // bottom T=0
Line(2) = {2, 3}; // right T=1
Line(3) = {3, 4}; // top flux
Line(4) = {4, 1}; // left T=0

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Transfinite Curve {1, 2, 3, 4} = 10;
Transfinite Surface{1};

// // tres faces isolada. a superior aquecida para 1
// Physical Curve("1;0") = {1,2,4}; 
// Physical Curve("0;1") = {3}; 

// 1d
Physical Curve("1;0") = {1,3}; 
// // Physical Curve("0;1") = {2}; 
Physical Curve("1;-1") = {2}; 
Physical Curve("0;0") = {4}; 



Physical Surface("Domain") = {1};

// Mesh
Mesh 2;
RecombineMesh;