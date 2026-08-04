// Unit square for BEM / IGA tests (Laplace)
// BC names for format2d: "type;value"  (q = -k dT/dn)
lc = 0.1;
ndiv = 10;

Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

Line(1) = {1, 2}; // bottom q=0
Line(2) = {2, 3}; // right  q=-1
Line(3) = {3, 4}; // top    q=0
Line(4) = {4, 1}; // left   T=0

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Transfinite Curve {1, 2, 3, 4} = ndiv;
Transfinite Surface {1};
Recombine Surface {1};

Physical Curve("1;0", 1) = {1, 3};
Physical Curve("1;-1", 2) = {2};
Physical Curve("0;0", 3) = {4};
Physical Surface("Domain", 4) = {1};

Mesh 2;
