function dyncomp_octave
% Headless Dynamic_Composite_Plate / test/EjemploDin_10.m, static.
% SS square [0,1]^2, [0/90/90/0], q=1, 2 quadratic els/edge, 1 centre internal.
% Domain Q,M: MATLAB DomainTermsRIM (aprfun=3 = r^2 log r) with RIM Gauss
% coarsened 8/12 -> 4/6 so Octave can finish. Wang KernelP unchanged.
more off;
warning('off', 'all');
thisdir = fileparts(mfilename('fullpath'));
matdir = '/home/lsc/Downloads/BEM_Plate_Shell_Book_Juseche-main/Dynamic_Composite_Plate';
addpath(fullfile(thisdir, 'dyncomp_octave'));  % patched DomainTermsRIM first
addpath(matdir);

global q CTE AT D K_Amatriz NPGIC NPGSF NPGID TOL DivElem rho hpl

PONTOS = [1  0.0  0.0
          2  1.0  0.0
          3  1.0  1.0
          4  0.0  1.0];
nel = 2;
LINHAS = [1  1  2  nel  1
          2  2  3  nel  1
          3  3  4  nel  1
          4  4  1  nel  1];
BCUin = [1  0  0  1  0  0  0  22
         2  0  0  1  0  0  0  22
         3  0  0  1  0  0  0  22
         4  0  0  1  0  0  0  22];
BCFin = [2  1  1  1  0.0  0.0  0.0];

CTE = [4000e3 2000e3 2000e3  1000e3  1000e3  500e3  0.25  0.25  0.25  0.0   0.025
       4000e3 2000e3 2000e3  1000e3  1000e3  500e3  0.25  0.25  0.25  90.0  0.025
       4000e3 2000e3 2000e3  1000e3  1000e3  500e3  0.25  0.25  0.25  90.0  0.025
       4000e3 2000e3 2000e3  1000e3  1000e3  500e3  0.25  0.25  0.25  0.0   0.025];
hpl = sum(CTE(:,11));
rho = 4000;
K_Amatriz = 5/6;
q = 1.0;
NPGIC = 6; NPGSF = 6; NPGID = 6;
DivElem = 4;
TOL = 1.0e4*eps;
aprfun = 3;   % f = r^2 log(r)
dA = 1;

[AT, D] = ContsLam(CTE, K_Amatriz);
fprintf('MATLAB Dynamic_Composite_Plate  EjemploDin_10 static\n');
fprintf('  SS [0,1]^2  [0/90/90/0]  h=%.3f  q=%.1f  rho=%.0f\n', hpl, q, rho);
fprintf('  D11=%.6f  D22=%.6f  D12=%.6f  D66=%.6f\n', D(1,1), D(2,2), D(1,2), D(3,3));
fprintf('  AT44=%.6f  AT45=%.6f  AT55=%.6f\n', AT(1,1), AT(1,2), AT(2,2));
fflush(stdout);

[ptocont, CoordCHP, ElemCHP, BCU, BCF, EqnType] = prepdata(PONTOS, LINHAS, BCUin, BCFin);
[NOS_GEO, ELEM_GEO] = NosGEOGen(PONTOS, LINHAS);
CoordINT = [1  0.5  0.5];
nnc = size(ptocont, 1);
nnd = size(CoordINT, 1);
fprintf('  nodes=%d  internals=%d  elements=%d\n', nnc, nnd, size(ElemCHP,1));
fflush(stdout);

fprintf('  assembling Hc,Gc ...\n'); fflush(stdout);
[Hc, Gc] = BuiltMatrix(ptocont, CoordCHP, ElemCHP, 0);
fprintf('  assembling Hd,Gd ...\n'); fflush(stdout);
[Hd, Gd] = BuiltMatrix(CoordINT, CoordCHP, ElemCHP, 1);
O = zeros(3*nnc, 3*nnd);
Iint = eye(3*nnd);
H = [Hc O; Hd Iint];
G = [Gc; Gd];
fprintf('  ||H||=%.4e  ||G||=%.4e\n', norm(H), norm(G));
fflush(stdout);

fprintf('  DomainTermsRIM (aprfun=3, Gauss 4/6) ...\n'); fflush(stdout);
[M, Q] = DomainTermsRIM(ELEM_GEO, ptocont, NOS_GEO, CoordINT, aprfun, dA, rho);
fprintf('  ||M||=%.4e  ||Q||=%.4e\n', norm(M), norm(Q));
fflush(stdout);

Usol = Statsolver(H, G, Q, BCU, BCF);
wc = Usol(3*(nnc+1));
wmax_b = max(abs(Usol(3:3:3*nnc)));
fprintf('  w_center=%.10e  max|w|_boundary=%.4e\n', wc, wmax_b);
psi = Usol(3*(nnc+1)-2:3*(nnc+1));
fprintf('  centre  psi_x=%.4e  psi_y=%.4e  w=%.4e\n', psi);

out = fullfile(thisdir, 'dyncomp_octave.csv');
csvwrite(out, [D(1,1) D(2,2) D(1,2) D(3,3) AT(1,1) AT(2,2) wc wmax_b norm(Q)]);
fid = fopen(fullfile(thisdir, 'dyncomp_octave.txt'), 'w');
fprintf(fid, 'D11=%.10e\nD22=%.10e\nD12=%.10e\nD66=%.10e\n', D(1,1), D(2,2), D(1,2), D(3,3));
fprintf(fid, 'A44=%.10e\nA55=%.10e\n', AT(1,1), AT(2,2));
fprintf(fid, 'w_center=%.16e\n', wc);
fprintf(fid, 'psi_x=%.16e\npsi_y=%.16e\n', psi(1), psi(2));
fprintf(fid, 'max_abs_w_boundary=%.16e\n', wmax_b);
fprintf(fid, 'normQ=%.16e\nnormM=%.16e\nnormH=%.16e\nnormG=%.16e\n', ...
    norm(Q), norm(M), norm(H), norm(G));
fclose(fid);
fprintf('Wrote %s\nDone.\n', out);
end
