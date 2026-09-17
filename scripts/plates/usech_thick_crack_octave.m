function usech_thick_crack_octave
% Headless Static_Thick_Cracked_Plate/test01.m (Dirgantara p.118).
% SS square [-1,1]^2, centre crack a=0.5, q=1, Dual BEM.
more off;
warning('off', 'all');
mat_dir = '/home/lsc/Downloads/BEM_Plate_Shell_Book_Juseche-main/Static_Thick_Cracked_Plate';
addpath(mat_dir);
addpath('/data/OneDrive/pesquisa/BEM_gmsh/scripts/plates');

global EE v hpl q EqnType CoordCHP ElemCHP MINDIV MAXDIV DD Lamd NDIV

PONTOS = [1   -1.0  -1.0
          2    1.0  -1.0
          3    1.0   1.0
          4   -1.0   1.0
          5   -0.50  0.0
          6    0.50  0.0];
LINHAS = [1  1  2  8  1
          2  2  3  8  1
          3  3  4  8  1
          4  4  1  8  1
          5  5  6  16 1
          6  6  5  16 2];
BCUin = [1  1 0 1  0 0 0  22
         2  0 1 1  0 0 0  22
         3  1 0 1  0 0 0  22
         4  0 1 1  0 0 0  22];
BCFin = [1  1 1 1  0.0  0.0  0.0
         3  1 1 1  0.0  0.0  0.0];
v = 0.3; EE = 1000e6; hpl = 0.1667; q = 1.0;
MINDIV = 1; MAXDIV = 10;
DD = EE*hpl^3/(12*(1-v^2));
Lamd = sqrt(10)/hpl;
no = [0 0];

fprintf('MATLAB Static_Thick_Cracked_Plate / test01\n');
fprintf('  SS [-1,1]^2  a=0.5  h=%.4f  E=%.3e  q=%.1f  Dual BEM\n', hpl, EE, q);
[CoordCOL, CoordCHP, ElemCHP, BCU, BCF, EqnType] = prepdata(PONTOS, LINHAS, BCUin, BCFin);
EqnType = EqnType(:);
nnod = size(CoordCOL, 1);
fprintf('  nodes=%d  elements=%d  HBIE nodes=%d\n', nnod, size(ElemCHP,1), sum(EqnType==2));
fflush(stdout);
fprintf('  assembling H,G,Q ...\n'); fflush(stdout);
[H, G, Q] = BuiltMatrix(CoordCOL, CoordCHP, ElemCHP, q, no);
fprintf('  solving ...\n'); fflush(stdout);
[Ucont, ~] = solverEqn(H, G, Q, BCU, BCF);
[K1b, K2b, K3b, Le, La, Lb] = usech_1051_calcsif(Ucont, CoordCHP, ElemCHP, EE, v, hpl);
a = 0.5;
fprintf('  K1b=%.6e  K2b=%.4e  K3b=%.4e\n', K1b, K2b, K3b);
fprintf('  Le=%.4f  La/Le=%.3f  Lb/Le=%.3f\n', Le, La/Le, Lb/Le);

% COD on face A (line 5) vs face B (line 6), same x
n5 = 3*LINHAS(5,4);
iA = (3*sum(LINHAS(1:4,4))+1):(3*sum(LINHAS(1:4,4))+n5);
iB = (iA(end)+1):(iA(end)+n5);
COD = [];
fprintf('\n  x          Δψx         Δψy          Δw\n');
for k = 1:n5
    i = iA(k); j = iB(n5+1-k);  % opposite traversal
    xx = CoordCOL(i,2);
    dpsi = [Ucont(3*i-2)-Ucont(3*j-2), Ucont(3*i-1)-Ucont(3*j-1), Ucont(3*i)-Ucont(3*j)];
    fprintf(' %+6.3f  %11.4e  %11.4e  %11.4e\n', xx, dpsi);
    COD = [COD; xx dpsi];
end
out = '/data/OneDrive/pesquisa/BEM_gmsh/scripts/plates/usech_thick_crack_octave.csv';
csvwrite(out, [COD, K1b*ones(size(COD,1),1), K2b*ones(size(COD,1),1), K3b*ones(size(COD,1),1)]);
fid = fopen('/data/OneDrive/pesquisa/BEM_gmsh/scripts/plates/usech_thick_crack_octave_sif.txt','w');
fprintf(fid, 'K1b=%.10e\nK2b=%.10e\nK3b=%.10e\nLe=%.10e\n', K1b, K2b, K3b, Le);
fclose(fid);
fprintf('Wrote %s\nDone.\n', out);
end
