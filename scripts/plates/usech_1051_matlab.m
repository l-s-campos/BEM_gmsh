% Headless Useche 10.5.1 for a/b = 0.6 and 0.8 (book mesh).
% Geometry matches scripts/plates/usech_1051.jl: b=1, h=0.5, c=2, Mo=1.
% Mesh: 8 BE / outer edge (32 outer) + 16 quadratic / crack face.
% F = K1b / (Mo * sqrt(pi*a))   [Table 10.1]
%
%   octave-cli --no-gui --quiet scripts/plates/usech_1051_matlab.m

more off;
warning("off", "all");
try, graphics_toolkit("gnuplot"); catch, end
try, set(0, "defaultfigurevisible", "off"); catch, end

this_dir = fileparts(mfilename("fullpath"));
mat_dir = "/home/lsc/Downloads/BEM_Plate_Shell_Book_Juseche-main/Static_Thick_Cracked_Plate";
addpath(mat_dir);
addpath(this_dir);

out_csv = fullfile(this_dir, "usech_1051_matlab.csv");
fid = fopen(out_csv, "w");
fprintf(fid, "ab,K1b,K2b,K3b,F,Le,La,Lb,nnod\n");

global EE v hpl q EqnType CoordCHP ElemCHP MINDIV MAXDIV DD Lamd NDIV

b = 1.0;
hpl = b / 2;
c = 2 * b;
EE = 2.1e5;
v = 0.3;
Mo = 1.0;
q = 0.0;
MINDIV = 1;
MAXDIV = 10;
DD = EE * hpl^3 / (12 * (1 - v^2));
Lamd = sqrt(10) / hpl;
no = [0 0];
nel_edge = 8;
nel_crack = 16;

printf("Useche 10.5.1 MATLAB/Octave  a/b=0.6,0.8  book mesh %d/edge %d/face\n", ...
       nel_edge, nel_crack);
printf("  plate [-%g,%g]x[-%g,%g]  h=%g  Mo=%g  E=%g  nu=%g\n", ...
       b, b, c, c, hpl, Mo, EE, v);
fflush(stdout);

abs_list = [0.6, 0.8];
for iab = 1:length(abs_list)
  ab = abs_list(iab);
  a = ab * b;
  printf("\n== a/b=%.1f  a=%g ==\n", ab, a);
  fflush(stdout);

  PONTOS = [1   -b   -c
            2    b   -c
            3    b    c
            4   -b    c
            5   -a    0.0
            6    a    0.0];
  % EQNTYPE 1 = CBIE (outer + face A), 2 = HBIE (face B)
  LINHAS = [1  1  2  nel_edge   1
            2  2  3  nel_edge   1
            3  3  4  nel_edge   1
            4  4  1  nel_edge   1
            5  5  6  nel_crack  1
            6  6  5  nel_crack  2];

  [CoordCOL, CoordCHP, ElemCHP, ~, ~, EqnType] = BdryElemGen(PONTOS, LINHAS);
  EqnType = EqnType(:);
  nnod = size(CoordCOL, 1);
  printf("  nodes=%d  elements=%d\n", nnod, size(ElemCHP, 1));
  fflush(stdout);

  % tractions (Mxn, Myn, Qn): Myn = Mo * ny on y=±c
  BCF = zeros(0, 7);
  for klin = [1, 3]
    if klin == 1
      nstart = 1;
    else
      nstart = 3 * sum(LINHAS(1:klin-1, 4)) + 1;
    end
    nend = nstart + 3 * LINHAS(klin, 4) - 1;
    My = Mo * (2 * (klin == 3) - 1);   % bottom ny=-1 → -Mo; top ny=+1 → +Mo
    for n = nstart:nend
      BCF = [BCF; n, 1, 1, 1, 0.0, My, 0.0];
    end
  end

  % three RBM pins on the outer wall (clone of pin_fsdt_rbm!)
  xy = CoordCOL(:, 2:3);
  on_outer = (abs(abs(xy(:,1)) - b) < 1e-8) | (abs(abs(xy(:,2)) - c) < 1e-8);
  ileft = find(on_outer & abs(xy(:,1) + b) < 1e-8);
  [~, k] = min(abs(xy(ileft, 2)));
  n_left = ileft(k);
  ibot = find(on_outer & abs(xy(:,2) + c) < 1e-8);
  [~, k] = min(abs(xy(ibot, 1)));
  n_bot = ibot(k);
  [~, ord] = sort(abs(xy(ibot, 1)));
  n_bot2 = ibot(ord(min(2, numel(ord))));
  if n_bot2 == n_bot && numel(ord) >= 3
    n_bot2 = ibot(ord(3));
  end
  BCU = [n_left  1 0 0  0 0 0
         n_bot   0 0 1  0 0 0
         n_bot2  0 1 0  0 0 0];
  printf("  pins  psi_x@%d  w@%d  psi_y@%d\n", n_left, n_bot, n_bot2);
  fflush(stdout);

  printf("  assembling H,G ...\n");
  fflush(stdout);
  [H, G, Q] = BuiltMatrix(CoordCOL, CoordCHP, ElemCHP, q, no);
  printf("  solving ...\n");
  fflush(stdout);
  [Ucont, ~] = solverEqn(H, G, Q, BCU, BCF);

  [K1b, K2b, K3b, Le, La, Lb] = usech_1051_calcsif(Ucont, CoordCHP, ElemCHP, EE, v, hpl);
  F = K1b / (Mo * sqrt(pi * a));
  printf("  K1b=%.6f  K2b=%.4e  K3b=%.4e\n", K1b, K2b, K3b);
  printf("  F=K1b/(Mo*sqrt(pi*a))=%.6f   book F=%.3f\n", F, 0.095 + 0.039 * (ab > 0.7));
  printf("  Le=%.6f  La/Le=%.4f  Lb/Le=%.4f\n", Le, La / Le, Lb / Le);
  fflush(stdout);
  fprintf(fid, "%.1f,%.8f,%.8e,%.8e,%.8f,%.8f,%.8f,%.8f,%d\n", ...
          ab, K1b, K2b, K3b, F, Le, La, Lb, nnod);
end

fclose(fid);
printf("\nWrote %s\nDone.\n", out_csv);
fflush(stdout);
