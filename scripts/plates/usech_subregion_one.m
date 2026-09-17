% MATLAB Static_Cracked_Laminated_Plate_Subregion / prueba.inp
% Region 1 only: [0,1]x[0,1] cantilever, E=70e3, nu=0, h=1.5, q=-1.
% 1 discontinuous quadratic element per edge (12 collocation nodes).
addpath(fullfile('/home/lsc/Downloads/BEM_Plate_Shell_Book_Juseche-main', ...
    'Static_Cracked_Laminated_Plate_Subregion/eqnbend/eqnplaca'));

global DD vp Lamd NDIV MINDIV MAXDIV
EEp = 70e3; vp = 0.0; hp = 1.5; q = -1.0;
DD = EEp*hp^3/(12*(1-vp^2));
Lamd = sqrt(10)/hp;
MINDIV = 1; MAXDIV = 4; NDIV = 4;

pontos = [1 0 0; 2 1 0; 3 1 1; 4 0 1];
linhas = [1 1 2 1; 2 2 3 1; 3 3 4 1; 4 4 1 1]; % 1 element / edge
nlinhasR = 4; node1 = 0; node2 = 0; elemt = 0;
coordis = []; coordcc = []; elem = [];
for i = 1:nlinhasR
    X1L = pontos(linhas(i,2),2); Y1L = pontos(linhas(i,2),3);
    X2L = pontos(linhas(i,3),2); Y2L = pontos(linhas(i,3),3);
    nelem = linhas(i,4);
    for e = -1:2/nelem:(1-2/nelem)
        X1 = 0.5*(1-e)*X1L + 0.5*(1+e)*X2L;
        Y1 = 0.5*(1-e)*Y1L + 0.5*(1+e)*Y2L;
        e2 = e + 2/nelem;
        X2 = 0.5*(1-e2)*X1L + 0.5*(1+e2)*X2L;
        Y2 = 0.5*(1-e2)*Y1L + 0.5*(1+e2)*Y2L;
        for ee = -2/3:2/3:2/3
            X = 0.5*(1-ee)*X1 + 0.5*(1+ee)*X2;
            Y = 0.5*(1-ee)*Y1 + 0.5*(1+ee)*Y2;
            node1 = node1+1;
            coordis(node1,:) = [node1 X Y];
        end
        for ee = -1:1:1
            X = 0.5*(1-ee)*X1 + 0.5*(1+ee)*X2;
            Y = 0.5*(1-ee)*Y1 + 0.5*(1+ee)*Y2;
            node2 = node2+1;
            coordcc(node2,:) = [node2 X Y];
        end
        elemt = elemt+1;
        elem(elemt,:) = [elemt, node1-2, node1-1, node1];
    end
end
Nnods = node1; Nelem = elemt;
[H,G] = BuiltMatrix(coordis, coordcc, elem, Nnods, Nelem);
Q = BuiltQvect(coordis, coordcc, elem, q, Nnods, Nelem);

ndof = 3*Nnods;
kin = false(ndof,1);
% line 4 = nodes 10:12, clamp RX, RY, UZ
for i = 10:12
    kin(3*i-2) = true;
    kin(3*i-1) = true;
    kin(3*i)   = true;
end
U = zeros(ndof,1);
t = zeros(ndof,1);
% clamp: u known 0, t unknown — swap those H columns for -G
A = H;
A(:, kin) = -G(:, kin);
x = A \ Q;
U(~kin) = x(~kin);
t(kin)  = x(kin);

fprintf('Octave region 1  Nnods=%d Nelem=%d  D=%.4f lambda=%.4f\n', Nnods, Nelem, DD, Lamd);
fprintf(' node     x        y         w\n');
wtip = 0; ntip = 0;
for i = 1:Nnods
    fprintf(' %4d  %7.3f  %7.3f  %12.4e\n', i, coordis(i,2), coordis(i,3), U(3*i));
    if abs(coordis(i,2)-1) < 1e-9
        wtip = wtip + U(3*i); ntip = ntip+1;
    end
end
wtip = wtip/max(ntip,1);
fprintf('mean w at x=1: %12.4e\n', wtip);
save('-ascii', 'scripts/plates/usech_subregion_one_octave.txt', 'U');
csvwrite('scripts/plates/usech_subregion_one_octave.csv', ...
    [coordis(:,2) coordis(:,3) U(1:3:end) U(2:3:end) U(3:3:end)]);
