function crack_multiregion_octave
% Two-subregion Reissner centre crack (Useche 10.5.1 geometry).
addpath(fullfile('/home/lsc/Downloads/BEM_Plate_Shell_Book_Juseche-main', ...
    'Static_Cracked_Laminated_Plate_Subregion/eqnbend/eqnplaca'));

global DD vp Lamd NDIV MINDIV MAXDIV
W = 1.0; H = 2.0; a = 0.2;
EEp = 2.1e5; vp = 0.3; hp = 0.5; Mo = 1.0;
DD = EEp*hp^3/(12*(1-vp^2));
Lamd = sqrt(10)/hp;
MINDIV = 1; MAXDIV = 4; NDIV = 4;
nI = 8; nV = 4; nT = 4;

ptsU = [-W 0; W 0; W H; -W H];
ptsL = [-W -H; W -H; W 0; -W 0];
nelsU = [nI nV nT nV];
nelsL = [nT nV nI nV];
[cU, ccU, eU] = mesh4(ptsU, nelsU);
[cL, ccL, eL] = mesh4(ptsL, nelsL);
nU = size(cU,1); nL = size(cL,1);
fprintf('mesh  nU=%d nL=%d  elsU=%d elsL=%d  D=%.4f\n', ...
    nU, nL, size(eU,1), size(eL,1), DD);
[H1,G1] = BuiltMatrix(cU, ccU, eU, nU, size(eU,1));
[H2,G2] = BuiltMatrix(cL, ccL, eL, nL, size(eL,1));

iU = find(abs(cU(:,3)) < 1e-9);
iL = find(abs(cL(:,3)) < 1e-9);
[~, pU] = sort(cU(iU,2));
[~, pL] = sort(cL(iL,2));
iU = iU(pU); iL = iL(pL);
if numel(iU) ~= numel(iL)
    error('interface node count mismatch %d vs %d', numel(iU), numel(iL));
end
isCrack = abs(cU(iU,2)) < a - 1e-12;
fprintf('interface nodes %d  crack %d  ligament %d\n', ...
    numel(iU), sum(isCrack), sum(~isCrack));

N1 = 3*nU; N2 = 3*nL; nUtot = N1+N2;
HH = blkdiag(H1, H2);
GG = blkdiag(G1, G2);
nX = 2*nUtot;
A = zeros(nX+32, nX);
b = zeros(nX+32, 1);
A(1:nUtot, 1:nUtot) = HH;
A(1:nUtot, nUtot+1:nX) = -GG;
row = nUtot;
known_t = false(nUtot,1);
known_u = false(nUtot,1);
[~, ip1] = min((cU(:,2)+W).^2 + (cU(:,3)-H/2).^2);
[~, ip2] = min((cL(:,2)-0).^2 + (cL(:,3)+H).^2);
[~, ip3] = min((cL(:,2)-0.3).^2 + (cL(:,3)+H).^2);
pin_dof = [3*ip1-2, N1+3*ip2, N1+3*ip3-1];
known_u(pin_dof) = true;

for k = 1:numel(iU)
    if ~isCrack(k), continue; end
    for c = 0:2
        d1 = 3*iU(k)-2+c; d2 = N1 + 3*iL(k)-2+c;
        row = row+1; A(row, nUtot+d1) = 1; b(row) = 0; known_t(d1) = true;
        row = row+1; A(row, nUtot+d2) = 1; b(row) = 0; known_t(d2) = true;
    end
end
for k = 1:numel(iU)
    if isCrack(k), continue; end
    for c = 0:2
        d1 = 3*iU(k)-2+c; d2 = N1 + 3*iL(k)-2+c;
        row = row+1; A(row, d1) = 1; A(row, d2) = -1; b(row) = 0;
        row = row+1; A(row, nUtot+d1) = 1; A(row, nUtot+d2) = 1; b(row) = 0;
        known_t(d1) = true; known_t(d2) = true;
    end
end

for i = 1:nU
    if any(iU == i), continue; end
    y = cU(i,3); ny = 0;
    if abs(y - H) < 1e-9, ny = 1; end
    for c = 0:2
        d = 3*i-2+c;
        if known_t(d) || known_u(d), continue; end
        val = 0;
        if ny ~= 0 && c == 1, val = Mo*ny; end
        row = row+1; A(row, nUtot+d) = 1; b(row) = val; known_t(d) = true;
    end
end
for i = 1:nL
    if any(iL == i), continue; end
    y = cL(i,3); ny = 0;
    if abs(y + H) < 1e-9, ny = -1; end
    for c = 0:2
        d = N1 + 3*i-2+c;
        if known_t(d) || known_u(d), continue; end
        val = 0;
        if ny ~= 0 && c == 1, val = Mo*ny; end
        row = row+1; A(row, nUtot+d) = 1; b(row) = val; known_t(d) = true;
    end
end
for d = pin_dof
    row = row+1; A(row, d) = 1; b(row) = 0;
end

fprintf('system rows=%d cols=%d\n', row, nX);
if row > nX
    A = A(1:nX,:); b = b(1:nX);
elseif row < nX
    A = A(1:row,:); b = b(1:row);
    warning('underdetermined by %d', nX-row);
end
x = A \ b;
U1 = x(1:N1); U2 = x(N1+1:nUtot);

fprintf('\n  x          Δψx         Δψy          Δw\n');
COD = [];
for k = 1:numel(iU)
    if ~isCrack(k), continue; end
    d1 = 3*iU(k)-2; d2 = 3*iL(k)-2;
    dpsi = [U1(d1)-U2(d2), U1(d1+1)-U2(d2+1), U1(d1+2)-U2(d2+2)];
    xx = cU(iU(k),2);
    fprintf(' %+6.3f  %11.4e  %11.4e  %11.4e\n', xx, dpsi);
    COD = [COD; xx dpsi];
end
csvwrite('/data/OneDrive/pesquisa/BEM_gmsh/scripts/plates/crack_multiregion_octave.csv', COD);
Crot = EEp*hp^3/48*sqrt(pi/2);
[~, p] = sort(COD(:,1));
COD = COD(p,:);
r = a - COD(:,1);
idx = find(r > 1e-6);
if numel(idx) >= 2
    iB = idx(end); iA = idx(end-1);
    rB = r(iB); rA = r(iA);
    KA = Crot * COD(iA,3) / sqrt(rA);
    KB = Crot * COD(iB,3) / sqrt(rB);
    K1 = rA/(rA-rB)*(KB - (rB/rA)*KA);
    F = abs(K1)/(Mo*sqrt(pi*a));
    fprintf('K1b=%.4e  F=%.4f  (book Dual ~0.99)  rA=%.3f rB=%.3f\n', K1, F, rA, rB);
end
fprintf('Done octave subregion.\n');
end

function [coordis, coordcc, elem] = mesh4(pts, nels)
    node1 = 0; node2 = 0; elemt = 0;
    coordis = []; coordcc = []; elem = [];
    for i = 1:4
        i1 = i; i2 = mod(i,4)+1;
        X1L = pts(i1,1); Y1L = pts(i1,2);
        X2L = pts(i2,1); Y2L = pts(i2,2);
        nelem = nels(i);
        step = 2/nelem;
        for e = -1:step:(1-step+1e-12)
            X1 = 0.5*(1-e)*X1L + 0.5*(1+e)*X2L;
            Y1 = 0.5*(1-e)*Y1L + 0.5*(1+e)*Y2L;
            e2 = e + step;
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
end
