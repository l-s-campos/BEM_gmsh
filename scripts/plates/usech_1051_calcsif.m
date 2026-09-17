function [K1b, K2b, K3b, Le, La, Lb] = usech_1051_calcsif(Usol, CoordCHP, ElemCHP, E, nu, thk)
% Headless CalcSIF.m (Dirgantara / Useche 10.15–10.16). Right tip.
for elem = 1:size(ElemCHP, 1) - 1
  CDX = CoordCHP(ElemCHP(elem, 2:4), 2) - CoordCHP(ElemCHP(elem + 1, 4:-1:2), 2);
  CDY = CoordCHP(ElemCHP(elem, 2:4), 3) - CoordCHP(ElemCHP(elem + 1, 4:-1:2), 3);
  if norm(CDX) < 1e-4 && norm(CDY) <= 1e-3
    break;
  end
end
nodes = [3 * elem - 2, 3 * elem - 1, 3 * elem, ...
         3 * (elem + 1) - 2, 3 * (elem + 1) - 1, 3 * (elem + 1)];
dx = CoordCHP(ElemCHP(elem, 4), 2) - CoordCHP(ElemCHP(elem, 2), 2);
dy = CoordCHP(ElemCHP(elem, 4), 3) - CoordCHP(ElemCHP(elem, 2), 3);
Le = sqrt(dx^2 + dy^2);
La = 5 / 6 * Le;
Lb = 1 / 2 * Le;
n1 = dx / Le;
n2 = dy / Le;
drxa = Usol(3 * nodes(1) - 2) - Usol(3 * nodes(6) - 2);
drya = Usol(3 * nodes(1) - 1) - Usol(3 * nodes(6) - 1);
dwa  = Usol(3 * nodes(1))     - Usol(3 * nodes(6));
drxb = Usol(3 * nodes(2) - 2) - Usol(3 * nodes(5) - 2);
dryb = Usol(3 * nodes(2) - 1) - Usol(3 * nodes(5) - 1);
dwb  = Usol(3 * nodes(2))     - Usol(3 * nodes(5));
dwaa = [drxa * n1 + drya * n2, drya * n1 - drxa * n2, dwa];
dwbb = [drxb * n1 + dryb * n2, dryb * n1 - drxb * n2, dwb];
C = zeros(3, 3);
C(1, 1) = E * thk^3 / 48.0 * sqrt(pi / 2.0);
C(2, 2) = C(1, 1);
C(3, 3) = 5 * E * thk / (24.0 * (1 + nu)) * sqrt(pi / 2.0);
Kaa = C * dwaa';
Kbb = C * dwbb';
Kaa = sqrt(1 / La) * Kaa;
Kbb = sqrt(1 / Lb) * Kbb;
ktip = La / (La - Lb) * (Kbb - Lb / La * Kaa);
% CalcSIF.m prints ktip(2) as K1b. CODE: ktip = [Crot*Δψt, Crot*Δψn, Csh*Δw].
K1b = ktip(2);
K2b = ktip(1);
K3b = ktip(3);
end
