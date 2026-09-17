% Step-by-step trace of dad_Contato_Bulk frictional incremental (Octave).
clear; close all; clc; more off; warning("off", "all");
try; graphics_toolkit("gnuplot"); catch; end
set(0, "defaultfigurevisible", "off");

this_dir = fileparts(mfilename("fullpath"));
out_dir = fullfile(this_dir, "..", "..", "plots", "cattaneo_mindlin", "octave_bulk");
mkdir(out_dir);
addpath("/data/OneDrive/Área de Trabalho/Contato");
addpath(this_dir);

try; dad_Contato_Bulk; catch err
  printf("dad graphics: %s\n", err.message);
end

[ELEM_GEO,ELEM_FIS,NOS_GEO,NOS_FIS,E,ni]=formata_dad(PONTOS,SEGMENTOS,MALHA,E,ni,tipo_prob);
CDC = gera_CDC(SEGMENTOS,MALHA,CCSeg,NOS_RES);
[normais,normais_nos]=calcula_normais(NOS_GEO,ELEM_GEO);
nos_subreg = calc_subregioes(MALHA,subregioes);
elem_interfaces = calc_interfaces(MALHA,interfaces);
printf("nodes=%d els=%d iface_els=%d  H,G ...\n", size(NOS_FIS,1), size(ELEM_FIS,1), elem_interfaces(1,3));
fflush(stdout);
[G,H]=monta_GeH_subregioes(ELEM_FIS,ELEM_GEO,NOS_FIS,NOS_GEO,E,ni,normais,nos_subreg);
[A,b] = aplica_cdc_subregioes(H,G,CDC,elem_interfaces);
nnos = size(NOS_FIS,1);
h=calc_gap_subreg(NOS_FIS,elem_interfaces);
P = 2*w*cargav;
East = E/(1-ni^2);
aH = sqrt(8*R*P/(pi*East));   % two similar bodies
p0H = 2*P/(pi*aH);
printf("H=%dx%d  A=%d  h[%.3e, %.3e]  n_h0=%d  P=%.4f aH=%.4f p0H=%.4f\n", ...
  size(H,1), size(H,2), size(A,1), min(h), max(h), sum(h==0), P, aH, p0H);
fflush(stdout);

nnos_c = 3*elem_interfaces(1,3);
i1 = elem_interfaces(1,1);
comp1 = 3*i1-2:3*i1-3+nnos_c;

% --- gap + normals ---
fid = fopen(fullfile(out_dir, "oct_gap.csv"), "w");
fprintf(fid, "x,y,nx,ny,h\n");
for i=1:nnos_c
  n = comp1(i);
  fprintf(fid, "%.12e,%.12e,%.12e,%.12e,%.12e\n", ...
    NOS_FIS(n,2), NOS_FIS(n,3), normais_nos(n,2), normais_nos(n,3), h(i));
end
fclose(fid);

% G self-term at centre-most contact node (local tn column = 2n-1)
[~, ic] = min(abs(NOS_FIS(comp1,2)));
nc = comp1(ic);
Gnn = G(2*nc-1, 2*nc-1);
Gnt = G(2*nc-1, 2*nc);
printf("centre node %d x=%.4f  G_nn=%.6e G_nt=%.6e  ||G||=%.4e ||H||=%.4e ||b||=%.4e\n", ...
  nc, NOS_FIS(nc,2), Gnn, Gnt, norm(G,"fro"), norm(H,"fro"), norm(b));
fflush(stdout);

% --- first verify from x=0 ---
[T,De,desl,trac]=inicializa_T_De(CDC);
x0 = zeros(size(A,1),1);
contato0 = verfica_contato_incremental_multicorpos(x0,h,elem_interfaces,nnos,mi,normais_nos,desl,trac);
printf("step0 verify  open=%d stick=%d slip=%d\n", sum(contato0==1), sum(contato0==3), sum(abs(contato0)==2));
fflush(stdout);

% --- incremental frictional, dump each step ---
fid = fopen(fullfile(out_dir, "oct_steps.csv"), "w");
fprintf(fid, "step,n_open,n_stick,n_slip,p0,qmax,nit\n");
x0 = zeros(size(A,1),1);
for s = 1:npassos
  nit = 0; dist = 1;
  while dist > 1e-8
    nit = nit + 1;
    contato = verfica_contato_incremental_multicorpos(x0,h,elem_interfaces,nnos,mi,normais_nos,desl,trac);
    [A,b2]=aplica_contato_incremental_multicorpos(A,b,h,contato,elem_interfaces,nnos,mi,normais_nos,desl,trac);
    y0 = A*x0 - b2;
    d0 = -A\y0;
    x = x0 + d0;
    dist = sqrt((x-x0)'*(x-x0));
    x0 = x;
    if nit > 80; break; end
  end
  [desl_dt,trac_dt,De_dt,T_dt] = reordena_subreg(x,CDC,elem_interfaces);
  desl = desl + desl_dt; trac = trac + trac_dt;
  De(:,2:end) = De(:,2:end) + De_dt(:,2:end);
  T(:,2:end) = T(:,2:end) + T_dt(:,2:end);
  x0 = x;

  tn = zeros(nnos_c,1); tt = zeros(nnos_c,1); xx = zeros(nnos_c,1);
  for i=1:nnos_c
    noglobal=comp1(i); xx(i)=NOS_FIS(noglobal,2);
    ii=ceil(noglobal/3); j=rem(noglobal,3); if j==0; j=3; end
    tn(i)=T(ii,2*j); tt(i)=T(ii,2*j+1);
  end
  cl = abs(contato) ~= 1;
  p0 = 0; Q = 0; Pline = 0; qmax = 0;
  if any(cl)
    p0 = max(-tn(cl));
    qmax = max(abs(tt(cl)));
  end
  n_o = sum(contato==1); n_s = sum(contato==3); n_l = sum(abs(contato)==2);
  fprintf(fid, "%d,%d,%d,%d,%.8e,%.8e,%d\n", s, n_o, n_s, n_l, p0, qmax, nit);
  printf("oct s=%2d  nit=%2d  o/s/l=%d/%d/%d  p0=%.3f  qmax=%.4f\n", s, nit, n_o, n_s, n_l, p0, qmax);
  fflush(stdout);
  if s==1 || s==npassos
    f2 = fopen(fullfile(out_dir, sprintf("oct_iface_s%d.csv", s)), "w");
    fprintf(f2, "x,tn,tt,state,h\n");
    for i=1:nnos_c
      fprintf(f2, "%.12e,%.12e,%.12e,%d,%.12e\n", xx(i), tn(i), tt(i), contato(i), h(i));
    end
    fclose(f2);
  end
end
fclose(fid);
printf("Done trace.\n");
