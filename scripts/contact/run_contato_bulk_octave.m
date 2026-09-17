% Headless Octave runner for dad_Contato_Bulk (Contato MATLAB).
% Frictionless (driver default) then frictional incremental.
%
%   octave-cli --no-gui --quiet scripts/contact/run_contato_bulk_octave.m

clear; close all; clc;
more off;
warning("off", "all");
try
  graphics_toolkit("gnuplot");
catch
end
set(0, "defaultfigurevisible", "off");

this_dir = fileparts(mfilename("fullpath"));
out_dir = fullfile(this_dir, "..", "..", "plots", "cattaneo_mindlin", "octave_bulk");
mkdir(out_dir);

contato_dir = "/data/OneDrive/Área de Trabalho/Contato";
addpath(contato_dir);
addpath(this_dir);
printf("Contato dir: %s\n", contato_dir);
fflush(stdout);

%% ---- data (plots in dad_* may fail; variables are set first) ----
try
  dad_Contato_Bulk;
catch err
  printf("dad_Contato_Bulk (graphics?) %s\n", err.message);
end
if ~exist("PONTOS", "var")
  error("dad_Contato_Bulk did not define PONTOS");
end
printf("w=%.4f R=%.1f cargav=%.4f mi=%.3f npassos=%d tipo_prob=%d ref contact els=%d\n", ...
  w, R, cargav, mi, npassos, tipo_prob, MALHA(1,2));
fflush(stdout);

[ELEM_GEO,ELEM_FIS,NOS_GEO,NOS_FIS,E,ni]= ...
    formata_dad(PONTOS,SEGMENTOS,MALHA,E,ni,tipo_prob);
CDC = gera_CDC(SEGMENTOS,MALHA,CCSeg,NOS_RES);
[normais,normais_nos]=calcula_normais(NOS_GEO,ELEM_GEO);
nos_subreg = calc_subregioes(MALHA,subregioes);
elem_interfaces = calc_interfaces(MALHA,interfaces);
PONTOS_INT = [];

printf("nodes=%d elems=%d interface els=%d  assembling H,G ...\n", ...
  size(NOS_FIS,1), size(ELEM_FIS,1), elem_interfaces(1,3));
fflush(stdout);
t0 = tic();
[G,H]=monta_GeH_subregioes(ELEM_FIS,ELEM_GEO,NOS_FIS,NOS_GEO,E,ni, ...
    normais,nos_subreg);
printf("H,G done in %.1f s  size %d\n", toc(t0), size(H,1));
fflush(stdout);

[A,b] = aplica_cdc_subregioes(H,G,CDC,elem_interfaces);
nnos = size(NOS_FIS,1);
h=calc_gap_subreg(NOS_FIS,elem_interfaces);
printf("A size=%d  gap h in [%.6f, %.6f]\n", size(A,1), min(h), max(h));
fflush(stdout);

P = 2*w*cargav;
East = E/(1-ni^2);
aH = sqrt(4*R*P/(pi*East));
p0H = 2*P/(pi*aH);
printf("P=%.4f  Hertz a=%.4f p0=%.4f  (E=%.1f ni=%.4f after formata_dad)\n", ...
  P, aH, p0H, E, ni);
fflush(stdout);

%% ---- frictionless incremental (driver default) ----
printf("\n=== frictionless incremental npassos=%d ===\n", npassos);
fflush(stdout);
x0=zeros(size(A,1),1);
t1 = tic();
[contato0,De0,T0,desl0,trac0]=newton_incremental_multicorpos_sem_atrito( ...
    x0,A,b,h,CDC,elem_interfaces,nnos,normais_nos,npassos);
printf("frictionless done in %.1f s\n", toc(t1));
fflush(stdout);
dump_contato_iface(fullfile(out_dir, "octave_bulk_mu0.csv"), NOS_FIS, T0, contato0, elem_interfaces, h);

%% ---- frictional incremental ----
printf("\n=== frictional incremental mi=%.3f npassos=%d ===\n", mi, npassos);
fflush(stdout);
x0=zeros(size(A,1),1);
t2 = tic();
[contato,De,T,desl,trac]=newton_incremental_multicorpos( ...
    x0,A,b,h,CDC,elem_interfaces,nnos,mi,normais_nos,npassos);
printf("frictional done in %.1f s\n", toc(t2));
fflush(stdout);
dump_contato_iface(fullfile(out_dir, "octave_bulk_mu.csv"), NOS_FIS, T, contato, elem_interfaces, h);

fid = fopen(fullfile(out_dir, "octave_bulk_meta.txt"), "w");
fprintf(fid, "P=%.12e\naH=%.12e\np0H=%.12e\nE=%.12e\nni=%.12e\nR=%.12e\nw=%.12e\ncargav=%.12e\nmi=%.12e\nnpassos=%d\ntipo_prob=%d\n", ...
  P, aH, p0H, E, ni, R, w, cargav, mi, npassos, tipo_prob);
fclose(fid);
printf("\nDone. outputs in %s\n", out_dir);
