% Dump 6x6 H,G of the three centre contact nodes (dad_Contato_Bulk).
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
printf("H,G ...\n"); fflush(stdout);
[G,H]=monta_GeH_subregioes(ELEM_FIS,ELEM_GEO,NOS_FIS,NOS_GEO,E,ni,normais,nos_subreg);

nnos_c = 3*elem_interfaces(1,3);
i1 = elem_interfaces(1,1);
comp1 = 3*i1-2:3*i1-3+nnos_c;
[~, ic] = min(abs(NOS_FIS(comp1,2)));
ids = comp1(ic-1:ic+1);
printf("centre nodes global=%s  x=%s\n", mat2str(ids), mat2str(NOS_FIS(ids,2)', 6));
fflush(stdout);

fid = fopen(fullfile(out_dir, "oct_Hcenter.csv"), "w");
fprintf(fid, "tag,i,j,Hxx,Hxy,Hyx,Hyy,Gxx,Gxy,Gyx,Gyy\n");
for a=1:3
  for b=1:3
    ia=ids(a); ib=ids(b);
    ri=2*ia-1; ci=2*ib-1;
    fprintf(fid, "oct,%d,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n", ...
      a, b, H(ri,ci), H(ri,ci+1), H(ri+1,ci), H(ri+1,ci+1), ...
      G(ri,ci), G(ri,ci+1), G(ri+1,ci), G(ri+1,ci+1));
  end
end
fclose(fid);
printf("H(left,centre) Hxx=%.4e Hxy=%.4e Hyx=%.4e Hyy=%.4e\n", ...
  H(2*ids(1)-1, 2*ids(2)-1), H(2*ids(1)-1, 2*ids(2)), ...
  H(2*ids(1),   2*ids(2)-1), H(2*ids(1),   2*ids(2)));
printf("G(left,centre) Gxx=%.4e Gxy=%.4e Gyx=%.4e Gyy=%.4e\n", ...
  G(2*ids(1)-1, 2*ids(2)-1), G(2*ids(1)-1, 2*ids(2)), ...
  G(2*ids(1),   2*ids(2)-1), G(2*ids(1),   2*ids(2)));
printf("Done.\n");
