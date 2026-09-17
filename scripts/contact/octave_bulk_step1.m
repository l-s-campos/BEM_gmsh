% Newton-by-Newton dump of load step 1 (dad_Contato_Bulk).
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

nnos_c = 3*elem_interfaces(1,3);
i1 = elem_interfaces(1,1);
i2 = elem_interfaces(1,2);
comp1 = 3*i1-2:3*i1-3+nnos_c;
comp2 = 3*i2-3+nnos_c:-1:3*i2-2;
printf("iface els=%d nodes=%d  h[%.3e, %.3e] n_h0=%d  min|x|=%.6e\n", ...
  elem_interfaces(1,3), nnos_c, min(h), max(h), sum(h==0), min(abs(NOS_FIS(comp1,2))));
fflush(stdout);

% closest 5
[~,ord] = sort(abs(NOS_FIS(comp1,2)));
printf("closest pairs:\n");
for k=1:5
  i = ord(k); n = comp1(i); m = comp2(i);
  printf("  xa=%+.6f xb=%+.6f h=%.6e nA=(%+.3f,%+.3f) nB=(%+.3f,%+.3f)\n", ...
    NOS_FIS(n,2), NOS_FIS(m,2), h(i), ...
    normais_nos(n,2), normais_nos(n,3), normais_nos(m,2), normais_nos(m,3));
end
fflush(stdout);

[T,De,desl,trac]=inicializa_T_De(CDC);
x0 = zeros(size(A,1),1);
contato0 = verfica_contato_incremental_multicorpos(x0,h,elem_interfaces,nnos,mi,normais_nos,desl,trac);
printf("step0 verify  open=%d stick=%d slip=%d\n", sum(contato0==1), sum(contato0==3), sum(abs(contato0)==2));
ix = find(contato0 ~= 1);
for k=1:length(ix)
  i = ix(k);
  printf("  closed x=%+.5f st=%d h=%.6e\n", NOS_FIS(comp1(i),2), contato0(i), h(i));
end
fflush(stdout);

fid = fopen(fullfile(out_dir, "oct_step1_newton.csv"), "w");
fprintf(fid, "tag,i,x,h,state,tn,tt\n");

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
  n_o = sum(contato==1); n_s = sum(contato==3); n_l = sum(abs(contato)==2);
  printf("oct it=%2d  dist=%.3e  o/s/l=%d/%d/%d\n", nit, dist, n_o, n_s, n_l);
  % neighbour kinematics from increment x (desl=0 on step 1)
  i1 = elem_interfaces(1,1); i2 = elem_interfaces(1,2);
  c1 = 3*i1-2:3*i1-3+nnos_c;
  c2 = 3*i2-3+nnos_c:-1:3*i2-2;
  [~,ord] = sort(abs(NOS_FIS(c1,2)));
  printf("  centre± kinematics (un,ut pad / spec):\n");
  for kk=1:min(5,length(ord))
    i = ord(kk); n1=c1(i); n2=c2(i);
    un1=x(2*n1-1); ut1=x(2*n1); un2=x(2*n2-1); ut2=x(2*n2);
    n1x=normais_nos(n1,2); n1y=normais_nos(n1,3);
    n2x=normais_nos(n2,2); n2y=normais_nos(n2,3);
    R1=[n1x,-n1y;n1y,n1x]; R2=[n2x,-n2y;n2y,n2x]; R=R2*R1';
    un21=R(1,:)*[un2;ut2]; ut21=R(2,:)*[un2;ut2];
    printf("    x=%+.5f un=(%.4e,%.4e) ut=(%.4e,%.4e) dun=%.4e dut=%.4e h=%.4e st=%d\n", ...
      NOS_FIS(n1,2), un1, un2, ut1, ut2, un1-un21, ut1-ut21, h(i), contato(i));
  end
  fflush(stdout);
  [desl_dt,trac_dt,De_dt,T_dt] = reordena_subreg(x,CDC,elem_interfaces);
  for i=1:nnos_c
    noglobal=comp1(i);
    xx=NOS_FIS(noglobal,2);
    ii=ceil(noglobal/3); j=rem(noglobal,3); if j==0; j=3; end
    tn=T_dt(ii,2*j); tt=T_dt(ii,2*j+1);
    if abs(contato(i)) == 1 && abs(tn) < 1e-14 && abs(tt) < 1e-14
      continue;
    end
    fprintf(fid, "oct_it%d,%d,%.12e,%.12e,%d,%.12e,%.12e\n", nit, i, xx, h(i), contato(i), tn, tt);
    printf("  oct_it%d i=%3d x=%+.5f st=%d tn=%8.3f tt=%8.4f h=%.3e\n", nit, i, xx, contato(i), tn, tt, h(i));
  end
  fflush(stdout);
  if nit > 20; break; end
end
fclose(fid);
printf("Done Octave step-1 Newtons, nit=%d\n", nit);
