function dump_contato_iface(fname, NOS_FIS, T, contato, elem_interfaces, h)
  nnos_contato = 3 * elem_interfaces(1, 3);
  i1 = elem_interfaces(1, 1);
  comp1 = 3 * i1 - 2 : 3 * i1 - 3 + nnos_contato;
  fid = fopen(fname, "w");
  fprintf(fid, "x,tn,tt,state,h\n");
  pmax = 0; ncl = 0; qmax = 0;
  for i = 1:nnos_contato
    noglobal = comp1(i);
    xno = NOS_FIS(noglobal, 2);
    ii = ceil(noglobal / 3);
    j = rem(noglobal, 3);
    if (j == 0); j = 3; end
    tn = T(ii, 2 * j);
    tt = T(ii, 2 * j + 1);
    st = 0;
    if i <= numel(contato); st = contato(i); end
    fprintf(fid, "%.12e,%.12e,%.12e,%d,%.12e\n", xno, tn, tt, st, h(i));
    if abs(st) ~= 1
      ncl = ncl + 1;
      pmax = max(pmax, -tn);
      qmax = max(qmax, abs(tt));
    end
  end
  fclose(fid);
  printf("wrote %s  n_closed=%d  p0=%.4f  max|tt|=%.4f\n", fname, ncl, pmax, qmax);
end
