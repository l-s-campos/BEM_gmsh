# Juliá Lerma (2025) §3.3.1 — tractive rolling of two identical spheres
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Printf

N = get(ENV, "JULIA_LERMA_FINE", "false") == "true" ? 91 : 31
r = rolling_spheres(N)
@printf "iso N=%d  P/P*=%.4f  a0=%.3f (3.5)  Qx/μP=%.3f  Qy=%.2e\n" N (1 - r.errP) r.hz.a r.Qx_over_μP r.Qy
o = rolling_spheres(min(N, 41); β=π / 4)
@printf "β=45°  Qx=%.4f  Qy=%.4e  (Qy must be ≠ 0)\n" o.Qx o.Qy
