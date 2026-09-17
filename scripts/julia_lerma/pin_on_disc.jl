# Juliá Lerma (2025) §3.1 — pin-on-disc sliding wear
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Printf

const FINE = get(ENV, "JULIA_LERMA_FINE", "false") == "true"
N = FINE ? 81 : 33
h = pin_hertz(N; L=FINE ? PIN.L : 0.6)
@printf "Hertz N=%d  P/P_hz=%.4f  pmax/p0=%.4f  σVM/p0=%.3f (ana 0.62)\n" N (1 - h.errP) (1 - h.errp) (h.σVM / h.hz.p0)
w = pin_wear(N; L=PIN.L, Δs=FINE ? 1.0 : 8.0, nsteps=FINE ? 80 : 8)
@printf "Argatov s=%.1f mm  w/w_arg=%.3f  a/a_arg=%.3f\n" w.s w.w / w.w_arg w.a / w.a_arg
o = pin_orthotropic(N; β=π/4, Δs=FINE ? 1.0 : 4.0, nsteps=FINE ? 20 : 4)
@printf "orthotropic β=45°  Qy=%.4e  wmax=%.4e\n" o.Qy o.w
