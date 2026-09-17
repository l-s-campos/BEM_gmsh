# Juliá Lerma (2025) §3.3.2 — twin discs
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Printf

fine = get(ENV, "JULIA_LERMA_FINE", "false") == "true"
nx, ny = fine ? (41, 121) : (13, 39)
r = twin_discs(nx; nx=nx, ny=ny, nrev=fine ? 5 : 1)
@printf "nx×ny=%d×%d  P/P*=%.4f  Qx=%.4f  Qy=%.3e  wmax=%.4e\n" nx ny (1 - r.errP) r.Qx r.Qy r.w
