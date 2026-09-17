# Juliá Lerma (2025) §3.2.2 — cyclic flat punch
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Printf

N = get(ENV, "JULIA_LERMA_FINE", "false") == "true" ? 41 : 25
r0 = flat_punch_static(N; μ=0.0)
@printf "Sneddon N=%d  P_num/P=%.4f  kn/kn0=%.4f  pmax/pcen=%.3f\n" N r0.P / PUNCH.P r0.kn_ratio r0.pmax / r0.pcen
rμ = flat_punch_static(N; μ=0.4)
@printf "μ=0.4  kn/kn0=%.4f  Mossakovskii=%.4f\n" rμ.kn_ratio r0.moss
cy = flat_punch_cycle(N; μ=0.2)
@printf "one cycle μ=0.2  wmax=%.4e\n" cy.w
