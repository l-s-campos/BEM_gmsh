# Juliá Lerma (2025) §3.2.1 — spherical punch fretting
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Printf

N = get(ENV, "JULIA_LERMA_FINE", "false") == "true" ? 61 : 21
for β in (0.0, π / 2)
    r = spherical_fretting(N; β=β)
    @printf "β=%5.1f°  P=%.3f  Qx=%.4f  n_slip=%d/%d  wmax=%.3e\n" (β * 180 / π) r.P r.Qx r.n_slip r.n_contact r.w
end
