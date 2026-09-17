# 12×6 bar, Δt=0.1, 4 periods: DRM f=r vs DRM MQ vs DIBEM r vs DIBEM MQ.
# julia --project=. scripts/plot_bar12x6_dibem_drm.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.6, framestyle=:box,
    grid=false, dpi=160, size=(880, 400), legendfontsize=8)

include(joinpath(@__DIR__, "plot_bar_rect12x6.jl"))

const Δt = 0.1
const Cmq = 0.01   # Samaan & Rashed (2007)

dad0 = make_dad()
@printf("elements=%d n=%d ni=%d nt=%d  Δt=%.2f  tf=%.0f  C=%.2g\n",
    length(dad0.elements), dad0.n, dad0.ni, dad0.nt, Δt, tf, Cmq)
H_G_full_direct(dad0; npg=12, threaded=false)
ana = ana_bar_sudden(; N=400, c=1.0, L=Lx)

function clip_blow(ux; lim=50)
    v = copy(ux)
    i0 = findfirst(i -> !isfinite(v[i]) || abs(v[i]) > lim, eachindex(v))
    if i0 !== nothing
        v[i0:end] .= NaN
    end
    return v, i0
end

function report(name, ux, ua, sys; t=nothing)
    ev = real.(eigvals(sys.M))
    uxc, i0 = clip_blow(ux)
    nn = count(<( -1e-8), ev)
    @printf("%-12s nneg(M̄)=%d  minλ=%.3e  max=%.3f  rel=%.3e  finite=%s  blow=%s\n",
        name, nn, minimum(ev),
        maximum(abs, filter(isfinite, ux); init=0.0),
        all(isfinite, ux) ? rel(ux, ua) : NaN,
        all(isfinite, ux),
        i0 === nothing ? "—" : @sprintf("t=%.2f step %d", t[i0], i0))
    return uxc
end

# DIBEM interpolates the whole integrand (Monta_M_RIMd).
# M_RIMd is used as H u + M ü = G t, so condensed mass_sign = −1.
dadDr = deepcopy(dad0)
DIBEM(dadDr; method=:dense, rbf=PHS(1; poly_deg=-1))
UDr, t, sysDr = houbolt_condensed!(dadDr, Δt, tf; mass_sign=-1)
ua = [ana.u(PROBE; t=ti) for ti in t]
uxDr = report("DIBEM r", probe_ux(UDr, dadDr), ua, sysDr; t=t)

dadDm = deepcopy(dad0)
DIBEM(dadDm; method=:dense, rbf=MQ(; C=Cmq, poly_deg=-1))
UDm, _, sysDm = houbolt_condensed!(dadDm, Δt, tf; mass_sign=-1)
uxDm = report("DIBEM MQ", probe_ux(UDm, dadDm), ua, sysDm; t=t)

dadRr = deepcopy(dad0)
drm_mass!(dadRr; flip=true, kernel=:r)
URr, _, sysRr = houbolt_condensed!(dadRr, Δt, tf)
uxRr = report("DRM r", probe_ux(URr, dadRr), ua, sysRr; t=t)

dadRm = deepcopy(dad0)
drm_mass!(dadRm; flip=true, kernel=:mq, C=Cmq)
URm, _, sysRm = houbolt_condensed!(dadRm, Δt, tf)
uxRm = report("DRM MQ", probe_ux(URm, dadRm), ua, sysRm; t=t)

plt = plot(t, ua; color=:black, ls=:dash, label="1D series")
plot!(plt, t, uxDr; color=:steelblue, label="DIBEM (\$r\$)")
plot!(plt, t, uxDm; color=:royalblue, ls=:dashdot, label="DIBEM (MQ, \$C=$Cmq\$)")
plot!(plt, t, uxRr; color=:darkorange, label="DRM (\$f=r\$)")
plot!(plt, t, uxRm; color=:firebrick, ls=:dot, label="DRM (MQ, \$C=$Cmq\$)")
plot!(plt; xlabel=L"t", ylabel=L"u_x(L, H/2)", ylim=(-1, 28),
    legend=:outertopright,
    title="12×6 bar  Houbolt  \$\\Delta t=0.1\$  4 periods")
mkpath(joinpath(projectdir(), "plots"))
out = joinpath(projectdir(), "plots", "bar12x6_dibem_drm_dt01")
savefig(plt, out * ".png")
savefig(plt, out * ".pdf")
println("wrote ", out * ".png")
