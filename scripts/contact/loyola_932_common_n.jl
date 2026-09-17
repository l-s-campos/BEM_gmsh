# §9.3.2 Step A with stiffness-weighted common normal n_AB.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.Contact
using BEM.MultiRegion
using LinearAlgebra, Statistics, Printf, Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "loyola_bulk_contact.jl"))

const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "loyola932_common_n")
mkpath(OUT)
const MR = BEM.MultiRegion
relerr(a, b) = abs(a - b) / max(abs(b), eps()) * 100

function pin_top_center_ux!(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0) || continue
        ax = abs(dad.Nodes[i][1])
        ax < bx && (bx = ax; best = i)
    end
    dad.BC[2best-1] = 0; dad.BV[2best-1] = 0.0
    return best
end

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10, verbose=true)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = zeros(ctx.N)
    for cp in pairs
        cp.state = 3; cp.ut_lock = 0.0
    end
    ok, nit, prev = false, 0, [cp.state for cp in pairs]
    for it in 1:maxiter
        nit = it
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        st = [cp.state for cp in pairs]
        n_flip = count(i -> st[i] != prev[i], eachindex(st))
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        verbose && @printf("  it=%2d  rel=%.3e  nflips=%d  o/s/l=%d/%d/%d\n", it, dist, n_flip,
            count(==(1), st), count(==(3), st), count(s -> abs(s) == 2, st))
        x0 = x; prev = st
        if dist < tol && n_flip == 0
            ok = true
            break
        elseif it >= 10 && n_flip <= 2 && dist < 0.2
            ok = true
            break
        end
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._update_contact_ut_locks!(pairs, prep, h, x0)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, nit)
end

par = loyola_bulk_params()
prob, _ = load_loyola_bulk_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
    tipo=2, nome="loyola932_nAB")
pin_top_center_ux!(prob)

# geometric n vs n_AB at pairing (before H,G / apply)
da, db = prob.regions
println("common-normal samples (pad n_geo vs n_AB), equal E:")
for cp in prob.contacts[1:3:end]
    nA, nB = da.Normal[cp.node_a], db.Normal[cp.node_b]
    nAB = contact_common_normal(nA, par.E, nB, par.E)
    x = da.Nodes[cp.node_a][1]
    abs(x) > 2.0 && continue
    @printf("  x=%+6.3f  n_geo=(%+.4f,%+.4f)  n_AB=(%+.4f,%+.4f)  Δangle=%.3f deg  gap0=%.5f\n",
        x, nA[1], nA[2], nAB[1], nAB[2],
        acosd(clamp(nA[1]*nAB[1] + nA[2]*nAB[2], -1, 1)), cp.gap0)
end

sol = contato_newton!(prob)
fr = contact_interface_xyτ(prob)
p = .-fr.tn; q = .-fr.tt
cl = abs.(fr.state) .!= 1
w = ones(length(fr.x)); w[1] = abs(fr.x[2]-fr.x[1]); w[end] = abs(fr.x[end]-fr.x[end-1])
for i in 2:length(fr.x)-1
    w[i] = 0.5 * abs(fr.x[i+1] - fr.x[i-1])
end
P = sum(p[cl] .* w[cl]); Q = sum(q[cl] .* w[cl]); p0 = maximum(p[cl])
μ = par.μ
util = mean(abs(fr.tt[i]) / max(μ * abs(fr.tn[i]), 1e-12) for i in eachindex(fr.state) if abs(fr.state[i]) != 1)
n_coul = count(i -> abs(fr.state[i]) != 1 && abs(fr.tt[i]) > 0.98 * μ * abs(fr.tn[i]), eachindex(fr.state))
println()
@printf("ok=%s it=%d  P=%.2f  Q=%+.2f  p0=%.2f (Hertz %.2f%%  thesis %.2f%%)\n",
    sol.ok, sol.nit, P, Q, p0, relerr(p0, par.p0_H), relerr(p0, par.p0_th))
@printf("st/sl/op=%d/%d/%d  coul=%d  util=%.3f  |q|_max/(f p0)=%.3f\n",
    count(==(3), fr.state), count(s -> abs(s)==2, fr.state), count(==(1), fr.state),
    n_coul, util, maximum(abs, q[cl]) / (μ * par.p0_H))
@printf("slave n after apply: sample n=(%.4f,%.4f)  master n=(%.4f,%.4f)\n",
    da.Normal[prob.contacts[div(end,2)].node_a]...,
    db.Normal[prob.contacts[div(end,2)].node_b]...)

aH, p0H = par.a_H, par.p0_H
default(linewidth=1.8, legendfontsize=8, grid=true, framestyle=:box)
xa = collect(range(-1.15aH, 1.15aH; length=400))
pH = cattaneo_pressure(xa, aH, p0H)
plt_p = plot(xa ./ aH, pH ./ p0H; color=:black, label="Hertz",
    xlabel="x/a", ylabel="p/p0", title="§9.3.2 A  common n_AB", ylims=(-0.05, 1.2))
scatter!(plt_p, fr.x ./ aH, p ./ p0H; ms=4, markerstrokewidth=0, label="BEM n_AB")
plt_q = plot(xlabel="x/a", ylabel="q/(f p0)", title="§9.3.2 A  shear", ylims=(-0.7, 0.7))
hline!(plt_q, [0.0]; color=:gray, ls=:dot, label=false)
scatter!(plt_q, fr.x ./ aH, q ./ (μ * p0H); ms=4, markerstrokewidth=0, label="BEM n_AB")
savefig(plt_p, joinpath(OUT, "p.png"))
savefig(plt_q, joinpath(OUT, "q.png"))
println("plots in $OUT")
