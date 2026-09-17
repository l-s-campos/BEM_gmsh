# 9.3.2: implementation suspects vs Loyola BEM (p0 err <1%, q matches Nowell).
# A) rigid pad-top ux=0 (holder, no stretch)
# B) common contact normal n=(0,±1), gap along ey
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Statistics, Printf, Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "loyola_bulk_contact.jl"))

const MR = BEM.MultiRegion
const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "wiggle_diag")
mkpath(OUT)

function contato_as!(prob; maxiter=40, npg=10)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    for cp in pairs
        cp.state = 3
        cp.ut_lock = 0.0
    end
    x0 = zeros(ctx.N)
    prev = fill(3, length(pairs))
    ok = false
    nit = 0
    for it in 1:maxiter
        nit = it
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        st = [cp.state for cp in pairs]
        n_flip = count(i -> st[i] != prev[i], eachindex(st))
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        x0 = x
        prev = st
        if dist < 1e-9 && n_flip == 0
            ok = true
            break
        elseif it >= 10 && n_flip <= 2 && dist < 0.2
            ok = true
            break
        end
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    return (; ok, nit)
end

function summarize(tag, prob, par)
    fr = contact_interface_xyτ(prob)
    cl = abs.(fr.state) .!= 1
    p = .-fr.tn; q = .-fr.tt
    μ = par.μ
    util = [abs(q[i]) / max(μ * abs(p[i]), 1e-12) for i in findall(cl)]
    @printf("%-28s  ok-n/a  closed=%d st/sl=%d/%d  p0=%.1f (Hertz %.1f, %+.1f%%)  mean|q|=%.1f  mean util=%.3f  n(util>0.95)=%d\n",
        tag, count(cl),
        count(==(3), fr.state[cl]), count(s -> abs(s)==2, fr.state[cl]),
        maximum(p[cl]; init=0.0), par.p0_H,
        100 * (maximum(p[cl]; init=0.0) / par.p0_H - 1),
        mean(abs, q[cl]), isempty(util) ? 0.0 : mean(util), count(>(0.95), util))
    return fr, p, q
end

par = loyola_bulk_params()
aH, p0H = par.a_H, par.p0_H
fp0 = par.μ * p0H

function loadA(; nome, rigid_top::Bool=false, common_n::Bool=false)
    prob, _ = load_loyola_bulk_contact(; μ=0.3, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        tipo=2, nome=nome)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    if rigid_top
        # holder: whole top ux=0 (no stretch), ty stays -cargav
        nset = 0
        for i in 1:dad.n
            if abs(dad.Nodes[i][2] - ymax) <= thr
                dad.BC[2i-1] = 0
                dad.BV[2i-1] = 0.0
                nset += 1
            end
        end
        @printf("  rigid top ux=0 on %d nodes\n", nset)
    else
        # single centre pin (current 932 A)
        best, bx = 0, Inf
        for i in 1:dad.n
            abs(dad.Nodes[i][2] - ymax) <= thr || continue
            ax = abs(dad.Nodes[i][1])
            ax < bx && (bx = ax; best = i)
        end
        dad.BC[2best-1] = 0
        dad.BV[2best-1] = 0.0
    end
    if common_n
        for i in 1:prob.regions[1].n
            if prob.regions[1].BC[2i-1] == 4 || prob.regions[1].BC[2i] == 4
                prob.regions[1].Normal[i] = Point2D(0.0, -1.0)
            end
        end
        for i in 1:prob.regions[2].n
            if prob.regions[2].BC[2i-1] == 4 || prob.regions[2].BC[2i] == 4
                prob.regions[2].Normal[i] = Point2D(0.0, 1.0)
            end
        end
        for cp in prob.contacts
            pa = prob.regions[1].Nodes[cp.node_a]
            pb = prob.regions[2].Nodes[cp.node_b]
            cp.gap0 = max(0.0, pb[2] - pa[2])   # along +ey, specimen above? specimen y=0, pad y>0
            # pad is above specimen: pa.y >= 0, pb.y = 0, gap = pa.y - pb.y along -n_pad
            # n_pad=(0,-1), (pb-pa)·n = (0 - ypad)*(-1) = ypad. Use that.
            cp.gap0 = max(0.0, pa[2] - pb[2])
        end
        @printf("  common n, gap0 ∈ [%.5f, %.5f]\n", extrema(cp.gap0 for cp in prob.contacts)...)
    end
    sol = contato_as!(prob)
    @printf("  converged=%s it=%d\n", sol.ok, sol.nit)
    return prob
end

println("="^72)
cases = []
println("\n[1] current 932 A: centre pin only")
p1 = loadA(; nome="i932pin")
push!(cases, ("centre pin", p1, summarize("centre pin", p1, par)...))

println("\n[2] rigid holder: ALL pad-top ux=0")
p2 = loadA(; nome="i932rig", rigid_top=true)
push!(cases, ("rigid top ux=0", p2, summarize("rigid top ux=0", p2, par)...))

println("\n[3] common normal n=(0,±1) + centre pin")
p3 = loadA(; nome="i932cn", common_n=true)
push!(cases, ("common n", p3, summarize("common n", p3, par)...))

println("\n[4] rigid top AND common normal")
p4 = loadA(; nome="i932both", rigid_top=true, common_n=true)
push!(cases, ("rigid+common n", p4, summarize("rigid+common n", p4, par)...))

pltq = plot(xlabel="x/a", ylabel="q/(f p0)", title="Step A shear — BC / normal frame",
    xlims=(-1.3, 1.3), ylims=(-1.2, 1.2), legend=:top, size=(720, 420))
hline!(pltq, [0.0, 1.0, -1.0]; color=:gray, ls=:dash, label=false)
pltp = plot(xlabel="x/a", ylabel="p/p0", title="Step A pressure",
    xlims=(-1.3, 1.3), ylims=(-0.05, 1.2), legend=:bottom, size=(720, 420))
xh = range(-1.2aH, 1.2aH; length=300)
plot!(pltp, collect(xh) ./ aH, [abs(x) < aH ? sqrt(1-(x/aH)^2) : 0.0 for x in xh];
    color=:black, lw=2, label="Hertz")
cols = [:steelblue, :green, :orange, :red]
for ((lab, _, fr, p, q), col) in zip(cases, cols)
    cl = abs.(fr.state) .!= 1
    scatter!(pltq, fr.x[cl] ./ aH, q[cl] ./ fp0; ms=4, color=col, label=lab)
    scatter!(pltp, fr.x[cl] ./ aH, p[cl] ./ p0H; ms=3, color=col, label=lab)
end
savefig(pltq, joinpath(OUT, "932_impl_q.png"))
savefig(pltp, joinpath(OUT, "932_impl_p.png"))
println("\n[5] rigid platen: all top ux=0 AND uy=uyA (from run 1)")
let
    # uyA from centre-pin frictional A
    dad = p1.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    uys = Float64[]
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= thr || continue
        has_cache(dad, :u) && push!(uys, dad.u[2i])
    end
    uyA = mean(uys)
    @printf("  uyA=%.6f from pin run; apply as rigid platen\n", uyA)
    p5, _ = load_loyola_bulk_contact(; μ=0.3, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        tipo=2, nome="i932plat")
    apply_pad_top_u!(p5, 0.0; uy=uyA)
    sol = contato_as!(p5)
    @printf("  converged=%s it=%d\n", sol.ok, sol.nit)
    summarize("rigid platen ux=uy", p5, par)
    fr = contact_interface_xyτ(p5)
    cl = abs.(fr.state) .!= 1
    p = .-fr.tn; q = .-fr.tt
    scatter!(pltq, fr.x[cl] ./ aH, q[cl] ./ fp0; ms=4, color=:purple, label="rigid platen")
    scatter!(pltp, fr.x[cl] ./ aH, p[cl] ./ p0H; ms=3, color=:purple, label="rigid platen")
    savefig(pltq, joinpath(OUT, "932_impl_q.png"))
    savefig(pltp, joinpath(OUT, "932_impl_p.png"))
    # ut of pad contact
    println("  pad contact ut (global ux) on closed nodes:")
    dad = p5.regions[1]
    xs = Float64[]; uts = Float64[]; uns = Float64[]
    for cp in p5.contacts
        abs(cp.state)==1 && continue
        i = cp.node_a
        push!(xs, dad.Nodes[i][1])
        push!(uts, dad.u[2i-1])
        push!(uns, dad.u[2i])
    end
    perm = sortperm(xs)
    for k in perm[1:3:end]
        @printf("    x=%+7.4f  ux=%+.5f  uy=%+.5f\n", xs[k], uts[k], uns[k])
    end
end
