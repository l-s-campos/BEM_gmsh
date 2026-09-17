# Nodal tipo=2 vs GL-weighted element mean vs tipo=1 vs Hertz.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf, Plots, Statistics

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

const MR = BEM.MultiRegion
const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "wiggle_diag")
mkpath(OUT)

function contato_as!(prob; tol=1e-9, maxiter=80, npg=10)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    for cp in pairs
        cp.state = 3
        cp.ut_lock = 0.0
    end
    x0 = zeros(ctx.N)
    ok = false
    for it in 1:maxiter
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        x0 = x
        dist < tol && (ok = true; break)
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    return nothing
end

function pin_top_ux!(dad)
    ymax = maximum(pt[2] for pt in dad.Nodes)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0) || continue
        ax = abs(dad.Nodes[i][1])
        ax < bx && (bx = ax; best = i)
    end
    dad.BC[2best - 1] = 0
    dad.BV[2best - 1] = 0.0
    return best
end

function setup_hertz(tipo)
    par5 = loyola_dad5d_params()
    prob, _ = load_dad_5d_contact(; μ=0.0, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        tipo=tipo, nome="wiggle_plot$tipo")
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0)
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i]   = 1; dad.BV[2i]   = -par5.cargav
        end
    end
    pin_top_ux!(dad)
    contato_as!(prob; npg=10)
    return prob, par5
end

function gl_element_mean(prob)
    dad = prob.regions[1]
    w = collect(Float64, dad.elem_weight)
    cmap = Dict{Int,Float64}(cp.node_a => -cp.tn for cp in prob.contacts)
    xs = Float64[]; ps = Float64[]
    for el in dad.elements
        ids = el.index
        all(haskey(cmap, i) for i in ids) || continue
        pp = [cmap[i] for i in ids]
        all(p -> p > 1.0, pp) || continue
        xmid = mean(dad.Nodes[i][1] for i in ids)
        pbar = sum(w[k] * pp[k] for k in eachindex(ids)) / sum(w)
        push!(xs, xmid); push!(ps, pbar)
    end
    perm = sortperm(xs)
    return xs[perm], ps[perm]
end

prob2, par5 = setup_hertz(2)
prob1, _ = setup_hertz(1)
aH, p0H = par5.a_H, par5.p0_H
phertz(x) = abs(x) < aH ? p0H * sqrt(max(0.0, 1 - (x / aH)^2)) : 0.0

fr2 = contact_interface_xyτ(prob2)
fr1 = contact_interface_xyτ(prob1)
xm, pm = gl_element_mean(prob2)

# weighted mean vs Hertz at element mid
@printf("tipo=2 nodal     max p=%.2f  (Hertz p0=%.2f, %+ .2f%%)\n",
    maximum(-fr2.tn), p0H, 100 * (maximum(-fr2.tn) / p0H - 1))
@printf("tipo=2 GL-mean   max p=%.2f  (%+ .2f%%)\n",
    maximum(pm), 100 * (maximum(pm) / p0H - 1))
@printf("tipo=1 nodal     max p=%.2f  (%+ .2f%%)\n",
    maximum(-fr1.tn), 100 * (maximum(-fr1.tn) / p0H - 1))
cl = abs.(fr2.state) .!= 1
pref = phertz.(fr2.x[cl])
@printf("tipo=2 nodal  L2 vs Hertz = %.4f\n",
    sqrt(mean(abs2, (-fr2.tn[cl]) .- pref)) / p0H)
pref1 = phertz.(fr1.x[abs.(fr1.state).!=1])
@printf("tipo=1 nodal  L2 vs Hertz = %.4f\n",
    sqrt(mean(abs2, (-fr1.tn[abs.(fr1.state).!=1]) .- pref1)) / p0H)
prefm = phertz.(xm)
@printf("tipo=2 GL-mean L2 vs Hertz = %.4f\n",
    sqrt(mean(abs2, pm .- prefm)) / p0H)

xa = range(-1.4 * aH, 1.4 * aH; length=400)
plt = plot(xa ./ aH, phertz.(xa) ./ p0H; color=:black, lw=2.5, label="Hertz",
    xlabel="x/a", ylabel="p/p₀", title="Step A — nodal quadratic vs element mean",
    legend=:bottom, size=(720, 480))
plot!(plt, fr2.x ./ aH, (-fr2.tn) ./ p0H; color=:steelblue, lw=1.2, marker=:circle,
    ms=4, label="tipo=2 nodal (3 GL / elem)")
plot!(plt, xm ./ aH, pm ./ p0H; color=:orange, lw=2, marker=:diamond, ms=6,
    label="tipo=2 GL-weighted element mean")
plot!(plt, fr1.x ./ aH, (-fr1.tn) ./ p0H; color=:green, lw=0, marker=:utriangle,
    ms=5, label="tipo=1 nodal")
xlims!(plt, -1.5, 1.5); ylims!(plt, -0.02, 1.12)
savefig(plt, joinpath(OUT, "hertz_nodal_vs_element_mean.png"))
savefig(plt, joinpath(OUT, "hertz_nodal_vs_element_mean.pdf"))
println("wrote ", joinpath(OUT, "hertz_nodal_vs_element_mean.png"))
