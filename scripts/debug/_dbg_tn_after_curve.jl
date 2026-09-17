# Re-check Hertz tn after CAD-aware quadratic mesh.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Statistics, Printf, Plots

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
    for it in 1:maxiter
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        x0 = x
        dist < tol && break
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
    dad.BC[2best-1] = 0
    dad.BV[2best-1] = 0.0
    return best
end

par5 = loyola_dad5d_params()
aH, p0H = par5.a_H, par5.p0_H
phertz(x) = abs(x) < aH ? p0H * sqrt(max(0.0, 1 - (x / aH)^2)) : 0.0

prob, _ = load_dad_5d_contact(; μ=0.0, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
    tipo=2, nome="tn_fix")
dad = prob.regions[1]
ymax = maximum(pt[2] for pt in dad.Nodes)
for i in 1:dad.n
    if abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0)
        dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
        dad.BC[2i]   = 1; dad.BV[2i]   = -par5.cargav
    end
end
pin_top_ux!(dad)
contato_as!(prob)

cmap = Dict(cp.node_a => cp for cp in prob.contacts)
println("per-element n_x and gap0 (should now vary); p=-tn")
oscs = Float64[]
for el in dad.elements
    ids = el.index
    all(haskey(cmap, i) for i in ids) || continue
    cps = [cmap[i] for i in ids]
    all(abs(cp.tn) > 1 for cp in cps) || continue
    xs = [dad.Nodes[i][1] for i in ids]
    maximum(abs, xs) > 0.3 && continue
    nx = [dad.Normal[i][1] for i in ids]
    gs = [cp.gap0 for cp in cps]
    ps = [-cp.tn for cp in cps]
    osc = ps[2] - 0.5 * (ps[1] + ps[3])
    push!(oscs, osc)
    @printf("  x=%s  n_x=%s  gap0=%s  p=%s  Δp=%+.1f\n",
        string(round.(xs; digits=4)),
        string(round.(nx; digits=5)),
        string(round.(gs; digits=6)),
        string(round.(ps; digits=1)), osc)
end
fr = contact_interface_xyτ(prob)
cl = abs.(fr.state) .!= 1
p = .-fr.tn
@printf("\nclosed=%d  p∈[%.1f, %.1f]  p0/Hertz=%+.2f%%  mean(mid-edge Δp)=%+.2f  max|Δp|=%.2f\n",
    count(cl), minimum(p[cl]), maximum(p[cl]),
    100 * (maximum(p[cl]) / p0H - 1),
    isempty(oscs) ? 0.0 : mean(oscs),
    isempty(oscs) ? 0.0 : maximum(abs, oscs))
pref = phertz.(fr.x[cl])
@printf("L2 vs Hertz = %.4f\n", sqrt(mean(abs2, p[cl] .- pref)) / p0H)

plt = plot(range(-1.4aH, 1.4aH; length=400) ./ aH,
    phertz.(range(-1.4aH, 1.4aH; length=400)) ./ p0H;
    color=:black, lw=2.5, label="Hertz", xlabel="x/a", ylabel="p/p₀",
    legend=:bottom, title="tipo=2 after CAD-aware quadratic mesh")
plot!(plt, fr.x ./ aH, p ./ p0H; marker=:circle, ms=4, lw=1, color=:steelblue,
    label="two-body −tn")
xlims!(-1.5, 1.5); ylims!(-0.02, 1.12)
savefig(plt, joinpath(OUT, "tn_after_cad_order.png"))
println("wrote ", joinpath(OUT, "tn_after_cad_order.png"))
