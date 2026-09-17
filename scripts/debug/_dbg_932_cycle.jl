# §9.3.2: which pair cycles, and why q scatters.
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

function pin_pad_ux!(prob)
    dad = prob.regions[1]
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

function contato_track!(prob; tol=1e-9, maxiter=40, npg=10, verbose=true)
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
    dad = prob.regions[1]
    for it in 1:maxiter
        nit = it
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        st = [cp.state for cp in pairs]
        flips = findall(i -> st[i] != prev[i], eachindex(st))
        if verbose && (!isempty(flips) || it <= 8 || it == maxiter)
            @printf("  it=%2d  o/s/l=%d/%d/%d  nflips=%d\n", it,
                count(==(1), st), count(==(3), st), count(s -> abs(s)==2, st),
                length(flips))
            for k in flips
                cp = pairs[k]
                x = dad.Nodes[cp.node_a][1]
                nx = sum(p.ndof for p in prep)
                kin = MR._contact_pair_kinematics(prep, cp, h[k], x0, k, nx)
                util = abs(kin.tt1) / max(cp.μ * abs(kin.tn1), 1e-12)
                @printf("    flip %3d  x=%+7.4f  %d→%d  tn=%+8.2f tt=%+8.2f  |tt|/μ|tn|=%.3f  gn=%+.3e  gt=%+.3e\n",
                    k, x, prev[k], st[k], kin.tn1, kin.tt1, util, kin.gn, kin.gt)
            end
        end
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        x0 = x
        prev = st
        dist < tol && length(flips) == 0 && (ok = true; break)
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, nit, x=x0, prep, pairs, h)
end

function dump_interface(tag, prob; aH)
    dad1 = prob.regions[1]
    dad2 = prob.regions[2]
    println("\n  $tag  closed nodes (x, p, q, util, state, Δx, n1x, n2x)")
    rows = []
    for cp in prob.contacts
        abs(cp.state) == 1 && continue
        i, j = cp.node_a, cp.node_b
        x = dad1.Nodes[i][1]
        p = -cp.tn
        q = -cp.tt
        util = abs(cp.tt) / max(cp.μ * abs(cp.tn), 1e-12)
        dx = dad1.Nodes[i][1] - dad2.Nodes[j][1]
        n1 = dad1.Normal[i]; n2 = dad2.Normal[j]
        push!(rows, (; x, p, q, util, st=cp.state, dx, n1x=n1[1], n2x=n2[1],
            n1y=n1[2], n2y=n2[2], i, j))
    end
    sort!(rows; by=r -> r.x)
    nslip = count(r -> abs(r.st)==2, rows)
    nstick = count(r -> r.st==3, rows)
    p0 = maximum(r.p for r in rows; init=0.0)
    qmax = maximum(abs(r.q) for r in rows; init=0.0)
    @printf("  n_closed=%d stick=%d slip=%d  p0=%.1f  |q|_max=%.1f  mean util=%.3f\n",
        length(rows), nstick, nslip, p0, qmax,
        isempty(rows) ? 0.0 : mean(r.util for r in rows))
    # Hertz-zone only
    hz = [r for r in rows if abs(r.x) < 1.05aH]
    !isempty(hz) && @printf("  Hertz-zone  n=%d  mean util=%.3f  n(util>0.95)=%d  max|Δx|=%.3e  n1x∈[%.4f,%.4f]\n",
        length(hz), mean(r.util for r in hz), count(r -> r.util > 0.95, hz),
        maximum(abs(r.dx) for r in hz),
        minimum(r.n1x for r in hz), maximum(r.n1x for r in hz))
    # print every node in Hertz zone
    for r in hz
        @printf("    x=%+7.4f  p=%7.1f  q=%+7.1f  util=%.3f  st=%2d  Δx=%+.2e  n1x=%+.4f n2x=%+.4f\n",
            r.x, r.p, r.q, r.util, r.st, r.dx, r.n1x, r.n2x)
    end
    return rows
end

function element_q_osc(prob)
    dad = prob.regions[1]
    cmap = Dict(cp.node_a => cp for cp in prob.contacts)
    oscp = Float64[]; oscq = Float64[]
    for el in dad.elements
        ids = el.index
        length(ids)==3 || continue
        all(haskey(cmap, i) for i in ids) || continue
        cps = [cmap[i] for i in ids]
        all(abs(cp.state)!=1 for cp in cps) || continue
        ps = [-cp.tn for cp in cps]
        qs = [-cp.tt for cp in cps]
        push!(oscp, ps[2] - 0.5*(ps[1]+ps[3]))
        push!(oscq, qs[2] - 0.5*(qs[1]+qs[3]))
    end
    @printf("  elem osc  Δp mid-edge: mean=%+.1f max||=%.1f   Δq: mean=%+.1f max||=%.1f  n=%d\n",
        isempty(oscp) ? 0.0 : mean(oscp), isempty(oscp) ? 0.0 : maximum(abs, oscp),
        isempty(oscq) ? 0.0 : mean(oscq), isempty(oscq) ? 0.0 : maximum(abs, oscq),
        length(oscp))
end

par = loyola_bulk_params()
aH, p0H = par.a_H, par.p0_H

function setup(; μ=0.3, tipo=2, nome="c932")
    prob, _ = load_loyola_bulk_contact(; μ=μ, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        ndiv_out=6, ndiv_bot=12, ndiv_bulk=10, tipo=tipo, nome=nome)
    pin_pad_ux!(prob)
    return prob
end

println("="^72)
println("[1] Default Step A  μ=0.3 tipo=2  (the cycling run)")
prob = setup(; nome="c932A")
sol = contato_track!(prob; maxiter=25, verbose=true)
@printf("  ok=%s it=%d\n", sol.ok, sol.nit)
dump_interface("A default", prob; aH=aH)
element_q_osc(prob)

println("\n[2] Frictionless Step A  μ=0 tipo=2")
prob0 = setup(; μ=0.0, nome="c932mu0")
sol0 = contato_track!(prob0; maxiter=20, verbose=true)
@printf("  ok=%s it=%d\n", sol0.ok, sol0.nit)
dump_interface("A μ=0", prob0; aH=aH)
element_q_osc(prob0)

println("\n[3] Poisson blocked: specimen left AND right ux=0, μ=0.3 tipo=2")
probR = setup(; nome="c932roll")
let
    dad2 = probR.regions[2]
    xmin = minimum(pt[1] for pt in dad2.Nodes)
    xmax = maximum(pt[1] for pt in dad2.Nodes)
    thr = 1e-9 * max(abs(xmax), 1.0)
    nL = 0
    nR = 0
    for i in 1:dad2.n
        x = dad2.Nodes[i][1]
        if abs(x - xmin) <= thr
            dad2.BC[2i-1] = 0; dad2.BV[2i-1] = 0.0
            nL += 1
        elseif abs(x - xmax) <= thr
            dad2.BC[2i-1] = 0; dad2.BV[2i-1] = 0.0
            nR += 1
        end
    end
    @printf("  pinned ux on left %d + right %d nodes\n", nL, nR)
end
solR = contato_track!(probR; maxiter=25, verbose=false)
@printf("  ok=%s it=%d\n", solR.ok, solR.nit)
dump_interface("A rollers both sides", probR; aH=aH)
element_q_osc(probR)

println("\n[4] tipo=1 μ=0.3  (linear, no 3-node facet bubble)")
prob1 = setup(; tipo=1, nome="c932t1")
sol1 = contato_track!(prob1; maxiter=25, verbose=false)
@printf("  ok=%s it=%d\n", sol1.ok, sol1.nit)
dump_interface("A tipo=1", prob1; aH=aH)

println("\n[5] element map around p-dips (μ=0 run)")
let
    dad = prob0.regions[1]
    cmap = Dict(cp.node_a => cp for cp in prob0.contacts)
    for el in dad.elements
        ids = el.index
        xs = [dad.Nodes[i][1] for i in ids]
        any(x -> abs(abs(x) - 0.80) < 0.12, xs) || continue
        ps = [haskey(cmap, i) ? -cmap[i].tn : NaN for i in ids]
        @printf("  el nodes=%s  x=%s  p=%s  n_x=%s\n",
            string(ids), string(round.(xs; digits=4)),
            string(round.(ps; digits=1)),
            string(round.([dad.Normal[i][1] for i in ids]; digits=4)))
    end
end

# plots: q/fp0 at A for the four cases
fp0 = par.μ * p0H
function xyq(prob)
    fr = contact_interface_xyτ(prob)
    cl = abs.(fr.state) .!= 1
    return fr.x[cl] ./ aH, (.-fr.tn[cl]) ./ p0H, (.-fr.tt[cl]) ./ fp0
end
plt = plot(xlabel="x/a", ylabel="q / (f p0)", title="Step A shear — what drives scatter",
    xlims=(-1.3, 1.3), ylims=(-1.2, 1.2), legend=:top, size=(720, 420))
hline!(plt, [0.0, 1.0, -1.0]; color=:gray, ls=:dash, label=false)
for (pr, lab, col) in (
        (prob, "default μ=0.3 t2", :steelblue),
        (prob0, "μ=0 t2", :black),
        (probR, "both-side ux=0", :green),
        (prob1, "tipo=1", :orange))
    ξ, _, qq = xyq(pr)
    scatter!(plt, ξ, qq; ms=4, color=col, label=lab)
end
savefig(plt, joinpath(OUT, "932_q_at_A.png"))

plt2 = plot(xlabel="x/a", ylabel="p / p0", title="Step A pressure",
    xlims=(-1.3, 1.3), ylims=(-0.05, 1.2), legend=:bottom, size=(720, 420))
xh = range(-1.2aH, 1.2aH; length=300)
ph = [abs(x)<aH ? sqrt(1-(x/aH)^2) : 0.0 for x in xh]
plot!(plt2, xh ./ aH, ph; color=:black, lw=2, label="Hertz")
for (pr, lab, col) in (
        (prob, "default", :steelblue),
        (prob0, "μ=0", :forestgreen),
        (probR, "ux=0 both sides", :orange))
    ξ, pp, _ = xyq(pr)
    scatter!(plt2, ξ, pp; ms=3, color=col, label=lab)
end
savefig(plt2, joinpath(OUT, "932_p_at_A.png"))
println("\nwrote $OUT/932_q_at_A.png  932_p_at_A.png")
