# Diagnose 3-point sawtooth in two-body contact p(x) (Loyola 9.3.1 Step A).
# Run:  julia --project=. scripts/debug/_dbg_contact_wiggle.jl
ENV["GKSwstype"] = "100"

using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra
using Statistics
using Printf
using Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))
include(datadir("Laplace", "Laplace_dad.jl"))

const MR = BEM.MultiRegion
const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "wiggle_diag")
mkpath(OUT)

function contato_as!(prob; tol=1e-9, maxiter=80, npg=10, verbose=false)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    for cp in pairs
        cp.state = 3
        cp.ut_lock = 0.0
    end
    x0 = zeros(ctx.N)
    ok = false
    nit = 0
    for it in 1:maxiter
        nit = it
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        verbose && @printf("    it=%2d  rel=%.3e  o/s/l=%d/%d/%d\n", it, dist,
            count(cp -> cp.state == 1, pairs),
            count(cp -> cp.state == 3, pairs),
            count(cp -> abs(cp.state) == 2, pairs))
        x0 = x
        dist < tol && (ok = true; break)
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    return (; ok, nit, x=x0, prep, pairs, h)
end

function pin_top_ux!(dad)
    ymax = maximum(pt[2] for pt in dad.Nodes)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0) || continue
        ax = abs(dad.Nodes[i][1])
        ax < bx && (bx = ax; best = i)
    end
    best == 0 && error("no top node")
    dad.BC[2best - 1] = 0
    dad.BV[2best - 1] = 0.0
    return best
end

function iface_p(prob)
    fr = contact_interface_xyτ(prob)
    return (x = fr.x, p = .-fr.tn, q = .-fr.tt, state = fr.state, tn = fr.tn, tt = fr.tt)
end

"""Group contact nodes of region 1 by element; report mid vs edge pressure."""
function element_triplets(prob)
    dad = prob.regions[1]
    cmap = Dict{Int,Int}()
    for (k, cp) in enumerate(prob.contacts)
        cmap[cp.node_a] = k
    end
    trips = NamedTuple[]
    for el in dad.elements
        ids = el.index
        all(haskey(cmap, i) for i in ids) || continue
        xs = [dad.Nodes[i][1] for i in ids]
        ps = [-prob.contacts[cmap[i]].tn for i in ids]
        ξs = collect(dad.element_type.nodes)
        # mid = smallest |ξ|
        imid = argmin(abs.(ξs))
        iedge = [j for j in eachindex(ξs) if j != imid]
        push!(trips, (; ids, xs, ps, ξs, p_mid = ps[imid], p_edge = mean(ps[iedge]),
            osc = ps[imid] - mean(ps[iedge])))
    end
    return trips
end

function bie_resid(prep, pairs, x)
    nx = sum(p.ndof for p in prep)
    out = NamedTuple[]
    for (ir, pr) in enumerate(prep)
        dad = pr.dad
        nd = pr.ndof
        u_loc = zeros(nd)
        t_loc = zeros(nd)
        BC, BV = pr.BC_ext, pr.BV_ext
        xr = x[pr.off+1:pr.off+nd]
        for dof in 1:nd
            inode = cld(dof, 2)
            if pr.is_contact_node[inode]
                u_loc[dof] = xr[dof]
            elseif BC[dof] == 0
                u_loc[dof] = BV[dof]
                t_loc[dof] = xr[dof]
            else
                t_loc[dof] = BV[dof]
                u_loc[dof] = xr[dof]
            end
        end
        for (k, cp) in enumerate(pairs)
            ot = nx + 4(k - 1)
            if cp.reg_a == ir
                t_loc[2cp.node_a-1] = x[ot+1]
                t_loc[2cp.node_a]   = x[ot+2]
            end
            if cp.reg_b == ir
                t_loc[2cp.node_b-1] = x[ot+3]
                t_loc[2cp.node_b]   = x[ot+4]
            end
        end
        Hloc, Gloc = transform_HG_local(dad.H, dad.G, dad)
        r = Hloc[1:nd, 1:nd] * u_loc - Gloc[1:nd, 1:nd] * t_loc
        rmax = maximum(abs, r)
        rrms = sqrt(mean(abs2, r))
        rc = Float64[]
        for i in 1:dad.n
            pr.is_contact_node[i] || continue
            push!(rc, abs(r[2i-1]), abs(r[2i]))
        end
        push!(out, (; ir, rmax, rrms, rcmax = isempty(rc) ? 0.0 : maximum(rc),
            rcrms = isempty(rc) ? 0.0 : sqrt(mean(abs2, rc))))
    end
    return out
end

function guiggiani_a_report(dad; nshow=12)
    poly = dad.element_type
    ξpoly = collect(poly.nodes)
    nbad = 0
    nchk = 0
    da = Float64[]
    ddist = Float64[]
    for el in dad.elements
        xj = dad.Nodes[el.index]
        for (k, src) in enumerate(el.index)
            pf = dad.Nodes[src]
            ξ0 = BEM._seed_1d(poly, xj, pf)
            a, _, dist = BEM.closest_point_1d(poly, xj, pf; ξ0=ξ0)
            nchk += 1
            ξexp = ξpoly[k]
            push!(da, abs(a - ξexp))
            push!(ddist, dist)
            if abs(a - ξexp) > 1e-6 || dist > 1e-10
                nbad += 1
                if nbad <= nshow
                    @printf("    BAD el nodes=%s k=%d ξexp=%+.5f a=%+.5f dist=%.3e seed=%+.5f\n",
                        string(el.index), k, ξexp, a, dist, ξ0)
                end
            end
        end
    end
    return (; nchk, nbad, max_da=maximum(da), max_dist=maximum(ddist),
        mean_da=mean(da))
end

function action_reaction(prep, pairs, x)
    nx = sum(p.ndof for p in prep)
    dn = Float64[]
    dt = Float64[]
    for (k, cp) in enumerate(pairs)
        abs(cp.state) == 1 && continue
        kin = MR._contact_pair_kinematics(prep, cp, cp.gap0, x, k, nx)
        # tn1 + R row1 t2 , tt1 + R row2 t2
        rn = kin.tn1 + kin.R[1, 1] * kin.tn2 + kin.R[1, 2] * kin.tt2
        rt = kin.tt1 + kin.R[2, 1] * kin.tn2 + kin.R[2, 2] * kin.tt2
        push!(dn, abs(rn))
        push!(dt, abs(rt))
    end
    return (; n=length(dn), maxn=isempty(dn) ? 0.0 : maximum(dn),
        maxt=isempty(dt) ? 0.0 : maximum(dt))
end

function summarize_p(tag, iface; p_ref=nothing)
    cl = abs.(iface.state) .!= 1
    p = iface.p
    ncl = count(cl)
    p0 = ncl == 0 ? 0.0 : maximum(p[cl])
    pmean = ncl == 0 ? 0.0 : mean(p[cl])
    @printf("  %-18s  closed=%3d  p∈[%.3f, %.3f]  mean=%.3f  p0=%.3f\n",
        tag, ncl,
        ncl == 0 ? 0.0 : minimum(p[cl]),
        ncl == 0 ? 0.0 : maximum(p[cl]),
        pmean, p0)
    if p_ref !== nothing && ncl > 0
        pref = p_ref.(iface.x[cl])
        osc = p[cl] .- pref
        @printf("           vs ref  rms=%.4f  max|Δ|=%.4f  (%.2f%% of p0_ref)\n",
            sqrt(mean(abs2, osc)), maximum(abs, osc),
            100 * maximum(abs, osc) / max(maximum(pref), 1e-12))
    end
    return nothing
end

# =============================================================================
println("="^72)
println(" Contact wiggle diagnosis")
println("="^72)

# -----------------------------------------------------------------------------
println("\n[0] Guiggiani collocation parameter a vs GL node (dad_5d tipo=2)")
prob0, _ = load_dad_5d_contact(; μ=0.0, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
    tipo=2, nome="wiggle_g")
dad0 = prob0.regions[1]
@printf("  poly nodes ξ = %s\n", string(collect(dad0.element_type.nodes)))
@printf("  n=%d  nelem=%d  n_contact=%d  elem_weight=%s\n",
    dad0.n, length(dad0.elements), length(prob0.contacts),
    string(collect(dad0.elem_weight)))
g0 = guiggiani_a_report(dad0)
@printf("  checked=%d  bad=%d  max|a-ξ|=%.3e  max dist=%.3e  mean|a-ξ|=%.3e\n",
    g0.nchk, g0.nbad, g0.max_da, g0.max_dist, g0.mean_da)
g1 = guiggiani_a_report(prob0.regions[2])
@printf("  body2 checked=%d  bad=%d  max|a-ξ|=%.3e  max dist=%.3e\n",
    g1.nchk, g1.nbad, g1.max_da, g1.max_dist)

# pairing x mismatch
dx = Float64[]
for cp in prob0.contacts
    xa = prob0.regions[1].Nodes[cp.node_a][1]
    xb = prob0.regions[2].Nodes[cp.node_b][1]
    push!(dx, abs(xa - xb))
end
@printf("  NTN |Δx| max=%.4e  mean=%.4e  unique masters=%d / %d\n",
    maximum(dx), mean(dx), length(unique(cp.node_b for cp in prob0.contacts)),
    length(prob0.contacts))

# -----------------------------------------------------------------------------
println("\n[1] Single-body local-frame compression (tipo=1 and tipo=2)")
println("    bottom u_n=0, t_t=0; top t_n=-1; recovered t_n on bottom")
for tipo in (1, 2)
    msh = quadrado_elasticity(; ndiv=8, show=false, nome="wiggle_sq$tipo", ordem=1)
    dad = format2d(msh, Elasticity(100.0, 0.3, 1.0; plane_strain=true);
        pontointerno=false, tipo=tipo)
    pbar = 1.0
    for i in 1:dad.n
        x, y = dad.Nodes[i]
        n = dad.Normal[i]
        if y < 1e-9 && abs(n[2]) > 0.5
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0.0
            dad.BC[2i]   = 1; dad.BV[2i]   = 0.0
        elseif y > 1 - 1e-9 && n[2] > 0.5
            dad.BC[2i-1] = 1; dad.BV[2i-1] = -pbar
            dad.BC[2i]   = 1; dad.BV[2i]   = 0.0
        else
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i]   = 1; dad.BV[2i]   = 0.0
        end
    end
    for i in 1:dad.n
        x, y = dad.Nodes[i]
        if x < 1e-9 && 0.4 < y < 0.6
            dad.BC[2i] = 0; dad.BV[2i] = 0.0
            break
        end
    end
    H_G_full_direct(dad, 12)
    solve(dad; frame=:local)
    tn_bot = Float64[]
    x_bot = Float64[]
    for i in 1:dad.n
        y = dad.Nodes[i][2]
        n = dad.Normal[i]
        if y < 1e-9 && abs(n[2]) > 0.5
            push!(tn_bot, dad.traction_local[2i-1])
            push!(x_bot, dad.Nodes[i][1])
        end
    end
    perm = sortperm(x_bot)
    tn_bot = tn_bot[perm]; x_bot = x_bot[perm]
    # skip corners (mixed BC pollution)
    interior = [i for i in eachindex(x_bot) if 0.05 < x_bot[i] < 0.95]
    tni = tn_bot[interior]
    @printf("  tipo=%d  n_bot=%d  tn∈[%.4f, %.4f]  mean=%.4f  std=%.4f  osc/mean=%.2f%%\n",
        tipo, length(tni), minimum(tni), maximum(tni), mean(tni), std(tni),
        100 * (maximum(tni) - minimum(tni)) / max(abs(mean(tni)), 1e-12))
    # mid vs edge on bottom elements
    if tipo == 2 && length(interior) >= 3
        oscs = Float64[]
        for k in 1:3:length(interior)-2
            trip = tni[k:k+2]
            # GL order along +x: edge, mid, edge
            push!(oscs, trip[2] - 0.5 * (trip[1] + trip[3]))
        end
        @printf("           mid-edge Δtn: mean=%+.4f  (positive ⇒ mid more compressive)\n",
            mean(oscs))
    end
end

# -----------------------------------------------------------------------------
println("\n[2] Two-block FLAT contact patch test (uniform ty=-100, μ=0)")
function load_flat_patch(; tipo=2, ndiv=6, p=100.0, μ=0.0)
    props = Elasticity(73.4e3, 0.33, 1.0; plane_strain=true)
    W, H, gap = 2.0, 1.0, 0.0
    msh_b = mesh_elastic_block(; x0=-W/2, y0=-H, W=W, H=H, ndiv_x=ndiv, ndiv_y=4, μ=μ,
        bottom_bc="0;0;0;0;8;1", top_bc="4;$μ;4;$μ;8;2",
        left_bc="1;0;1;0;8;3", right_bc="1;0;1;0;8;4", nome="wiggle_flat_b")
    msh_t = mesh_elastic_block(; x0=-W/2, y0=gap, W=W, H=H, ndiv_x=ndiv, ndiv_y=4, μ=μ,
        bottom_bc="4;$μ;4;$μ;9;1", top_bc="1;0;1;$(-p);9;2",
        left_bc="1;0;1;0;9;3", right_bc="1;0;1;0;9;4", nome="wiggle_flat_t")
    dad_b = format2d(msh_b, props; pontointerno=false, tipo=tipo)
    dad_t = format2d(msh_t, props; pontointerno=false, tipo=tipo)
    dad_b.name = "bottom"; dad_t.name = "top"
    # region 1 = top (loaded), region 2 = bottom (fixed) — matches dad_5d order
    prob = MultiRegionProblem([dad_t, dad_b]; name="flat_patch")
    pair_contacts!(prob; method=:ntn, slave_reg=1, master_reg=2)
    pin_top_ux!(prob.regions[1])
    return prob
end

for tipo in (1, 2)
    println("  -- tipo=$tipo")
    prob = load_flat_patch(; tipo=tipo, ndiv=6, p=100.0, μ=0.0)
    @printf("     n=%d/%d  pairs=%d  gap0∈[%.3e, %.3e]\n",
        prob.regions[1].n, prob.regions[2].n, length(prob.contacts),
        extrema(cp.gap0 for cp in prob.contacts)...)
    sol = contato_as!(prob; npg=12, verbose=false)
    iface = iface_p(prob)
    summarize_p("flat tipo=$tipo", iface; p_ref=x -> 100.0)
    trips = element_triplets(prob)
    if !isempty(trips)
        @printf("     n_elem=%d  mean(p_mid-p_edge)=%+.3f  max|osc|=%.3f\n",
            length(trips), mean(t.osc for t in trips),
            maximum(abs(t.osc) for t in trips))
        for (k, t) in enumerate(trips[1:min(3, length(trips))])
            @printf("       el %d  x=%s  p=%s  osc=%+.3f\n",
                k, string(round.(t.xs; digits=3)), string(round.(t.ps; digits=3)), t.osc)
        end
    end
    br = bie_resid(sol.prep, sol.pairs, sol.x)
    for b in br
        @printf("     BIE r%d  max=%.3e  rms=%.3e  contact max=%.3e\n",
            b.ir, b.rmax, b.rrms, b.rcmax)
    end
    ar = action_reaction(sol.prep, sol.pairs, sol.x)
    @printf("     action-reaction  max|tn1+R t2|=%.3e  max|tt1+R t2|=%.3e  ok=%s it=%d\n",
        ar.maxn, ar.maxt, sol.ok, sol.nit)
    if tipo == 2
        plt = plot(iface.x, iface.p; seriestype=:scatter, ms=4, label="two-body t_n",
            xlabel="x", ylabel="p", title="flat patch tipo=2 (expect p=100)")
        hline!([100.0]; color=:black, ls=:dash, label="p=100")
        savefig(plt, joinpath(OUT, "flat_patch_tipo2.png"))
    end
end

# -----------------------------------------------------------------------------
println("\n[3] Hertz two-cylinder Step A  (dad_5d, P via ty=-100)")
par5 = loyola_dad5d_params()
aH, p0H = par5.a_H, par5.p0_H
phertz(x) = abs(x) < aH ? p0H * sqrt(1 - (x / aH)^2) : 0.0

function run_hertz(; tipo=2, μ=0.0, npg=10, ndiv_c=12, tag="h")
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=ndiv_c, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        tipo=tipo, nome="wiggle_$tag")
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0)
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i]   = 1; dad.BV[2i]   = -par5.cargav
        end
    end
    pin_top_ux!(dad)
    @printf("  %s  tipo=%d μ=%.2f npg=%d  n=%d/%d  NPc=%d\n",
        tag, tipo, μ, npg, prob.regions[1].n, prob.regions[2].n, length(prob.contacts))
    sol = contato_as!(prob; npg=npg, verbose=true)
    iface = iface_p(prob)
    summarize_p(tag, iface; p_ref=phertz)
    trips = element_triplets(prob)
    closed_trips = [t for t in trips if all(p -> p > 1.0, t.ps)]
    if !isempty(closed_trips)
        @printf("     closed elems=%d  mean(p_mid-p_edge)=%+.2f  max|osc|=%.2f  (%.1f%% of p0H)\n",
            length(closed_trips), mean(t.osc for t in closed_trips),
            maximum(abs(t.osc) for t in closed_trips),
            100 * maximum(abs(t.osc) for t in closed_trips) / p0H)
        # print Hertz-zone elements
        for t in closed_trips
            xmax = maximum(abs, t.xs)
            xmax > 1.2 * aH && continue
            @printf("       x=%s  p=%s  Hertz@mid=%.1f  osc=%+.1f\n",
                string(round.(t.xs; digits=3)),
                string(round.(t.ps; digits=1)),
                phertz(t.xs[argmin(abs.(t.ξs))]), t.osc)
        end
    end
    br = bie_resid(sol.prep, sol.pairs, sol.x)
    for b in br
        @printf("     BIE r%d  max=%.3e  rms=%.3e  contact max=%.3e\n",
            b.ir, b.rmax, b.rrms, b.rcmax)
    end
    ar = action_reaction(sol.prep, sol.pairs, sol.x)
    @printf("     action-reaction maxn=%.3e maxt=%.3e  ok=%s it=%d\n",
        ar.maxn, ar.maxt, sol.ok, sol.nit)
    # gn residual on closed
    nx = sum(p.ndof for p in sol.prep)
    gns = Float64[]
    for (k, cp) in enumerate(sol.pairs)
        abs(cp.state) == 1 && continue
        kin = MR._contact_pair_kinematics(sol.prep, cp, sol.h[k], sol.x, k, nx)
        push!(gns, abs(kin.gn))
    end
    @printf("     |gn| closed max=%.3e mean=%.3e\n",
        isempty(gns) ? 0.0 : maximum(gns), isempty(gns) ? 0.0 : mean(gns))
    return (; iface, trips, sol, tag, tipo, μ, npg)
end

runs = []
push!(runs, run_hertz(; tipo=2, μ=0.0, npg=10, tag="t2mu0npg10"))
push!(runs, run_hertz(; tipo=2, μ=0.0, npg=20, tag="t2mu0npg20"))
push!(runs, run_hertz(; tipo=1, μ=0.0, npg=10, tag="t1mu0npg10"))
push!(runs, run_hertz(; tipo=2, μ=0.3, npg=10, tag="t2mu3npg10"))

plt = plot(xlabel="x/a", ylabel="p/p0", title="Step A pressure — wiggle diagnosis",
    legend=:bottom)
xa = range(-1.5 * aH, 1.5 * aH; length=400)
plot!(plt, xa ./ aH, phertz.(xa) ./ p0H; color=:black, lw=2, label="Hertz")
cols = [:steelblue, :orange, :green, :red, :purple]
for (i, r) in enumerate(runs)
    plot!(plt, r.iface.x ./ aH, r.iface.p ./ p0H; seriestype=:scatter, ms=3,
        color=cols[i], label=r.tag)
end
savefig(plt, joinpath(OUT, "hertz_wiggle_compare.png"))
println("\nWrote $OUT")
println("done")
