# Dump dad_5d shear symmetry / oscillation diagnostics at B and C
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf
using Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

const FIGDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM_contact\figures"
mkpath(FIGDIR)

par = loyola_dad5d_params()
μ = par.μ
u_max = 0.03

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10)
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = has_cache(prob.regions[1], :contact_x) &&
         length(prob.regions[1].contact_x) == ctx.N ?
         collect(Float64, prob.regions[1].contact_x) : zeros(ctx.N)
    if !has_cache(prob.regions[1], :contact_x) || length(prob.regions[1].contact_x) != ctx.N
        for cp in pairs; cp.state = 3; cp.ut_lock = 0.0; end
        x0 = zeros(ctx.N)
    end
    ok = false
    for it in 1:maxiter
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0); x0 = x
        dist < tol && (ok = true; break)
    end
    BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    BEM._update_contact_ut_locks!(pairs, prep, h, x0)
    BEM._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, prep, pairs, x=x0, h)
end

function collect_rows(sol)
    prep, pairs, x, h = sol.prep, sol.pairs, sol.x, sol.h
    da = prep[1].dad
    nx = sum(p.ndof for p in prep)
    rows = NamedTuple[]
    for (k, cp) in enumerate(pairs)
        kin = BEM._contact_pair_kinematics(prep, cp, h[k], x, k, nx)
        n1 = da.Normal[cp.node_a]
        _, t1 = local_basis2d(n1)
        xa = da.Nodes[cp.node_a]
        push!(rows, (; k, x=xa[1], y=xa[2], st=Int(cp.state), tn=kin.tn1, tt=kin.tt1,
            gn=kin.gn, gt=kin.gt, ut_lock=cp.ut_lock, un1=kin.un1, ut1=kin.ut1,
            un2=kin.un2, ut2=kin.ut2, n1x=n1[1], n1y=n1[2], t1x=t1[1], t1y=t1[2],
            gap0=cp.gap0, node_a=cp.node_a, node_b=cp.node_b))
    end
    return sort(rows; by=r -> r.x)
end

function dump_rows(label, rows)
    println("\n==== $label ====")
    @printf("%3s %8s %6s %10s %10s %10s %10s %9s %9s %8s %6s\n",
        "k", "x", "st", "tn", "tt", "p", "q", "ut1", "ut2", "t1x", "node")
    for r in rows
        abs(r.x) > 2.5 && abs(r.st) == 1 && continue
        @printf("%3d %8.4f %6d %10.3f %10.3f %10.3f %10.3f %9.5f %9.5f %8.3f %6d\n",
            r.k, r.x, r.st, r.tn, r.tt, -r.tn, -r.tt, r.ut1, r.ut2, r.t1x, r.node_a)
    end
    cl = [r for r in rows if abs(r.st) != 1 && abs(r.x) < 2.0]
    println("closed |x|<2: ", length(cl))
    for r in cl
        r.x < -1e-6 || continue
        j = argmin(abs(s.x + r.x) for s in cl)
        s = cl[j]
        abs(r.x) < 1.4 || continue
        @printf("  sym x=%+.4f/%+.4f  q=%+8.2f/%+8.2f dq=%7.2f  p=%7.1f/%7.1f  st=%d/%d\n",
            r.x, s.x, -r.tt, -s.tt, (-r.tt)-(-s.tt), -r.tn, -s.tn, r.st, s.st)
    end
    # checkerboard: successive q differences
    if length(cl) >= 3
        qs = [-r.tt for r in cl]
        d2 = [qs[i+1] - 2qs[i] + qs[i-1] for i in 2:length(qs)-1]
        @printf("q curvature |d2| mean=%.2f max=%.2f (high ⇒ oscillation)\n",
            mean(abs, d2), maximum(abs, d2))
    end
    return cl
end

function main()
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=16, ndiv_f=8, ndiv_s=6, ndiv_top=12, nome="sym")
    da = prob.regions[1]

    println("MESH / PAIRING")
    xs = sort([da.Nodes[cp.node_a][1] for cp in prob.contacts])
    @printf("n_pairs=%d  x[%.4f,%.4f]  sum x=%.3e  max|x+rev|=%.3e\n",
        length(xs), first(xs), last(xs), sum(xs), maximum(abs.(xs .+ reverse(xs))))
    g = sort(prob.contacts; by=cp -> da.Nodes[cp.node_a][1])
    for i in 1:min(6, length(g)÷2)
        L, R = g[i], g[end+1-i]
        @printf("  gap0 x=%+.4f/%+.4f  g=%.6f/%.6f  nodes %d/%d\n",
            da.Nodes[L.node_a][1], da.Nodes[R.node_a][1], L.gap0, R.gap0,
            L.node_a, R.node_a)
    end
    println("element_type=", da.element_type, "  ne=", length(da.elements),
        "  first indices=", da.elements[1].index)
    # collocation pattern: print successive contact node spacing
    xc = [da.Nodes[cp.node_a][1] for cp in g if abs(da.Nodes[cp.node_a][1]) < 1.5]
    println("spacings near contact: ", round.(diff(xc); digits=4))
    # are collocation nodes discontinuous (repeated coords)?
    pts = [(round(da.Nodes[i][1]; digits=10), round(da.Nodes[i][2]; digits=10)) for i in 1:length(da.Nodes)]
    println("unique collocation pts=", length(unique(pts)), " / ", length(pts))

    println("\n[A]")
    sol = contato_newton!(prob)
    dump_rows("A", collect_rows(sol))
    _, uyA = dad5d_top_u_mean(prob)
    @printf("uyA=%.6f\n", uyA)

    apply_dad5d_bulk_ux!(prob, 0.0; uy=uyA)
    contato_newton!(prob)

    println("\n[B]")
    for s in 1:6
        apply_dad5d_bulk_ux!(prob, u_max * s/6; uy=uyA)
        sol = contato_newton!(prob)
    end
    rowsB = collect_rows(sol)
    clB = dump_rows("B", rowsB)

    println("\n[C]")
    for s in 1:6
        apply_dad5d_bulk_ux!(prob, u_max * (1 - s/6); uy=uyA)
        sol = contato_newton!(prob)
    end
    rowsC = collect_rows(sol)
    clC = dump_rows("C", rowsC)

    # figure
    plts = Plots.Plot[]
    for (j, (lab, rows)) in enumerate((("B", rowsB), ("C", rowsC)))
        cl = [r for r in rows if abs(r.st) != 1]
        x = [r.x for r in cl]; q = [-r.tt for r in cl]; p = [-r.tn for r in cl]
        st = [r.st for r in cl]
        pplt = plot(x, q;
            color=:steelblue, linewidth=2, label="AS q",
            xlabel="x", ylabel="q=-t_t", title="dad_5d raw $lab",
            size=(460, 360), framestyle=:box, legend=:topright)
        scatter!(pplt, x, q; color=:steelblue, markersize=7, label="")
        slip = findall(s -> abs(s) == 2, st)
        isempty(slip) || scatter!(pplt, x[slip], q[slip]; color=:crimson, markersize=11, label="slip")
        if lab == "B" && length(x) >= 2
            w = ones(length(x))
            w[1] = abs(x[2]-x[1]); w[end] = abs(x[end]-x[end-1])
            for i in 2:length(x)-1; w[i] = 0.5*abs(x[i+1]-x[i-1]); end
            P = sum(p .* w); Q = sum(q .* w); p0 = maximum(p); a = 2P/(π*max(p0,eps()))
            qA = cattaneo_shear(collect(x), a, p0, Q, μ, abs(P))
            plot!(pplt, x, qA; color=:black, linestyle=:dash, label="Cattaneo", linewidth=1.5)
            @printf("B: P=%.1f Q=%.1f a=%.3f p0=%.1f\n", P, Q, a, p0)
        end
        push!(plts, pplt)
    end
    fig = plot(plts...; layout=(1, 2), size=(920, 360))
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_raw_q_BC.png"))
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_raw_q_BC.pdf"))
    println("saved ", joinpath(FIGDIR, "fig_dad5d_raw_q_BC.png"))
end

main()
