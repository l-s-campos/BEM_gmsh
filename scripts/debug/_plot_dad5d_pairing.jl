# Plot NTN contact node pairing on dad_5d
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf
using Plots

include(datadir("elastico", "dad_5d_contact.jl"))

const FIGDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM_contact\figures"
mkpath(FIGDIR)

function main()
    prob, par = load_dad_5d_contact(; ndiv_c=16, ndiv_f=8, ndiv_s=6, ndiv_top=12,
        tipo=2, nome="pairplot")
    da, db = prob.regions
    pairs = prob.contacts

    # all boundary nodes for outline
    xu = [p[1] for p in da.Nodes]; yu = [p[2] for p in da.Nodes]
    xl = [p[1] for p in db.Nodes]; yl = [p[2] for p in db.Nodes]

    # contact nodes
    cu = Int[]; cl = Int[]
    for i in 1:da.n
        (da.BC[2i-1] == BEM.BC_CONTACT || da.BC[2i] == BEM.BC_CONTACT) && push!(cu, i)
    end
    for i in 1:db.n
        (db.BC[2i-1] == BEM.BC_CONTACT || db.BC[2i] == BEM.BC_CONTACT) && push!(cl, i)
    end

    xs = Float64[]; ys = Float64[]
    xm = Float64[]; ym = Float64[]
    dx = Float64[]; gap = Float64[]; idx = Int[]
    for (k, cp) in enumerate(pairs)
        pa = da.Nodes[cp.node_a]
        pb = db.Nodes[cp.node_b]
        push!(xs, pa[1]); push!(ys, pa[2])
        push!(xm, pb[1]); push!(ym, pb[2])
        push!(dx, abs(pa[1] - pb[1]))
        push!(gap, cp.gap0)
        push!(idx, k)
    end

    println("n_pairs = ", length(pairs))
    @printf("Δx: mean=%.3e  max=%.3e\n", mean(dx), maximum(dx))
    @printf("gap0: min=%.4f max=%.4f\n", minimum(gap), maximum(gap))

    # --- full geometry + all pair links ---
    p1 = plot(;
        xlabel="x", ylabel="y", title="dad_5d NTN pairs (all)",
        aspect_ratio=:equal, framestyle=:box, legend=:topright, size=(500, 360))
    scatter!(p1, xu, yu; color=:steelblue, alpha=0.35, markersize=4, label="upper nodes")
    scatter!(p1, xl, yl; color=:darkorange, alpha=0.35, markersize=4, label="lower nodes")
    scatter!(p1, [da.Nodes[i][1] for i in cu], [da.Nodes[i][2] for i in cu];
        color=:steelblue, markersize=7, label="upper contact")
    scatter!(p1, [db.Nodes[i][1] for i in cl], [db.Nodes[i][2] for i in cl];
        color=:darkorange, markersize=7, label="lower contact")
    for k in eachindex(pairs)
        plot!(p1, [xs[k], xm[k]], [ys[k], ym[k]];
            color=:gray30, alpha=0.55, linewidth=0.8, label="")
    end

    # --- zoom on contact strip ---
    p2 = plot(;
        xlabel="x", ylabel="y", title="contact strip (zoom)",
        aspect_ratio=:equal, framestyle=:box, legend=:topright,
        ylims=(-0.6, 0.6), size=(500, 360))
    scatter!(p2, [da.Nodes[i][1] for i in cu], [da.Nodes[i][2] for i in cu];
        color=:steelblue, markersize=9, label="slave (upper)")
    scatter!(p2, [db.Nodes[i][1] for i in cl], [db.Nodes[i][2] for i in cl];
        color=:darkorange, markersize=9, label="master (lower)")
    # color links by |Δx|
    dmax = max(maximum(dx), eps())
    for k in eachindex(pairs)
        c = dx[k] / dmax
        plot!(p2, [xs[k], xm[k]], [ys[k], ym[k]];
            color=RGB(c, 0.15, 1 - c), linewidth=1.4, label="")
    end
    # normals at a few pairs
    perm = sortperm(xs)
    qx = Float64[]; qy = Float64[]; qu = Float64[]; qv = Float64[]
    for k in perm[1:4:end]
        cp = pairs[k]
        n = da.Normal[cp.node_a]
        nn = n / (norm(n) + eps())
        push!(qx, xs[k]); push!(qy, ys[k])
        push!(qu, 0.25*nn[1]); push!(qv, 0.25*nn[2])
    end
    isempty(qx) || quiver!(p2, qx, qy; quiver=(qu, qv), color=:navy, label="")

    # --- x_slave vs x_master ---
    p3 = plot(;
        xlabel="x_slave", ylabel="x_master",
        title="pairing map  (ideal = diagonal)",
        aspect_ratio=:equal, framestyle=:box, legend=false, size=(500, 360))
    scatter!(p3, xs, xm; color=:purple, markersize=6, label="")
    xlo, xhi = extrema(vcat(xs, xm))
    plot!(p3, [xlo, xhi], [xlo, xhi]; color=:black, linestyle=:dash, linewidth=1, label="")
    # index labels near contact
    for k in eachindex(pairs)
        abs(xs[k]) > 1.5 && continue
        annotate!(p3, [(xs[k], xm[k], Plots.text(string(idx[k]), 7, :left))])
    end

    # --- gap0 and |Δx| along x ---
    p4 = plot(;
        xlabel="x_slave", ylabel="gap0 / |Δx|×10³",
        title="gap0 and pairing mismatch",
        framestyle=:box, legend=:topright, size=(500, 360))
    perm = sortperm(xs)
    plot!(p4, xs[perm], gap[perm]; color=:steelblue, linewidth=2, label="gap0")
    scatter!(p4, xs[perm], gap[perm]; color=:steelblue, markersize=5, label="")
    plot!(p4, xs[perm], 1e3 .* dx[perm]; color=:crimson, linewidth=2, label="|Δx|×10³")

    # print table near Hertz zone
    println("\npairs with |x|<1.5:")
    @printf("%4s %8s %8s %10s %10s %8s %8s\n", "k", "x_s", "x_m", "y_s", "y_m", "Δx", "gap0")
    for k in perm
        abs(xs[k]) > 1.5 && continue
        @printf("%4d %8.4f %8.4f %10.5f %10.5f %8.2e %8.5f\n",
            idx[k], xs[k], xm[k], ys[k], ym[k], xs[k]-xm[k], gap[k])
    end

    fig = plot(p1, p2, p3, p4; layout=(2, 2), size=(1000, 720))
    out = joinpath(FIGDIR, "fig_dad5d_pairing")
    savefig(fig, out * ".png")
    savefig(fig, out * ".pdf")
    println("\nsaved ", out * ".png")
    display(fig)
end

main()
