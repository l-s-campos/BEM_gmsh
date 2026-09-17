# Is tn the wrong quantity on curved contact?
# Discriminators:
#   (A) dump n, J, gap0, tn, ty on Hertz elements
#   (B) flat blocks + parabolic Hertz gap  (curve vs gap)
#   (C) cylinders with contact n forced to ±e_y  (tn := ty)
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
    return (; x=x0, prep, pairs, h)
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

function apply_top_ty!(dad, ty)
    ymax = maximum(pt[2] for pt in dad.Nodes)
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0)
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i]   = 1; dad.BV[2i]   = ty
        end
    end
    pin_top_ux!(dad)
    return nothing
end

"""Assembly (Legendre) normal and J at a collocation node of element `el`."""
function assembly_nj(dad, el, k)
    poly = dad.element_type
    xj = dad.Nodes[el.index]
    g = BEM._geom_1d(poly, xj, poly.nodes[k])
    g === nothing && return (n = dad.Normal[el.index[k]], J = el.Jacobian[k])
    return (n = Point2D(g[4][1], g[4][2]), J = g[2])
end

function dump_hertz_elements(prob; nprint=6)
    dad = prob.regions[1]
    dad2 = prob.regions[2]
    cmap = Dict(cp.node_a => cp for cp in prob.contacts)
    println("  el  loc  x        n_stored             n_assembly           n_circle            Jst/Jas  gap0     tn       ty      t·e_y")
    nshow = 0
    for el in dad.elements
        ids = el.index
        all(haskey(cmap, i) for i in ids) || continue
        xs = [dad.Nodes[i][1] for i in ids]
        maximum(abs, xs) > 0.25 && nshow >= nprint && continue
        all(abs(cmap[i].tn) > 1 for i in ids) || continue
        nshow += 1
        for (k, i) in enumerate(ids)
            cp = cmap[i]
            ns = dad.Normal[i]
            na = assembly_nj(dad, el, k)
            # upper circle centre (0, R)
            R = 70.0
            p = dad.Nodes[i]
            nc = p - Point2D(0.0, R)
            nc = nc / (norm(nc) + eps())
            # stored n should ≈ outward = (x, y-R)/R  (down)
            ty = has_cache(dad, :traction) ? dad.traction[2i] : NaN
            tx = has_cache(dad, :traction) ? dad.traction[2i-1] : NaN
            @printf("  %2d %3d %7.4f  (%7.4f,%7.4f)  (%7.4f,%7.4f)  (%7.4f,%7.4f)  %6.4f  %8.5f %8.2f %8.2f %8.2f\n",
                nshow, k, p[1], ns[1], ns[2], na.n[1], na.n[2], nc[1], nc[2],
                el.Jacobian[k] / max(na.J, eps()), cp.gap0, -cp.tn, ty, ty)
        end
        # mid vs edge tn and n_y
        nys = [dad.Normal[i][2] for i in ids]
        tns = [-cmap[i].tn for i in ids]
        @printf("     n_y=%s  p=-tn=%s  Δn_y(mid-edge)=%+.4e  Δp=%+.2f\n",
            string(round.(nys; digits=6)), string(round.(tns; digits=1)),
            nys[2] - 0.5 * (nys[1] + nys[3]), tns[2] - 0.5 * (tns[1] + tns[3]))
        nshow >= nprint && break
    end
    return nothing
end

function gl_mean_p(prob)
    dad = prob.regions[1]
    w = collect(Float64, dad.elem_weight)
    cmap = Dict(cp.node_a => -cp.tn for cp in prob.contacts)
    xs = Float64[]; ps = Float64[]
    for el in dad.elements
        ids = el.index
        all(haskey(cmap, i) for i in ids) || continue
        pp = [cmap[i] for i in ids]
        all(>(1.0), pp) || continue
        push!(xs, mean(dad.Nodes[i][1] for i in ids))
        push!(ps, sum(w[k] * pp[k] for k in eachindex(ids)) / sum(w))
    end
    perm = sortperm(xs)
    return xs[perm], ps[perm]
end

function osc_report(tag, prob)
    dad = prob.regions[1]
    cmap = Dict(cp.node_a => -cp.tn for cp in prob.contacts)
    oscs = Float64[]
    pmax = 0.0
    for el in dad.elements
        ids = el.index
        length(ids) == 3 || continue
        all(haskey(cmap, i) for i in ids) || continue
        pp = [cmap[i] for i in ids]
        all(>(1.0), pp) || continue
        pmax = max(pmax, maximum(pp))
        push!(oscs, pp[2] - 0.5 * (pp[1] + pp[3]))
    end
    fr = contact_interface_xyτ(prob)
    cl = abs.(fr.state) .!= 1
    tys = Float64[]
    if has_cache(dad, :traction)
        for cp in prob.contacts
            abs(cp.state) == 1 && continue
            i = cp.node_a
            push!(tys, dad.traction[2i])
        end
    end
    @printf("  %-28s  closed=%d  p=-tn ∈ [%.1f, %.1f]  mean(mid-edge Δp)=%+.2f  max|Δp|=%.2f",
        tag, count(cl),
        any(cl) ? minimum(.-fr.tn[cl]) : 0.0,
        any(cl) ? maximum(.-fr.tn[cl]) : 0.0,
        isempty(oscs) ? 0.0 : mean(oscs),
        isempty(oscs) ? 0.0 : maximum(abs, oscs))
    if !isempty(tys)
        @printf("  ty ∈ [%.1f, %.1f]", minimum(tys), maximum(tys))
    end
    println()
    return oscs
end

par5 = loyola_dad5d_params()
aH, p0H = par5.a_H, par5.p0_H
phertz(x) = abs(x) < aH ? p0H * sqrt(max(0.0, 1 - (x / aH)^2)) : 0.0

println("="^72)
println("[A] Hertz cylinders — n, J, gap0, tn vs ty")
probA, _ = load_dad_5d_contact(; μ=0.0, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
    tipo=2, nome="tnA")
apply_top_ty!(probA.regions[1], -par5.cargav)
contato_as!(probA)
osc_report("Hertz curved, tn=t·n", probA)
dump_hertz_elements(probA)

# also compare -tn vs +ty vs -t·n_circle
dadA = probA.regions[1]
xs = Float64[]; p_tn = Float64[]; p_ty = Float64[]; p_nc = Float64[]
for cp in probA.contacts
    abs(cp.state) == 1 && continue
    i = cp.node_a
    p = dadA.Nodes[i]
    ncirc = p - Point2D(0.0, par5.R)
    ncirc = ncirc / (norm(ncirc) + eps())
    tg = SVector(dadA.traction[2i-1], dadA.traction[2i])
    push!(xs, p[1])
    push!(p_tn, -cp.tn)
    push!(p_ty, tg[2])          # upper: compression ⇒ ty > 0
    push!(p_nc, -dot(tg, ncirc))
end
perm = sortperm(xs)
xs, p_tn, p_ty, p_nc = xs[perm], p_tn[perm], p_ty[perm], p_nc[perm]
@printf("  max|p_tn - p_ty|=%.4f   max|p_tn - p_ncirc|=%.4f   max|p_ty - Hertz|=%.2f  max|p_tn - Hertz|=%.2f\n",
    maximum(abs, p_tn .- p_ty), maximum(abs, p_tn .- p_nc),
    maximum(abs, p_ty .- phertz.(xs)), maximum(abs, p_tn .- phertz.(xs)))

println("\n[B] Flat blocks + parabolic gap (same R, tipo=2)")
function load_flat_parab(; tipo=2, ndiv=16)
    props = Elasticity(par5.E, par5.ν, 1.0; plane_strain=true)
    W, H = 2 * par5.w, par5.w
    msh_b = mesh_elastic_block(; x0=-W/2, y0=-H, W=W, H=H, ndiv_x=ndiv, ndiv_y=4, μ=0.0,
        bottom_bc="0;0;0;0;8;1", top_bc="4;0;4;0;8;2",
        left_bc="1;0;1;0;8;3", right_bc="1;0;1;0;8;4", nome="tnB_b")
    msh_t = mesh_elastic_block(; x0=-W/2, y0=0.0, W=W, H=H, ndiv_x=ndiv, ndiv_y=4, μ=0.0,
        bottom_bc="4;0;4;0;9;1", top_bc="1;0;1;$(-par5.cargav);9;2",
        left_bc="1;0;1;0;9;3", right_bc="1;0;1;0;9;4", nome="tnB_t")
    dad_b = format2d(msh_b, props; pontointerno=false, tipo=tipo)
    dad_t = format2d(msh_t, props; pontointerno=false, tipo=tipo)
    prob = MultiRegionProblem([dad_t, dad_b]; name="flat_parab")
    pair_contacts!(prob; method=:ntn, slave_reg=1, master_reg=2)
    # two-cylinder equivalent gap: x²/(2R*) with R* = R/2 ⇒ x²/R
    apply_parabolic_contact_gap!(prob; R=par5.R / 2, gap_min=0.0, x0=0.0, method=:ntn,
        slave_reg=1, master_reg=2)
    pin_top_ux!(prob.regions[1])
    return prob
end
probB = load_flat_parab()
@printf("  pairs=%d  gap0 ∈ [%.5f, %.5f]  n_y unique=%s\n",
    length(probB.contacts), extrema(cp.gap0 for cp in probB.contacts)...,
    string(unique(round(probB.regions[1].Normal[cp.node_a][2]; digits=8) for cp in probB.contacts)))
contato_as!(probB)
osc_report("flat + parabolic gap", probB)
frB = contact_interface_xyτ(probB)

println("\n[C] Cylinders, contact n forced to ±e_y  (tn ≡ ±ty)")
probC, _ = load_dad_5d_contact(; μ=0.0, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
    tipo=2, nome="tnC")
apply_top_ty!(probC.regions[1], -par5.cargav)
# overwrite contact normals only (local frame / gap); H,G kernels still use interpolant n
for (dad, nfix) in ((probC.regions[1], Point2D(0.0, -1.0)),
                    (probC.regions[2], Point2D(0.0, 1.0)))
    for i in 1:dad.n
        if dad.BC[2i-1] == 4 || dad.BC[2i] == 4
            dad.Normal[i] = nfix
        end
    end
end
# rebuild gap0 along ±e_y
for cp in probC.contacts
    pa = probC.regions[1].Nodes[cp.node_a]
    pb = probC.regions[2].Nodes[cp.node_b]
    n̂ = probC.regions[1].Normal[cp.node_a]
    cp.gap0 = max(0.0, dot(pb - pa, n̂ / (norm(n̂) + eps())))
end
contato_as!(probC)
osc_report("curved geom, n=±e_y", probC)

println("\n[D] Cylinders, gap0 replaced by analytical x²/R, normals untouched")
probD, _ = load_dad_5d_contact(; μ=0.0, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
    tipo=2, nome="tnD")
apply_top_ty!(probD.regions[1], -par5.cargav)
for cp in probD.contacts
    x = probD.regions[1].Nodes[cp.node_a][1]
    cp.gap0 = x^2 / par5.R   # two equal cylinders, R* = R/2
end
contato_as!(probD)
osc_report("curved, analytical gap0", probD)

# plots
frA = contact_interface_xyτ(probA)
frC = contact_interface_xyτ(probC)
frD = contact_interface_xyτ(probD)
xa = range(-1.4aH, 1.4aH; length=400)
plt = plot(xa ./ aH, phertz.(xa) ./ p0H; color=:black, lw=2.5, label="Hertz",
    xlabel="x/a", ylabel="p/p₀", legend=:bottom, size=(740, 500),
    title="What is tn?  curved vs flat-parabolic vs n=±e_y")
plot!(plt, frA.x ./ aH, (.-frA.tn) ./ p0H; marker=:circle, ms=3, lw=1, color=:steelblue,
    label="cylinders  p=−tn")
plot!(plt, xs ./ aH, p_ty ./ p0H; marker=:x, ms=3, lw=0, color=:red,
    label="cylinders  p=+ty")
plot!(plt, frB.x ./ aH, (.-frB.tn) ./ p0H; marker=:diamond, ms=4, lw=1, color=:orange,
    label="flat + parabolic gap")
plot!(plt, frC.x ./ aH, (.-frC.tn) ./ p0H; marker=:utriangle, ms=4, lw=0, color=:green,
    label="cylinders n=±e_y")
xlims!(-1.5, 1.5); ylims!(-0.02, 1.15)
savefig(plt, joinpath(OUT, "tn_definition.png"))
println("\nwrote ", joinpath(OUT, "tn_definition.png"))
