#!/usr/bin/env julia
# Cost of true element distance vs node/AABB bounds for the near/far test.
using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))
using BEM, LinearAlgebra, StaticArrays, Printf, Statistics

include(joinpath(dirname(dirname(@__DIR__)), "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "data", "Laplace", "cube_mesh.jl"))

function aabb(nodes)
    lo = nodes[1]
    hi = nodes[1]
    @inbounds for k in 2:length(nodes)
        lo = min.(lo, nodes[k])
        hi = max.(hi, nodes[k])
    end
    return lo, hi
end

@inline function dist_aabb(p, lo, hi)
    d2 = 0.0
    @inbounds for a in eachindex(p)
        v = p[a]
        if v < lo[a]
            Δ = lo[a] - v
            d2 += Δ * Δ
        elseif v > hi[a]
            Δ = v - hi[a]
            d2 += Δ * Δ
        end
    end
    return sqrt(d2)
end

@inline function dist_node(p, nodes)
    m = Inf
    @inbounds for k in eachindex(nodes)
        d = norm(p - nodes[k])
        d < m && (m = d)
    end
    return m
end

@inline function dist_chord(p, a, b)
    ab = b - a
    t = clamp(dot(p - a, ab) / (dot(ab, ab) + eps()), 0.0, 1.0)
    return norm(p - (a + t * ab))
end

@inline function dist_gauss(p, ys)
    m = Inf
    @inbounds for y in ys
        d = norm(p - y)
        d < m && (m = d)
    end
    return m
end

function classify_counts(dad, geos, factor)
    lims = factor
    n_near_node = 0
    n_far_box = 0
    n_amb = 0
    n_false_far = 0  # node says far, true d < factor*L
    ntot = 0
    @inbounds for i in 1:dad.nt
        x = point(dad, i)
        for g in geos
            ntot += 1
            L = g.el.Length
            lim = factor * L
            dn = dist_node(x, g.nodes)
            lo, hi = aabb(g.nodes)
            db = dist_aabb(x, lo, hi)
            dch = dist_chord(x, g.nodes[1], g.nodes[end])
            if dn < lim
                n_near_node += 1
            elseif db >= lim
                n_far_box += 1
            else
                n_amb += 1
                dch < lim && (n_false_far += 1)
            end
        end
    end
    return (; ntot, n_near_node, n_far_box, n_amb, n_false_far)
end

function bench_one(label, dad)
    println("\n", label, "  nt=", dad.nt, "  ne=", length(dad.elements),
        "  deg=", degree(dad.element_type), "  dim=", dad.dimension)
    BEM._init_quadrature!(dad, dad.dimension == 2 ? 16 : 8)
    geos = BEM._rim_build_elements(dad)
    el = dad.elements[1]
    nodes = dad.Nodes[el.index]
    poly = dad.element_type
    pf = point(dad, 1)
    ys = geos[1].y
    lo, hi = aabb(nodes)

    # warmup
    dist_node(pf, nodes); dist_aabb(pf, lo, hi)
    dist_chord(pf, nodes[1], nodes[end]); dist_gauss(pf, ys)
    if dad.dimension == 2
        closest_point_1d(poly, nodes, pf)
    else
        closest_point_2d(poly, nodes, pf)
    end

    nrep = 50_000
    function tsec(f)
        f()
        t0 = time_ns()
        for _ in 1:nrep
            f()
        end
        return (time_ns() - t0) / nrep
    end
    t_node = tsec(() -> dist_node(pf, nodes))
    t_aabb = tsec(() -> dist_aabb(pf, lo, hi))
    t_ch = tsec(() -> dist_chord(pf, nodes[1], nodes[end]))
    t_g = tsec(() -> dist_gauss(pf, ys))
    t_nwt = if dad.dimension == 2
        tsec(() -> closest_point_1d(poly, nodes, pf))
    else
        tsec(() -> closest_point_2d(poly, nodes, pf))
    end
    @printf("  ns/call   node=%6.0f  aabb=%6.0f  chord=%6.0f  gauss-min=%6.0f  Newton=%8.0f\n",
        t_node, t_aabb, t_ch, t_g, t_nwt)
    @printf("  Newton / node = %.0fx    chord / node = %.2fx    aabb / node = %.2fx\n",
        t_nwt / t_node, t_ch / t_node, t_aabb / t_node)

    for f in (0.75, 2.0)
        c = classify_counts(dad, geos, f)
        @printf("  factor=%.2f  near-by-node=%.1f%%  far-by-AABB=%.1f%%  ambiguous=%.1f%%  (false-far if node-only=%.1f%% of all)\n",
            f, 100c.n_near_node / c.ntot, 100c.n_far_box / c.ntot,
            100c.n_amb / c.ntot, 100c.n_false_far / c.ntot)
    end
end

dad2 = format2d(quadrado(ndiv=12, show=false, nome="ndc2"), Laplace(1.0); pontointerno=true)
bench_one("Laplace 2-D linear ndiv=12", dad2)

msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="ndc3")
dad3 = format3d(msh, Laplace(1.0); pontointerno=false)
set_internal_nodes!(dad3, vec(cube_interior_grid(1.0, 2)))
bench_one("Laplace 3-D cube ndiv=2", dad3)

msh4 = mesh_unit_cube(; L=1.0, ndiv=4, nome="ndc3b")
dad4 = format3d(msh4, Laplace(1.0); pontointerno=false)
set_internal_nodes!(dad4, vec(cube_interior_grid(1.0, 2)))
bench_one("Laplace 3-D cube ndiv=4", dad4)
println("\nDone.")
