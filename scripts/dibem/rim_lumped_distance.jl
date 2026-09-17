#!/usr/bin/env julia
# Where nodal RIM lumping breaks vs Gauss, as a function of source–element
# distance over element size (d/L).
#
#   julia --project=. -t 8 scripts/dibem/rim_lumped_distance.jl
#
# `_near_element` flags near if any *node* is within `factor * Length`.
# This script reports pair-wise |lump − Gauss| vs true d/L (min distance to
# Gauss points) and vs node d/L, plus a global ID sweep over `near_factor`.

using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))

using BEM
using LinearAlgebra
using StaticArrays
using Printf
using Statistics: median, quantile

const PROJECT = dirname(dirname(@__DIR__))
include(joinpath(PROJECT, "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(PROJECT, "data", "Laplace", "cube_mesh.jl"))

const BINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, Inf)
const FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)

section(t) = (println(); println("="^72); println(t); println("="^72))

function _dmin(x, pts)
    m = Inf
    @inbounds for p in pts
        d = norm(x - p)
        d < m && (m = d)
    end
    return m
end

const _RBF = PHS(3; poly_deg=1)

function _laplace_gauss(x, g, props, dimv)
    accF = 0.0
    accD = 0.0
    @inbounds for q in eachindex(g.wJ)
        wJ = g.wJ[q]
        wJ == 0 && continue
        y = g.y[q]
        r = y - x
        R = norm(r)
        R < 1e-14 && continue
        wJn = wJ * BEM._rim_factor(g.n[q], r, R, dimv)
        accF += BEM.int(_RBF, x, y) * wJn
        accD += BEM.radial_integral(props, R, dimv) * wJn
    end
    return accF, accD
end

function _laplace_lump(x, g, props, dimv)
    accF = 0.0
    accD = 0.0
    @inbounds for j in eachindex(g.xj)
        xj = g.xj[j]
        r = xj - x
        R = norm(r)
        R < 1e-10 && continue
        wJn = g.wj[j] * BEM._rim_factor(g.nj[j], r, R, dimv)
        accF += BEM.int(_RBF, x, xj) * wJn
        accD += BEM.radial_integral(props, R, dimv) * wJn
    end
    return accF, accD
end

function _elast_gauss(x, g, props, dimv::Val{D}) where {D}
    BEM._dibem_elast_near_ID(x, g, PHS(3; poly_deg=1), props, dimv)
end
function _elast_lump(x, g, props, dimv::Val{D}) where {D}
    BEM._dibem_elast_far_ID(x, g, PHS(3; poly_deg=1), props, dimv)
end

_rel(a, b) = abs(a - b) / (abs(b) + 1e-16)
_relmat(A, B) = norm(A - B) / (norm(B) + 1e-16)

function _bin_index(x)
    for i in 1:length(BINS)-1
        BINS[i] ≤ x < BINS[i+1] && return i
    end
    return length(BINS) - 1
end

function _bin_label(i)
    a, b = BINS[i], BINS[i+1]
    b == Inf && return @sprintf("[%4.2f,  inf)", a)
    return @sprintf("[%4.2f, %4.2f)", a, b)
end

mutable struct BinAcc
    n::Int
    errs::Vector{Float64}
    dL::Vector{Float64}
    BinAcc() = new(0, Float64[], Float64[])
end

function _push!(b::BinAcc, err, dL)
    b.n += 1
    push!(b.errs, err)
    push!(b.dL, dL)
end

function _summarize(bins)
    println("  d/L bin              n     median      p90      p99      max")
    for i in 1:length(bins)
        b = bins[i]
        b.n == 0 && continue
        e = sort(b.errs)
        n = length(e)
        med = e[max(1, n ÷ 2)]
        p90 = e[max(1, Int(ceil(0.90 * n)))]
        p99 = e[max(1, Int(ceil(0.99 * n)))]
        @printf("  %-18s %6d  %8.1e  %8.1e  %8.1e  %8.1e\n",
            _bin_label(i), b.n, med, p90, p99, e[end])
    end
end

function collect_pairs_laplace(dad; npg=16)
    BEM._init_quadrature!(dad, npg)
    geos = BEM._rim_build_elements(dad)
    dimv = Val(Int(dad.dimension))
    props = dad.properties
    bins_true = [BinAcc() for _ in 1:length(BINS)-1]
    bins_node = [BinAcc() for _ in 1:length(BINS)-1]
    n_on = 0
    worst = (err=0.0, dL=NaN, dnodeL=NaN, i=0, eidx=0)
    @inbounds for i in 1:dad.nt
        x = point(dad, i)
        for (eidx, g) in enumerate(geos)
            d_true = _dmin(x, g.y)
            d_node = _dmin(x, g.nodes)
            L = max(g.el.Length, 1e-16)
            _, Ig = _laplace_gauss(x, g, props, dimv)
            _, Il = _laplace_lump(x, g, props, dimv)
            err = _rel(Il, Ig)
            dL = d_true / L
            dnL = d_node / L
            _push!(bins_true[_bin_index(dL)], err, dL)
            _push!(bins_node[_bin_index(dnL)], err, dnL)
            if d_true < 1e-12
                n_on += 1
            elseif err > worst.err
                worst = (err=err, dL=dL, dnodeL=dnL, i=i, eidx=eidx)
            end
        end
    end
    return bins_true, bins_node, n_on, worst
end

function collect_pairs_elast(dad; npg=12)
    ηs, ws = gausslegendre(npg)
    geos = BEM._rim_build_elements(dad, ηs, ws)
    D = dad.dimension
    dimv = Val(D)
    props = dad.properties
    bins_true = [BinAcc() for _ in 1:length(BINS)-1]
    bins_node = [BinAcc() for _ in 1:length(BINS)-1]
    n_on = 0
    worst = (err=0.0, dL=NaN, dnodeL=NaN, i=0, eidx=0)
    @inbounds for i in 1:dad.nt
        x = point(dad, i)
        for (eidx, g) in enumerate(geos)
            d_true = _dmin(x, g.y)
            d_node = _dmin(x, g.nodes)
            L = max(g.el.Length, 1e-16)
            _, Ig = _elast_gauss(x, g, props, dimv)
            _, Il = _elast_lump(x, g, props, dimv)
            err = _relmat(Il, Ig)
            dL = d_true / L
            dnL = d_node / L
            _push!(bins_true[_bin_index(dL)], err, dL)
            _push!(bins_node[_bin_index(dnL)], err, dnL)
            if d_true < 1e-12
                n_on += 1
            elseif err > worst.err
                worst = (err=err, dL=dL, dnodeL=dnL, i=i, eidx=eidx)
            end
        end
    end
    return bins_true, bins_node, n_on, worst
end

function factor_sweep_laplace(dad; npg=16)
    rbf = PHS(3; poly_deg=1)
    BEM._init_quadrature!(dad, npg)
    geos = BEM._rim_build_elements(dad)
    IFref = zeros(dad.nt)
    IDref = zeros(dad.nt)
    BEM._dibem_accumulate_IF_ID!(IFref, IDref, dad, rbf; threaded=true,
        geos=geos, near_factor=Inf)  # all Gauss
    println("  factor    n_near%    ‖ΔIF‖/‖IF‖    ‖ΔID‖/‖ID‖")
    for f in FACTORS
        IF = zeros(dad.nt)
        ID = zeros(dad.nt)
        BEM._dibem_accumulate_IF_ID!(IF, ID, dad, rbf; threaded=true,
            geos=geos, near_factor=float(f))
        n_near = 0
        ntot = 0
        @inbounds for i in 1:dad.nt
            x = point(dad, i)
            for g in geos
                ntot += 1
                BEM._near_element(x, g.nodes, g.el; factor=float(f)) && (n_near += 1)
            end
        end
        @printf("  %5.2f    %6.1f%%     %9.2e     %9.2e\n",
            f, 100n_near / max(ntot, 1),
            norm(IF - IFref) / (norm(IFref) + 1e-16),
            norm(ID - IDref) / (norm(IDref) + 1e-16))
    end
end

function factor_sweep_elast(dad; npg=12)
    rbf = PHS(3; poly_deg=1)
    IFref, IDref, _ = BEM._dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=true,
        rim=:full_gauss)
    geos = BEM._rim_build_elements(dad, gausslegendre(npg)...)
    println("  factor    n_near%    ‖ΔIF‖/‖IF‖    ‖ΔID‖/‖ID‖")
    for f in FACTORS
        IF, ID, _ = BEM._dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=true,
            rim=:lumped, near_factor=float(f))
        n_near = 0
        ntot = 0
        @inbounds for i in 1:dad.nt
            x = point(dad, i)
            for g in geos
                ntot += 1
                BEM._near_element(x, g.nodes, g.el; factor=float(f)) && (n_near += 1)
            end
        end
        @printf("  %5.2f    %6.1f%%     %9.2e     %9.2e\n",
            f, 100n_near / max(ntot, 1),
            norm(IF - IFref) / (norm(IFref) + 1e-16),
            norm(ID - IDref) / (norm(IDref) + 1e-16))
    end
end

function report_pairs(name, bins_true, bins_node, n_on, worst, nt, ne)
    println("  $name   nt=$nt  nelem=$ne  on-element pairs (d≈0): $n_on")
    println("  worst off-element pair:  rel=$(worst.err)  d/L=$(worst.dL)  d_node/L=$(worst.dnodeL)  src=$(worst.i)  el=$(worst.eidx)")
    println("\n  vs true d/L (min distance to Gauss points / Length)")
    _summarize(bins_true)
    println("\n  vs node d/L (what `_near_element` uses)")
    _summarize(bins_node)
    # at default factor=2, which lumped pairs (d_node/L ≥ 2) still have large error?
    n_far = 0
    n_bad3 = 0
    n_bad2 = 0
    n_bad1 = 0
    min_dL_bad3 = Inf
    for i in 1:length(bins_node)
        b = bins_node[i]
        BINS[i] < 2.0 && continue  # these are classified near at factor=2
        for (e, dL) in zip(b.errs, b.dL)
            n_far += 1
            if e > 1e-3
                n_bad3 += 1
                dL < min_dL_bad3 && (min_dL_bad3 = dL)
            end
            e > 1e-2 && (n_bad2 += 1)
            e > 1e-1 && (n_bad1 += 1)
        end
    end
    println("\n  among pairs lumped at factor=2 (d_node/L ≥ 2):")
    @printf("    n=%d   err>1e-3: %d (%.2f%%)   err>1e-2: %d   err>1e-1: %d\n",
        n_far, n_bad3, 100n_bad3 / max(n_far, 1), n_bad2, n_bad1)
    if isfinite(min_dL_bad3)
        @printf("    smallest true d/L with lumped err>1e-3: %.3f\n", min_dL_bad3)
    else
        println("    no lumped pair with err>1e-3")
    end
end

function laplace2d()
    msh = quadrado(ndiv=12, show=false, nome="rim_d_l2")
    dad = format2d(msh, Laplace(1.0); pontointerno=true)
    section("Laplace 2-D  ndiv=12  n=$(dad.n) ni=$(dad.ni) nelem=$(length(dad.elements))")
    bt, bn, non, w = collect_pairs_laplace(dad; npg=16)
    report_pairs("Laplace 2-D PHS3+U*", bt, bn, non, w, dad.nt, length(dad.elements))
    println("\n  global IF,ID vs all-Gauss, sweeping near_factor")
    factor_sweep_laplace(dad; npg=16)
end

function laplace3d()
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="rim_d_l3")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    section("Laplace 3-D  cube ndiv=2  n=$(dad.n) ni=$(dad.ni) nelem=$(length(dad.elements))")
    bt, bn, non, w = collect_pairs_laplace(dad; npg=8)
    report_pairs("Laplace 3-D PHS3+U*", bt, bn, non, w, dad.nt, length(dad.elements))
    println("\n  global IF,ID vs all-Gauss, sweeping near_factor")
    factor_sweep_laplace(dad; npg=8)
end

function elast2d()
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = quadrado_elasticity(ndiv=12, show=false, nome="rim_d_e2")
    dad = format2d(msh, props; pontointerno=true)
    section("Kelvin 2-D  ndiv=12  n=$(dad.n) ni=$(dad.ni) nelem=$(length(dad.elements))")
    bt, bn, non, w = collect_pairs_elast(dad; npg=12)
    report_pairs("Kelvin 2-D U*", bt, bn, non, w, dad.nt, length(dad.elements))
    println("\n  global IF,ID vs all-Gauss, sweeping near_factor")
    factor_sweep_elast(dad; npg=12)
end

function elast3d()
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="rim_d_e3", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    section("Kelvin 3-D  cube ndiv=2  n=$(dad.n) ni=$(dad.ni) nelem=$(length(dad.elements))")
    bt, bn, non, w = collect_pairs_elast(dad; npg=8)
    report_pairs("Kelvin 3-D U*", bt, bn, non, w, dad.nt, length(dad.elements))
    println("\n  global IF,ID vs all-Gauss, sweeping near_factor")
    factor_sweep_elast(dad; npg=8)
end

function main()
    println("RIM lumped vs Gauss  vs  d/L")
    println("  near if any node is within factor*Length (default factor=2)")
    println("  true d = min distance to element Gauss points")
    laplace2d()
    elast2d()
    laplace3d()
    elast3d()
    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
