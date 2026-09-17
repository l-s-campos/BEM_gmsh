# Interior point → boundary: Granados maps on a fine T=x square.
# (1) representation T = Gq − HT (c = 1) from a fixed boundary solve
# (2) coupled internal collocation (all approach points in one assembly)
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Topology: interior_grad_T

const MAPS = (:plain, :euclid, :csinh, :sinhsinh, :tangent, :p3c, :tanp3c)
const DS = [10.0^k for k in -1:-1:-8]

function interior_potential(dad, pf)
    # c u = ∫ U q − ∫ T* u, with c = −∑ T* (row-sum; +1 inside, 0 outside)
    n = dad.n
    T, q = dad.T, dad.q
    h = zeros(float(eltype(T)), n)
    g = zeros(float(eltype(T)), n)
    @inbounds for el in dad.elements
        xj = dad.Nodes[el.index]
        nn = length(el.index)
        hloc = zeros(eltype(h), nn)
        gloc = zeros(eltype(g), nn)
        integrate_element(dad, el, xj, pf, hloc, gloc)
        for (a, j) in enumerate(el.index)
            h[j] += hloc[a]
            g[j] += gloc[a]
        end
    end
    c = -sum(h)
    return (dot(g, view(q, 1:n)) - dot(h, view(T, 1:n))) / c
end

function print_table(title, ds, errs, maps)
    println("\n", title)
    @printf("  %-8s", "d")
    for nf in maps
        @printf("  %11s", String(nf))
    end
    println()
    for (i, d) in enumerate(ds)
        @printf("  %8.0e", d)
        for e in errs[i]
            @printf("  %11.3e", e)
        end
        println()
    end
end

ndiv, npg = 40, 16
msh = quadrado(ndiv=ndiv, show=false, nome="nf_approach_n$(ndiv)")
println("=== mesh ndiv=$ndiv  npg=$npg  L=$(1 / ndiv) ===")
dad = format2d(msh, Laplace(1.0); pontointerno=false)
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
set_cache!(dad; nearfield=:sinhsinh)
assemble!(dad, npg; threaded=false)
solve(dad)
@printf("  boundary  rel_error(T)=%.4e  rel_error(q)=%.4e  n=%d\n",
    rel_error(dad), rel_error_flux(dad), dad.n)
set_cache!(dad; nearfield=:sinhsinh)
@printf("  centre T(0.5,0.5)=%.6f  (ana 0.5)\n",
    interior_potential(dad, Point2D(0.5, 0.5)))

function sweep_rep(dad, pts, Tana, gana)
    eT = [zeros(length(MAPS)) for _ in pts]
    eG = [zeros(length(MAPS)) for _ in pts]
    for (k, pf) in enumerate(pts)
        for (j, nf) in enumerate(MAPS)
            set_cache!(dad; nearfield=nf)
            eT[k][j] = abs(interior_potential(dad, pf) - Tana[k])
            g = interior_grad_T(dad, [pf])[1]
            eG[k][j] = norm(g - gana)
        end
    end
    return eT, eG
end

gana = SA[1.0, 0.0]
pts_bot = [Point2D(0.5, d) for d in DS]
eT, eG = sweep_rep(dad, pts_bot, fill(0.5, length(DS)), gana)
print_table("representation |T−0.5|     (0.5, d) → y=0", DS, eT, MAPS)
print_table("representation |∇T−(1,0)|  (0.5, d) → y=0", DS, eG, MAPS)

pts_r = [Point2D(1 - d, 0.5) for d in DS]
eTr, eGr = sweep_rep(dad, pts_r, [1 - d for d in DS], gana)
print_table("representation |T−(1−d)|   (1−d, 0.5) → x=1", DS, eTr, MAPS)
print_table("representation |∇T−(1,0)|  (1−d, 0.5) → x=1", DS, eGr, MAPS)

# One assembly per map; internals do not couple (H_ii is row-sum only).
println("\ncollocation |T_internal−0.5|  (0.5, d) → y=0")
@printf("  %-8s", "d")
for nf in MAPS
    @printf("  %11s", String(nf))
end
println()
col = Dict{Symbol,Vector{Float64}}()
for nf in MAPS
    dadc = format2d(msh, Laplace(1.0); pontointerno=false)
    attach_analytical!(dadc, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    set_internal_nodes!(dadc, pts_bot)
    set_cache!(dadc; nearfield=nf)
    assemble!(dadc, npg; threaded=false)
    solve(dadc)
    col[nf] = abs.(dadc.T[dadc.n+1:end] .- 0.5)
end
for (i, d) in enumerate(DS)
    @printf("  %8.0e", d)
    for nf in MAPS
        @printf("  %11.3e", col[nf][i])
    end
    println()
end
