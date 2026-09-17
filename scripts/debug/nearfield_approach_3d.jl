# 3-D nearly-singular maps: parent-square kernels + cube T=z approach.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays

include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "cube_mesh.jl"))

# Closed form on parent square [-1,1]², source (a,b,d), n = ê_z.
function _rect_1R(x, y, z)
    R = hypot(x, y, z)
    rx, ry = hypot(x, z), hypot(y, z)
    t1 = rx > 0 ? x * asinh(y / rx) : 0.0
    t2 = ry > 0 ? y * asinh(x / ry) : 0.0
    t3 = (abs(z) < 1e-16 || R < 1e-16) ? 0.0 : z * atan(x * y / (z * R))
    return t1 + t2 - t3
end
function _rect_zR3(x, y, z)
    R = hypot(x, y, z)
    (abs(z) < 1e-16 || R < 1e-16) && return 0.0
    return atan(x * y / (z * R))
end
function I1R(a, b, d)
    xs = (-1 - a, 1 - a); ys = (-1 - b, 1 - b)
    return _rect_1R(xs[2], ys[2], d) - _rect_1R(xs[1], ys[2], d) -
           _rect_1R(xs[2], ys[1], d) + _rect_1R(xs[1], ys[1], d)
end
function IzR3(a, b, d)
    xs = (-1 - a, 1 - a); ys = (-1 - b, 1 - b)
    return _rect_zR3(xs[2], ys[2], d) - _rect_zR3(xs[1], ys[2], d) -
           _rect_zR3(xs[2], ys[1], d) + _rect_zR3(xs[1], ys[1], d)
end

function Iquad(mode, a, b, d, n, f)
    ξ, η, w = BEM._nearfield_2d(a, b, d; n=n, mode=mode)
    s = 0.0
    @inbounds for i in eachindex(w)
        s += w[i] * f(ξ[i], η[i])
    end
    return s
end

println("=== parent square [-1,1]²  n=8  source (a,b,d) ===")
@printf("  %-22s  %-8s  %10s %10s %10s %10s\n", "geom", "d", "plain", "tensor", "polar", "auto")
geoms = (
    ("face centre", 0.0, 0.0),
    ("offset face", 0.5, 0.3),
    ("near edge", 0.95, 0.0),
    ("near corner", 0.95, 0.95),
    ("outside edge", 1.4, 0.0),
    ("outside corner", 1.4, 1.4),
)
for (label, a, b) in geoms
    for d in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
        t1, t3 = I1R(a, b, d), IzR3(a, b, d)
        f1 = (ξ, η) -> 1 / hypot(ξ - a, η - b, d)
        f3 = (ξ, η) -> d / hypot(ξ - a, η - b, d)^3
        @printf("  %-22s  %8.0e", "$label 1/R", d)
        for mode in (:plain, :tensor, :polar, :auto)
            Iq = Iquad(mode, a, b, d, 8, f1)
            @printf("  %10.2e", abs(Iq - t1) / max(abs(t1), 1e-16))
        end
        println()
        @printf("  %-22s  %8.0e", "$label z/R³", d)
        for mode in (:plain, :tensor, :polar, :auto)
            Iq = Iquad(mode, a, b, d, 8, f3)
            @printf("  %10.2e", abs(Iq - t3) / max(abs(t3), 1e-16))
        end
        println()
    end
end

function interior_T(dad, pf)
    n = dad.n
    h = zeros(n); g = zeros(n)
    @inbounds for el in dad.elements
        xj = dad.Nodes[el.index]
        nn = length(el.index)
        hloc = zeros(nn); gloc = zeros(nn)
        integrate_element(dad, el, xj, pf, hloc, gloc)
        for (a, j) in enumerate(el.index)
            h[j] += hloc[a]; g[j] += gloc[a]
        end
    end
    c = -sum(h)
    return (dot(g, view(dad.q, 1:n)) - dot(h, view(dad.T, 1:n))) / c
end

println("\n=== cube T=z  ndiv=4  npg=8 ===")
msh = mesh_cube(; L=1.0, ndiv=4, nome="nf3d_cube")
dad = format3d(msh, Laplace(1.0); pontointerno=false)
attach_analytical!(dad, ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0]))
assemble!(dad; npg=8, threaded=false)
solve(dad)
@printf("  boundary relT=%.3e  relq=%.3e  n=%d  ne=%d\n",
    rel_error(dad), rel_error_flux(dad), dad.n, length(dad.elements))

using BEM.Topology: interior_grad_T
gana = SA[0.0, 0.0, 1.0]
approaches = (
    ("face  (0.5,0.5,d)", d -> Point3D(0.5, 0.5, d), d -> d),
    ("edge  (0.5,d,d)",   d -> Point3D(0.5, d, d),   d -> d),
    ("corner (d,d,d)",    d -> Point3D(d, d, d),     d -> d),
    ("side  (d,0.5,0.5)", d -> Point3D(d, 0.5, 0.5), d -> 0.5),
)
maps = (:plain, :tensor, :polar, :auto)
@printf("  %-20s %-8s", "geom", "d")
for nf in maps
    @printf("  %10s", "T-" * String(nf))
end
for nf in maps
    @printf("  %10s", "G-" * String(nf))
end
println()
for (label, pfun, Tana) in approaches
    for d in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
        pf = pfun(d)
        Ta = Tana(d)
        @printf("  %-20s %8.0e", label, d)
        eT = Float64[]; eG = Float64[]
        for nf in maps
            set_cache!(dad; nearfield=nf)
            push!(eT, abs(interior_T(dad, pf) - Ta))
            g = interior_grad_T(dad, [pf])[1]
            push!(eG, norm(g - gana))
        end
        for e in eT
            @printf("  %10.2e", e)
        end
        for e in eG
            @printf("  %10.2e", e)
        end
        println()
    end
end
