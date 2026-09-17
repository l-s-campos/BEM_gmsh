#!/usr/bin/env julia
# Follow-up: quantify elem_geom cost and dump hottest Profile frames.
using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))
using BEM
using BEM.Plate
using StaticArrays
using LinearAlgebra
using Printf
using Profile

const TP = BEM.Plate.ThinPlate
const Point2D = SVector{2,Float64}

function quiet(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f()
        end
    end
end

function elem_geom_cached(el, ξ, poly_geo)
    Ng, dNg = BEM.shapefun(poly_geo, ξ)
    geo = el.geo
    x = zero(geo[1])
    dx = zero(geo[1])
    @inbounds for k in eachindex(geo)
        x += Ng[1, k] * geo[k]
        dx += dNg[1, k] * geo[k]
    end
    J = norm(dx)
    n = J > 0 ? Point2D(dx[2] / J, -dx[1] / J) : Point2D(0.0, 0.0)
    return x, J, n
end

function elem_geom_linear(el, ξ)
    g0, g1 = el.geo[1], el.geo[end]
    N0 = (1 - ξ) / 2
    N1 = (1 + ξ) / 2
    x = N0 * g0 + N1 * g1
    dx = (g1 - g0) / 2
    J = norm(dx)
    n = Point2D(dx[2] / J, -dx[1] / J)
    return x, J, n
end

mesh = build_square_plate(; a=1.0, n_el=8, bc="SSSS",
    props=ThinPlateProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0),
    corner_bc='F', n_internal=1, p=2)
el = mesh.elements[1]
poly_geo = BEM.Equispaced(length(el.geo) - 1)
ξ = 0.1
N = 50_000

TP.elem_geom(el, ξ)
elem_geom_cached(el, ξ, poly_geo)
elem_geom_linear(el, ξ)

function bench(label, f)
    f()
    GC.gc(false)
    t0 = time_ns()
    for _ in 1:N
        f()
    end
    dt = (time_ns() - t0) / 1e9
    a = @allocated f()
    @printf("  %-28s  %8.2f ns/call   alloc %d B\n", label, 1e9 * dt / N, a)
end

println("Geometry sampling (n=$(N))")
bench("elem_geom (current)", () -> TP.elem_geom(el, ξ))
bench("elem_geom cached poly", () -> elem_geom_cached(el, ξ, poly_geo))
bench("elem_geom linear edge", () -> elem_geom_linear(el, ξ))
bench("plate_kernels iso", () -> plate_kernels(el.geo[1], mesh.nodes[1],
    mesh.Normal[1], mesh.Normal[1], mesh.props))

# Accuracy: linear vs Equispaced on a straight square-plate edge
pg, J, n = TP.elem_geom(el, ξ)
pgL, JL, nL = elem_geom_linear(el, ξ)
@printf("\n  |Δx|=%.3e  |ΔJ|=%.3e  |Δn|=%.3e  (straight edge)\n",
    norm(pg - pgL), abs(J - JL), norm(n - nL))

# CPU profile, hottest frames last
println("\nHottest Profile frames (assemble_plate! :guiggiani n_el=8)")
quiet(() -> assemble_plate!(build_square_plate(; a=1.0, n_el=8, bc="SSSS",
    props=ThinPlateProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0),
    corner_bc='F', n_internal=1); npg=12, singular=:guiggiani))
mesh2 = build_square_plate(; a=1.0, n_el=8, bc="SSSS",
    props=ThinPlateProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0),
    corner_bc='F', n_internal=1)
Profile.init(n=10^8, delay=0.0005)
Profile.clear()
@profile quiet(() -> assemble_plate!(mesh2; npg=12, singular=:guiggiani))
io = IOBuffer()
Profile.print(io; format=:flat, sortedby=:count, C=false)
lines = filter(!isempty, split(String(take!(io)), '\n'))
println(join(lines[1:min(6, length(lines))], '\n'))
println("  ...")
println(join(lines[max(1, end - 35):end], '\n'))

# Count samples mentioning key names
txt = join(lines, '\n')
for key in ("elem_geom", "shapefun", "Equispaced", "interpolation_matrix",
            "plate_kernels", "compute_q_el", "guiggiani", "_plate_Nwt",
            "assemble_w_row", "zeros")
    n = count(l -> occursin(key, l), lines)
    println("  frames containing $(rpad(key, 24)) $n")
end
