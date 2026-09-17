# Surface DIBEM vs closed form (parent square) and polar (cube interior T=z).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays

include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "cube_mesh.jl"))

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
I1(d) = _rect_1R(1, 1, d) - _rect_1R(-1, 1, d) - _rect_1R(1, -1, d) + _rect_1R(-1, -1, d)
I3(d) = _rect_zR3(1, 1, d) - _rect_zR3(-1, 1, d) - _rect_zR3(1, -1, d) + _rect_zR3(-1, -1, d)

function _square_dad()
    poly = BEM.Equispaced(1)
    nodes = [Point3D(-1.0, -1.0, 0.0), Point3D(1.0, -1.0, 0.0),
             Point3D(-1.0, 1.0, 0.0), Point3D(1.0, 1.0, 0.0)]
    nrm = [Point3D(0.0, 0.0, 1.0) for _ in 1:4]
    el = Element([1, 2, 3, 4], ones(4), 2.0, 1)
    dad = BEMdata(; name="surf_dibem_sq", dimension=3, elements=[el],
        element_type=poly, elem_weight=SA[1.0, 1.0, 1.0, 1.0],
        collocation=nodes, Normal=nrm, properties=Laplace(1.0),
        BC=ones(Int, 4), BV=zeros(4), n=4, ni=0, nt=4)
    qq, ww = BEM.gausslegendre(8)
    set_cache!(dad; qsi=qq, w=ww, nearfield=:dibem)
    return dad, el, nodes
end

println("=== PHS3 weights / IF ===")
ξc, ηc = BEM._surf_dibem_centers(8)
c = BEM._surf_dibem_weights(ξc, ηc)
@printf("  n=%d  sum(c)-4 = %.3e  c·ξ = %.3e  c·η = %.3e\n",
    length(c), sum(c) - 4, dot(c, ξc), dot(c, ηc))

println("=== parent square  source (0,0,d)  nedge=8 ===")
@printf("  %8s  %10s  %10s  %10s  %10s\n", "d", "ID U rel", "ID T rel", "∑g rel", "∑h rel")
dad, el, nodes = _square_dad()
for d in (0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5)
    t1, t3 = I1(d), I3(d)
    IDu = BEM._id_radial_parent(0.0, 0.0, d, 1)
    IDt = d * BEM._id_radial_parent(0.0, 0.0, d, 3)
    h = zeros(4); g = zeros(4)
    pf = Point3D(0.0, 0.0, d)
    fill!(h, 0); fill!(g, 0)
    BEM.integrate_element_dibem!(h, g, dad, el, nodes, pf)
    gex, hex = t1 / (4π), -t3 / (4π)
    @printf("  %8.0e  %10.2e  %10.2e  %10.2e  %10.2e\n", d,
        abs(IDu - t1) / t1, abs(IDt - t3) / t3,
        abs(sum(g) - gex) / abs(gex), abs(sum(h) - hex) / abs(hex))
end

println("=== nodal split vs polar ===")
@printf("  %8s  %10s  %10s\n", "d", "rel g", "rel h")
for d in (0.5, 0.1, 1e-2, 1e-3, 1e-5)
    pf = Point3D(0.0, 0.0, d)
    hd = zeros(4); gd = zeros(4)
    BEM.integrate_element_dibem!(hd, gd, dad, el, nodes, pf)
    set_cache!(dad; nearfield=:polar)
    hp = zeros(4); gp = zeros(4)
    integrate_element(dad, el, nodes, pf, hp, gp)
    set_cache!(dad; nearfield=:dibem)
    @printf("  %8.0e  %10.2e  %10.2e\n", d,
        norm(gd - gp) / max(norm(gp), 1e-16),
        norm(hd - hp) / max(norm(hp), 1e-16))
end

println("=== cube T=z interior approach  ndiv=2  npg=8 ===")
msh = mesh_cube(; L=1.0, ndiv=2, nome="surf_dibem_cube")
dadc = format3d(msh, Laplace(1.0); pontointerno=false)
attach_analytical!(dadc, ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0]))
assemble!(dadc; npg=8, threaded=false)
solve(dadc)
function _intT(dad, pf)
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
    c0 = -sum(h)
    return (dot(g, view(dad.q, 1:n)) - dot(h, view(dad.T, 1:n))) / c0
end
@printf("  %8s  %10s  %10s  %10s\n", "d", "polar", "dibem", "plain")
for d in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    pf = Point3D(0.25, 0.25, d)
    set_cache!(dadc; nearfield=:polar)
    epol = abs(_intT(dadc, pf) - d)
    set_cache!(dadc; nearfield=:dibem)
    edib = abs(_intT(dadc, pf) - d)
    set_cache!(dadc; nearfield=:plain)
    epl = abs(_intT(dadc, pf) - d)
    @printf("  %8.0e  %10.2e  %10.2e  %10.2e\n", d, epol, edib, epl)
end
