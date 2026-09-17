# P3 with exact circular geometry at every Gauss / SST sample.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function make_mesh(; load=:tx)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3ex"); lc=80.0
    c=gmsh.model.geo.addPoint(0,0,0,lc)
    p1=gmsh.model.geo.addPoint(300,0,0,lc); p2=gmsh.model.geo.addPoint(600,0,0,lc)
    p3=gmsh.model.geo.addPoint(0,600,0,lc); p4=gmsh.model.geo.addPoint(0,300,0,lc)
    b=gmsh.model.geo.addLine(p1,p2); o=gmsh.model.geo.addCircleArc(p2,c,p3)
    t=gmsh.model.geo.addLine(p3,p4); inn=gmsh.model.geo.addCircleArc(p4,c,p1)
    cl=gmsh.model.geo.addCurveLoop([b,o,t,inn]); s1=gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(b,3); gmsh.model.mesh.setTransfiniteCurve(t,3)
    gmsh.model.mesh.setTransfiniteCurve(o,11); gmsh.model.mesh.setTransfiniteCurve(inn,11)
    bc = load===:ty ? "1;0;1;-1000" : "1;1000;1;0"
    gmsh.model.addPhysicalGroup(1,[b],-1,bc)
    gmsh.model.addPhysicalGroup(1,[t],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inn,o],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out=datadir("elastico","p3_exact.msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

# --- exact polar geometry for nodes that sit on a circle about the origin ---
function _is_circle(nodes; tol=0.02)
    rs = norm.(nodes)
    R = sum(rs) / length(rs)
    R < 50 && return false
    (maximum(rs) - minimum(rs)) / R > tol && return false
    θ = [atan(n[2], n[1]) for n in nodes]
    return maximum(θ) - minimum(θ) > 0.02
end

function _circle_at(nodes, poly, ξ)
    rs = norm.(nodes)
    R = sum(rs) / length(rs)
    θs = [atan(n[2], n[1]) for n in nodes]
    N, dN = BEM.shapefun(poly, ξ)
    θ = 0.0; dθ = 0.0
    @inbounds for k in eachindex(nodes)
        θ += N[1, k] * θs[k]
        dθ += dN[1, k] * θs[k]
    end
    s, c = sincos(θ)
    pg = Point2D(R * c, R * s)
    dx = Point2D(-R * s * dθ, R * c * dθ)
    J = norm(dx)
    nrm = Point2D(dx[2], -dx[1]) / J   # tan2normal
    return pg, dx, J, nrm, view(N, 1, :)
end

@eval BEM function _geom_1d(poly, nodes, a)
    if Main._is_circle(nodes)
        pg, dx, J, n, Nrow = Main._circle_at(nodes, poly, a)
        J < 1e-30 && return nothing
        t = dx / J
        return Nrow, J, t, n
    end
    N, dN = shapefun(poly, a)
    nN = size(N, 2)
    x = zero(eltype(nodes))
    dx = zero(eltype(nodes))
    @inbounds for k in 1:nN
        x += N[1, k] * nodes[k]
        dx += dN[1, k] * nodes[k]
    end
    J = norm(dx)
    J < 1e-30 && return nothing
    t = dx / J
    n = SVector{2,Float64}(t[2], -t[1])
    return view(N, 1, :), J, t, n
end

@eval BEM function _sample_kernel(dad, poly, nodes, pf, ξ, f)
    if Main._is_circle(nodes)
        pg, dx, J, nrm, Nrow = Main._circle_at(nodes, poly, ξ)
        J < 1e-30 && return nothing
        r = pg - pf
        norm(r) < 1e-30 && return nothing
        U, T = f(dad, r, nrm)
        return U, T, Nrow, J
    end
    N, dN = shapefun(poly, ξ)
    pg = zero(eltype(nodes))
    dx = zero(eltype(nodes))
    @inbounds for k in eachindex(nodes)
        pg += N[1, k] * nodes[k]
        dx += dN[1, k] * nodes[k]
    end
    J = norm(dx)
    J < 1e-30 && return nothing
    r = pg - pf
    norm(r) < 1e-30 && return nothing
    nrm = tan2normal(dx / J)
    U, T = f(dad, r, nrm)
    return U, T, view(N, 1, :), J
end

@eval BEM function _quad_geom(dad, elem, x::AbstractVector{<:Point2D}, pf::Point2D)
    eta, ww = transform(dad, elem, x, pf)
    if Main._is_circle(x)
        n = length(eta)
        nN = length(x)
        N = zeros(n, nN)
        r = Vector{Point2D}(undef, n)
        nrm = Vector{Point2D}(undef, n)
        wwJ = similar(ww)
        poly = dad.element_type
        @inbounds for i in 1:n
            pg, dx, J, ni, Ni = Main._circle_at(x, poly, eta[i])
            N[i, :] .= Ni
            r[i] = pg - pf
            nrm[i] = ni
            wwJ[i] = J * ww[i]
        end
        return N, r, nrm, wwJ
    end
    N, dN = shapefun(dad.element_type, eta)
    pg = N * x
    dx = dN * x
    r = pg .- Ref(pf)
    J = norm.(dx)
    nrm = tan2normal.(dx ./ J)
    return N, r, nrm, J .* ww
end

function go(load)
    msh = make_mesh(; load=load)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    H_G_hyper(dad; npg=50, threaded=false); solve(dad)
    rel = norm(dad.u .- uc) / (norm(uc) + 1e-30)
    @printf("exact-circle load=%-3s  n=%d  CBIE=%10.3f mm (%6.2f cm)  HBIE=%10.3f mm (%6.2f cm)  rel=%.3e  cond=%.2e\n",
        load, dad.n, maximum(abs, uc), maximum(abs, uc)/10,
        maximum(abs, dad.u), maximum(abs, dad.u)/10, rel, cond(Matrix(dad.A)))
end

println("P3 exact circular geometry")
go(:tx)
go(:ty)
println("done")
