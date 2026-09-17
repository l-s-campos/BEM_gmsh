# SST vs Guiggiani on anisotropic P3, plus remainder regularity.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

const MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334,
              η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
    η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

# --- 1. remainder K-K* as ξ→a on a curved arc ---
println("="^64)
println(" remainder regularity on a circular quadratic")
poly = BEM.Legendre(2)
θ = (0.0, π/10, π/5)
X = [Point2D(600*cos(t), 600*sin(t)) for t in θ]
a = poly.nodes[1]   # first GL node (as in disc. collocation)
qsi, w = BEM.gausslegendre(16)
N0, dN0 = BEM.shapefun(poly, a)
g0 = BEM._geom_1d(poly, X, a)
Nrow, J0, t0, n0 = g0
pf = (N0 * X)[1]
Uh, Th = BEM._lekh_Uh_Th_lead(props.params, n0)
invJ = 1 / J0

function fpair(ξ)
    N, dN = BEM.shapefun(poly, ξ)
    pg = (N * X)[1]; dx = (dN * X)[1]; J = norm(dx)
    nrm = Point2D(dx[2], -dx[1]) / J
    r = pg - pf
    norm(r) < 1e-30 && return zeros(2, 6), zeros(2, 6)
    U, T = fundamental_hyper(props, pg, pf, nrm, n0)
    U = BEM._to_smat(U); T = BEM._to_smat(T)
    Fg = zeros(2, 6); Fh = zeros(2, 6)
    for j in 1:3
        cols = (2j-1):(2j)
        Fg[:, cols] .= U .* (N[1, j] * J)
        Fh[:, cols] .= T .* (N[1, j] * J)
    end
    return Fg, Fh
end

println("  a=$(a)  J0=$(J0)  n=$(n0)  pf=$(pf)")
println("  δ          ||Fh-K*||     ||δ(Fh-K*)||   ||δ²(Fh-K*)||  ||Fh||")
for δ in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
    ξ = a + δ
    abs(ξ) > 1 && continue
    Fg, Fh = fpair(ξ)
    Ks = zeros(2, 6)
    for j in 1:3
        c0 = 2j
        for β in 1:2, α in 1:2
            Ks[α, 2j-2+β] = Th[α, β] * (Nrow[j] / (δ*δ*J0) + dN0[1, j] / (δ*J0))
        end
    end
    dF = Fh - Ks
    @printf("  %.1e   %10.3e    %10.3e    %10.3e   %10.3e\n",
        δ, norm(dF), norm(δ*dF), norm(δ^2*dF), norm(Fh))
end

IgS, IhS = sst_GH(fpair, a; qsi=qsi, w=w, props=props, poly=poly, nodes=X)
IgG, IhG = guiggiani_GH(fpair, a; order_G=-1, order_H=-2, qsi=qsi, w=w,
    props=props, poly=poly, nodes=X)
println("\n  on-element  ||Ih_SST||=$(norm(IhS))  ||Ih_Gui||=$(norm(IhG))  rel=$(norm(IhS-IhG)/(norm(IhG)+1e-30))")
println("  on-element  ||Ig_SST||=$(norm(IgS))  ||Ig_Gui||=$(norm(IgG))  rel=$(norm(IgS-IgG)/(norm(IgG)+1e-30))")

# straight control
println("\n--- straight unit element (SST should match Guiggiani) ---")
polys = BEM.Equispaced(1)
ns = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
as = 0.0
pfs = Point2D(0.5, 0.0)
nel = Point2D(0.0, -1.0)
fstr = ξ -> begin
    N, dN = BEM.shapefun(polys, ξ)
    pg = N[1,1]*ns[1]+N[1,2]*ns[2]
    dx = dN[1,1]*ns[1]+dN[1,2]*ns[2]; J=norm(dx)
    nrm = Point2D(dx[2], -dx[1])/J
    U, T = fundamental_hyper(props, pg, pfs, nrm, nel)
    U=BEM._to_smat(U); T=BEM._to_smat(T)
    Fg=zeros(2,4); Fh=zeros(2,4)
    for j in 1:2
        Fg[:, 2j-1:2j] .= U .* (N[1,j]*J)
        Fh[:, 2j-1:2j] .= T .* (N[1,j]*J)
    end
    return Fg, Fh
end
IsS, IhSs = sst_GH(fstr, as; qsi=qsi, w=w, props=props, poly=polys, nodes=ns)
IsG, IhGs = guiggiani_GH(fstr, as; order_G=-1, order_H=-2, qsi=qsi, w=w,
    props=props, poly=polys, nodes=ns)
println("  ||Ih_SST-Ih_Gui||/||Gui||=$(norm(IhSs-IhGs)/(norm(IhGs)+1e-30))")
println("  ||Ig_SST-Ig_Gui||/||Gui||=$(norm(IsS-IsG)/(norm(IsG)+1e-30))")

# --- 2. full P3: SST vs Guiggiani ---
println("\n" * "="^64)
println(" P3 full system: force Guiggiani vs SST")

function p3_mesh()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sstinv"); lc=80.0
    c = gmsh.model.geo.addPoint(0.0,0.0,0.0,lc)
    p1 = gmsh.model.geo.addPoint(300.0,0.0,0.0,lc)
    p2 = gmsh.model.geo.addPoint(600.0,0.0,0.0,lc)
    p3 = gmsh.model.geo.addPoint(0.0,600.0,0.0,lc)
    p4 = gmsh.model.geo.addPoint(0.0,300.0,0.0,lc)
    b = gmsh.model.geo.addLine(p1,p2)
    o = gmsh.model.geo.addCircleArc(p2,c,p3)
    t = gmsh.model.geo.addLine(p3,p4)
    inn = gmsh.model.geo.addCircleArc(p4,c,p1)
    cl = gmsh.model.geo.addCurveLoop([b,o,t,inn])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(b,3); gmsh.model.mesh.setTransfiniteCurve(t,3)
    gmsh.model.mesh.setTransfiniteCurve(o,11); gmsh.model.mesh.setTransfiniteCurve(inn,11)
    gmsh.model.addPhysicalGroup(1,[b],-1,"1;1000;1;0")
    gmsh.model.addPhysicalGroup(1,[t],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inn,o],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out = datadir("elastico","sst_inv_p3.msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end
msh = p3_mesh()

dad = format2d(msh, props; tipo=2, pontointerno=false)
assemble!(dad; npg=16, threaded=false); solve(dad)
uc = copy(dad.u)
println("  CBIE max|u|=$(maximum(abs, uc))")

H_G_hyper(dad; npg=16, threaded=false); solve(dad)
us = copy(dad.u)
println("  HBIE SST        max|u|=$(maximum(abs, us))  rel vs CBIE=$(norm(us.-uc)/(norm(uc)+1e-30))  cond=$(cond(Matrix(dad.A)))")

dad = format2d(msh, props; tipo=2, pontointerno=false)
set_cache!(dad; singular=:guiggiani)
H_G_hyper(dad; npg=16, threaded=false); solve(dad)
ug = copy(dad.u)
println("  HBIE Guiggiani  max|u|=$(maximum(abs, ug))  rel vs CBIE=$(norm(ug.-uc)/(norm(uc)+1e-30))  cond=$(cond(Matrix(dad.A)))")
println("done")
