# P3 HBIE: isotropic vs aniso, linear vs quad, kernel FD, cond.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

const MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)

function p3_mesh()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3p"); lc=80.0
    c = gmsh.model.geo.addPoint(0.0,0.0,0.0,lc)
    p1 = gmsh.model.geo.addPoint(300.0,0.0,0.0,lc)
    p2 = gmsh.model.geo.addPoint(600.0,0.0,0.0,lc)
    p3 = gmsh.model.geo.addPoint(0.0,600.0,0.0,lc)
    p4 = gmsh.model.geo.addPoint(0.0,300.0,0.0,lc)
    bottom = gmsh.model.geo.addLine(p1,p2)
    outer = gmsh.model.geo.addCircleArc(p2,c,p3)
    top = gmsh.model.geo.addLine(p3,p4)
    inner = gmsh.model.geo.addCircleArc(p4,c,p1)
    cl = gmsh.model.geo.addCurveLoop([bottom,outer,top,inner])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bottom,3); gmsh.model.mesh.setTransfiniteCurve(top,3)
    gmsh.model.mesh.setTransfiniteCurve(outer,11); gmsh.model.mesh.setTransfiniteCurve(inner,11)
    gmsh.model.addPhysicalGroup(1,[bottom],-1,"1;0;1;-1000")  # ty = -P (paper arrows down)
    gmsh.model.addPhysicalGroup(1,[top],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inner,outer],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out = datadir("elastico","p3_probe.msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

function run(lab, props; tipo=2, bie=:cbie, npg=16)
    dad = format2d(msh, props; tipo=tipo, pontointerno=false)
    if bie === :hbie
        H_G_hyper(dad; npg=npg, threaded=false)
    else
        assemble!(dad; npg=npg, threaded=false)
    end
    solve(dad)
    A = dad.A
    println("  $lab  tipo=$tipo n=$(dad.n)  max|u|=$(maximum(abs, dad.u))  cond(A)=$(cond(Matrix(A)))")
    return dad
end

# FD: nξ · C : ∇ξ {U,T} vs fundamental_hyper
function fd_hyper()
    p = lekhnitskii_engineering(MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2)
    props = AnisotropicElasticity(p)
    y = Point2D(1.0, 0.4)
    ξ = Point2D(0.1, 0.2)
    ny = Point2D(0.0, 1.0)
    nξ = Point2D(1.0, 0.0)
    h = 1e-6
    Uh, Th = let kp = fundamental_hyper(props, y, ξ, ny, nξ)
        BEM._to_smat(kp.U), BEM._to_smat(kp.T)
    end
    function strain_cols(Uξ, Uξx, Uξy)
        # U is 2x2 (disp i, force k); derivative wrt ξ
        # σ = C : ε(U_k) for each force column k, then nξ · σ
        C = props.params.C
        outU = zeros(2,2)
        for k in 1:2
            ε = SVector(Uξx[1,k], Uξy[2,k], Uξy[1,k]+Uξx[2,k])
            σ = C * ε  # Voigt σ11,σ22,σ12
            outU[1,k] = nξ[1]*σ[1] + nξ[2]*σ[3]
            outU[2,k] = nξ[1]*σ[3] + nξ[2]*σ[2]
        end
        return outU
    end
    U0, T0 = let kp = fundamental(props, y, ξ, ny); BEM._to_smat(kp.U), BEM._to_smat(kp.T) end
    Ux, Tx = let kp = fundamental(props, y, ξ + Point2D(h,0), ny); BEM._to_smat(kp.U), BEM._to_smat(kp.T) end
    Uy, Ty = let kp = fundamental(props, y, ξ + Point2D(0,h), ny); BEM._to_smat(kp.U), BEM._to_smat(kp.T) end
    # ∇ξ U ≈ -(U(ξ+h)-U(ξ))/h if U depends on (y-ξ); our FD is +∂/∂ξ_coord of second arg
    Uξx = (Ux - U0) / h
    Uξy = (Uy - U0) / h
    Tξx = (Tx - T0) / h
    Tξy = (Ty - T0) / h
    UhFD = strain_cols(U0, Uξx, Uξy)
    ThFD = strain_cols(T0, Tξx, Tξy)
    println("  FD Uh rel=$(norm(Uh.-UhFD)/(norm(Uh)+1e-30))")
    println("  FD Th rel=$(norm(Th.-ThFD)/(norm(Th)+1e-30))")
    println("  Uh=\n$Uh\n  UhFD=\n$UhFD")
    println("  Th=\n$Th\n  ThFD=\n$ThFD")
end

println("=== kernel FD ===")
fd_hyper()

msh = p3_mesh()
aniso = AnisotropicElasticity(lekhnitskii_engineering(MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
    η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
# isotropic-ish: E=E1, ν=0.334
E = MAT1.E1; ν = MAT1.ν12
iso = Elasticity(E, ν, 1.0)  # plane stress map via effective_nu

println("\n=== P3 solves ===")
run("aniso cbie", aniso; tipo=2, bie=:cbie, npg=16)
run("aniso hbie", aniso; tipo=2, bie=:hbie, npg=50)
run("aniso hbie lin", aniso; tipo=1, bie=:hbie, npg=50)
run("aniso cbie lin", aniso; tipo=1, bie=:cbie, npg=16)
run("iso cbie", iso; tipo=2, bie=:cbie, npg=16)
run("iso hbie", iso; tipo=2, bie=:hbie, npg=50)
println("done")
