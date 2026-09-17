# HBIE: rigid-body row-sum (paper Table 7, cij=0) vs residual on Cordeiro problems.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

const MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
const MAT2 = (E1=9.81e3, E2=0.41e3, G12=0.74e3, ν12=-0.01, η12_1=0.0, η12_2=0.0)

function _rect_mesh(; Lx, Ly, nlong, nshort, nome)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / 2
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2); l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4); l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(l1, nlong)
    gmsh.model.mesh.setTransfiniteCurve(l3, nlong)
    gmsh.model.mesh.setTransfiniteCurve(l2, nshort)
    gmsh.model.mesh.setTransfiniteCurve(l4, nshort)
    gmsh.model.mesh.setTransfiniteSurface(s1)
    gmsh.model.mesh.setRecombine(2, s1)
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", nome * ".msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

function _props(mat; plane_strain=false, kwargs...)
    AnisotropicElasticity(lekhnitskii_engineering(mat.E1, mat.E2, mat.G12, mat.ν12;
        η12_1=mat.η12_1, η12_2=mat.η12_2, plane_strain=plane_strain, kwargs...))
end

function rigid_rowsum!(H, n; dim=2)
    @views for i in 1:n
        ii = BEM.expand(i, dim)
        H[ii, ii] .= 0.0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end
    return H
end

function rowsum_norms(H, n)
    e1 = zeros(2n); e2 = zeros(2n)
    for i in 1:n; e1[2i-1] = 1; e2[2i] = 1; end
    return norm(H * e1), norm(H * e2)
end

function residual(dad, H, G)
    u = zeros(2 * dad.n); t = zeros(2 * dad.n)
    ana = dad.analytical
    for i in 1:dad.n
        ui = ana(dad.Nodes[i])
        ti = ana.q(dad.Nodes[i], dad.Normal[i])
        u[2i-1:2i] .= ui; t[2i-1:2i] .= ti
    end
    r = H * u - G * t
    return norm(r) / (norm(H * u) + 1e-30), norm(r), maximum(abs, r)
end

function neu_err(dad, vert)
    uana = analytical(dad); unum = dad.u
    mask = trues(length(unum))
    for i in vert; mask[2i-1] = false; mask[2i] = false; end
    return rel_error(dad), norm(unum[mask] .- uana[mask]) / (norm(uana[mask]) + 1e-30)
end

function run_p1()
    println("\n===== P1 tension =====")
    Lx, Ly, p = 500.0, 200.0, 100.0
    props = _props(MAT1); D = inv(props.params.C)
    ana = AnalyticalSolution("p1",
        (x; t=0.0) -> SVector(D[1,1]*p*x[1] + D[1,3]*p*x[2], D[1,2]*p*x[2]);
        q = (x, n; t=0.0) -> SVector(p*n[1], 0.0))
    dad = format2d(_rect_mesh(; Lx, Ly, nlong=3, nshort=2, nome="rs_p1"), props; tipo=2, pontointerno=false)
    vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1e-6 || dad.Nodes[i][1] > Lx-1e-6]
    neu = setdiff(1:dad.n, vert)
    apply_analytical_bc!(dad, ana, neu)
    for (lab, npg, rb) in (("hbie-24", 24, false), ("hbie-24-rb", 24, true),
                           ("hbie-50", 50, false), ("hbie-50-rb", 50, true))
        H, G = H_G_hyper(dad; npg=npg, threaded=false)
        rb && rigid_rowsum!(H, dad.n)
        set_cache!(dad; H, G)
        relr, nr, mr = residual(dad, H, G)
        println("  $lab  row|H e|=$(rowsum_norms(H, dad.n))  rel(Hu-Gt)=$(relr)  maxres=$(mr)")
        solve(dad)
        _, e = neu_err(dad, vert)
        println("         rel(Neu)=$(e)  max|u|=$(maximum(abs, dad.u))")
    end
end

function run_p2()
    println("\n===== P2 beam =====")
    L, h, p = 2000.0, 500.0, 10.0
    Lx, Ly = 2L, h
    props = _props(MAT2; plane_strain=true)
    a11 = inv(props.params.C)[1,1]; a12 = inv(props.params.C)[1,2]
    ufun = function (x; t=0.0)
        x1 = x[1]-L; x2 = x[2]-h/2
        SVector((2p/h)*a11*x1*x2, (p/h)*(a12*x2^2 - a11*x1^2 + a11*L^2))
    end
    qfun = function (x, n; t=0.0)
        x2 = x[2]-h/2; σx = (2p/h)*x2; SVector(σx*n[1], 0.0)
    end
    ana = AnalyticalSolution("p2", ufun; q=qfun)
    dad = format2d(_rect_mesh(; Lx, Ly, nlong=3, nshort=2, nome="rs_p2"), props; tipo=2, pontointerno=false)
    vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1e-6 || dad.Nodes[i][1] > Lx-1e-6]
    apply_analytical_bc!(dad, ana, setdiff(1:dad.n, vert))
    for (lab, npg, rb) in (("hbie-24", 24, false), ("hbie-24-rb", 24, true),
                           ("hbie-50", 50, false), ("hbie-50-rb", 50, true))
        H, G = H_G_hyper(dad; npg=npg, threaded=false)
        rb && rigid_rowsum!(H, dad.n)
        set_cache!(dad; H, G)
        relr, nr, mr = residual(dad, H, G)
        println("  $lab  row|H e|=$(rowsum_norms(H, dad.n))  rel(Hu-Gt)=$(relr)  maxres=$(mr)")
        solve(dad)
        _, e = neu_err(dad, vert)
        println("         rel(Neu)=$(e)  max|u|=$(maximum(abs, dad.u))")
    end
    assemble!(dad; npg=16, threaded=false); solve(dad)
    _, e = neu_err(dad, vert)
    println("  cbie-16  rel(Neu)=$(e)")
end

function run_p3()
    println("\n===== P3 quarter =====")
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("rs_p3"); lc=80.0
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
    gmsh.model.addPhysicalGroup(1,[bottom],-1,"1;1000;1;0")
    gmsh.model.addPhysicalGroup(1,[top],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inner,outer],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out = datadir("elastico","rs_p3.msh"); mkpath(dirname(out)); gmsh.write(out); gmsh.finalize()
    dad = format2d(out, _props(MAT1); tipo=2, pontointerno=false)
    for (lab, npg, rb) in (("hbie-24", 24, false), ("hbie-24-rb", 24, true),
                           ("hbie-50", 50, false), ("hbie-50-rb", 50, true))
        H, G = H_G_hyper(dad; npg=npg, threaded=false)
        rb && rigid_rowsum!(H, dad.n)
        set_cache!(dad; H, G)
        println("  $lab  row|H e|=$(rowsum_norms(H, dad.n))")
        solve(dad)
        println("         max|u|=$(maximum(abs, dad.u)) mm")
    end
    assemble!(dad; npg=16, threaded=false); solve(dad)
    println("  cbie-16  max|u|=$(maximum(abs, dad.u)) mm")
end

run_p1(); run_p2(); run_p3()
println("done")
