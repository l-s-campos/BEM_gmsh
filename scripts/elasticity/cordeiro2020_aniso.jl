# Cordeiro & Leonel, Eng. Anal. Bound. Elem. 119 (2020) 214–224
# Analytic SST coefficients (eqs. 34–37) for 2-D Lekhnitskii CBIE / HBIE.
#
# On-element HBIE is Guiggiani with Cordeiro SST leading tensors as Laurent
# coefficients. H_G_hyper enforces cij=0 (Table 7). Default npg=50.
#
#   julia --project=. scripts/elasticity/cordeiro2020_aniso.jl
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^64)
println(" Cordeiro & Leonel 2020 — anisotropic Guiggiani / SST")
println("="^64)

# Paper Table 1/8. ν listed as ν21 is used as ν12 = −ε2/ε1 under σ1
# (matches Fig. 6 Uy scale; ν21 = ν12 E2/E1 would be ~0.027).
# Fernández 2012 / Cordeiro 2015 §7.2: Ex, Ey, Gxy, νyx, ηxy,x, ηxy,y.
# νyx=0.344 is the large Poisson (D12=-νyx/Ex). Literal −νyx/Ey is not PD.
# EABE 2020 Table 1 printed 0.334 for the same material.
const MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.344,
              η12_1=1.255, η12_2=-0.031)   # MPa, mm
const MAT2 = (E1=9.81e3, E2=0.41e3, G12=0.74e3, ν12=-0.01,
              η12_1=0.0, η12_2=0.0)

function _rect_mesh(; Lx, Ly, nlong, nshort, nome)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / 2
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
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
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function _props(mat; plane_strain=false, kwargs...)
    p = lekhnitskii_engineering(mat.E1, mat.E2, mat.G12, mat.ν12;
        η12_1=mat.η12_1, η12_2=mat.η12_2, plane_strain=plane_strain, kwargs...)
    return AnisotropicElasticity(p)
end

function _neu_error(dad, vert)
    uana = analytical(dad)
    unum = dad.u
    mask = trues(length(unum))
    for i in vert
        mask[2i - 1] = false
        mask[2i] = false
    end
    return rel_error(dad),
        norm(unum[mask] .- uana[mask]) / (norm(uana[mask]) + 1e-30)
end

function _assemble_solve!(dad; bie=:cbie, npg=16)
    if bie === :hbie
        H_G_hyper(dad; npg=npg, threaded=false)
    else
        assemble!(dad; npg=npg, threaded=false)
    end
    return solve(dad)
end

# ---------------------------------------------------------------------------
# Problem 1 — uniform tension (Fig. 4), 0.5 m × 0.2 m, p = 100 MPa
# ---------------------------------------------------------------------------
function problem1()
    println("\n--- Problem 1: uniform tension, material 1 (plane stress) ---")
    Lx, Ly, p = 500.0, 200.0, 100.0          # mm, MPa
    props = _props(MAT1)
    D = inv(props.params.C)
    ana = AnalyticalSolution("cordeiro-p1",
        (x; t=0.0) -> SVector(D[1, 1] * p * x[1] + D[1, 3] * p * x[2],
                              D[1, 2] * p * x[2]);
        q = (x, n; t=0.0) -> SVector(p * n[1], 0.0),
        description="u1=D11 p x1 + D16 p x2, u2=D12 p x2")
    msh = _rect_mesh(; Lx=Lx, Ly=Ly, nlong=3, nshort=2, nome="cordeiro_p1")
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    # Dirichlet on vertical sides (paper); traction-free top/bottom
    vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1e-6 || dad.Nodes[i][1] > Lx - 1e-6]
    neu = setdiff(1:dad.n, vert)
    apply_analytical_bc!(dad, ana, neu)
    uana_max = D[1, 1] * p * Lx + D[1, 3] * p * Ly
    out = NamedTuple[]
    for bie in (:cbie, :hbie)
        npg = bie === :hbie ? 50 : 16
        _assemble_solve!(dad; bie=bie, npg=npg)
        erru, e_neu = _neu_error(dad, vert)
        umax = maximum(abs, dad.u)
        println("  $(bie): nodes=$(dad.n)  rel(all)=$(erru)  rel(Neu)=$(e_neu)  max|u|=$(umax) mm")
        println("         ana corner u1 = $(uana_max) mm  (paper Fig.6 Ux peak ≈ 0.60 mm)")
        push!(out, (; bie, erru, e_neu, umax))
    end
    return out
end

# ---------------------------------------------------------------------------
# Problem 2 — linear traction / bending (Fig. 9), 4 m × 0.5 m, p = 10 MPa
# ---------------------------------------------------------------------------
function problem2()
    println("\n--- Problem 2: linear traction, material 2 (plane strain) ---")
    L, h, p = 2000.0, 500.0, 10.0            # half-length L, height h; mm, MPa
    Lx, Ly = 2L, h
    props = _props(MAT2; plane_strain=true)
    a11 = inv(props.params.C)[1, 1]
    a12 = inv(props.params.C)[1, 2]
    # origin at the geometric centre (Lekhnitskii beam)
    ufun = function (x; t=0.0)
        x1 = x[1] - L
        x2 = x[2] - h / 2
        u1 = (2p / h) * a11 * x1 * x2
        u2 = (p / h) * (a12 * x2^2 - a11 * x1^2 + a11 * L^2)
        return SVector(u1, u2)
    end
    qfun = function (x, n; t=0.0)
        x2 = x[2] - h / 2
        σx = (2p / h) * x2
        return SVector(σx * n[1], 0.0)
    end
    ana = AnalyticalSolution("cordeiro-p2", ufun; q=qfun,
        description="Lekhnitskii beam σx=(2p/h) x2")
    msh = _rect_mesh(; Lx=Lx, Ly=Ly, nlong=3, nshort=2, nome="cordeiro_p2")
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1e-6 || dad.Nodes[i][1] > Lx - 1e-6]
    neu = setdiff(1:dad.n, vert)
    apply_analytical_bc!(dad, ana, neu)
    out = NamedTuple[]
    for bie in (:cbie, :hbie)
        npg = bie === :hbie ? 50 : 16
        _assemble_solve!(dad; bie=bie, npg=npg)
        erru, e_neu = _neu_error(dad, vert)
        umax = maximum(abs, dad.u)
        println("  $(bie): nodes=$(dad.n)  rel(all)=$(erru)  rel(Neu)=$(e_neu)  max|u|=$(umax) mm")
        push!(out, (; bie, erru, e_neu, umax))
    end
    return out
end

# ---------------------------------------------------------------------------
# Problem 3 — 90° arc (Fig. 13), qualitative CBIE solve
# ---------------------------------------------------------------------------
function problem3()
    println("\n--- Problem 3: quarter annulus, material 1, P = 1 GPa ---")
    Ri, Ro, P = 600.0, 900.0, 1000.0         # mm, MPa  (0.6 m / 0.9 m)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cordeiro_p3")
    lc = 80.0
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(Ri, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Ro, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(0.0, Ro, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ri, 0.0, lc)
    bottom = gmsh.model.geo.addLine(p1, p2)                 # loaded
    outer = gmsh.model.geo.addCircleArc(p2, c, p3)
    top = gmsh.model.geo.addLine(p3, p4)                    # fixed
    inner = gmsh.model.geo.addCircleArc(p4, c, p1)
    cl = gmsh.model.geo.addCurveLoop([bottom, outer, top, inner])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bottom, 3)
    gmsh.model.mesh.setTransfiniteCurve(top, 3)
    gmsh.model.mesh.setTransfiniteCurve(outer, 11)
    gmsh.model.mesh.setTransfiniteCurve(inner, 11)
    # Fig. 13a: downward traction on the horizontal edge (ty = −P), u=0 on x=0.
    gmsh.model.addPhysicalGroup(1, [bottom], -1, "1;0;1;-$P")
    gmsh.model.addPhysicalGroup(1, [top], -1, "0;0;0;0")
    gmsh.model.addPhysicalGroup(1, [inner, outer], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", "cordeiro_p3.msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    props = _props(MAT1)
    dad = format2d(out, props; tipo=2, pontointerno=false)
    outp = NamedTuple[]
    uc = nothing
    for bie in (:cbie, :hbie)
        npg = bie === :hbie ? 16 : 16
        _assemble_solve!(dad; bie=bie, npg=npg)
        umax = maximum(abs, dad.u)
        extra = ""
        if bie === :cbie
            uc = copy(dad.u)
        else
            rel = norm(dad.u .- uc) / (norm(uc) + 1e-30)
            extra = "  rel vs CBIE=$(rel)  cond=$(cond(Matrix(dad.A)))"
        end
        ux = extrema(getindex.(Ref(dad.u), 1:2:length(dad.u)))
        uy = extrema(getindex.(Ref(dad.u), 2:2:length(dad.u)))
        println("  $(bie): nodes=$(dad.n)  Ux=$(ux[1]/10) cm  Uy=$(uy[1]/10) cm  max|u|=$(umax/10) cm$(extra)")
        push!(outp, (; bie, umax))
    end
    return outp
end

r1 = problem1()
r2 = problem2()
r3 = problem3()
println("\nSummary (paper: both CBIE and HBIE overlay the analytics)")
for (lab, r) in (("P1", r1), ("P2", r2))
    for t in r
        println("  $lab $(t.bie)  rel(Neu)=$(t.e_neu)")
    end
end
for t in r3
    println("  P3 $(t.bie)  max|u|=$(t.umax) mm")
end
