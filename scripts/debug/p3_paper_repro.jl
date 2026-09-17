# Reproduce Cordeiro & Leonel 2020 P3 (Figs 13–15) and Tables 2–5.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

const MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334,
              η12_1=1.255, η12_2=-0.031)
const Ri, Ro, P = 300.0, 600.0, 1000.0   # mm, MPa

function rotate_C(C, θdeg)
    θ = deg2rad(θdeg)
    m, n = cos(θ), sin(θ)
    Rot = @SMatrix [
        m^2   n^2    2m*n
        n^2   m^2   -2m*n
       -m*n   m*n    m^2-n^2
    ]
    return inv(Rot) * C * inv(Rot')
end

function props_mat1(; θ=0.0, plane_strain=false)
    p = lekhnitskii_engineering(MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
        η12_1=MAT1.η12_1, η12_2=MAT1.η12_2, plane_strain=plane_strain,
        ν31=0.40, ν32=0.25, η12_3=0.50)
    C = θ == 0 ? p.C : rotate_C(p.C, θ)
    return AnisotropicElasticity(lekhnitskii_params(C))
end

function p3_mesh(; load=:ty)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cordeiro_p3_repro")
    lc = 80.0
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(Ri, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Ro, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(0.0, Ro, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ri, 0.0, lc)
    bottom = gmsh.model.geo.addLine(p1, p2)
    outer = gmsh.model.geo.addCircleArc(p2, c, p3)
    top = gmsh.model.geo.addLine(p3, p4)
    inner = gmsh.model.geo.addCircleArc(p4, c, p1)
    cl = gmsh.model.geo.addCurveLoop([bottom, outer, top, inner])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bottom, 3)
    gmsh.model.mesh.setTransfiniteCurve(top, 3)
    gmsh.model.mesh.setTransfiniteCurve(outer, 11)
    gmsh.model.mesh.setTransfiniteCurve(inner, 11)
    bc = load === :ty ? "1;0;1;$(-P)" : "1;$P;1;0"
    gmsh.model.addPhysicalGroup(1, [bottom], -1, bc)
    gmsh.model.addPhysicalGroup(1, [top], -1, "0;0;0;0")
    gmsh.model.addPhysicalGroup(1, [inner, outer], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", "cordeiro_p3_repro.msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function path_w(xy)
    x, y = xy[1], xy[2]
    r = hypot(x, y)
    # paper Fig. 13b: w=0 at outer-top, along outer → bottom → inner → vertical
    if abs(r - Ro) < 8 && x > 8 && y > 8
        θ = atan(x, y)                 # 0 at top, π/2 at bottom-right
        return Ro * θ / 10             # mm → cm
    elseif y < 8 && x > Ri - 8
        return Ro * (π / 2) / 10 + (Ro - x) / 10
    elseif abs(r - Ri) < 8 && x > 8 && y > 8
        θ = atan(x, y)
        return (Ro * π / 2 + (Ro - Ri) + Ri * (π / 2 - θ)) / 10
    else
        # vertical fixed edge x≈0
        return (Ro * π / 2 + (Ro - Ri) + Ri * π / 2 + (Ro - y)) / 10
    end
end

function run_bie!(dad; bie, npg)
    if bie === :hbie
        H_G_hyper(dad; npg=npg, threaded=false)
    else
        assemble!(dad; npg=npg, threaded=false)
    end
    solve(dad)
    return dad
end

function summarize(dad, label)
    u = dad.u
    ux = u[1:2:end]
    uy = u[2:2:end]
    @printf("%-28s n=%3d  max|u|=%10.3f mm = %8.3f cm  maxUx=%8.3f cm  maxUy=%8.3f cm  minUx=%8.3f  minUy=%8.3f\n",
        label, dad.n, maximum(abs, u), maximum(abs, u) / 10,
        maximum(ux) / 10, maximum(uy) / 10, minimum(ux) / 10, minimum(uy) / 10)
end

# ---------------------------------------------------------------------------
println("="^72)
println(" μ for MAT1 (plane stress)")
for θ in (0, 90)
    pr = props_mat1(; θ=θ)
    println("  θ=$(θ)°  μ=$(pr.params.mi)  C11=$(pr.params.C[1,1])  C22=$(pr.params.C[2,2])")
end

# ---------------------------------------------------------------------------
println("\n" * "="^72)
println(" Tables 2–5: on-element DG/DH sums (highlighted outer-bottom element)")
msh = p3_mesh(; load=:ty)
props = props_mat1()
dad = format2d(msh, props; tipo=2, pontointerno=false)
# first outer-arc element: after 2 bottom elements
el = dad.elements[3]
xj = dad.Nodes[el.index]
println("  elem 3 nodes: $(xj)")
println("  paper: DGsing≈-0.00348  DHsing≈0.20936  DGhyp≈-0.06112  DHhyp≈-381.81087")

function elem_sums(dad, el, xj, i; hyper)
    dim = 2
    nf = dad.Normal[i]
    pf = dad.Nodes[i]
    hloc = zeros(dim, dim * length(el.index))
    gloc = zeros(dim, dim * length(el.index))
    f = if hyper
        (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
    else
        fundamental
    end
    orders = hyper ? (-1, -2) : nothing
    BEM._init_quadrature!(dad, 50)
    BEM.integrate_element(dad, el, xj, pf, hloc, gloc, f; orders=orders, source=i)
    return sum(gloc), sum(hloc)
end

for (lab, hyper) in (("CBIE", false), ("HBIE", true))
    println("  --- $lab ---")
    for i in el.index
        sg, sh = elem_sums(dad, el, xj, i; hyper=hyper)
        @printf("    src=%2d  at (%.1f, %.1f)  sum(DG)=%12.5f  sum(DH)=%12.5f\n",
            i, dad.Nodes[i][1], dad.Nodes[i][2], sg, sh)
    end
end

# also try θ=90
props90 = props_mat1(; θ=90)
dad90 = format2d(msh, props90; tipo=2, pontointerno=false)
el90 = dad90.elements[3]
xj90 = dad90.Nodes[el90.index]
println("  --- HBIE θ=90 ---")
for i in el90.index
    sg, sh = elem_sums(dad90, el90, xj90, i; hyper=true)
    @printf("    src=%2d  at (%.1f, %.1f)  sum(DG)=%12.5f  sum(DH)=%12.5f\n",
        i, dad90.Nodes[i][1], dad90.Nodes[i][2], sg, sh)
end

# ---------------------------------------------------------------------------
println("\n" * "="^72)
println(" P3 CBIE / HBIE  (paper Fig.14 peak ≈ −80 cm)")
println(" load ty=-P as Fig.13 arrows; also tx=P; θ=0 and 90")

for (load, θ, npg_h) in (
        (:ty, 0, 50), (:ty, 90, 50), (:tx, 0, 50), (:tx, 90, 50),
        (:ty, 90, 16), (:ty, 90, 8),
    )
    mshc = p3_mesh(; load=load)
    pr = props_mat1(; θ=θ)
    dadc = format2d(mshc, pr; tipo=2, pontointerno=false)
    run_bie!(dadc; bie=:cbie, npg=16)
    uc = copy(dadc.u)
    summarize(dadc, "CBIE load=$(load) θ=$(θ)")
    run_bie!(dadc; bie=:hbie, npg=npg_h)
    rel = norm(dadc.u .- uc) / (norm(uc) + 1e-30)
    summarize(dadc, "HBIE npg=$(npg_h) load=$(load) θ=$(θ)")
    @printf("    rel(HBIE vs CBIE)=%.3e  cond=%.3e\n", rel, cond(Matrix(dadc.A)))
end

println("\ndone")
