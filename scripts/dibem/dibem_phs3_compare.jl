# DIBEM (PHS3 + poly) vs cell mass. Known body force Mb, analytical ü residual.
# julia --project=. scripts/dibem/dibem_phs3_compare.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

rel(a, b) = norm(a - b) / (norm(b) + 1e-30)
const rbf3 = PHS(3; poly_deg=1)

println("="^72)
println(" DIBEM (PHS3)  vs  cells")
println("="^72)

# ---------- known b on the unit square (elasticity) ----------
msh = Base.invokelatest(quadrado_elasticity; ndiv=8, show=false, nome="phs3_b")
de = format2d(msh, Elasticity(1.0, 0.3, 1.0; plane_strain=true);
    tipo=1, pontointerno=true)
H_G_full_direct(de; npg=10, threaded=false)
cm = build_cell_mass(deepcopy(de); npg=12)
Md = DIBEM(deepcopy(de); method=:dense, rbf=rbf3)
pts = all_points(de)
forces = (
    ("(1,0)", p -> SVector(1.0, 0.0)),
    ("(x,0)", p -> SVector(p[1], 0.0)),
    ("sin e_x", p -> SVector(sin(π * p[1]) * sin(π * p[2]), 0.0)),
)
println("\nElasticity square  Mb vs M_cell b_cell")
println("  force       DIBEM          cells P")
for (name, bfun) in forces
    bn = zeros(2 * de.nt)
    bc = zeros(2 * length(cm.cells))
    @inbounds for i in 1:de.nt
        v = bfun(pts[i]); bn[2i-1] = v[1]; bn[2i] = v[2]
    end
    @inbounds for k in eachindex(cm.cells)
        v = bfun(cm.cells[k].centroid); bc[2k-1] = v[1]; bc[2k] = v[2]
    end
    dref = cm.M_cell * bc
    @printf("  %-10s  %.3e      %.3e\n", name,
        rel(Md * bn, dref), rel(cm.M * bn, dref))
end

# ---------- analytical ü residual on the bar ----------
function elast_fields!(u, tv, ddu, dad, t, meta)
    pts = all_points(dad)
    @inbounds for i in eachindex(pts)
        f = bar_sudden_fields(pts[i], t; N=400, c=meta.c, L=meta.L)
        u[2i-1] = f.u; u[2i] = 0
        ddu[2i-1] = f.ddu; ddu[2i] = 0
    end
    @inbounds for i in 1:dad.n
        f = bar_sudden_fields(dad.Nodes[i], t; N=400, c=meta.c, L=meta.L)
        tv[2i-1] = meta.E * f.dudx * dad.Normal[i][1]
        tv[2i] = 0
    end
end

dadE, metaE = elasticity_bar_sudden(; ndiv=8, n_int=4, ν=0.0)
H_G_full_direct(dadE; npg=8, threaded=false)
H, G = Matrix(dadE.H), Matrix(dadE.G)
Md = DIBEM(deepcopy(dadE); method=:dense, rbf=rbf3)
Mc = build_cell_mass(deepcopy(dadE); npg=8).M
ndof, nb = 2 * dadE.nt, 2 * dadE.n
println("\nElasticity bar  r = H u + M ü − G t")
println("  t     ||ü||       DIBEM ||r||/Σ   cells ||r||/Σ")
for t in (0.25, 0.5, 1.0)
    u = zeros(ndof); tv = zeros(nb); ddu = zeros(ndof)
    elast_fields!(u, tv, ddu, dadE, t, metaE)
    Hu = H * u; Gt = G * tv
    function nrm(M)
        r = Hu .+ (M * ddu) .- Gt
        return norm(r) / (norm(Hu) + norm(M * ddu) + norm(Gt) + 1e-30)
    end
    @printf("  %.2f  %.3e   %.3e         %.3e\n",
        t, norm(ddu), nrm(Md), nrm(Mc))
end
println("\nDone.")
