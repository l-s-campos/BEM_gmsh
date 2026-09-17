# Why Burton–Miller degrades the interior Laplace bar.
# Operator norms, HBIE residual, M+αM′ spectrum. Same 12×6 mesh as the plot.
#
#   julia --project=. scripts/bm_diagnose.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

const Lx, Ly = 12.0, 6.0
const NPG = 12
const RBF = PHS(1; poly_deg=-1)
const PROBE = Point2D(Lx, Ly / 2)

function mesh_bar_12x6(; ndivx=13, ndivy=7, ordem=1, nome="bm_diag12")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 1.0
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(l1, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l3, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l2, ndivy)
    gmsh.model.mesh.setTransfiniteCurve(l4, ndivy)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [l1, l3], -1, "1;0")
    gmsh.model.addPhysicalGroup(1, [l2], -1, "1;-1")
    gmsh.model.addPhysicalGroup(1, [l4], -1, "0;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

relres(r, parts...) = norm(r) / (sum(norm, parts) + 1e-30)
nneg(A) = count(<( -1e-8), real.(eigvals(Matrix(A))))

function fill_bar!(u, q, ddu, dad, t)
    pts = all_points(dad)
    @inbounds for i in eachindex(pts)
        f = bar_sudden_fields(pts[i], t; N=400, c=1.0, L=Lx)
        u[i] = f.u
        ddu[i] = f.ddu
    end
    @inbounds for i in 1:dad.n
        f = bar_sudden_fields(dad.Nodes[i], t; N=400, c=1.0, L=Lx)
        q[i] = -f.dudx * dad.Normal[i][1]
    end
    return u, q, ddu
end

println("=== Laplace 12×6  diagnose Burton–Miller ===")
dad = format2d(mesh_bar_12x6(), Laplace(1.0); tipo=1, pontointerno=true)
n, nt = dad.n, dad.nt
hmin = minimum(el.Length for el in dad.elements)
@printf("n=%d nt=%d  hmin=%.3f  L=%.0f  |i/κ₁|=2L/π=%.2f\n",
    n, nt, hmin, Lx, 2Lx / π)

H_G_full_direct(dad; npg=NPG, threaded=true)
Hp, Gp = H_G_hyper(dad; npg=NPG, threaded=true)
H, G = Matrix(dad.H), Matrix(dad.G)
Hb, Gb = H[1:n, :], G[1:n, :]

@printf("\n--- kernel scaling (Frobenius) ---\n")
@printf("  ||H||_F     = %.3e    ||H_bdry||_F = %.3e\n", norm(H), norm(Hb))
@printf("  ||H′||_F    = %.3e    ||H′||/||H_b|| = %.3f\n", norm(Hp), norm(Hp) / norm(Hb))
@printf("  ||G||_F     = %.3e    ||G_bdry||_F = %.3e\n", norm(G), norm(Gb))
@printf("  ||G′||_F    = %.3e    ||G′||/||G_b|| = %.3f\n", norm(Gp), norm(Gp) / norm(Gb))
@printf("  mean|diag H′| = %.3e   mean|diag G′| = %.3e\n",
    mean(abs, diag(Hp[:, 1:n])), mean(abs, diag(Gp)))

tana = 6.0
u = zeros(nt); q = zeros(n); ddu = zeros(nt)
fill_bar!(u, q, ddu, dad, tana)

println("\n--- analytical residual at t=$tana  (r = Hu−Gq−Mü, r′ = H′u−G′q−M′ü) ---")
@printf("  %-8s  %10s  %10s  %10s  %8s  %8s  %8s\n",
    "mass", "||r||/Σ", "||r′||/Σ", "||r+αr′||", "nneg M", "nneg Mα", "||M′||/||M||")
for mass in (:drm, :dibem, :cells)
    d = deepcopy(dad)
    set_cache!(d; H=H, G=G)
    M, Mp = if mass === :drm
        drm = build_drm_matrices(d, RBF; npg=NPG)
        drm.M, drm_hyper_mass(drm, Hp, Gp)
    elseif mass === :dibem
        DIBEM(d; method=:dense, rbf=RBF)
        Matrix(d.M), dibem_hyper_mass(d; rbf=RBF)
    else
        cm = build_cell_mass(d; npg=NPG)
        cm.M, cell_hyper_mass(d; npg=NPG, P=cm.P, cells=cm.cells)
    end
    r = H * u - G * q - M * ddu
    rp = Hp * u - Gp * q - view(Mp, 1:n, :) * ddu
    α = 1.0
    rα = r[1:n] .+ α .* rp
    Mc = combine_burton_miller(H, G, M, Hp, Gp, Mp, α; n=n)[3]
    @printf("  %-8s  %10.3e  %10.3e  %10.3e  %8d  %8d  %8.2f\n",
        mass,
        relres(r, H * u, G * q, M * ddu),
        relres(rp, Hp * u, Gp * q, view(Mp, 1:n, :) * ddu),
        relres(rα, r[1:n], α .* rp),
        nneg(M), nneg(Mc),
        norm(Mp) / (norm(M) + 1e-30))
end

println("\n--- α sweep, cells only (stable method): Houbolt Δt=0.1, tf=48 (1 period) ---")
cm = let d = deepcopy(dad)
    set_cache!(d; H=H, G=G)
    build_cell_mass(d; npg=NPG)
end
Mp = cell_hyper_mass(dad; npg=NPG, P=cm.P, cells=cm.cells)
M = cm.M
ana = ana_bar_sudden(; N=400, c=1.0, L=Lx)
Δt, tf1 = 0.1, 48.0
@printf("  %8s  %10s  %10s  %s\n", "α", "u_end_err", "max|u|", "blow")
for α in (0.0, 0.1, 0.5, 1.0, hmin, 2Lx / π)
    d = deepcopy(dad)
    Hc, Gc, Mc = combine_burton_miller(H, G, M, Hp, Gp, Mp, α; n=n)
    set_cache!(d; H=Hc, G=Gc, M=Mc)
    solve_Houbolt(d, Δt, tf1)
    pts = all_points(d)
    ip = argmin(norm(p - PROBE) for p in pts)
    ux = d.T[ip, :]
    ua = [float(ana.u(PROBE; t=ti)) for ti in d.time]
    i0 = findfirst(i -> !isfinite(ux[i]) || abs(ux[i]) > 80, eachindex(ux))
    ok = all(isfinite, ux)
    err = ok && i0 === nothing ? norm(ux .- ua) / (norm(ua) + 1e-30) : NaN
    mx = maximum(abs, filter(isfinite, ux); init=0.0)
    blow = i0 === nothing ? "—" : @sprintf("t=%.1f", d.time[i0])
    @printf("  %8.3f  %10.3e  %10.3e  %s\n", α, err, mx, blow)
end
println("\nClassic exterior BM uses α = i/κ (complex, |α|∼2L/π for mode 1).")
println("Here α is real. Interior bar has no fictitious frequencies to cancel.")
