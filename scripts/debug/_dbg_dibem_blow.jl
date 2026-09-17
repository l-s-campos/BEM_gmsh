# Diagnose internals source + why DIBEM blows on mixed Kovarik.
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Statistics
using Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(joinpath(@__DIR__, "..", "sbm_drm", "sbm_drm_vs_dibem.jl"))

function report_points(tag, dad)
    cells = has_cache(dad, :cells) ? dad.cache[:cells] : nothing
    nc = cells === nothing ? 0 : length(cells)
    cents = nc == 0 ? Point2D[] : [c.centroid for c in cells]
    ints = Point2D[Point2D(p) for p in dad.internalNodes]
    same = nc > 0 && length(ints) == nc &&
           maximum(norm(ints[i] - cents[i]) for i in 1:nc) < 1e-12
    xmin, xmax = extrema(p[1] for p in ints)
    ymin, ymax = extrema(p[2] for p in ints)
    @printf("%-18s n=%3d ni=%3d cells=%3d  using_format2d_centroids=%s  bbox=[%.3f,%.3f]x[%.3f,%.3f]\n",
            tag, dad.n, dad.ni, nc, same, xmin, xmax, ymin, ymax)
    return dad
end

function mass_stats(tag, dad, rbf)
    H_G_full_direct(dad; npg=12, threaded=false)
    DIBEM(dad; method=:dense, rbf=rbf)
    M = Matrix(dad.M)
    ev = eigvals(M)
    evr = real.(ev)
    @printf("  %-22s  condM=%.2e  λmin=%.2e  λmax=%.2e  nneg=%d  |M|_F=%.2e  max|c|=%.2e\n",
            tag, cond(M), minimum(evr), maximum(evr), count(<(0), evr),
            norm(M), maximum(abs, dad.cache[:dibem_c]))
    return M
end

function march_max(dad, u0; κ, Δt, tf, rbf)
    sol = bem_dibem_heat(dad, u0; κ=κ, Δt=Δt, tf=tf, scheme=:houbolt, rbf=rbf)
    mx = [maximum(abs, sol.U[:, k]) for k in 1:size(sol.U, 2)]
    return sol, mx
end

println("=== where internals come from ===")
msh = quadrado(ndiv=10, show=false, nome="dbg_sin", ordem=1)
dad_s = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
report_points("sine format2d", dad_s)

msh2 = rect_diffusion_mesh(; Lx=3, Ly=3, nb_side=12, left_neumann=true, nome="dbg_ex2")
dad_c = format2d(msh2, Laplace(1.0); tipo=1, pontointerno=true)
report_points("ex2 format2d", dad_c)
dad_g = deepcopy(dad_c)
set_internal_nodes!(dad_g, interior_grid(3.0, 3.0, 5))
report_points("ex2 interior_grid", dad_g)

println()
println("=== DIBEM mass on mixed ex2 (interior_grid 5×5) ===")
r_nopoly = PHS(3; poly_deg=-1)
r_poly = PHS(3; poly_deg=2)   # package default
dad1 = deepcopy(dad_g)
dad2 = deepcopy(dad_g)
mass_stats("PHS3 poly=-1", dad1, r_nopoly)
mass_stats("PHS3 poly=2", dad2, r_poly)

println()
println("=== DIBEM mass on mixed ex2 (format2d centroids) ===")
dad3 = deepcopy(dad_c)
dad4 = deepcopy(dad_c)
mass_stats("centroids poly=-1", dad3, r_nopoly)
mass_stats("centroids poly=2", dad4, r_poly)

println()
println("=== first-step growth, mixed ex2 interior_grid ===")
N, ni = dad_g.n, dad_g.ni
u0 = fill(30.0, N + ni); u0[1:N] .= 0
for (lab, rbf) in (("poly=-1", r_nopoly), ("poly=2", r_poly))
    sol, mx = march_max(deepcopy(dad_g), u0; κ=1.25, Δt=1.2/40, tf=1.2, rbf=rbf)
    @printf("  %s  max|u| steps 1,2,4,10,end = %.3e  %.3e  %.3e  %.3e  %.3e\n",
            lab, mx[1], mx[min(2,end)], mx[min(4,end)], mx[min(10,end)], mx[end])
end

println()
println("=== mixed ex2 RMSE at tf (format2d centroids vs interior_grid) ===")
function _ex2_rmse(dad; rbf)
    N, ni = dad.n, dad.ni
    u0 = fill(30.0, N + ni); u0[1:N] .= 0
    pts = vcat([Point2D(p) for p in dad.Nodes], [Point2D(p) for p in dad.internalNodes])
    uex = [exact_ex2(p[1], p[2], 1.2; κ=1.25, Lx=3.0, Ly=3.0) for p in pts]
    sol = bem_dibem_heat(deepcopy(dad), u0; κ=1.25, Δt=1.2/40, tf=1.2,
                         scheme=:houbolt, rbf=rbf)
    ii = (N + 1):(N + ni)
    return sqrt(mean(abs2, sol.U[ii, end] .- uex[ii])), maximum(abs, sol.U[:, end])
end
r_i, m_i = _ex2_rmse(dad_g; rbf=r_nopoly)
r_c, m_c = _ex2_rmse(dad_c; rbf=r_nopoly)
@printf("  interior_grid  int RMSE=%.3e  max|u|=%.3e\n", r_i, m_i)
@printf("  centroids      int RMSE=%.3e  max|u|=%.3e\n", r_c, m_c)

println()
println("=== first-step growth, mixed ex2 format2d centroids ===")
Nc, nic = dad_c.n, dad_c.ni
u0c = fill(30.0, Nc + nic); u0c[1:Nc] .= 0
for (lab, rbf) in (("poly=-1", r_nopoly), ("poly=2", r_poly))
    sol, mx = march_max(deepcopy(dad_c), u0c; κ=1.25, Δt=1.2/40, tf=1.2, rbf=rbf)
    @printf("  %s  max|u| steps 1,2,4,10,end = %.3e  %.3e  %.3e  %.3e  %.3e\n",
            lab, mx[1], mx[min(2,end)], mx[min(4,end)], mx[min(10,end)], mx[end])
end
