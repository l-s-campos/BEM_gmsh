# 3D Poisson on the unit cube via DIBEM: ∇²u = 6, u = |x|².
using DrWatson
@quickactivate :BEM
using Printf
using LinearAlgebra
include(datadir("Laplace", "cube_mesh.jl"))

const L = 1.0
const NPG = 8
const NDIVS = (2, 3, 4)
const FSRC = 6.0  # ∇²(|x|²) = 6 in 3D

function _rate(h, e)
    r = fill(NaN, length(e))
    for i in 2:length(e)
        (e[i-1] > 0 && e[i] > 0 && h[i-1] != h[i]) || continue
        r[i] = log(e[i-1] / e[i]) / log(h[i-1] / h[i])
    end
    return r
end

println("="^72)
println(" 3D Poisson DIBEM — unit cube, u = |x|², ∇²u = 6")
println("="^72)

ana = ana_poisson_r2(; k=1.0, dim=3)
hs = Float64[]; eT = Float64[]; eQ = Float64[]; ns = Int[]; nds = Int[]
for ndiv in NDIVS
    msh = mesh_unit_cube(; L=L, ndiv=ndiv, nome="cv3d_po_$ndiv")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(L, 2)))
    apply_analytical_bc!(dad, ana)
    assemble!(dad; npg=NPG, threaded=true)
    solve_poisson_dibem!(dad, FSRC; rbf=PHS(3; poly_deg=1), npg=NPG)
    push!(nds, ndiv)
    push!(ns, dad.n)
    push!(hs, L / ndiv)
    push!(eT, rel_error(dad))
    push!(eQ, rel_error_flux(dad))
    @printf("  ndiv=%d  n=%4d  ni=%2d  rel T=%.3e  rel q=%.3e\n",
        ndiv, dad.n, dad.ni, eT[end], eQ[end])
end
rT = _rate(hs, eT)
rQ = _rate(hs, eQ)
println()
@printf("  %4s  %6s  %8s  %9s  %6s  %9s  %6s\n",
    "ndiv", "n", "h", "rel T", "rateT", "rel q", "rateQ")
for i in eachindex(nds)
    rTs = isnan(rT[i]) ? "—" : @sprintf("%.2f", rT[i])
    rQs = isnan(rQ[i]) ? "—" : @sprintf("%.2f", rQ[i])
    @printf("  %4d  %6d  %8.4f  %9.2e  %6s  %9.2e  %6s\n",
        nds[i], ns[i], hs[i], eT[i], rTs, eQ[i], rQs)
end
println("\nDone.")
