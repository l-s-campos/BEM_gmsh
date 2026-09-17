# 3D Kelvin collocation: uniform-strain patch on the unit cube.
#
#   u = ε · x  with εxx = 0.01, other ε = 0  (Dirichlet all faces)
#
using DrWatson
@quickactivate :BEM
using Printf
using LinearAlgebra
include(datadir("Laplace", "cube_mesh.jl"))

const L = 1.0
const NPG = 8
const NDIVS = (2, 3, 4)
const E = 1.0
const ν = 0.3
const εxx = 0.01

function _rate(h, e)
    length(e) < 2 && return fill(NaN, length(e))
    r = fill(NaN, length(e))
    for i in 2:length(e)
        (e[i-1] > 0 && e[i] > 0 && h[i-1] != h[i]) || continue
        r[i] = log(e[i-1] / e[i]) / log(h[i-1] / h[i])
    end
    return r
end

println("="^72)
println(" 3D elasticity BEM — unit cube strain patch (npg=$NPG, tipo=1 quads)")
println("  u = ($(εxx) x, 0, 0),  E=$E, ν=$ν")
println("="^72)

props = Elasticity(E, ν, 1.0; plane_strain=true)
ana = ana_elasticity_patch(; E=E, ν=ν, εxx=εxx, dim=3)

function _not_x_faces(dad)
    return [i for i in 1:dad.n if abs(dad.Normal[i][1]) < 0.5]
end

hs = Float64[]; eU = Float64[]; eT = Float64[]; ns = Int[]; nds = Int[]
for ndiv in NDIVS
    msh = mesh_unit_cube(; L=L, ndiv=ndiv, nome="cv3d_el_$ndiv",
        bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    # Dirichlet on x=0,L (prescribed u); Neumann traction on y and z faces
    apply_analytical_bc!(dad, ana, _not_x_faces(dad))
    assemble!(dad; npg=NPG, threaded=true)
    solve(dad)
    push!(nds, ndiv)
    push!(ns, dad.n)
    push!(hs, L / ndiv)
    push!(eU, rel_error(dad))
    push!(eT, rel_error_flux(dad))
    @printf("  ndiv=%d  n=%4d  ndof=%5d  rel u=%.3e  rel t=%.3e\n",
        ndiv, dad.n, 3 * dad.n, eU[end], eT[end])
end
rU = _rate(hs, eU)
rT = _rate(hs, eT)
println()
@printf("  %4s  %6s  %10s  %9s  %6s  %9s  %6s\n",
    "ndiv", "n", "h", "rel u", "rate u", "rel t", "rate t")
for i in eachindex(nds)
    rUs = isnan(rU[i]) ? "—" : @sprintf("%.2f", rU[i])
    rTs = isnan(rT[i]) ? "—" : @sprintf("%.2f", rT[i])
    @printf("  %4d  %6d  %10.4f  %9.2e  %6s  %9.2e  %6s\n",
        nds[i], ns[i], hs[i], eU[i], rUs, eT[i], rTs)
end
println("\nDone. Mixed patch: Dirichlet on x-faces, traction on y/z-faces.")
