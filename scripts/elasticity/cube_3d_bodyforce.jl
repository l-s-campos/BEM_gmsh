# 3D Kelvin DIBEM body force on the unit cube.
# Manufactured: u = (x², 0, 0),  b = −2(λ+2μ) e_x  (Navier).
using DrWatson
@quickactivate :BEM
using Printf
using LinearAlgebra
include(datadir("Laplace", "cube_mesh.jl"))

const L = 1.0
const NPG = 8
const NDIVS = (2, 3)
const E = 1.0
const ν = 0.3

function _rate(h, e)
    r = fill(NaN, length(e))
    for i in 2:length(e)
        (e[i-1] > 0 && e[i] > 0 && h[i-1] != h[i]) || continue
        r[i] = log(e[i-1] / e[i]) / log(h[i-1] / h[i])
    end
    return r
end

println("="^72)
println(" 3D elasticity DIBEM — cube, u=(x²,0,0), constant body force")
println("="^72)

props = Elasticity(E, ν, 1.0; plane_strain=true)
λ, μ = props.lambda, props.mu
bval = -2 * (λ + 2μ)
bf = p -> SVector(bval, 0.0, 0.0)

hs = Float64[]; es = Float64[]; ns = Int[]; nds = Int[]
for ndiv in NDIVS
    msh = mesh_unit_cube(; L=L, ndiv=ndiv, nome="cv3d_bf_$ndiv",
        bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(L, 2)))
    for i in 1:dad.n
        p = dad.Nodes[i]
        dad.BC[3*(i-1)+1:3*i] .= 0
        dad.BV[3*(i-1)+1] = p[1]^2
        dad.BV[3*(i-1)+2] = 0.0
        dad.BV[3*(i-1)+3] = 0.0
    end
    assemble!(dad; npg=NPG, threaded=true)
    DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=NPG)
    solve_thermoelastic!(dad; bodyforce=bf, θ=0.0)
    err = 0.0
    den = 0.0
    for i in 1:dad.ni
        p = dad.internalNodes[i]
        ui = dad.uint[3*(i-1)+1:3*i]
        ue = SA[p[1]^2, 0.0, 0.0]
        err += sum(abs2, ui .- ue)
        den += sum(abs2, ue)
    end
    rel = sqrt(err / max(den, eps()))
    push!(nds, ndiv)
    push!(ns, dad.n)
    push!(hs, L / ndiv)
    push!(es, rel)
    @printf("  ndiv=%d  n=%4d  ni=%2d  ndof=%5d  rel u_int=%.3e\n",
        ndiv, dad.n, dad.ni, 3 * dad.nt, rel)
end
rates = _rate(hs, es)
println()
@printf("  %4s  %6s  %8s  %10s  %6s\n", "ndiv", "n", "h", "rel u_int", "rate")
for i in eachindex(nds)
    rs = isnan(rates[i]) ? "—" : @sprintf("%.2f", rates[i])
    @printf("  %4d  %6d  %8.4f  %10.2e  %6s\n", nds[i], ns[i], hs[i], es[i], rs)
end
println("\nDone.")
