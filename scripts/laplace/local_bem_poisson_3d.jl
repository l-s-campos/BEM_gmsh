# 3D local BEM Poisson on the unit cube: ∇²u = 6, u = |x|².
using DrWatson
@quickactivate :BEM
using Printf
using LinearAlgebra
include(datadir("Laplace", "cube_mesh.jl"))

const L = 1.0
const NPG = 8
const FSRC = 6.0

println("="^72)
println(" 3D local BEM — unit cube, u = |x|², ∇²u = 6")
println("="^72)

ana = ana_poisson_r2(; k=1.0, dim=3)
for ndiv in (2, 3)
    msh = mesh_unit_cube(; L=L, ndiv=ndiv, nome="lbem3d_$ndiv")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(L, 2)))
    apply_analytical_bc!(dad, ana)
    solve_local_bem!(dad, FSRC; npg=NPG, source=:local)
    err = 0.0
    den = 0.0
    @inbounds for k in 1:dad.ni
        p = dad.internalNodes[k]
        ui = dad.T[dad.n + k]
        ue = sum(abs2, p)
        err += (ui - ue)^2
        den += ue^2
    end
    eint = sqrt(err / max(den, eps()))
    @printf("  ndiv=%d  n=%4d  ni=%2d  rel T_int=%.3e  rel q=%.3e\n",
        ndiv, dad.n, dad.ni, eint, rel_error_flux(dad))
end
println("\nDone.")
