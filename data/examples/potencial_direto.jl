# Port of BEM.jl atual potencial_direto benchmarks (Gmsh + format2d)
using DrWatson
@quickactivate :BEM
using LinearAlgebra

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))

println("="^60)
println(" Example: potencial_direto suite (selected problems)")
println("="^60)

function check(name, msh, ana; keep=false, tol=5e-2, ndiv_note="")
    dad = format2d(msh, Laplace(1.0); tipo=2, pontointerno=false)
    keep ? apply_bc_keep_type!(dad, ana) : attach_analytical!(dad, ana)
    H_G_full_direct(dad; npg=12, threaded=false)
    solve(dad)
    err = rel_error(dad)
    println("  $name$ndiv_note  n=$(dad.n)  rel_error = $err")
    @assert err < tol "rel_error $err ≥ $tol for $name"
    return err
end

# 1) T = x on unit square
check("potencial1d", potencial1d_mesh(; ndiv=12, nome="ex_p1d"), ana_potencial1d(); tol=1e-2)

# 2) Laquini-3 (smooth Dirichlet) — should be accurate
check("laquini3", laquini3_mesh(; ndiv=16, nome="ex_laq3"), ana_laquini3(); tol=5e-2)

# 3) Quarter annulus radial
check("quarto_circ", quarto_circ_mesh(; ndiv=12, nome="ex_qcirc"), ana_quarto_circ(); tol=5e-2)

# 4) Moulton (singular) — coarser tolerance
check("placa_moulton", placa_moulton_mesh(; ndiv=16, nome="ex_moulton"), ana_moulton();
      keep=true, tol=0.15)

println("OK.")
