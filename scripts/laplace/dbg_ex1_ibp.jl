# Example 1: S1 Hessian vs S3 IBP vs aniso FS, same internal_grid.
using LinearAlgebra
using StaticArrays
using Printf
include(joinpath(@__DIR__, "orthotropic_dibem_ex1_grid_ex3_rot.jl"))

k1, k2 = 2.0, 0.5
K = @SMatrix [k1 0.0; 0.0 k2]
ufun = (x, y) -> example1_u(x, y; k1=k1, k2=k2)
gfun = (x, y) -> example1_grad(x, y; k1=k1, k2=k2)
path = quadrado(; nome="dbg_e1_n80", ndiv=21, show=false, ordem=1)

function run_var(label; fs=:iso, nx=8, ny=8, strat=:hess, rbf=RBF, nlocal=21)
    dad = make_dad(path, fs, K)
    internal_grid!(dad, nx, ny; d_min=0.01)
    apply_mixed_ex1!(dad; k1=k1, k2=k2)
    assemble!(dad; npg=NPG, threaded=true)
    if fs === :iso
        DIBEM(dad; rbf=rbf, threaded=true)
        if strat === :hess
            solve_anisotropic_dibem!(dad, K; rbf=rbf, npg=NPG, nlocal=nlocal)
        elseif strat === :ibp
            solve_anisotropic_ibp!(dad, K; rbf=rbf, npg=NPG, nlocal=nlocal)
        elseif strat === :hessg
            Hess = rbf_diff_ops(all_points(dad), rbf; nlocal=nothing)[2]
            # fall back to library with large nlocal ≈ global
            solve_anisotropic_dibem!(dad, K; rbf=rbf, npg=NPG, nlocal=dad.nt - 1)
        end
    else
        solve(dad)
    end
    e = mrpe_ex1(dad, ufun, gfun, K)
    # flux error vs distance to corner on the left edge
    idx = edge_idx(dad, :left; skip_corners=false)
    qa = [pkg_flux(dad.Normal[i], K, gfun(dad.Nodes[i][1], dad.Nodes[i][2])) for i in idx]
    qe = abs.(dad.q[idx] .- qa)
    y = [dad.Nodes[i][2] for i in idx]
    @printf("%-28s ni=%4d  T_int=%7.3f  T_top=%7.3f  T_rgt=%7.3f  q_L=%7.3f  q_B=%7.3f  qLmax=%.3f @y=%.3f\n",
        label, e["ni"], e["eT_int_pct"], e["eT_top_pct"], e["eT_right_pct"],
        e["eq_left_pct"], e["eq_bot_pct"], maximum(qe), y[argmax(qe)])
    flush(stdout)
    return e
end

r1 = PHS(3; poly_deg=1)
r2 = PHS(3; poly_deg=2)
println("nel=80  MRPE %")
run_var("aniso FS 4x4"; fs=:aniso, nx=4, ny=4)
run_var("aniso FS 24x24"; fs=:aniso, nx=24, ny=24)
for (nx, ny) in ((4, 4), (8, 8), (24, 24))
    println("-- $(nx)x$(ny) --")
    run_var("S1 p1 nloc21  $(nx)x$(ny)"; nx=nx, ny=ny, strat=:hess, rbf=r1, nlocal=21)
    run_var("S1 p1 nloc41  $(nx)x$(ny)"; nx=nx, ny=ny, strat=:hess, rbf=r1, nlocal=41)
    run_var("S1 p2 nloc41  $(nx)x$(ny)"; nx=nx, ny=ny, strat=:hess, rbf=r2, nlocal=41)
    run_var("S3 IBP p1 n21 $(nx)x$(ny)"; nx=nx, ny=ny, strat=:ibp, rbf=r1, nlocal=21)
    run_var("S3 IBP p2 n41 $(nx)x$(ny)"; nx=nx, ny=ny, strat=:ibp, rbf=r2, nlocal=41)
    run_var("S3 IBP p2 glob $(nx)x$(ny)"; nx=nx, ny=ny, strat=:ibp, rbf=r2, nlocal=max(4*nx*ny, 80))
end
