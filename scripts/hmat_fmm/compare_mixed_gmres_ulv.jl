# Mixed 2×2: GMRES with no preconditioner vs Schur GMRES with ulv(A), ulv(D).
#
#   julia --project=. scripts/hmat_fmm/compare_mixed_gmres_ulv.jl
#   MIXED_NDIV=2500  (boundary n ≈ 4*(ndiv-1) ≈ 1e4)
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Krylov
using LinearMaps
using BEM.HMatrices

const NDIV = parse(Int, get(ENV, "MIXED_NDIV", "1250"))
const RTOL = parse(Float64, get(ENV, "MIXED_RTOL", "1e-8"))
const NMAX = parse(Int, get(ENV, "MIXED_NMAX", "64"))
const ITMAX = parse(Int, get(ENV, "MIXED_ITMAX", "400"))

function main()
    Random.seed!(1)
    println("mesh ndiv=", NDIV)
    tmesh = @elapsed msh = quadrado(ndiv=NDIV, show=false, nome="cmp_mixed_1e4")
    dad = format2d(msh, Laplace(1.0); pontointerno=false)
    attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
    tasm = @elapsed assemble!(dad, 8)
    applyBC(dad; blocks=true)
    Aop = dad.A
    b = dad.b
    idx = dad.bc_idx
    nu_, nq_ = nu(idx), nq(idx)
    n = nu_ + nq_
    @printf("n=%d  nu=%d  nq=%d  mesh %.1fs  assemble %.1fs\n",
        n, nu_, nq_, tmesh, tasm)
    println("HSS rtol=", RTOL, " nmax=", NMAX, " GMRES itmax=", ITMAX)
    flush(stdout)

    println("\n== GMRES on 2x2, no preconditioner ==")
    t_g = @elapsed xg, stg = Krylov.gmres(Aop, b; atol=1e-8, rtol=1e-8,
        itmax=ITMAX, history=true)
    rg = norm(Aop * xg - b) / (norm(b) + 1e-14)
    @printf("  time %.2fs  iters %s  status=%s  resid=%.2e\n",
        t_g, stg.niter, stg.status, rg)
    flush(stdout)

    println("\n== ULV on A and D, GMRES on Schur (B,C HSS matvec) ==")
    bl = Aop.blocks
    t_den = @elapsed begin
        A11 = Matrix{Float64}(bl.Huu)
        A12 = .-Matrix{Float64}(bl.Guq)
        A21 = Matrix{Float64}(bl.Hqu)
        A22 = .-Matrix{Float64}(bl.Gqq)
    end
    pts_u = [dad.collocation[i] for i in idx.u]
    pts_q = [dad.collocation[i] for i in idx.q]
    @printf("  densify tiles %.2fs\n", t_den)

    kw = (; rtol=RTOL, method=:id, Tmax=Inf, symm=:n)
    spl = PrincipalComponentSplitter(; nmax=NMAX)
    t_hss = @elapsed begin
        tu = ClusterTree(pts_u, spl)
        tq = ClusterTree(pts_q, spl)
        HA = assemble_hss(A11, tu; kw...)
        FA = ulv(HA)
        HD = assemble_hss(A22, tq; kw...)
        FD = ulv(HD)
        HB = assemble_hss(A12, tu, tq; kw...)
        HC = assemble_hss(A21, tq, tu; kw...)
    end
    @printf("  HSS+ULV A,D and HSS B,C  %.2fs\n", t_hss)
    println("    A ", HA, "  D ", HD)
    println("    B ", HB, "  C ", HC)
    flush(stdout)

    T = Float64
    Smap = LinearMap{T}((y, v) -> (y .= HD * v .- HC * (FA \ (HB * v)); y),
        nq_, nq_; ismutating=true)
    b1 = b[1:nu_]
    b2 = b[(nu_ + 1):n]
    t_rhs = @elapsed begin
        w = FA \ copy(b1)
        rhs = b2 .- HC * w
    end
    t_s = @elapsed x2, sts = Krylov.gmres(Smap, rhs; atol=1e-8, rtol=1e-8,
        itmax=ITMAX, history=true, M=FD, ldiv=true)
    t_x1 = @elapsed x1 = FA \ (b1 .- HB * x2)
    xu = similar(b)
    copyto!(view(xu, 1:nu_), x1)
    copyto!(view(xu, (nu_ + 1):n), x2)
    ru = norm(Aop * xu - b) / (norm(b) + 1e-14)
    @printf("  Schur GMRES time %.2fs  (rhs %.2fs + x1 %.2fs)  iters %s  status=%s\n",
        t_s, t_rhs, t_x1, sts.niter, sts.status)
    @printf("  setup+solve %.2fs  resid vs 2x2=%.2e\n",
        t_den + t_hss + t_rhs + t_s + t_x1, ru)

    println("\n== summary ==")
    @printf("  GMRES no Pl     %8.2fs  iters=%s  resid=%.2e\n", t_g, stg.niter, rg)
    @printf("  ULV Schur GMRES %8.2fs  setup=%.2fs solve=%.2fs  iters=%s  resid=%.2e\n",
        t_den + t_hss + t_rhs + t_s + t_x1, t_den + t_hss, t_rhs + t_s + t_x1,
        sts.niter, ru)
    return
end

main()
