# Time H-matrix vs H² at N=1e4: assembly + GMRES + LU (factor + solve).
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Dates
using BEM.HMatrices

const N1D = parse(Int, get(ENV, "SOLVE_N1D", "100"))  # 100^2 = 10000
const NMAX = parse(Int, get(ENV, "SOLVE_NMAX", "32"))
const RTOL = parse(Float64, get(ENV, "SOLVE_RTOL", "1e-6"))
const GMRES_RTOL = parse(Float64, get(ENV, "SOLVE_GMRES_RTOL", "1e-8"))
const GMRES_ITMAX = parse(Int, get(ENV, "SOLVE_GMRES_ITMAX", "400"))
const THREADS = get(ENV, "SOLVE_THREADS", "1") in ("1", "true", "yes")

function _pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

# Symmetric shifted log ≈ Kd + Kd' + 8I (SPD enough for unpivoted LU).
function _kernel(pts)
    return KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
end

function _stamp(msg)
    @printf("[%s] %s\n", Dates.format(Dates.now(), "HH:MM:SS"), msg)
    flush(stdout)
    return
end

function _resid(A, u, b)
    return norm(A * u - b) / (norm(b) + 1e-14)
end

function warmup!()
    _stamp("warmup N=64")
    pts = _pts(8)
    K = _kernel(pts)
    tree = ClusterTree(pts, hmatrix_splitter(; nmax=16); cube=true)
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    b = randn(length(pts))
    gmres_h(H, b; rtol=1e-8, itmax=50, history=false)
    F = lu(H; rtol=1e-6, threads=false)
    F \ copy(b)
    tree2 = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
    A2 = assemble_h2(K, tree2; rtol=1e-6, threads=false)
    gmres_h(A2, b; rtol=1e-8, itmax=50, history=false)
    F2 = lu(A2; method=:nested, rtol=1e-6)
    F2 \ copy(b)
    return
end

function main()
    Random.seed!(1)
    n1d = N1D
    pts = _pts(n1d)
    n = length(pts)
    K = _kernel(pts)
    b = randn(n)
    x = randn(n)
    @printf("Solve timing  N=%d  (grid %d×%d)  nmax=%d  rtol=%g\n", n, n1d, n1d, NMAX, RTOL)
    println("  ", Dates.now())
    println("  threads=", Threads.nthreads(), "  ACA/NNCA threads=", THREADS)
    println("  GMRES rtol=", GMRES_RTOL, "  itmax=", GMRES_ITMAX)
    println()
    warmup!()

    # ---- H-matrix ----------------------------------------------------------
    _stamp("H-matrix: assemble")
    t_tree = @elapsed treeH = ClusterTree(pts, hmatrix_splitter(; nmax=NMAX); cube=true)
    t_asmH = @elapsed H = assemble_hmatrix(K, treeH, treeH;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=RTOL),
        threads=THREADS)
    y = similar(x)
    t_mvH = @elapsed mul!(y, H, x)
    @printf("  tree=%.3fs  asm=%.3fs  matvec=%.3fs  mem=%.1f MB  ratio=%.2f\n",
        t_tree, t_asmH, t_mvH, Base.summarysize(H) / 2^20, compression_ratio(H))
    flush(stdout)

    _stamp("H-matrix: GMRES")
    t_gmH = @elapsed xgH, stH = gmres_h(H, b; rtol=GMRES_RTOL, itmax=GMRES_ITMAX, history=true)
    r_gmH = _resid(H, xgH, b)
    @printf("  gmres=%.3fs  iters=%s  status=%s  resid=%.3e  total(asm+gmres)=%.3fs\n",
        t_gmH, stH.niter, stH.status, r_gmH, t_asmH + t_gmH)
    flush(stdout)

    _stamp("H-matrix: LU factor")
    t_luH = @elapsed FH = lu(H; rtol=RTOL, threads=THREADS)
    _stamp("H-matrix: LU solve")
    t_solH = @elapsed uH = FH \ copy(b)
    r_luH = _resid(H, uH, b)
    @printf("  lu=%.3fs  solve=%.3fs  resid=%.3e  total(asm+lu+solve)=%.3fs\n",
        t_luH, t_solH, r_luH, t_asmH + t_luH + t_solH)
    flush(stdout)

    # ---- H² / NNCA ---------------------------------------------------------
    _stamp("H2: assemble")
    t_tree2 = @elapsed tree2 = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
    t_asm2 = @elapsed A2 = assemble_h2(K, tree2; rtol=RTOL, threads=THREADS)
    t_mv2 = @elapsed mul!(y, A2, x)
    @printf("  tree=%.3fs  asm=%.3fs  matvec=%.3fs  mem=%.1f MB  maxrank=%d\n",
        t_tree2, t_asm2, t_mv2, Base.summarysize(A2) / 2^20, maxrank(A2))
    flush(stdout)

    _stamp("H2: GMRES")
    t_gm2 = @elapsed xg2, st2 = gmres_h(A2, b; rtol=GMRES_RTOL, itmax=GMRES_ITMAX, history=true)
    r_gm2 = _resid(A2, xg2, b)
    @printf("  gmres=%.3fs  iters=%s  status=%s  resid=%.3e  total(asm+gmres)=%.3fs\n",
        t_gm2, st2.niter, st2.status, r_gm2, t_asm2 + t_gm2)
    flush(stdout)

    _stamp("H2: nested LU factor")
    t_lu2 = @elapsed F2 = lu(A2; method=:nested, rtol=RTOL)
    _stamp("H2: nested LU solve")
    t_sol2 = @elapsed u2 = F2 \ copy(b)
    r_lu2 = _resid(A2, u2, b)
    @printf("  lu=%.3fs  solve=%.3fs  resid=%.3e  total(asm+lu+solve)=%.3fs\n",
        t_lu2, t_sol2, r_lu2, t_asm2 + t_lu2 + t_sol2)
    flush(stdout)

    println()
    println("="^78)
    @printf("%-10s %10s %10s %10s %10s %10s %10s\n",
        "", "asm", "matvec", "GMRES", "LU", "LUsolve", "GMRES tot")
    @printf("%-10s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
        "H-matrix", t_asmH, t_mvH, t_gmH, t_luH, t_solH, t_asmH + t_gmH)
    @printf("%-10s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
        "H2", t_asm2, t_mv2, t_gm2, t_lu2, t_sol2, t_asm2 + t_gm2)
    println()
    @printf("%-10s  GMRES resid=%9.2e  iters=%s   LU resid=%9.2e   LU total=%.3fs\n",
        "H-matrix", r_gmH, stH.niter, r_luH, t_asmH + t_luH + t_solH)
    @printf("%-10s  GMRES resid=%9.2e  iters=%s   LU resid=%9.2e   LU total=%.3fs\n",
        "H2", r_gm2, st2.niter, r_lu2, t_asm2 + t_lu2 + t_sol2)
    return
end

main()
