using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using StaticArrays
using BEM.HMatrices

function kernel(pts)
    return KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
end

function run(n1d; nmax=32, rtol=1e-6)
    Random.seed!(1)
    xs = range(0.0, 1.0; length=n1d)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    K = kernel(pts)
    b = randn(n)
    tree2 = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    treeH = ClusterTree(pts, hmatrix_splitter(; nmax=nmax); cube=true)

    t_h2 = @elapsed H2 = assemble_h2(K, tree2; rtol=rtol, threads=true)
    t_H = @elapsed H = assemble_hmatrix(K, treeH, treeH;
        adm=StrongAdmissibilityStd(2.0), comp=PartialACA(; rtol=rtol), threads=true)

    t_srs = @elapsed Fs = srs_factor(K, tree2; rtol=rtol, rank=40)
    t_hlu = @elapsed FH = lu(H; rtol=rtol, threads=true)

    uH = FH \ copy(b)
    rH = norm(H * uH - b) / (norm(b) + 1e-14)
    rH2 = norm(H2 * uH - b) / (norm(b) + 1e-14)

    t0 = @elapsed x0, st0 = gmres_h(H2, b; rtol=1e-8, itmax=400, history=true)
    r0 = norm(H2 * x0 - b) / (norm(b) + 1e-14)

    tS = @elapsed xS, stS = gmres_h(H2, b; Pl=Fs, rtol=1e-8, itmax=80, history=true)
    rS = norm(H2 * xS - b) / (norm(b) + 1e-14)

    tL = @elapsed xL, stL = gmres_h(H2, b; Pl=FH, rtol=1e-8, itmax=80, history=true)
    rL = norm(H2 * xL - b) / (norm(b) + 1e-14)

    tHH = @elapsed xHH, stHH = gmres_h(H, b; Pl=FH, rtol=1e-8, itmax=20, history=true)
    rHH = norm(H * xHH - b) / (norm(b) + 1e-14)

    println("="^72)
    @printf("N=%d  nmax=%d  rtol=%g\n", n, nmax, rtol)
    @printf("  assemble  H2=%.3fs  H=%.3fs   factor  SRS=%.3fs  H-LU=%.3fs\n",
        t_h2, t_H, t_srs, t_hlu)
    @printf("  H-LU as direct:  ‖Hu-b‖/‖b‖=%.2e  ‖H2 u-b‖/‖b‖=%.2e\n", rH, rH2)
    @printf("  GMRES on H2     %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(st0.niter), t0, r0, t_h2 + t0, st0.status)
    @printf("  GMRES H2+SRS    %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(stS.niter), tS, rS, t_h2 + t_srs + tS, stS.status)
    @printf("  GMRES H2+H-LU   %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(stL.niter), tL, rL, t_h2 + t_H + t_hlu + tL, stL.status)
    @printf("  GMRES H +H-LU   %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(stHH.niter), tHH, rHH, t_H + t_hlu + tHH, stHH.status)
    flush(stdout)
    return
end

println("threads=", Threads.nthreads())
run(16)
run(32)
run(48)
run(100)
