# GMRES follow-up: log kernel with more iters, plus SPD Gaussian that actually solves.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random, StaticArrays, BEM.HMatrices

function pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function run(label, K, pts, nmax, itmax; do_lu=true)
    n = length(pts)
    b = randn(n)
    println("=== $label  N=$n  itmax=$itmax ===")
    treeH = ClusterTree(pts, hmatrix_splitter(; nmax=nmax); cube=true)
    tH = @elapsed H = assemble_hmatrix(K, treeH, treeH;
        adm=StrongAdmissibilityStd(2.0), comp=PartialACA(; rtol=1e-6), threads=true)
    tgmH = @elapsed xH, stH = gmres_h(H, b; rtol=1e-8, itmax=itmax, history=true)
    rH = norm(H * xH - b) / (norm(b) + 1e-14)
    @printf("  H-matrix asm=%.3fs  gmres=%.3fs  iters=%s  status=%s  resid=%.3e  tot=%.3fs\n",
        tH, tgmH, stH.niter, stH.status, rH, tH + tgmH)
    if !isempty(stH.residuals)
        rs = stH.residuals
        @printf("    residual[1,10,end]=%.3e  %.3e  %.3e\n", rs[1], rs[min(10, length(rs))], rs[end])
    end
    flush(stdout)

    tree2 = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    t2 = @elapsed A2 = assemble_h2(K, tree2; rtol=1e-6, threads=true)
    tgm2 = @elapsed x2, st2 = gmres_h(A2, b; rtol=1e-8, itmax=itmax, history=true)
    r2 = norm(A2 * x2 - b) / (norm(b) + 1e-14)
    @printf("  H2       asm=%.3fs  gmres=%.3fs  iters=%s  status=%s  resid=%.3e  tot=%.3fs\n",
        t2, tgm2, st2.niter, st2.status, r2, t2 + tgm2)
    if !isempty(st2.residuals)
        rs = st2.residuals
        @printf("    residual[1,10,end]=%.3e  %.3e  %.3e\n", rs[1], rs[min(10, length(rs))], rs[end])
    end
    flush(stdout)

    if do_lu
        tluH = @elapsed FH = lu(H; rtol=1e-6, threads=true)
        tsolH = @elapsed uH = FH \ copy(b)
        @printf("  H-matrix LU=%.3fs  solve=%.3fs  resid=%.3e  tot=%.3fs\n",
            tluH, tsolH, norm(H * uH - b) / (norm(b) + 1e-14), tH + tluH + tsolH)
        flush(stdout)
        tlu2 = @elapsed F2 = lu(A2; method=:nested, rtol=1e-6)
        tsol2 = @elapsed u2 = F2 \ copy(b)
        @printf("  H2       LU=%.3fs  solve=%.3fs  resid=%.3e  tot=%.3fs\n",
            tlu2, tsol2, norm(A2 * u2 - b) / (norm(b) + 1e-14), t2 + tlu2 + tsol2)
        flush(stdout)
    end
    return
end

Random.seed!(1)
P = pts(100)
Klog = KernelMatrix(P, P) do a, b
    r = hypot(a[1] - b[1], a[2] - b[2])
    return r < 1e-30 ? 16.0 : 2.0 * log(r)
end
Kgauss = KernelMatrix(P, P) do a, b
    d2 = sum(abs2, a - b)
    return exp(-8 * d2) + (d2 < 1e-30 ? 2.0 : 0.0)
end

run("shifted-log", Klog, P, 32, 2000; do_lu=false)
run("gaussian+ridge", Kgauss, P, 32, 400; do_lu=true)
