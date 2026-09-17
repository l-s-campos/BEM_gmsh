using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random, StaticArrays, BEM.HMatrices
import BEM.HMatrices: h2_foreach, isdense_h2, isuniform, issplit, h2node

function pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function kernel(P)
    return KernelMatrix(P, P) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
end

function countk(N)
    nd = nu = nsf = ns = 0
    h2_foreach(N) do node
        if isdense_h2(node)
            nd += 1
        elseif isuniform(node)
            nu += 1
            node.s_full && (nsf += 1)
        else
            ns += 1
        end
    end
    return (; nd, nu, nsf, ns)
end

function run(n1d; nmax=32, rtol=1e-6)
    Random.seed!(1)
    P = pts(n1d)
    n = length(P)
    K = kernel(P)
    b = randn(n)
    tree2 = ClusterTree(P, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    t_asm = @elapsed A = assemble_h2(K, tree2; rtol=rtol, threads=true)
    root = h2node(A)
    c0 = countk(root)
    t_lu = @elapsed F = lu(A; method=:nested, rtol=rtol)
    c1 = countk(F.factors)
    t_sol = @elapsed u = F \ copy(b)
    rel = norm(A * u - b) / (norm(b) + 1e-14)
    treeH = ClusterTree(P, hmatrix_splitter(; nmax=nmax); cube=true)
    t_asmH = @elapsed H = assemble_hmatrix(K, treeH, treeH;
        adm=StrongAdmissibilityStd(2.0), comp=PartialACA(; rtol=rtol), threads=true)
    t_luH = @elapsed FH = lu(H; rtol=rtol, threads=true)
    t_solH = @elapsed uH = FH \ copy(b)
    relH = norm(H * uH - b) / (norm(b) + 1e-14)
    @printf("N=%5d  H2 asm=%.3f lu=%.3f sol=%.3f resid=%.2e  dens/unif/sfull/split %d/%d/%d/%d -> %d/%d/%d/%d\n",
        n, t_asm, t_lu, t_sol, rel, c0.nd, c0.nu, c0.nsf, c0.ns, c1.nd, c1.nu, c1.nsf, c1.ns)
    @printf("         H  asm=%.3f lu=%.3f sol=%.3f resid=%.2e\n", t_asmH, t_luH, t_solH, relH)
    flush(stdout)
    return (; n, t_lu, t_luH, rel)
end

println("threads=", Threads.nthreads())
run(16)
run(32)
run(48)
run(64)
run(80)
run(100)
