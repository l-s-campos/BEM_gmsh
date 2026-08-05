# Profile H² matvec allocations (workspace + Rk tmp)
using BenchmarkTools
using LinearAlgebra
using BEM

function _pts(nside)
    xs = range(0.0, 1.0; length=nside)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

pts = _pts(16)  # n=256
K = KernelMatrix(pts, pts) do x, y
    d2 = sum(abs2, x - y)
    exp(-6 * d2) + (d2 < 1e-30 ? 2.0 : 0.0)
end
tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=24))
H2 = assemble_h2(K, tree; rtol=1e-5, far_method=:aca,
    comp=PartialACA(; rtol=1e-5), alpha=0.5)
n = length(pts)
x = randn(n)
y = zeros(n)
X = randn(n, 8)
Y = zeros(n, 8)

# warmup (builds workspace)
mul!(y, H2, x)
mul!(Y, H2, X)

println("H2 ", H2)
println("single RHS:")
display(@benchmark mul!($y, $H2, $x) samples=50)
println("\nmulti RHS s=8:")
display(@benchmark mul!($Y, $H2, $X) samples=30)

# nested LR residual smoke
b = randn(n)
F = lu(H2; method=:nested, rtol=1e-4)
xx = ldiv!(F, copy(b))
println("\nnested LR residual ", norm(H2 * xx - b) / (norm(b) + 1e-14))
nf = count(nd -> isuniform(nd) && nd.s_full, h2_leaves(F.factors))
nd = count(isdense_h2, h2_leaves(F.factors))
println("factor leaves: dense=$nd s_full_rk=$nf / $(h2_nleaves(F.factors))")
