# SAFRAN Experiment 1 vs BEM NNCA (same points, log kernel, ε=10^{-9}).
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using StaticArrays
using BEM.HMatrices

const N1D = 5
const L = 1.0
const BSOFT = 1e-4
const RTOL = 1e-9
const NLEVELS = 2:5
const NWARM = 3
const NRUN = 10

function leaf_centers(nLevels, L)
    boxes = [SVector(0.0, 0.0)]
    boxR = float(L)
    for _ in 1:nLevels
        shift = 0.5 * boxR
        new = SVector{2,Float64}[]
        for c in boxes
            push!(new, SVector(c[1] - shift, c[2] - shift))
            push!(new, SVector(c[1] + shift, c[2] - shift))
            push!(new, SVector(c[1] + shift, c[2] + shift))
            push!(new, SVector(c[1] - shift, c[2] + shift))
        end
        boxes = new
        boxR *= 0.5
    end
    return boxes
end

function safran_uniform_2d(nLevels; n1d=N1D, L=L)
    radius = L / 2^nLevels
    nodes1d = [-L + 2L * (k + 1) / (n1d + 1) for k in 0:(n1d - 1)]
    pts = SVector{2,Float64}[]
    for c in leaf_centers(nLevels, L)
        for j in 1:n1d, k in 1:n1d
            push!(pts, SVector(nodes1d[k] * radius + c[1], nodes1d[j] * radius + c[2]))
        end
    end
    return pts
end

function kernel_safran(x, y)
    R2 = (x[1] - y[1])^2 + (x[2] - y[2])^2
    R2 < 1e-10 && return 0.0
    R = sqrt(R2)
    if R < BSOFT
        return (R * log(R) - 1) / (BSOFT * log(BSOFT) - 1)
    else
        return log(R) / log(BSOFT)
    end
end

function sampled_rel(H, K, x; nsample=min(200, length(x)))
    n = length(x)
    y = H * x
    Random.seed!(1)
    rows = nsample >= n ? collect(1:n) : sort(randperm(n)[1:nsample])
    yd = zeros(length(rows))
    @inbounds for (ii, i) in enumerate(rows)
        s = 0.0
        for j in 1:n
            s += K[i, j] * x[j]
        end
        yd[ii] = s
    end
    return norm(y[rows] - yd) / (norm(yd) + 1e-14)
end

println("Julia threads = ", Threads.nthreads())
# compile
let
    pts = safran_uniform_2d(2)
    K = KernelMatrix(kernel_safran, pts, pts)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=N1D * N1D, tight=false); cube=true)
    H = assemble_h2(K, tree; rtol=RTOL, threads=false)
    H * randn(length(pts))
end

@printf "%6s %8s %12s %12s %8s %10s\n" "lev" "N" "Ta" "Tm" "rank" "rel"
for nLevels in NLEVELS
    pts = safran_uniform_2d(nLevels)
    n = length(pts)
    nmax = N1D * N1D
    K = KernelMatrix(kernel_safran, pts, pts)
    t_asm = @elapsed begin
        tree = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
        H = assemble_h2(K, tree; rtol=RTOL, threads=true)
    end
    x = randn(n)
    y = similar(x)
    for _ in 1:NWARM
        mul!(y, H, x)
    end
    t0 = time_ns()
    for _ in 1:NRUN
        mul!(y, H, x)
    end
    t_mv = (time_ns() - t0) / NRUN / 1e9
    rel = sampled_rel(H, K, x; nsample=n <= 6400 ? n : 200)
    @printf "%6d %8d %12.4e %12.4e %8.1f %10.3e\n" nLevels n t_asm t_mv H.avg_rank rel
    flush(stdout)
end
