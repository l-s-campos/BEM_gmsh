using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using StaticArrays
using BEM.HMatrices
using BEM.FMM

# FMM log + diagonal ridge, original point order.
struct RidgeOp{T, A} <: AbstractMatrix{T}
    A::A
    α::T
end
Base.size(R::RidgeOp) = size(R.A)
Base.getindex(R::RidgeOp, i::Int, j::Int) = R.A[i, j] + (i == j ? R.α : zero(R.α))
function LinearAlgebra.mul!(y::AbstractVector, R::RidgeOp, x::AbstractVector)
    mul!(y, R.A, x)
    @inbounds for i in eachindex(y)
        y[i] += R.α * x[i]
    end
    return y
end
function LinearAlgebra.mul!(Y::AbstractMatrix, R::RidgeOp, X::AbstractMatrix)
    mul!(Y, R.A, X)
    @inbounds for j in axes(Y, 2), i in axes(Y, 1)
        Y[i, j] += R.α * X[i, j]
    end
    return Y
end
function LinearAlgebra.mul!(y::AbstractVector, Rt::Adjoint{<:Any, <:RidgeOp}, x::AbstractVector)
    R = parent(Rt)
    mul!(y, adjoint(R.A), x)
    @inbounds for i in eachindex(y)
        y[i] += R.α * x[i]
    end
    return y
end
function LinearAlgebra.mul!(Y::AbstractMatrix, Rt::Adjoint{<:Any, <:RidgeOp}, X::AbstractMatrix)
    R = parent(Rt)
    mul!(Y, adjoint(R.A), X)
    @inbounds for j in axes(Y, 2), i in axes(Y, 1)
        Y[i, j] += R.α * X[i, j]
    end
    return Y
end

function run(n1d; nmax=32, ridge=8.0, rank=16, p=16)
    Random.seed!(1)
    xs = range(0.0, 1.0; length=n1d)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    P = Matrix{Float64}(undef, 2, n)
    @inbounds for j in 1:n
        P[1, j] = pts[j][1]
        P[2, j] = pts[j][2]
    end
    K = KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    b = randn(n)

    t_fmm = @elapsed Fmm = fmm_laplace2d_matrix(P; eps=1e-6, nmax=nmax, tree=tree)
    A = RidgeOp(Fmm, ridge)
    t_srs = @elapsed Fs = srs_factor(K, tree; rtol=1e-6, rank=rank)

    t0 = @elapsed x0, st0 = gmres_h(A, b; rtol=1e-8, itmax=400, history=true)
    r0 = norm(A * x0 - b) / (norm(b) + 1e-14)
    t1 = @elapsed x1, st1 = gmres_h(A, b; Pl=Fs, rtol=1e-8, itmax=40, history=true)
    r1 = norm(A * x1 - b) / (norm(b) + 1e-14)

    t_mv = NaN
    t2 = NaN
    st2n = "-"
    st2s = "skip"
    r2 = NaN
    nelim = 0
    nroot = 0
    nsteps = 0
    try
        t_mv = @elapsed Fmv = srs_factor_matvec(A, tree; rtol=1e-6, rank=rank, p=p)
        nelim = sum(s -> length(s.R), Fmv.steps; init=0)
        nroot = length(Fmv.root_idx)
        nsteps = length(Fmv.steps)
        t2 = @elapsed x2, st2 = gmres_h(A, b; Pl=Fmv, rtol=1e-8, itmax=40, history=true)
        r2 = norm(A * x2 - b) / (norm(b) + 1e-14)
        st2n, st2s = string(st2.niter), string(st2.status)
    catch e
        st2s = sprint(showerror, e)
    end

    println("="^72)
    @printf("N=%d  nmax=%d  rank=%d\n", n, nmax, rank)
    @printf("  FMM asm=%.3fs  SRS(entries)=%.3fs  SRS(matvec)=%s  steps=%d elim=%d root=%d\n",
        t_fmm, t_srs, isnan(t_mv) ? "fail" : @sprintf("%.3fs", t_mv), nsteps, nelim, nroot)
    @printf("  GMRES FMM          %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(st0.niter), t0, r0, t_fmm + t0, st0.status)
    @printf("  GMRES FMM+SRS(K)   %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(st1.niter), t1, r1, t_fmm + t_srs + t1, st1.status)
    @printf("  GMRES FMM+SRS(mv)  %4s it  %s  resid=%s  tot=%s  %s\n",
        st2n, isnan(t2) ? "  n/a " : @sprintf("%.3fs", t2),
        isnan(r2) ? "n/a" : @sprintf("%.2e", r2),
        (isnan(t_fmm) || isnan(t_mv) || isnan(t2)) ? "n/a" :
            @sprintf("%.3fs", t_fmm + t_mv + t2), st2s)
    flush(stdout)
end

println("threads=", Threads.nthreads())
run(16; nmax=8, rank=6, p=6)
run(32; nmax=16, rank=10, p=10)
run(48; nmax=16, rank=12, p=12)
