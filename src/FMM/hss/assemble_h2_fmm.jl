# =============================================================================
# Nested H² via FMM matvecs → HMatrices.H2Matrix (HARA-H²)
# =============================================================================

"""
    assemble_h2_fmm(A, tree; kwargs...) -> H2Matrix

Build a nested [`H2Matrix`](@ref) from a **matvec-capable** operator `A`
(typically an [`FMMKernelMatrix`](@ref) or scaled FMM) using
[`hara_h2`](@ref) — no dense fill of far blocks.

# Keywords
- `rtol`, `rank`, `nsample`, `alpha` — passed to `hara_h2`
- `global_index=true` — tree permutation vs exterior ordering of `A`
- `orthog=true`, `compress=true` — post-process nested bases
"""
function assemble_h2_fmm(
        A::AbstractMatrix,
        tree;
        rtol::Real = 1e-4,
        rank::Int = 48,
        nsample::Int = 64,
        alpha::Real = 0.5,
        global_index::Bool = true,
        orthog::Bool = true,
        compress::Bool = true,
        kwargs...,
    )
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("assemble_h2_fmm requires square A"))
    length(tree) == n || throw(DimensionMismatch(
        "tree length $(length(tree)) ≠ matrix size $n — for block DOFs use expand_tree"))
    S = KernelMatvecSampler(A)
    return hara_h2(S, tree;
        rtol=float(rtol),
        rank=rank,
        nsample=nsample,
        alpha=float(alpha),
        global_index=global_index,
        orthog=orthog,
        compress=compress,
        kwargs...)
end

"""
    assemble_h2_fmm(points; kernel=:laplace2d, kwargs...)

One-shot: build cluster tree + FMM kernel + nested H² via HARA.

# Keywords
- `kernel` — `:laplace2d`, `:laplace3d`, `:yukawa3d`, or `:kelvin2d`
- `μ`, `ν` — required for `:kelvin2d`
- `κ` — Yukawa wave number
- `eps`, `nmax`, `η` — FMM
- `rtol`, `rank`, `nsample`, `alpha` — H² HARA
- `scale` — optional scalar multiply on the FMM operator (e.g. BEM `−1/(2πk)`)
- `splitter` — default `PrincipalComponentSplitter(nmax=nmax)`
"""
function assemble_h2_fmm(
        points::AbstractMatrix{<:Real};
        kernel::Symbol = :laplace2d,
        μ = nothing,
        ν = nothing,
        κ::Float64 = 1.0,
        eps::Float64 = 1e-8,
        nmax::Int = 40,
        η::Float64 = 1.0,
        rtol::Real = 1e-4,
        rank::Int = 48,
        nsample::Int = 64,
        alpha::Real = 0.5,
        scale::Union{Nothing,Real} = nothing,
        splitter = nothing,
        orthog::Bool = true,
        compress::Bool = true,
    )
    d, n = size(points)
    pts_sv = [SVector{d,Float64}(ntuple(k -> Float64(points[k, i]), d)) for i in 1:n]
    spl = splitter === nothing ? PrincipalComponentSplitter(; nmax=nmax) : splitter
    tree = ClusterTree(pts_sv, spl)

    A = if kernel === :laplace2d
        d == 2 || throw(ArgumentError("laplace2d needs 2×N points"))
        fmm_laplace2d_matrix(points; eps=eps, nmax=nmax, η=η)
    elseif kernel === :laplace3d
        d == 3 || throw(ArgumentError("laplace3d needs 3×N points"))
        fmm_laplace3d_matrix(points; eps=eps, nmax=nmax, η=η)
    elseif kernel === :yukawa3d
        d == 3 || throw(ArgumentError("yukawa3d needs 3×N points"))
        fmm_yukawa3d_matrix(points, κ; eps=eps, nmax=nmax, η=η)
    elseif kernel === :kelvin2d
        d == 2 || throw(ArgumentError("kelvin2d needs 2×N points"))
        (μ === nothing || ν === nothing) &&
            throw(ArgumentError("kelvin2d requires keywords μ and ν"))
        raw = fmm_kelvin2d_matrix(points; μ=μ, ν=ν, eps=eps, nmax=nmax, η=η)
        etree = expand_tree(tree, 2)
        Aop = scale === nothing ? raw : _scale_fmm_op(raw, float(scale))
        return assemble_h2_fmm(Aop, etree; rtol=rtol, rank=rank, nsample=nsample,
            alpha=alpha, orthog=orthog, compress=compress)
    else
        throw(ArgumentError("unknown kernel $kernel"))
    end

    Aop = scale === nothing ? A : _scale_fmm_op(A, float(scale))
    return assemble_h2_fmm(Aop, tree; rtol=rtol, rank=rank, nsample=nsample,
        alpha=alpha, orthog=orthog, compress=compress)
end

"""
    assemble_h2_fmm_kernel(K, pts; kwargs...)

H² of matvec kernel `K` on point cloud `pts` (builds tree). Use
`blocksize=2` for node-major vectorial DOFs (Kelvin).
"""
function assemble_h2_fmm_kernel(
        K::AbstractMatrix,
        pts::AbstractVector;
        rtol = 1e-4,
        rank = 48,
        nsample = 64,
        alpha = 0.5,
        nmax = 32,
        blocksize::Int = 1,
        global_index::Bool = true,
        orthog::Bool = true,
        compress::Bool = true,
    )
    splitter = PrincipalComponentSplitter(; nmax=nmax)
    tree = ClusterTree(collect(pts), splitter)
    btree = blocksize == 1 ? tree : expand_tree(tree, blocksize)
    length(btree) == size(K, 1) || throw(DimensionMismatch(
        "expanded tree length $(length(btree)) ≠ size(K,1)=$(size(K,1))"))
    return assemble_h2_fmm(K, btree;
        rtol=rtol, rank=rank, nsample=nsample, alpha=alpha,
        global_index=global_index, orthog=orthog, compress=compress)
end

# Local scale helper (avoid depending on BEM._ScaledFMM name from FMM)
struct _FMMScale{TA} <: AbstractMatrix{Float64}
    A::TA
    α::Float64
end
Base.size(S::_FMMScale) = size(S.A)
Base.size(S::_FMMScale, d) = size(S.A, d)
function LinearAlgebra.mul!(y::AbstractVector, S::_FMMScale, x::AbstractVector)
    mul!(y, S.A, x)
    y .*= S.α
    return y
end
function LinearAlgebra.mul!(Y::AbstractMatrix, S::_FMMScale, X::AbstractMatrix)
    mul!(Y, S.A, X)
    Y .*= S.α
    return Y
end
function LinearAlgebra.mul!(y::AbstractVector, St::Adjoint{<:Any,<:_FMMScale}, x::AbstractVector)
    S = parent(St)
    mul!(y, adjoint(S.A), x)
    y .*= S.α
    return y
end
function LinearAlgebra.mul!(Y::AbstractMatrix, St::Adjoint{<:Any,<:_FMMScale}, X::AbstractMatrix)
    S = parent(St)
    mul!(Y, adjoint(S.A), X)
    Y .*= S.α
    return Y
end
_scale_fmm_op(A, α::Float64) = isone(α) ? A : _FMMScale(A, α)
