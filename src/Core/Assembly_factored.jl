# Hierarchical / factored H,G infrastructure (physics-agnostic operators)
#
#   Bare kernels Du, Dq compressed separately; quadrature weights outside:
#     G x = Du (w ∘ x) + diag_G ∘ x
#     H x = Dq (w_H ∘ x) + diag_H ∘ x
#
# Same pattern as DIBEM  M x = D (c ∘ x) + diag ∘ x.
# Physics-specific kernels and H_G_Hmat live under Laplace/ (or Elasticity/).

export ColWeightedOp, MixedBCOperator, node_weights, free_term

"""
    node_weights(dad::BEMdata) -> Vector{Float64}

Integration weights for each boundary collocation node (`Jacobian × quad weight`).
"""
function node_weights(dad::BEMdata)
    w = zeros(dad.n)
    for elem in dad.elements
        for (k, node) in enumerate(elem.index)
            w[node] = elem.Jacobian[k] * dad.elem_weight[k]
        end
    end
    return w
end

"""
    free_term(dad, i) -> Float64

Diagonal free-term entry placed in ``H_ii`` for discontinuous collocation
(scalar problems).

| location | ``c`` | ``H_ii = free_term`` |
|----------|-------|----------------------|
| boundary | ``1/2`` | ``-1/2`` |
| internal | ``1`` | ``-1`` |
"""
@inline free_term(dad::BEMdata, i::Integer) = i <= dad.n ? -0.5 : -1.0

# ---------------------------------------------------------------------------
# Column-weighted operator  A x = K (w ∘ x) + diag ∘ x
# ---------------------------------------------------------------------------

"""
    ColWeightedOp

Matrix-free
```
A x = K (w ∘ x) + d ∘ x
```
with compressed bare kernel `K` and column weights `w` (length = `ncols`).
Diagonal free-term / regularisation lives in `d` (length = `nrows`).

Same structure as [`DibemFactoredOperator`](@ref) (`w` ↔ `c`).
"""
mutable struct ColWeightedOp{TK} <: AbstractMatrix{Float64}
    K::TK
    w::Vector{Float64}
    d::Vector{Float64}
    nrows::Int
    ncols::Int
    """Optional sparse near-field correction: full ∫ − pointwise kernel×w."""
    corr::Union{Nothing, SparseMatrixCSC{Float64,Int}}
end

ColWeightedOp(K, w, d, nrows, ncols) =
    ColWeightedOp(K, w, d, nrows, ncols, nothing)

Base.size(A::ColWeightedOp) = (A.nrows, A.ncols)
Base.IndexStyle(::Type{<:ColWeightedOp}) = IndexCartesian()

function Base.getindex(A::ColWeightedOp, i::Int, j::Int)
    val = A.w[j] * A.K[i, j]
    if i == j && j <= length(A.d)
        val += A.d[i]
    end
    if A.corr !== nothing && 1 <= i <= size(A.corr, 1) && 1 <= j <= size(A.corr, 2)
        val += A.corr[i, j]
    end
    return val
end

function LinearAlgebra.mul!(y::AbstractVector, A::ColWeightedOp, x::AbstractVector)
    length(x) == A.ncols && length(y) == A.nrows || throw(DimensionMismatch())
    mul!(y, A.K, A.w .* x)
    n = min(A.nrows, A.ncols, length(A.d))
    @inbounds for i in 1:n
        y[i] += A.d[i] * x[i]
    end
    if A.corr !== nothing
        mul!(y, A.corr, x, 1, 1)  # y += corr * x
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::ColWeightedOp, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, A, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, A, x)
        y .= β .* y .+ α .* t
    end
    return y
end

Base.:*(A::ColWeightedOp, x::AbstractVector) = mul!(similar(x, Float64, A.nrows), A, x)

# ---------------------------------------------------------------------------
# Mixed-BC linear operator for hierarchical / factored H,G (scalar)
# ---------------------------------------------------------------------------

"""
    MixedBCOperator

Matrix-free mixed BC operator for hierarchical or factored `H` (`nt×nt`) and
`G` (`nt×n`):

- Dirichlet dof `j`: column `-G[:,j]`, unknown `q_j`
- Neumann / internal: column `H[:,j]`, unknown `T_j`
"""
struct MixedBCOperator{TH,TG} <: AbstractMatrix{Float64}
    H::TH
    G::TG
    BC::Vector{Int}
    n::Int
    nt::Int
end

Base.size(A::MixedBCOperator) = (A.nt, A.nt)

function LinearAlgebra.mul!(y::AbstractVector, A::MixedBCOperator, x::AbstractVector)
    T = zeros(eltype(x), A.nt)
    q = zeros(eltype(x), A.n)
    @inbounds for j in 1:A.n
        if A.BC[j] == 0
            q[j] = x[j]
        else
            T[j] = x[j]
        end
    end
    @inbounds for j in (A.n+1):A.nt
        T[j] = x[j]
    end
    mul!(y, A.H, T)
    yg = A.G * q
    y .-= yg
    return y
end

Base.:*(A::MixedBCOperator, x::AbstractVector) = mul!(similar(x, size(A, 1)), A, x)

"""Write diagonal of an H-matrix using `fdiag(global_index) -> value`."""
function _set_diagonal!(fdiag, Hmat::HMatrix)
    piv = HMatrices.pivot(Hmat)
    for block in HMatrices.nodes(Hmat)
        HMatrices.hasdata(block) || continue
        HMatrices.isadmissible(block) && continue
        data = HMatrices.data(block)
        data isa Matrix || continue

        irange = HMatrices.rowrange(block) .- piv[1] .+ 1
        jrange = HMatrices.colrange(block) .- piv[2] .+ 1
        irangeg = HMatrices.rowperm(Hmat)[irange]
        jrangeg = HMatrices.colperm(Hmat)[jrange]
        for (iloc, ig) in enumerate(irangeg)
            ig > size(Hmat, 2) && continue
            for (jloc, jg) in enumerate(jrangeg)
                if ig == jg && jloc <= size(data, 2) && iloc <= size(data, 1)
                    data[iloc, jloc] = fdiag(ig)
                end
            end
        end
    end
    return nothing
end
