# Shared DIBEM infrastructure (scalar factored form + F-solve + format helpers)
#
# Physics-specific pieces stay in:
#   Laplace/Domain.jl, Domain_fast.jl
#   Elasticity/Domain.jl, Domain_fast.jl

export DibemFactoredOperator, DibemFMMOperator, DibemFKernel

# ---------------------------------------------------------------------------
# RBF Gram kernel
# ---------------------------------------------------------------------------

"""RBF Gram ``F[i,j] = φ(‖x_i−x_j‖²)``."""
struct DibemFKernel{R,P} <: AbstractMatrix{Float64}
    rbf::R
    points::Vector{P}
end
Base.size(K::DibemFKernel) = (length(K.points), length(K.points))
Base.getindex(K::DibemFKernel, i::Int, j::Int) =
    float(K.rbf(sqeuclidean(K.points[i], K.points[j])))

# ---------------------------------------------------------------------------
# Factored operator  M x = D (c ∘ x) + diag ∘ x
# ---------------------------------------------------------------------------

"""
    DibemFactoredOperator

Unified matrix-free scalar DIBEM `M`:

```
M x = D (c ∘ x) + diag ∘ x ,    diag = ID − D c
```

Alias: [`DibemFMMOperator`](@ref) (historical name).
"""
struct DibemFactoredOperator{TD} <: AbstractMatrix{Float64}
    n::Int
    c::Vector{Float64}
    diag::Vector{Float64}
    D::TD
    method::Symbol
end

const DibemFMMOperator = DibemFactoredOperator

Base.size(A::DibemFactoredOperator) = (A.n, A.n)
Base.IndexStyle(::Type{<:DibemFactoredOperator}) = IndexCartesian()

function Base.getindex(A::DibemFactoredOperator, i::Int, j::Int)
    i == j && return A.diag[i]
    return A.c[j] * A.D[i, j]
end

function LinearAlgebra.mul!(y::AbstractVector, A::DibemFactoredOperator, x::AbstractVector)
    length(x) == A.n && length(y) == A.n || throw(DimensionMismatch())
    mul!(y, A.D, A.c .* x)
    @inbounds for i in 1:A.n
        y[i] += A.diag[i] * x[i]
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::DibemFactoredOperator, x::AbstractVector,
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

Base.:*(A::DibemFactoredOperator, x::AbstractVector) = mul!(similar(x, Float64), A, x)

"""Build factored M from compressed D, weights c, Galerkin ID."""
function _dibem_factored_M(D, c::Vector{Float64}, ID::Vector{Float64}, method::Symbol)
    n = length(c)
    rowsum_off = D * c
    diagv = .-rowsum_off .+ ID
    return DibemFactoredOperator(n, c, diagv, D, method)
end

# ---------------------------------------------------------------------------
# Format map + F solve
# ---------------------------------------------------------------------------

function _dibem_struct_format(method::Symbol)
    m = method
    m in (:hmatrix, :Hmat, :hmat, :H, :HMatrix) && return :H
    m in (:hodlr, :HODLR, :HODLRMatrix) && return :HODLR
    m in (:hss, :HSS, :HSSMatrix) && return :HSS
    m in (:hbs, :HBS, :HBSMatrix) && return :HBS
    m in (:h2, :H2, :H2Matrix) && return :H2
    m in (:blr, :BLR) && return :BLR
    m in (:dense,) && return :dense
    m in (:fmm, :FMM) && return :fmm
    return nothing
end

function _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=:dense, atol=1e-6,
        rtol=1e-6, nmax=32, eta=3.0, threads=true, rank=typemax(Int),
        hss_method=:dense, alpha=1.0)
    nt = length(IF)
    if f_method === :dense || (f_method === :auto && nt <= 256)
        F = zeros(nt, nt)
        @inbounds for j in 1:nt, i in 1:nt
            F[i, j] = rbf(sqeuclidean(pts[i], pts[j]))
        end
        ε = 1e-12 * (tr(F) / nt + 1)
        @inbounds for i in 1:nt
            F[i, i] += ε
        end
        set_cache!(dad; dibem_F=F)
        return F \ IF
    end

    splitter = PrincipalComponentSplitter(; nmax=nmax)
    tree = ClusterTree(pts, splitter)
    KF = DibemFKernel(rbf, pts)
    fmt = _dibem_struct_format(f_method)
    fmt === nothing && (fmt = :H)
    if fmt === :H2
        return _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=:dense, atol=atol, rtol=rtol)
    end
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)
    adm = StrongAdmissibilityStd(; eta=eta)
    Fst = assemble_structured(KF, tree; format=fmt, adm=adm, comp=comp,
        threads=threads, rtol=rtol, rank=rank, alpha=alpha,
        method=hss_method, global_index=true)
    if fmt === :H && Fst isa HMatrices.HMatrix
        _hmat_add_diag_ridge!(Fst, 1e-12)
    end
    c, stats = Krylov.gmres(Fst, IF; atol=rtol, rtol=rtol, itmax=max(4nt, 200))
    if !stats.solved
        @warn "DIBEM: GMRES(F) incomplete" f_method stats.status
    end
    set_cache!(dad; dibem_F_h=Fst)
    return c
end

function _hmat_add_diag_ridge!(Hmat::HMatrices.HMatrix, ε::Float64)
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
        for (iloc, ig) in enumerate(irangeg), (jloc, jg) in enumerate(jrangeg)
            if ig == jg && iloc <= size(data, 1) && jloc <= size(data, 2)
                data[iloc, jloc] += ε * (tr(data) / max(size(data, 1), 1) + 1)
            end
        end
    end
    return nothing
end

"""Scale FMM kernel: `(αA)x = α(Ax)`."""
struct _ScaledFMM{TA} <: AbstractMatrix{Float64}
    A::TA
    α::Float64
end
Base.size(S::_ScaledFMM) = size(S.A)
Base.size(S::_ScaledFMM, d) = size(S.A, d)
Base.IndexStyle(::Type{<:_ScaledFMM}) = IndexCartesian()
Base.getindex(S::_ScaledFMM, i::Int, j::Int) = S.α * S.A[i, j]
function LinearAlgebra.mul!(y::AbstractVector, S::_ScaledFMM, x::AbstractVector)
    mul!(y, S.A, x)
    y .*= S.α
    return y
end
function LinearAlgebra.mul!(y::AbstractVector, S::_ScaledFMM, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, S, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, S, x)
        y .= β .* y .+ α .* t
    end
    return y
end
Base.:*(S::_ScaledFMM, x::AbstractVector) = mul!(similar(x, Float64), S, x)
function LinearAlgebra.mul!(Y::AbstractMatrix, S::_ScaledFMM, X::AbstractMatrix)
    mul!(Y, S.A, X)
    Y .*= S.α
    return Y
end
function LinearAlgebra.mul!(y::AbstractVector, St::Adjoint{<:Any,<:_ScaledFMM}, x::AbstractVector)
    S = parent(St)
    mul!(y, adjoint(S.A), x)
    y .*= S.α
    return y
end
function LinearAlgebra.mul!(Y::AbstractMatrix, St::Adjoint{<:Any,<:_ScaledFMM}, X::AbstractMatrix)
    S = parent(St)
    mul!(Y, adjoint(S.A), X)
    Y .*= S.α
    return Y
end

"""
    assemble_hss_fmm_kernel(K, pts; rtol, rank, nmax, oversampling, blocksize=1)

HMatrices HSS of matvec-capable kernel `K` on points `pts` via
`assemble_hss(...; method=:fmm)`. Use `blocksize=2` for node-major vectorial
DOFs (elasticity Kelvin).
"""
function assemble_hss_fmm_kernel(
    K::AbstractMatrix,
    pts::AbstractVector;
    rtol=1e-6,
    rank=48,
    nmax=32,
    oversampling=10,
    blocksize::Int=1,
    global_index::Bool=true,
)
    splitter = PrincipalComponentSplitter(; nmax=nmax)
    tree = ClusterTree(collect(pts), splitter)
    btree = blocksize == 1 ? tree : expand_tree(tree, blocksize)
    length(btree) == size(K, 1) || throw(DimensionMismatch(
        "expanded tree length $(length(btree)) ≠ size(K,1)=$(size(K,1))"))
    return assemble_hss(float(eltype(K)), K, btree;
        method=:fmm, rtol=rtol, rank=rank, oversampling=oversampling,
        global_index=global_index)
end
