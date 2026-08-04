# =============================================================================
# Fast DIBEM — unified factored form (all compression backends)
# =============================================================================
#
# Discrete DIBEM (Domain.jl dense):
#   F c = IF ,   M_ij = c_j u*(x_i,x_j)  (i≠j) ,   M_ii = ID_i − ∑_{j≠i} M_ij
#
# Equivalent **factored** matvec (used by every compressed backend):
#   M x = D (c ∘ x) + diag ∘ x
#   diag = ID − D c
#
# where D_ij = u*(x_i, x_j) (i≠j), D_ii = 0  — the same single-layer kernel as G,
# without boundary quadrature weights.  G is rectangular (boundary cols × w_j);
# D is square on all collocation poles.  Compression applies only to D (and F);
# the vector c carries all DIBEM-specific information.
#
# Backends for D: :hmatrix | :hodlr | :hss | :hbs | :h2 | :fmm
# F solve: dense or H-matrix GMRES
# =============================================================================

export DIBEM, DIBEM_Hmat, DIBEM_HODLR, DIBEM_HSS, DIBEM_HBS, DIBEM_H2
export DIBEM_FMM, DibemFactoredOperator, DibemFMMOperator, dibem!

# ---------------------------------------------------------------------------
# Geometry / IF, ID
# ---------------------------------------------------------------------------

function _dibem_collocation_points(dad::BEMdata)
    return isempty(dad.internalNodes) ? collect(dad.Nodes) :
           vcat(collect(dad.Nodes), collect(dad.internalNodes))
end

function _dibem_IF_ID(dad::BEMdata{<:Laplace}, rbf)
    nt = dad.nt
    props = dad.properties
    pts = _dibem_collocation_points(dad)
    IF = zeros(nt)
    ID = zeros(nt)
    @inbounds for i in 1:nt
        x = pts[i]
        for elem in dad.elements
            for j in eachindex(elem.index)
                ind = elem.index[j]
                xj = dad.Nodes[ind]
                r = xj - x
                R = norm(r)
                R < 1e-10 && continue
                wJn = dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
                IF[i] += int(rbf, x, xj) * wJn
                ID[i] += _galerkin_n_dot_gradG(props, R) * wJn
            end
        end
    end
    return IF, ID, pts
end

# ---------------------------------------------------------------------------
# Kernels: F (RBF) and D (plain u* — same family as G)
# ---------------------------------------------------------------------------

"""RBF Gram ``F[i,j] = φ(‖x_i−x_j‖²)``."""
struct DibemFKernel{R,P} <: AbstractMatrix{Float64}
    rbf::R
    points::Vector{P}
end
Base.size(K::DibemFKernel) = (length(K.points), length(K.points))
Base.getindex(K::DibemFKernel, i::Int, j::Int) =
    float(K.rbf(sqeuclidean(K.points[i], K.points[j])))

"""
Square single-layer kernel on all collocation points (DIBEM factor `D`):

```
D_ij = u*(x_i, x_j) = fundamental(props, x_j−x_i, ·).U   (i≠j),   D_ii = 0
```

Same `u*` as the BEM `G` matrix; `G` additionally multiplies boundary weights
and keeps only boundary columns.
"""
struct DibemUStarKernel{P,Prop} <: AbstractMatrix{Float64}
    points::Vector{P}
    props::Prop
    n_dummy::P
end
Base.size(K::DibemUStarKernel) = (length(K.points), length(K.points))
function Base.getindex(K::DibemUStarKernel, i::Int, j::Int)
    i == j && return 0.0
    r = K.points[j] - K.points[i]
    norm(r) < 1e-15 && return 0.0
    return float(fundamental(K.props, r, K.n_dummy).U)
end

function _dibem_ustar_kernelmatrix(pts, props, n_dummy)
    function ustar(x, y)::Float64
        r = y - x
        R = norm(r)
        R < 1e-15 && return 0.0
        return float(fundamental(props, r, n_dummy).U)
    end
    return KernelMatrix{typeof(ustar), typeof(pts), typeof(pts), Float64}(ustar, pts, pts)
end

# ---------------------------------------------------------------------------
# Factored operator  M x = D (c ∘ x) + diag ∘ x
# ---------------------------------------------------------------------------

"""
    DibemFactoredOperator

Unified matrix-free DIBEM `M` for **every** compressed backend:

```
M x = D (c ∘ x) + diag ∘ x ,    diag = ID − D c
```

`D` is any matvec-capable compression of plain `u*` (H, HODLR, HSS, H², FMM).
`c` is the only DIBEM-specific vector (`F c = IF`).

Alias: [`DibemFMMOperator`](@ref) (historical name).
"""
struct DibemFactoredOperator{TD} <: AbstractMatrix{Float64}
    n::Int
    c::Vector{Float64}
    diag::Vector{Float64}
    D::TD
    method::Symbol
end

const DibemFMMOperator = DibemFactoredOperator  # backward compatible

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
    rowsum_off = D * c                    # (D c)_i = ∑_j D_ij c_j
    diagv = .-rowsum_off .+ ID
    return DibemFactoredOperator(n, c, diagv, D, method)
end

# ---------------------------------------------------------------------------
# F solve: F c = IF
# ---------------------------------------------------------------------------

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

    # structured F
    splitter = PrincipalComponentSplitter(; nmax=nmax)
    tree = ClusterTree(pts, splitter)
    KF = DibemFKernel(rbf, pts)
    fmt = _dibem_struct_format(f_method)
    fmt === nothing && (fmt = :H)
    if fmt === :H2
        # RBF Gram + H² proxies is unreliable → dense
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

# ---------------------------------------------------------------------------
# Compress plain D = u*
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

"""
Assemble compressed single-layer `D` on all collocation points (same `u*` as G).
"""
function _dibem_compress_D(dad, pts, method::Symbol; atol=1e-6, rtol=1e-6,
        nmax=32, eta=3.0, threads=true, rank=typemax(Int), alpha=1.0,
        hss_method=:dense, eps=1e-6)
    props = dad.properties
    k = float(props.k)
    dim = dad.dimension
    n_dummy = pts[1] isa SVector{2} ? SVector(0.0, 1.0) : SVector(0.0, 0.0, 1.0)
    nt = length(pts)
    fmt = _dibem_struct_format(method)

    if fmt === :fmm
        d = length(pts[1])
        Pmat = Matrix{Float64}(undef, d, nt)
        @inbounds for j in 1:nt, a in 1:d
            Pmat[a, j] = pts[j][a]
        end
        if dim == 2
            raw = FMM.fmm_laplace2d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
            return _ScaledFMM(raw, -1 / (2π * k))
        else
            raw = FMM.fmm_laplace3d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
            return _ScaledFMM(raw, 1 / k)
        end
    end

    if fmt === :dense || fmt === nothing
        D = zeros(nt, nt)
        KD = DibemUStarKernel(pts, props, n_dummy)
        @inbounds for j in 1:nt, i in 1:nt
            D[i, j] = KD[i, j]
        end
        return D
    end

    splitter = PrincipalComponentSplitter(; nmax=nmax)
    tree = ClusterTree(pts, splitter)
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)
    adm = StrongAdmissibilityStd(; eta=eta)

    if fmt === :H2
        KD = _dibem_ustar_kernelmatrix(pts, props, n_dummy)
        return assemble_h2(Float64, KD, tree; rtol=rtol, rank=rank, alpha=alpha,
            global_index=true, symmetric=true)
    end

    KD = DibemUStarKernel(pts, props, n_dummy)
    return assemble_structured(KD, tree; format=fmt, adm=adm, comp=comp,
        threads=threads, rtol=rtol, rank=rank, alpha=alpha,
        method=hss_method, global_index=true)
end

"""Scale FMM kernel: `(αA)x = α(Ax)`."""
struct _ScaledFMM{TA}
    A::TA
    α::Float64
end
Base.size(S::_ScaledFMM) = size(S.A)
Base.size(S::_ScaledFMM, d) = size(S.A, d)
Base.getindex(S::_ScaledFMM, i::Int, j::Int) = S.α * S.A[i, j]
function LinearAlgebra.mul!(y::AbstractVector, S::_ScaledFMM, x::AbstractVector)
    mul!(y, S.A, x)
    y .*= S.α
    return y
end
Base.:*(S::_ScaledFMM, x::AbstractVector) = mul!(similar(x, Float64), S, x)

# ---------------------------------------------------------------------------
# Unified compressed DIBEM
# ---------------------------------------------------------------------------

"""
    DIBEM_compressed(dad; method=:hmatrix, rbf=PHS(), f_method=:auto, kwargs...)

Factored DIBEM for any compression of `D = u*`:

1. `c` from `F c = IF` (`f_method` = `:dense` / `:hmatrix` / `:hodlr` / …)
2. `D` = compressed single-layer on all poles (`method`)
3. `M = DibemFactoredOperator(D, c, ID)`
"""
function DIBEM_compressed(
    dad::BEMdata{<:Laplace};
    method::Symbol = :hmatrix,
    rbf = PHS(),
    f_method::Symbol = :auto,
    atol = 1e-6,
    rtol = 1e-6,
    nmax = 32,
    eta = 3.0,
    threads = true,
    rank = typemax(Int),
    alpha = 1.0,
    hss_method = :dense,
    eps = 1e-6,
    f_nmax = nothing,
)
    IF, ID, pts = _dibem_IF_ID(dad, rbf)
    nt = dad.nt

    # default F backend: dense if small, else same structured family as D (not FMM/H2)
    fm = f_method
    if fm === :auto
        fm = nt <= 256 ? :dense :
             method in (:fmm, :FMM, :h2, :H2) ? :hmatrix : method
    end

    @info "DIBEM_compressed" method f_method=fm nt
    c = _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=fm, atol=atol, rtol=rtol,
        nmax = something(f_nmax, nmax), eta=eta, threads=threads, rank=rank,
        hss_method=hss_method, alpha=alpha)

    D = _dibem_compress_D(dad, pts, method; atol=atol, rtol=rtol, nmax=nmax,
        eta=eta, threads=threads, rank=rank, alpha=alpha, hss_method=hss_method,
        eps=eps)

    M = _dibem_factored_M(D, c, ID, method)
    set_cache!(dad; M=M, dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_rbf=rbf,
        dibem_method=method)
    return M
end

# Named entry points (all factored)
DIBEM_Hmat(dad; kwargs...)  = DIBEM_compressed(dad; method=:hmatrix, kwargs...)
DIBEM_HODLR(dad; kwargs...) = DIBEM_compressed(dad; method=:hodlr, kwargs...)
DIBEM_HSS(dad; kwargs...)   = DIBEM_compressed(dad; method=:hss, kwargs...)
DIBEM_HBS(dad; kwargs...)   = DIBEM_compressed(dad; method=:hbs, kwargs...)
DIBEM_H2(dad; kwargs...)    = DIBEM_compressed(dad; method=:h2, f_method=:dense, kwargs...)
DIBEM_FMM(dad; kwargs...)   = DIBEM_compressed(dad; method=:fmm, kwargs...)

# ---------------------------------------------------------------------------
# Unified DIBEM entry
# ---------------------------------------------------------------------------

"""
    DIBEM(dad; method=:dense, rbf=PHS(), kwargs...)

| `method` | `D = u*` compression | `M` type |
|----------|----------------------|----------|
| `:dense` | dense | dense `Matrix` (`Domain.jl`) |
| `:hmatrix` | H-matrix | [`DibemFactoredOperator`](@ref) |
| `:hodlr` | HODLR | factored |
| `:hss` / `:hbs` | HSS | factored |
| `:h2` | H² | factored |
| `:fmm` | FMM | factored |

All compressed methods share

```
M x = D (c ∘ x) + (ID − D c) ∘ x
```

with the same `D ∼ u*` kernel family as the BEM `G` matrix (square poles, no `w_j`).
"""
function DIBEM(dad::BEMdata{<:Laplace}; method::Symbol=:dense, rbf=PHS(), kwargs...)
    if method === :dense
        return DIBEM_dense(dad; rbf=rbf)
    elseif _dibem_struct_format(method) !== nothing || method in (:fmm, :FMM)
        return DIBEM_compressed(dad; method=method, rbf=rbf, kwargs...)
    else
        throw(ArgumentError(
            "DIBEM method must be :dense, :hmatrix, :hodlr, :hss, :hbs, :h2, or :fmm; got $method"))
    end
end

const dibem! = DIBEM
