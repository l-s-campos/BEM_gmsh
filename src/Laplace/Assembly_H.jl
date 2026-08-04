# Hierarchical H,G assembly — factored form (same idea as DIBEM)
#
#   Bare kernels (geometry only, from fundamental):
#     Du_ij = u*(x_i, x_j)           single layer   (G factor)
#     Dq_ij = ∂u*/∂n_j (x_i, x_j)    double layer   (H factor)
#
#   Quadrature weights w_j on boundary (0 on internal cols of H):
#     G x = Du (w ∘ x) + diag_G ∘ x     (G is nt×n)
#     H x = Dq (w∘x) + diag_H ∘ x     (H is nt×nt; w=0 on internal)
#
# Compression applies only to Du, Dq. Weights and free-term diagonals are
# outside — same pattern as DIBEM  M x = D (c ∘ x) + diag ∘ x.
#
export H_G_Hmat, corrige_diagonais!, MixedBCOperator, ColWeightedOp
export node_weights
# all_points / point live in Structures.jl

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

# ---------------------------------------------------------------------------
# Bare kernels (no weights) — Fundamental.jl
# ---------------------------------------------------------------------------

"""
Double-layer bare kernel: ``(Dq)_{ij} = ∂u*/∂n_j(x_i,x_j)`` (no `w_j`).
Internal columns and the diagonal are 0 (free term later).
"""
struct LaplaceDqKernel{P,Prop} <: AbstractMatrix{Float64}
    points::Vector{P}
    normals::Vector{P}
    props::Prop
    n_boundary::Int
end
Base.size(K::LaplaceDqKernel) = (length(K.points), length(K.points))
function Base.getindex(K::LaplaceDqKernel, i::Int, j::Int)
    (i == j || j > K.n_boundary) && return 0.0
    r = K.points[j] - K.points[i]
    norm(r) < 1e-15 && return 0.0
    # fundamental.T = ∂u*/∂n_j  (double layer density kernel)
    return float(fundamental(K.props, r, K.normals[j]).T)
end

"""
Single-layer bare kernel on boundary columns: ``(Du)_{ij} = u*(x_i,x_j)`` (no `w_j`).
Size `(n_total × n_boundary)`.
"""
struct LaplaceDuKernel{P,Prop} <: AbstractMatrix{Float64}
    points::Vector{P}
    props::Prop
    n_boundary::Int
    n_dummy::P
end
Base.size(K::LaplaceDuKernel) = (length(K.points), K.n_boundary)
function Base.getindex(K::LaplaceDuKernel, i::Int, j::Int)
    i == j && return 0.0
    r = K.points[j] - K.points[i]
    norm(r) < 1e-15 && return 0.0
    return float(fundamental(K.props, r, K.n_dummy).U)
end

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
end

Base.size(A::ColWeightedOp) = (A.nrows, A.ncols)
Base.IndexStyle(::Type{<:ColWeightedOp}) = IndexCartesian()

function Base.getindex(A::ColWeightedOp, i::Int, j::Int)
    val = A.w[j] * A.K[i, j]
    if i == j && j <= length(A.d)
        val += A.d[i]
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
# H-matrix assembly (factored)
# ---------------------------------------------------------------------------

"""
    H_G_Hmat(dad; atol=1e-6, nmax=32, eta=3.0, threads=true, format=:H)

Assemble hierarchical ``H`` and ``G`` in **factored** form:

1. Compress bare kernels `Dq = ∂u*/∂n`, `Du = u*` (no quadrature weights)
2. Wrap with boundary weights `w`:
   - `H = ColWeightedOp(Dq, w_H, d_H)`  (`w_H = 0` on internal columns)
   - `G = ColWeightedOp(Du, w, d_G)`
3. [`corrige_diagonais!`](@ref) fills free-term diagonals `d_H`, `d_G`

`format` is forwarded to structured assembly of the bare kernels (`:H` default;
`:HODLR`, `:HSS`, `:H2` also work when square trees match).

Stores `H`, `G` in `dad.cache` and returns them.
"""
function H_G_Hmat(
    dad::BEMdata{<:Laplace};
    atol = 1e-6,
    rtol = 0.0,
    nmax = 32,
    eta = 3.0,
    threads = true,
    format::Symbol = :H,
    rank = typemax(Int),
    alpha = 1.0,
    hss_method = :dense,
)
    points = collect(all_points(dad))
    w = node_weights(dad)
    n = dad.n
    nt = length(points)
    props = dad.properties
    n_dummy = points[1] isa SVector{2} ? SVector(0.0, 1.0) : SVector(0.0, 0.0, 1.0)

    # column weights for H: boundary w, internal 0
    wH = zeros(nt)
    wH[1:n] .= w

    splitter = PrincipalComponentSplitter(; nmax=nmax)
    Xclt = ClusterTree(points, splitter)
    Yclt_H = ClusterTree(copy(points), splitter)
    Yclt_G = ClusterTree(collect(dad.Nodes), splitter)

    adm = StrongAdmissibilityStd(; eta=eta)
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)

    KDq = LaplaceDqKernel(points, dad.Normal, props, n)
    KDu = LaplaceDuKernel(points, props, n, n_dummy)

    fmt = format
    @info "H_G_Hmat factored" format=fmt nt n

    Dq, Du = _assemble_HG_bare(KDq, KDu, Xclt, Yclt_H, Yclt_G, fmt;
        adm=adm, comp=comp, threads=threads, rtol=rtol, rank=rank,
        alpha=alpha, hss_method=hss_method)

    H = ColWeightedOp(Dq, wH, zeros(nt), nt, nt)
    G = ColWeightedOp(Du, copy(w), zeros(nt), nt, n)

    corrige_diagonais!(dad, H, G)
    set_cache!(dad; H=H, G=G, H_bare=Dq, G_bare=Du, bem_weights=w)
    return H, G
end

function _assemble_HG_bare(KDq, KDu, Xclt, Yclt_H, Yclt_G, fmt::Symbol;
        adm, comp, threads, rtol, rank, alpha, hss_method)
    if fmt in (:H, :HMatrix, :hmatrix)
        Dq = assemble_hmatrix(KDq, Xclt, Yclt_H; adm=adm, comp=comp, threads=threads)
        Du = assemble_hmatrix(KDu, Xclt, Yclt_G; adm=adm, comp=comp, threads=threads)
        return Dq, Du
    end
    # structured square tree for H; G may be rectangular — use H-matrix path for G
    # if format is HODLR/HSS (those assemblers expect matching trees)
    treeH = Xclt
    if fmt in (:HODLR, :hodlr, :HSS, :hss, :HBS, :hbs)
        Dq = assemble_structured(KDq, treeH; format=fmt, adm=adm, comp=comp,
            threads=threads, rtol=rtol, rank=rank, method=hss_method, global_index=true)
        # rectangular G: fall back to H-matrix (row tree full, col tree boundary)
        Du = assemble_hmatrix(KDu, Xclt, Yclt_G; adm=adm, comp=comp, threads=threads)
        return Dq, Du
    elseif fmt in (:H2, :h2)
        # Double-layer needs n(y) on proxies — not available; use H-matrix for both.
        # (Single-layer Du could be H² via KernelMatrix(fundamental.U); kept uniform.)
        Dq = assemble_hmatrix(KDq, Xclt, Yclt_H; adm=adm, comp=comp, threads=threads)
        Du = assemble_hmatrix(KDu, Xclt, Yclt_G; adm=adm, comp=comp, threads=threads)
        return Dq, Du
    else
        throw(ArgumentError("H_G_Hmat format must be :H, :HODLR, :HSS, or :H2; got $fmt"))
    end
end

"""
    corrige_diagonais!(dad, H, G)

Fill free-term diagonals on [`ColWeightedOp`](@ref) (or classical `HMatrix`).

- `H`: constant-field identity → `d_i = -∑_j H_ij` (with current d=0)
- `G`: linear-field identity with `H`
"""
function corrige_diagonais!(dad::BEMdata{<:Laplace}, H::ColWeightedOp, G::ColWeightedOp)
    n = dad.n
    nt = dad.nt
    k = float(dad.properties.k)

    # H free term: row sums of off-diagonal part
    hsum = H * ones(nt)   # with d=0 → Dq*(wH.*1)
    @inbounds for i in 1:nt
        H.d[i] = -hsum[i]
    end

    pts = all_points(dad)
    if dad.dimension == 2
        xlin = [p[1] + p[2] for p in pts]
        qlin = [-k * (dad.Normal[j][1] + dad.Normal[j][2]) for j in 1:n]
    else
        xlin = [p[1] + p[2] + p[3] for p in pts]
        qlin = [-k * sum(dad.Normal[j]) for j in 1:n]
    end
    resid = H * xlin - G * qlin
    @inbounds for i in 1:n
        denom = qlin[i]
        G.d[i] = abs(denom) < 1e-14 ? 0.0 : resid[i] / denom
    end
    return nothing
end

# Classical HMatrix path (legacy)
function corrige_diagonais!(dad::BEMdata{<:Laplace}, Hmat::HMatrix, Gmat::HMatrix)
    n = dad.n
    nt = dad.nt
    k = float(dad.properties.k)

    hsum = Hmat * ones(nt)
    _set_diagonal!(Hmat) do i
        -hsum[i]
    end

    pts = all_points(dad)
    if dad.dimension == 2
        xlin = [p[1] + p[2] for p in pts]
        qlin = [-k * (dad.Normal[j][1] + dad.Normal[j][2]) for j in 1:n]
    else
        xlin = [p[1] + p[2] + p[3] for p in pts]
        qlin = [-k * sum(dad.Normal[j]) for j in 1:n]
    end
    resid = Hmat * xlin - Gmat * qlin
    _set_diagonal!(Gmat) do i
        i > n && return 0.0
        denom = qlin[i]
        abs(denom) < 1e-14 && return 0.0
        return resid[i] / denom
    end
    return nothing
end

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

# ---------------------------------------------------------------------------
# Mixed-BC linear operator for hierarchical / factored H,G
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

"""
Build RHS `b` consistent with [`MixedBCOperator`](@ref):
`b = -H * T_known + G * q_known`.
"""
function mixed_bc_rhs(H, G, dad::BEMdata{<:Laplace})
    Tknown = zeros(dad.nt)
    qknown = zeros(dad.n)
    @inbounds for j in 1:dad.n
        if dad.BC[j] == 0
            Tknown[j] = dad.BV[j]
        else
            qknown[j] = dad.BV[j]
        end
    end
    return -(H * Tknown) + (G * qknown)
end
