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
export correct_nearfield!, node_weights, free_term
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
    nearfield::Bool = true,
    near_factor::Real = 2.0,
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

    # Near-field: full element integration vs pointwise when r < near_factor * L
    nearfield && correct_nearfield!(dad, H, G; factor=near_factor)
    corrige_diagonais!(dad, H, G)
    set_cache!(dad; H=H, G=G, H_bare=Dq, G_bare=Du, bem_weights=w)
    return H, G
end

"""
    correct_nearfield!(dad, H, G; factor=2.0)

For compressed / factored [`ColWeightedOp`](@ref) operators, correct all
entries involving element `e` when

```
min_k ‖x_source − x_node_k‖ < factor · Length(e)
```

(same criterion as dense assembly). Correction per column `j` of element `e`:

```
ΔH_ij = h_j^∫ − (∂u*/∂n_j) w_j
ΔG_ij = g_j^∫ − u*_j w_j
```

where `(h^∫, g^∫) =` [`integrate_element`](@ref) (Dumont when singular).

- **G**: use integrated values for near/self columns (including ``G_ii`` from
  the singular single-layer integral — there is no free term in G).
- **H**: use integrated double-layer values (self-element is 0 only if the
  element is straight; curved → nonzero). The jump/free term
  c = 1/2 or 1 is **not** inside Dumont; it is added on the diagonal by
  [`corrige_diagonais!`](@ref) / [`free_term`](@ref) as H_ii = -c.

Pass `nearfield=false` to [`H_G_Hmat`](@ref) to skip.
"""
function correct_nearfield!(
    dad::BEMdata{<:Laplace},
    H::ColWeightedOp,
    G::ColWeightedOp;
    factor::Real = 2.0,
)
    nt = H.nrows
    n = G.ncols
    n == dad.n || throw(DimensionMismatch("G columns"))
    # COO accumulators
    Ih = Int[]; Jh = Int[]; Vh = Float64[]
    Ig = Int[]; Jg = Int[]; Vg = Float64[]

    elems = dad.elements
    Xel = [[dad.Nodes[j] for j in elem.index] for elem in elems]

    nn_max = maximum(length(el) for el in elems; init=3)
    hloc = zeros(nn_max)
    gloc = zeros(nn_max)

    @inbounds for i in 1:nt
        pf = point(dad, i)
        for (ej, elem) in enumerate(elems)
            xj = Xel[ej]
            # same trigger as Assembly_full dense path
            r0 = euclidean(pf, xj[1])
            r0 < factor * elem.Length || continue

            nn = length(elem)
            fill!(hloc, 0.0)
            fill!(gloc, 0.0)
            hv = @view hloc[1:nn]
            gv = @view gloc[1:nn]
            integrate_element(dad, elem, xj, pf, hv, gv)

            for k in 1:nn
                j = elem.index[k]
                j > n && continue
                # Pointwise far kernel×w (0 on diagonal — bare K has K_ii=0).
                # Do not index H.K/G.K (HMatrix disables getindex).
                hp = 0.0
                gp = 0.0
                if i != j
                    r_node = dad.Nodes[j] - pf
                    n_node = dad.Normal[j]
                    U, Tker = fundamental(dad, r_node, n_node)
                    hp = Tker * H.w[j]
                    gp = U * G.w[j]
                end
                # Full element integral (Dumont/sinh). On a *straight* self-element
                # h_self≈0; on curved geometry h_self can be nonzero. Free term
                # c is NOT in Dumont — it is added on H.d via free_term().
                dh = hv[k] - hp
                dg = gv[k] - gp
                if abs(dh) > 0
                    push!(Ih, i); push!(Jh, j); push!(Vh, dh)
                end
                if abs(dg) > 0
                    push!(Ig, i); push!(Jg, j); push!(Vg, dg)
                end
            end
        end
    end

    H.corr = isempty(Vh) ? spzeros(nt, nt) : sparse(Ih, Jh, Vh, nt, nt)
    G.corr = isempty(Vg) ? spzeros(nt, n) : sparse(Ig, Jg, Vg, nt, n)
    return nothing
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
    free_term(dad, i) -> Float64

Diagonal free-term entry placed in ``H_ii`` for discontinuous collocation.

Analytical jump coefficients are ``c=1/2`` (smooth boundary) and ``c=1``
(domain). This code stores **``H_ii = -c``** so that, with the double-layer
sign convention of [`fundamental`](@ref), constant fields satisfy
``∑_j H_ij ≈ 0`` (same as the classical row-sum fix).

| location | ``c`` | ``H_ii = free_term`` |
|----------|-------|----------------------|
| boundary | ``1/2`` | ``-1/2`` |
| internal | ``1`` | ``-1`` |
"""
@inline free_term(dad::BEMdata, i::Integer) = i <= dad.n ? -0.5 : -1.0

"""
    corrige_diagonais!(dad, H, G; free_term=:explicit)

Diagonal coefficients for [`ColWeightedOp`](@ref) / `HMatrix`.

# Keywords
- `free_term = :explicit` (default): discontinuous elements —
  ``H_ii = 1/2`` on the boundary, ``H_ii = 1`` inside the domain.
- `free_term = :rowsum`: ``H_ii = -∑_{j≠i} H_ij`` (classical identity; useful if
  far-field is approximate and you want to enforce ``H 1 = 0`` exactly).

`G_ii` is always from the linear-field identity (not a free term).
"""
function corrige_diagonais!(dad::BEMdata{<:Laplace}, H::ColWeightedOp, G::ColWeightedOp;
        free_term::Symbol = :explicit)
    n = dad.n
    nt = dad.nt
    k = float(dad.properties.k)

    # --- H diagonal (free term) ---
    if free_term === :explicit
        # discontinuous collocation: c = 1/2 (boundary), c = 1 (domain)
        @inbounds for i in 1:nt
            H.d[i] = BEM.free_term(dad, i)
        end
    elseif free_term === :rowsum
        fill!(H.d, 0.0)
        hsum = H * ones(nt)   # off-diagonal + corr only
        @inbounds for i in 1:nt
            H.d[i] = -hsum[i]
        end
    else
        throw(ArgumentError("free_term must be :explicit or :rowsum, got $free_term"))
    end

    # --- G diagonal (linear field; not a free-term coefficient) ---
    pts = all_points(dad)
    if dad.dimension == 2
        xlin = [p[1] + p[2] for p in pts]
        qlin = [-k * (dad.Normal[j][1] + dad.Normal[j][2]) for j in 1:n]
    else
        xlin = [p[1] + p[2] + p[3] for p in pts]
        qlin = [-k * sum(dad.Normal[j]) for j in 1:n]
    end
    fill!(G.d, 0.0)
    resid = H * xlin - G * qlin
    @inbounds for i in 1:n
        denom = qlin[i]
        G.d[i] = abs(denom) < 1e-14 ? 0.0 : resid[i] / denom
    end
    return nothing
end

function corrige_diagonais!(dad::BEMdata{<:Laplace}, Hmat::HMatrix, Gmat::HMatrix;
        free_term::Symbol = :explicit)
    n = dad.n
    nt = dad.nt
    k = float(dad.properties.k)

    if free_term === :explicit
        _set_diagonal!(Hmat) do i
            BEM.free_term(dad, i)
        end
    elseif free_term === :rowsum
        # zero diagonal then row-sum (legacy H-matrix path)
        _set_diagonal!(Hmat) do _
            0.0
        end
        hsum = Hmat * ones(nt)
        _set_diagonal!(Hmat) do i
            -hsum[i]
        end
    else
        throw(ArgumentError("free_term must be :explicit or :rowsum, got $free_term"))
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
