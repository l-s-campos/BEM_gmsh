# =============================================================================
# Anchor Net method (Cai, Nagy, Xi, SIMAX 2022 / arXiv:2102.05215)
# Used as geometric landmark selection for data-driven kernel compression
# (Cai et al. "dd" preprint in D:\dd difeng).
# =============================================================================

"""
    struct AdaptiveTensorGrid

Low-discrepancy set on a box via an *adaptive tensor grid*: nonnegative integer
`p` controls resolution through ``i_1+\\cdots+i_d = p+d`` with ``i_k\\ge 1``
(Section 4.4.1 of the anchor-net paper). Avoids the ``p^d`` explosion of full
tensor grids.
"""
struct AdaptiveTensorGrid
    p::Int
end

"""
    low_discrepancy_points(box, n; method=:adaptive_tensor) -> Vector{SVector}

Generate approximately `n` low-discrepancy points inside `box`.
"""
function low_discrepancy_points(
        box::HyperRectangle{N, T},
        n::Integer;
        method::Symbol = :auto,
    ) where {N, T}
    n = max(Int(n), 1)
    # adaptive tensor grids are best in low-d; Halton scales to high-d
    if method === :auto
        method = N <= 4 ? :adaptive_tensor : :halton
    end
    if method === :adaptive_tensor
        return _adaptive_tensor_grid(box, n)
    elseif method === :halton
        return _halton_in_box(box, n)
    else
        throw(ArgumentError("unknown low-discrepancy method $(repr(method))"))
    end
end

"""Choose p so that the adaptive tensor grid has about `n` nodes in dimension N."""
function _p_for_target_nodes(n::Int, ::Val{N}) where {N}
    # crude search: |nodes| grows like O((p+d choose d) * something); use DOF bound ((p+d)/d)^d
    p = 0
    while true
        est = _adaptive_tensor_dof_bound(p, N)
        est >= n && return p
        p += 1
        p > 10_000 && return p
    end
end

_adaptive_tensor_dof_bound(p::Int, d::Int) = round(Int, (p + d)^d / d^d)

"""
All compositions i₁+…+i_d = p+d with i_k ≥ 1, and the corresponding tensor-grid
nodes (uniform in each dimension) inside `box`.
"""
function _adaptive_tensor_grid(box::HyperRectangle{N, T}, ntarget::Int) where {N, T}
    lo = low_corner(box)
    hi = high_corner(box)
    # expand tiny boxes
    δ = hi .- lo
    lo_a = collect(Float64, lo)
    hi_a = collect(Float64, hi)
    for k in 1:N
        if δ[k] <= zero(T)
            lo_a[k] -= 1e-12
            hi_a[k] += 1e-12
        end
    end
    lo = SVector{N, T}(lo_a)
    hi = SVector{N, T}(hi_a)
    p = _p_for_target_nodes(ntarget, Val(N))
    # collect multi-indices with sum(i) = p+N, i_k ≥ 1
    target = p + N
    idxs = Vector{NTuple{N, Int}}()
    _compositions!(idxs, N, target)
    pts = SVector{N, Float64}[]
    for i_tup in idxs
        # for each multi-index, build 1D nodes and tensor product
        grids = ntuple(k -> _uniform_1d(Float64(lo[k]), Float64(hi[k]), i_tup[k]), N)
        _tensor_push!(pts, grids)
    end
    # if still short (small p), fall back to Halton fill
    if length(pts) < ntarget
        append!(pts, _halton_in_box(box, ntarget - length(pts)))
    end
    # subsample evenly if too many
    if length(pts) > ntarget
        step = length(pts) / ntarget
        pts = [pts[clamp(round(Int, (j - 0.5) * step), 1, length(pts))] for j in 1:ntarget]
    end
    return pts
end

function _uniform_1d(a::Float64, b::Float64, m::Int)
    m <= 1 && return [(a + b) / 2]
    return range(a, b; length = m) |> collect
end

function _compositions!(out, d::Int, target::Int, prefix = Int[])
    if d == 1
        push!(out, Tuple(vcat(prefix, target)))
        return out
    end
    # i_k ≥ 1 and remaining ≥ d-1
    for i in 1:(target - (d - 1))
        _compositions!(out, d - 1, target - i, vcat(prefix, i))
    end
    return out
end

function _tensor_push!(pts, grids::NTuple{N, Vector{Float64}}) where {N}
    # recursive cartesian product
    function rec(k, acc)
        if k > N
            push!(pts, SVector{N, Float64}(acc...))
            return
        end
        for v in grids[k]
            rec(k + 1, (acc..., v))
        end
    end
    rec(1, ())
    return pts
end

# ---- Halton (fallback / optional LDS) --------------------------------------

function _halton_in_box(box::HyperRectangle{N, T}, n::Int) where {N, T}
    lo = low_corner(box)
    hi = high_corner(box)
    δ = hi .- lo
    primes = _first_primes(N)
    pts = Vector{SVector{N, Float64}}(undef, n)
    @inbounds for j in 1:n
        coords = ntuple(k -> Float64(lo[k]) + Float64(δ[k]) * _halton(j, primes[k]), N)
        pts[j] = SVector{N, Float64}(coords)
    end
    return pts
end

function _halton(i::Int, base::Int)
    f, r, invb = 1.0, 0.0, 1 / base
    x = i
    while x > 0
        f *= invb
        r += f * (x % base)
        x = div(x, base)
    end
    return r
end

function _first_primes(n::Int)
    primes = Int[]
    k = 2
    while length(primes) < n
        if all(k % p != 0 for p in primes)
            push!(primes, k)
        end
        k += 1
    end
    return primes
end

# =============================================================================
# Algorithm 4.1 — Anchor net construction
# =============================================================================

"""
    anchor_net(X, m; lds=:adaptive_tensor) -> Vector{SVector}

Construct an anchor net ``\\mathcal{A}_X`` of nominal size `m` for point set `X`
(Algorithm 4.1 of Cai–Nagy–Xi).

1. Build a coarse low-discrepancy set ``T`` of size ``O(m)`` in the bounding box
2. Voronoi-partition ``X`` among sites in ``T``
3. In each nonempty cell, place a fine low-discrepancy set with size proportional
   to the cell volume
"""
function anchor_net(
        X::AbstractVector{<:SVector{N, T}},
        m::Integer;
        lds::Symbol = :auto,
        coarse_factor::Int = 2,
    ) where {N, T}
    n = length(X)
    m = max(Int(m), 1)
    n == 0 && return SVector{N, Float64}[]
    box0 = bounding_box(X)
    s = min(n, max(m * coarse_factor, m))
    Tset = low_discrepancy_points(box0, s; method = lds)
    s = length(Tset)
    # partition X by nearest t ∈ T
    groups = [Int[] for _ in 1:s]
    @inbounds for j in 1:n
        xj = X[j]
        best, bid = Inf, 1
        for k in 1:s
            d = _dist2(xj, Tset[k])
            if d < best
                best = d
                bid = k
            end
        end
        push!(groups[bid], j)
    end
    nonempty = findall(!isempty, groups)
    # volumes (Lebesgue measure of bounding boxes)
    vols = Float64[]
    boxes = HyperRectangle{N, T}[]
    for i in nonempty
        Gi = @view X[groups[i]]
        bi = bounding_box(Gi)
        push!(boxes, bi)
        push!(vols, _box_volume(bi))
    end
    Vtot = sum(vols)
    Vtot <= 0 && (Vtot = 1.0)
    # allocate fine nets proportional to volume
    A = SVector{N, Float64}[]
    remaining = m
    for (t, i) in enumerate(nonempty)
        Mi = t == length(nonempty) ? remaining :
             max(1, round(Int, m * vols[t] / Vtot))
        Mi = min(Mi, remaining)
        remaining -= Mi
        Mi <= 0 && continue
        append!(A, low_discrepancy_points(boxes[t], Mi; method = lds))
    end
    return A
end

_dist2(a::SVector{N}, b::SVector{N}) where {N} = sum(abs2, a .- b)

function _box_volume(box::HyperRectangle{N}) where {N}
    δ = high_corner(box) .- low_corner(box)
    v = 1.0
    for k in 1:N
        v *= max(Float64(δ[k]), 0.0)
    end
    return v
end

# =============================================================================
# Algorithm 4.2 — Anchor net method (landmark selection on the data)
# =============================================================================

"""
    anchor_net_sample(X, m; kwargs...) -> (indices, points)

Select `m` landmark **data points** from `X` by the anchor-net method
(Algorithm 4.2): build an anchor net, then snap each net node to its nearest
point of `X`. Complexity ``O(m d n)``.

Returns the unique indices into `X` and the corresponding points.
"""
function anchor_net_sample(
        X::AbstractVector{<:SVector{N, T}},
        m::Integer;
        kwargs...,
    ) where {N, T}
    n = length(X)
    m = clamp(Int(m), 1, max(n, 1))
    n == 0 && return Int[], SVector{N, T}[]
    AX = anchor_net(X, m; kwargs...)
    chosen = Int[]
    seen = falses(n)
    @inbounds for y in AX
        best, bid = Inf, 1
        for k in 1:n
            d = _dist2(X[k], y)
            if d < best
                best = d
                bid = k
            end
        end
        if !seen[bid]
            seen[bid] = true
            push!(chosen, bid)
        end
    end
    # if collisions reduced the count, fill with FPS-style farthest points
    while length(chosen) < m
        best_d, best_i = -1.0, 0
        for k in 1:n
            seen[k] && continue
            # distance to current set
            dmin = Inf
            for j in chosen
                dmin = min(dmin, _dist2(X[k], X[j]))
            end
            if dmin > best_d
                best_d = dmin
                best_i = k
            end
        end
        best_i == 0 && break
        seen[best_i] = true
        push!(chosen, best_i)
    end
    return chosen, X[chosen]
end

"""
    farthest_point_sample(X, m) -> (indices, points)

Farthest-point sampling (FPS) baseline, complexity ``O(m^2 d n)``.
"""
function farthest_point_sample(X::AbstractVector{<:SVector{N, T}}, m::Integer) where {N, T}
    n = length(X)
    m = clamp(Int(m), 1, max(n, 1))
    n == 0 && return Int[], SVector{N, T}[]
    # start from point closest to centroid
    μ = sum(X) / n
    _, i0 = findmin(i -> _dist2(X[i], μ), 1:n)
    chosen = [i0]
    min_d2 = [_dist2(X[i], X[i0]) for i in 1:n]
    min_d2[i0] = -1.0
    while length(chosen) < m
        i_next = argmax(min_d2)
        push!(chosen, i_next)
        @inbounds for i in 1:n
            min_d2[i] < 0 && continue
            min_d2[i] = min(min_d2[i], _dist2(X[i], X[i_next]))
        end
        min_d2[i_next] = -1.0
    end
    return chosen, X[chosen]
end

# =============================================================================
# Data-driven kernel compression (dd-preprint Algorithms fac1 / fac2)
# =============================================================================

"""
    struct DataDrivenLR{T}

One- or two-sided data-driven low-rank factor of a kernel matrix:
``K \\approx U B V'`` with factors stored densely.

- one-sided column skeleton: ``K \\approx U K[I,:]``  (`V = I`, `B` unused)
- two-sided: ``K \\approx K[:,J] pinv(K[I,J]) K[I,:]`` stored as `U,B,V`
"""
struct DataDrivenLR{T} <: AbstractMatrix{T}
    U::Matrix{T}
    B::Matrix{T}
    V::Matrix{T}
    row_idx::Vector{Int}
    col_idx::Vector{Int}
    mode::Symbol   # :one_sided or :two_sided
end

Base.size(A::DataDrivenLR) = (size(A.U, 1), size(A.V, 1))
Base.eltype(::DataDrivenLR{T}) where {T} = T

function LinearAlgebra.mul!(
        y::AbstractVector, A::DataDrivenLR, x::AbstractVector,
        α::Number = 1, β::Number = 0,
    )
    if A.mode === :one_sided
        # y = α U (V' x) + β y  with V' x = x[col_idx] if B empty? 
        # stored as U * K[I,:] so V holds K[I,:]' i.e. V = K[I,:]'
        tmp = A.V' * x
        return mul!(y, A.U, tmp, α, β)
    else
        tmp = A.V' * x
        tmp = A.B * tmp
        return mul!(y, A.U, tmp, α, β)
    end
end

function Base.Matrix(A::DataDrivenLR{T}) where {T}
    if A.mode === :one_sided
        return A.U * A.V'
    else
        return A.U * A.B * A.V'
    end
end

"""
    dd_onesided(K, X, Y; rank=r, method=:anchor_net, oversample=2)

One-sided data-driven factorization (dd-preprint Algorithm fac1):
1. Select ``r'`` landmarks ``S\\subset Y`` geometrically (`:anchor_net` or `:fps`)
2. Form ``K_{X S}`` and compute a stable ID / thin QR so
   ``K \\approx U K_{I Y}`` with ``|I|=r``.
"""
function dd_onesided(
        K::AbstractMatrix{T},
        X::AbstractVector{<:SVector},
        Y::AbstractVector{<:SVector};
        rank::Int = 20,
        method::Symbol = :anchor_net,
        oversample::Int = 2,
    ) where {T}
    m, n = size(K)
    @assert length(X) == m && length(Y) == n
    r = min(rank, m, n)
    r2 = min(n, max(r, oversample * r))
    # geometric selection on Y
    idxS = _select_indices(Y, r2, method)
    # K_XS : m × r2
    KXS = Matrix{T}(undef, m, length(idxS))
    getblock!(KXS, K, 1:m, idxS)
    # Fixed-rank row ID via column-pivoted QR on KXS'
    F = qr(KXS', ColumnNorm())
    ract = min(r, size(KXS, 1), size(KXS, 2), length(F.p))
    Jloc = Vector{Int}(F.p[1:ract])
    rest = setdiff(collect(1:m), Jloc)
    if isempty(rest)
        U = Matrix{T}(I, m, ract)
    else
        E = Matrix{T}(KXS[rest, :] / KXS[Jloc, :])
        P = vcat(Jloc, rest)
        Utmp = [Matrix{T}(I, ract, ract); E]
        U = zeros(T, m, ract)
        U[P, :] = Utmp
    end
    # V' = K[I, :]  so V = K[I,:]'
    KIY = Matrix{T}(undef, length(Jloc), n)
    getblock!(KIY, K, Jloc, 1:n)
    V = Matrix{T}(KIY')   # n × r
    return DataDrivenLR{T}(U, Matrix{T}(I, size(U, 2), size(U, 2)), V, Jloc, idxS, :one_sided)
end

"""
    dd_twosided(K, X, Y; rank=r, method=:anchor_net)

Two-sided data-driven factorization (dd-preprint Algorithm fac2):
``K \\approx K_{X S_2}\\,K_{S_1 S_2}^{+}\\,K_{S_1 Y}`` with geometric subsets.
"""
function dd_twosided(
        K::AbstractMatrix{T},
        X::AbstractVector{<:SVector},
        Y::AbstractVector{<:SVector};
        rank::Int = 20,
        method::Symbol = :anchor_net,
    ) where {T}
    m, n = size(K)
    @assert length(X) == m && length(Y) == n
    r = min(rank, m, n)
    I1 = _select_indices(X, r, method)
    I2 = _select_indices(Y, r, method)
    K_XS2 = Matrix{T}(undef, m, length(I2))
    getblock!(K_XS2, K, 1:m, I2)
    K_S1Y = Matrix{T}(undef, length(I1), n)
    getblock!(K_S1Y, K, I1, 1:n)
    K_S1S2 = Matrix{T}(undef, length(I1), length(I2))
    getblock!(K_S1S2, K, I1, I2)
    # stable pseudoinverse via truncated SVD
    F = svd(K_S1S2)
    tol = eps(real(T)) * maximum(F.S) * max(size(K_S1S2)...)
    k = count(s -> s > tol, F.S)
    k = max(k, 1)
    B = F.V[:, 1:k] * Diagonal(inv.(F.S[1:k])) * F.U[:, 1:k]'
    return DataDrivenLR{T}(K_XS2, B, Matrix{T}(K_S1Y'), I1, I2, :two_sided)
end

function _select_indices(X::AbstractVector{<:SVector}, r::Int, method::Symbol)
    if method === :anchor_net
        idx, _ = anchor_net_sample(X, r)
        return idx
    elseif method === :fps
        idx, _ = farthest_point_sample(X, r)
        return idx
    elseif method === :uniform || method === :random
        n = length(X)
        return sort(randperm(n)[1:min(r, n)])
    else
        throw(ArgumentError("unknown selection method $(repr(method)); use :anchor_net, :fps, or :uniform"))
    end
end

"""
    AnchorNetCompressor(; rank=20, method=:anchor_net, side=:one)

Callable compressor compatible with HMatrices leaf compression style:
`comp(K, rowtree, coltree)` returns an [`RkMatrix`](@ref) built by data-driven
selection of landmarks with the anchor-net (or FPS) method.
"""
Base.@kwdef struct AnchorNetCompressor
    rank::Int = 20
    method::Symbol = :anchor_net
    side::Symbol = :one   # :one or :two
    oversample::Int = 2
end

function (comp::AnchorNetCompressor)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    X = collect(elements(rowtree))
    Y = collect(elements(coltree))
    # local index ranges
    ir = index_range(rowtree)
    jr = index_range(coltree)
    # view of K on the block
    Kb = _BlockView(K, ir, jr)
    r = min(comp.rank, length(ir), length(jr))
    if comp.side === :one
        F = dd_onesided(Kb, X, Y; rank = r, method = comp.method, oversample = comp.oversample)
        return RkMatrix(F.U, F.V)
    else
        F = dd_twosided(Kb, X, Y; rank = r, method = comp.method)
        # merge B into factors: U_new = U*B, V stays
        return RkMatrix(Matrix(F.U * F.B), F.V)
    end
end

function (comp::AnchorNetCompressor)(K, irange::AbstractRange, jrange::AbstractRange, bufs = nothing)
    # without geometry, fall back to PartialACA on the block
    return PartialACA(; rank = comp.rank)(K, irange, jrange, bufs)
end

# thin matrix view for a sub-block without copying K
struct _BlockView{K, T} <: AbstractMatrix{T}
    parent::K
    ir::UnitRange{Int}
    jr::UnitRange{Int}
end
_BlockView(K, ir, jr) = _BlockView{typeof(K), eltype(K)}(K, ir, jr)
Base.size(B::_BlockView) = (length(B.ir), length(B.jr))
Base.getindex(B::_BlockView, i::Int, j::Int) = B.parent[B.ir[i], B.jr[j]]
function getblock!(out, B::_BlockView, irange_, jrange_)
    ir = irange_ isa Colon ? (1:size(B, 1)) : irange_
    jr = jrange_ isa Colon ? (1:size(B, 2)) : jrange_
    for (jl, j) in enumerate(jr), (il, i) in enumerate(ir)
        out[il, jl] = B[i, j]
    end
    return out
end
