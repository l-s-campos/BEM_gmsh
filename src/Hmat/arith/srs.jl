# Deterministic strong recursive skeletonization (Yesypenko–Martinsson /
# Minden–Ho–Ying): ID far-field of each box, eliminate redundant indices
# against neighbors, walk the tree upward. Inverse is a product of sparse
# factors, used as a GMRES left preconditioner via `ldiv!`.

"""One box elimination: redundant `R`, skeleton `S`, active neighbors `N`."""
struct SRSStep{T}
    R::Vector{Int}
    S::Vector{Int}
    N::Vector{Int}
    T::Matrix{T}             # |R| × |S|,  A[R, F] ≈ T * A[S, F]
    Frr::LU{T, Matrix{T}}    # LU of X_RR
    X_rest_R::Matrix{T}      # |S|+|N| × |R|
    X_R_rest::Matrix{T}      # |R| × |S|+|N|
end

"""
    SRSFactor

Approximate inverse from strong recursive skeletonization.
`ldiv!(F, x)` applies `F ≈ A⁻¹` in place (Krylov left preconditioner).
"""
struct SRSFactor{T}
    steps::Vector{SRSStep{T}}
    root_idx::Vector{Int}
    Froot::LU{T, Matrix{T}}
    n::Int
end

Base.size(F::SRSFactor) = (F.n, F.n)
Base.size(F::SRSFactor, d::Integer) = d == 1 || d == 2 ? F.n : 1
Base.eltype(::SRSFactor{T}) where {T} = T

function Base.show(io::IO, F::SRSFactor)
    nelim = sum(s -> length(s.R), F.steps; init = 0)
    return print(io, "SRSFactor{", eltype(F), "} n=", F.n,
        " steps=", length(F.steps), " eliminated=", nelim,
        " root=", length(F.root_idx))
end
Base.show(io::IO, ::MIME"text/plain", F::SRSFactor) = show(io, F)

# ---- interpolative decomposition (row ID) ------------------------------------

function _srs_row_id(A::AbstractMatrix{T}; rtol = 1e-6, rank = typemax(Int)) where {T}
    m, n = size(A)
    m == 0 && return Int[], Int[], zeros(T, 0, 0)
    n == 0 && return collect(1:m), Int[], zeros(T, m, 0)
    F = qr!(Matrix(transpose(A)), ColumnNorm())
    rd = abs.(diag(F.R))
    isempty(rd) && return collect(1:m), Int[], zeros(T, m, 0)
    τ = float(rtol) * rd[1]
    k = count(s -> s > τ, rd)
    k = clamp(k, 1, min(m, n, Int(rank), length(rd)))
    p = F.p
    Sloc = p[1:k]
    Rloc = p[(k + 1):end]
    isempty(Rloc) && return Int[], Sloc, zeros(T, 0, k)
    Tm = A[Rloc, :] / A[Sloc, :]
    return Rloc, Sloc, Matrix{T}(Tm)
end

# ---- kernel / modified near-field cache --------------------------------------

"""Overlay of modified entries on a kernel `K`. Far field stays `K[i,j]`."""
struct SRSCache{T, K}
    n::Int
    K::K
    mod::Dict{UInt64, T}
end

@inline _srs_key(i::Int, j::Int, n::Int) = UInt64(i) + UInt64(n) * UInt64(j - 1)

function _srs_get(c::SRSCache{T}, i::Int, j::Int) where {T}
    got = get(c.mod, _srs_key(i, j, c.n), nothing)
    got !== nothing && return got
    return T(c.K[i, j])
end

function _srs_block(c::SRSCache{T}, I::Vector{Int}, J::Vector{Int}) where {T}
    M = if c.K isa KernelMatrix
        _rskelf_block(c.K, I, J)
    else
        M0 = Matrix{T}(undef, length(I), length(J))
        @inbounds for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
            M0[ii, jj] = T(c.K[i, j])
        end
        M0
    end
    if !isempty(c.mod)
        n = c.n
        @inbounds for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
            got = get(c.mod, _srs_key(i, j, n), nothing)
            got !== nothing && (M[ii, jj] = got)
        end
    end
    return M
end

function _srs_set!(c::SRSCache{T}, I::Vector{Int}, J::Vector{Int}, M::AbstractMatrix) where {T}
    @inbounds for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        c.mod[_srs_key(i, j, c.n)] = T(M[ii, jj])
    end
    return c
end

_srs_glob(box::ClusterTree) = collect(loc2glob(box)[index_range(box)])

function _srs_active_in(box::ClusterTree, active::AbstractVector{Bool})
    g = _srs_glob(box)
    return filter(i -> active[i], g)
end

# ---- one box -----------------------------------------------------------------

function _srs_process_box!(
        cache::SRSCache{T},
        active::AbstractVector{Bool},
        Bidx::Vector{Int},
        Nidx::Vector{Int},
        Fidx::Vector{Int};
        rtol,
        rank,
    ) where {T}
    length(Bidx) < 2 && return nothing
    # Far sample for ID: IL points, padded from remaining active if needed
    Fuse = Fidx
    if length(Fuse) < 2 * length(Bidx)
        extra = Int[]
        for i in eachindex(active)
            active[i] || continue
            (i in Bidx || i in Nidx) && continue
            push!(extra, i)
            length(Fuse) + length(extra) >= 8 * length(Bidx) && break
        end
        Fuse = isempty(Fuse) ? extra : vcat(Fuse, extra)
    end
    isempty(Fuse) && return nothing

    ABF = _srs_block(cache, Bidx, Fuse)
    AFB = _srs_block(cache, Fuse, Bidx)
    Yid = hcat(ABF, transpose(AFB))
    Rloc, Sloc, Tt = _srs_row_id(Yid; rtol = rtol, rank = rank)
    isempty(Rloc) && return nothing
    Ridx = Bidx[Rloc]
    Sidx = Bidx[Sloc]
    rest = vcat(Sidx, Nidx)
    isempty(rest) && return nothing

    ARR0 = _srs_block(cache, Ridx, Ridx)
    ARS0 = _srs_block(cache, Ridx, Sidx)
    ASR0 = _srs_block(cache, Sidx, Ridx)
    ASS = _srs_block(cache, Sidx, Sidx)
    ARN0 = _srs_block(cache, Ridx, Nidx)
    ANR0 = _srs_block(cache, Nidx, Ridx)
    ASN = _srs_block(cache, Sidx, Nidx)
    ANS = _srs_block(cache, Nidx, Sidx)
    ANN = _srs_block(cache, Nidx, Nidx)
    TtS = transpose(Tt)
    # F (right): columns R ← R − S T′
    ARR1 = ARR0 - ARS0 * TtS
    ASR1 = ASR0 - ASS * TtS
    ANR1 = ANR0 - ANS * TtS
    # E (left): rows R ← R − T S
    ARR = ARR1 - Tt * ASR1
    ARS = ARS0 - Tt * ASS
    ARN = ARN0 - Tt * ASN
    ASR, ANR = ASR1, ANR1

    Frr = try
        lu(ARR)
    catch
        lu(ARR + T(1e-12) * I)
    end
    X_R_rest = hcat(ARS, ARN)
    X_rest_R = vcat(ASR, ANR)
    # Schur update on rest×rest: Xrr -= X_rest_R * (ARR \ X_R_rest)
    Urest = Frr \ X_R_rest
    Sch = vcat(hcat(ASS, ASN), hcat(ANS, ANN)) - X_rest_R * Urest
    ns = length(Sidx)
    nn = length(Nidx)
    _srs_set!(cache, Sidx, Sidx, Sch[1:ns, 1:ns])
    if nn > 0
        _srs_set!(cache, Sidx, Nidx, Sch[1:ns, (ns + 1):end])
        _srs_set!(cache, Nidx, Sidx, Sch[(ns + 1):end, 1:ns])
        _srs_set!(cache, Nidx, Nidx, Sch[(ns + 1):end, (ns + 1):end])
    end
    for i in Ridx
        active[i] = false
    end
    return SRSStep{T}(Ridx, Sidx, Nidx, Tt, Frr, X_rest_R, X_R_rest)
end

# ---- factor ------------------------------------------------------------------

"""
    srs_factor(K, tree; rtol=1e-6, rank=typemax(Int)) -> SRSFactor

Deterministic strong recursive skeletonization using kernel entries `K[i,j]`
(global point order) and the geometric neighbor / interaction lists of `tree`.
"""
function srs_factor(
        K::AbstractMatrix{T},
        tree::ClusterTree;
        rtol = 1e-6,
        rank = typemax(Int),
    ) where {T}
    n = size(K, 1)
    size(K, 2) == n || throw(DimensionMismatch("srs_factor needs a square kernel"))
    node_id(tree) == 0 && assign_node_ids!(tree)
    neigh, il, id2 = neighbor_il_lists(tree)
    levels = nodes_by_depth(tree)
    active = trues(n)
    cache = SRSCache{T, typeof(K)}(n, K, Dict{UInt64, T}())
    steps = SRSStep{T}[]

    for lev in Iterators.reverse(levels)
        isempty(lev) && continue
        depth(first(lev)) == 0 && continue  # skip root; handled as dense tail
        for box in lev
            id = node_id(box)
            Bidx = _srs_active_in(box, active)
            length(Bidx) < 2 && continue
            Nidx = Int[]
            for qid in neigh[id]
                append!(Nidx, _srs_active_in(id2[qid], active))
            end
            unique!(Nidx)
            filter!(i -> !(i in Bidx), Nidx)
            Fidx = Int[]
            for qid in il[id]
                append!(Fidx, _srs_active_in(id2[qid], active))
            end
            unique!(Fidx)
            step = _srs_process_box!(cache, active, Bidx, Nidx, Fidx; rtol = rtol, rank = rank)
            step !== nothing && push!(steps, step)
        end
    end

    root_idx = findall(active)
    if isempty(root_idx)
        root_idx = Int[1]
        active[1] = true
    end
    Droot = _srs_block(cache, root_idx, root_idx)
    Droot = Droot + T(1e-14) * I
    Froot = lu(Droot)
    return SRSFactor{T}(steps, root_idx, Froot, n)
end

# ---- apply inverse -----------------------------------------------------------

function _srs_apply_E!(s::SRSStep, x::AbstractVector)
    # x_R -= T * x_S
    isempty(s.R) && return x
    xR = view(x, s.R)
    xS = view(x, s.S)
    mul!(xR, s.T, xS, -one(eltype(x)), true)
    return x
end

function _srs_apply_L!(s::SRSStep, x::AbstractVector)
    isempty(s.R) && return x
    rest = vcat(s.S, s.N)
    isempty(rest) && return x
    u = s.Frr \ view(x, s.R)
    mul!(view(x, rest), s.X_rest_R, u, -one(eltype(x)), true)
    return x
end

function _srs_apply_U!(s::SRSStep, x::AbstractVector)
    isempty(s.R) && return x
    rest = vcat(s.S, s.N)
    isempty(rest) && return x
    v = s.X_R_rest * view(x, rest)
    ldiv!(s.Frr, v)
    view(x, s.R) .-= v
    return x
end

function _srs_apply_F!(s::SRSStep, x::AbstractVector)
    # x_S -= T' * x_R
    isempty(s.S) && return x
    mul!(view(x, s.S), adjoint(s.T), view(x, s.R), -one(eltype(x)), true)
    return x
end

function LinearAlgebra.ldiv!(F::SRSFactor{T}, x::AbstractVector{T}) where {T}
    length(x) == F.n || throw(DimensionMismatch())
    for s in F.steps
        _srs_apply_E!(s, x)
        _srs_apply_L!(s, x)
    end
    for s in F.steps
        ldiv!(s.Frr, view(x, s.R))
    end
    ldiv!(F.Froot, view(x, F.root_idx))
    for s in Iterators.reverse(F.steps)
        _srs_apply_U!(s, x)
        _srs_apply_F!(s, x)
    end
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, F::SRSFactor, x::AbstractVector)
    y === x || copyto!(y, x)
    return ldiv!(F, y)
end

function Base.:\(F::SRSFactor, b::AbstractVector)
    x = copy(b)
    ldiv!(F, x)
    return x
end

# ---- randomized sketches (matrix-free RSRS) ----------------------------------

# E = [I -T; 0 I] on [R,S].  F = [I 0; -T' I] = E'.
# V^{-1} = L E,  W = U^{-1} F^{-1}.
# Sketch updates (Yesypenko–Martinsson Alg. 1):
#   Y ← L E Y,   Ω ← U^{-1} F^{-1} Ω
#   Z ← U' F' Z, Ψ ← L^{-*} E^{-*} Ψ

function _srs_E!(s::SRSStep, M::AbstractMatrix)
    mul!(view(M, s.R, :), s.T, view(M, s.S, :), -one(eltype(M)), true)
    return M
end
function _srs_Finv!(s::SRSStep, M::AbstractMatrix)
    mul!(view(M, s.S, :), adjoint(s.T), view(M, s.R, :), one(eltype(M)), true)
    return M
end
function _srs_Fadj!(s::SRSStep, M::AbstractMatrix)
    mul!(view(M, s.R, :), s.T, view(M, s.S, :), -one(eltype(M)), true)
    return M
end
function _srs_Einvadj!(s::SRSStep, M::AbstractMatrix)
    mul!(view(M, s.S, :), adjoint(s.T), view(M, s.R, :), one(eltype(M)), true)
    return M
end
function _srs_rest(s::SRSStep)
    return vcat(s.S, s.N)
end
function _srs_L!(s::SRSStep, M::AbstractMatrix)
    rest = _srs_rest(s)
    isempty(rest) && return M
    U = s.Frr \ Matrix(view(M, s.R, :))
    mul!(view(M, rest, :), s.X_rest_R, U, -one(eltype(M)), true)
    return M
end
function _srs_Uinv!(s::SRSStep, M::AbstractMatrix)
    rest = _srs_rest(s)
    isempty(rest) && return M
    V = s.X_R_rest * view(M, rest, :)
    ldiv!(s.Frr, V)
    view(M, s.R, :) .+= V
    return M
end
function _srs_Uadj!(s::SRSStep, M::AbstractMatrix)
    rest = _srs_rest(s)
    isempty(rest) && return M
    W = s.Frr' \ Matrix(view(M, s.R, :))
    mul!(view(M, rest, :), adjoint(s.X_R_rest), W, -one(eltype(M)), true)
    return M
end
function _srs_Linvadj!(s::SRSStep, M::AbstractMatrix)
    rest = _srs_rest(s)
    isempty(rest) && return M
    W = adjoint(s.X_rest_R) * view(M, rest, :)
    ldiv!(s.Frr', W)
    view(M, s.R, :) .+= W
    return M
end

function _srs_sample(A, Ω::AbstractMatrix{T}) where {T}
    n, s = size(Ω)
    Y = Matrix{T}(undef, size(A, 1), s)
    x = Vector{T}(undef, n)
    y = Vector{T}(undef, size(A, 1))
    @inbounds for j in 1:s
        copyto!(x, view(Ω, :, j))
        mul!(y, A, x)
        copyto!(view(Y, :, j), y)
    end
    return Y
end

function _srs_null(M::AbstractMatrix{T}; keep::Int = 0, rtol = 1e-12) where {T}
    p, sdim = size(M)
    (p == 0 || sdim == 0) && return zeros(T, sdim, 0)
    if p >= sdim
        F = svd(M; full = true)
        σ1 = isempty(F.S) ? zero(real(T)) : F.S[1]
        r = count(σ -> σ > rtol * σ1, F.S)
        nnull = sdim - r
        keep > 0 && (nnull = min(nnull, keep))
        nnull < 1 && return zeros(T, sdim, 0)
        return F.V[:, (r + 1):(r + nnull)]
    end
    # Fat Ω_{B∪N}: nullspace from trailing Q of pivoted QR of Ω'.
    F = qr(M', ColumnNorm())
    rd = abs.(diag(F.R))
    σ1 = isempty(rd) ? zero(real(T)) : rd[1]
    r = count(σ -> σ > rtol * σ1, rd)
    r = clamp(r, 0, p)
    nnull = sdim - r
    keep > 0 && (nnull = min(nnull, keep))
    nnull < 1 && return zeros(T, sdim, 0)
    E = zeros(T, sdim, nnull)
    @inbounds for j in 1:nnull
        E[r + j, j] = one(T)
    end
    return lmul!(F.Q, E)
end

"""X ≈ Y Ω† with Ω of size k×s, Y of size m×s (block extraction)."""
function _srs_extract(Y::AbstractMatrix{T}, Ω::AbstractMatrix{T}; rtol = 1e-12) where {T}
    size(Y, 2) == size(Ω, 2) || throw(DimensionMismatch("sketch width mismatch"))
    m, k = size(Y, 1), size(Ω, 1)
    (m == 0 || k == 0 || size(Ω, 2) == 0) && return zeros(T, m, k)
    # `rtol` reserved for a truncated SVD pinv; `/` is the full-row least-squares †.
    isfinite(rtol) || throw(ArgumentError("rtol must be finite"))
    return Y / Ω
end

function _srs_max_bn(levels, neigh, id2)
    mx = 0
    for lev in levels
        isempty(lev) && continue
        depth(first(lev)) == 0 && continue
        for box in lev
            nb = length(index_range(box))
            nn = 0
            @inbounds for qid in neigh[node_id(box)]
                nn += length(index_range(id2[qid]))
            end
            mx = max(mx, nb + nn)
        end
    end
    return mx
end

"""
    srs_factor_matvec(A, tree; rtol=1e-6, rank=40, p=40, samples=0) -> SRSFactor

Matrix-free RSRS (Yesypenko–Martinsson Alg. 1). One pair of Gaussian sketches
`Y = A Ω`, `Z = A' Ψ` is drawn, then updated in place with `E, F, L, U` after
each box. Far-field ID uses block nullification; near blocks use block extraction.

`A` must support `mul!(y, A, x)` in the same index order as `tree` (`loc2glob`).
Typical `A`: H²/`NNCAMatrix`, or an FMM kernel matrix. Singular kernels (zero
diagonal) should be wrapped with a diagonal shift before factoring and solving.

`samples ≤ 0` chooses `s = min(N-1, max|B∪N| + k + p)` so each leaf near
field has a nontrivial nullspace (paper eq. (39)). Pass `samples=40*rank+p`
for the 2D volume constant in the paper. For nonsymmetric operators set
`symmetric=false` (needs `A'`).
"""
function srs_factor_matvec(
        A::AbstractMatrix{T},
        tree::ClusterTree;
        rtol = 1e-6,
        rank = 40,
        p = 40,
        samples::Integer = 0,
        symmetric::Bool = true,
    ) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("srs_factor_matvec needs a square operator"))
    k = Int(rank)
    po = Int(p)
    node_id(tree) == 0 && assign_node_ids!(tree)
    neigh, _, id2 = neighbor_il_lists(tree)
    levels = nodes_by_depth(tree)
    maxbn = _srs_max_bn(levels, neigh, id2)
    s_need = maxbn + k + po
    s = samples > 0 ? min(n - 1, Int(samples)) : min(n - 1, s_need)
    s < 8 && throw(ArgumentError("srs_factor_matvec needs more samples (got s=$s)"))
    Ω = randn(T, n, s)
    Ψ = randn(T, n, s)
    Y = _srs_sample(A, Ω)
    Z = symmetric ? _srs_sample(A, Ψ) : _srs_sample(A', Ψ)
    active = trues(n)
    steps = SRSStep{T}[]
    extr_rtol = max(float(rtol), 1e-12)

    for lev in Iterators.reverse(levels)
        isempty(lev) && continue
        depth(first(lev)) == 0 && continue
        for box in lev
            id = node_id(box)
            Bidx = _srs_active_in(box, active)
            length(Bidx) < 2 && continue
            Nidx = Int[]
            for qid in neigh[id]
                append!(Nidx, _srs_active_in(id2[qid], active))
            end
            unique!(Nidx)
            filter!(i -> !(i in Bidx), Nidx)
            BN = vcat(Bidx, Nidx)
            length(BN) >= s && continue
            Nb = _srs_null(Ω[BN, :]; keep = k + po, rtol = extr_rtol)
            Np = _srs_null(Ψ[BN, :]; keep = k + po, rtol = extr_rtol)
            size(Nb, 2) < 2 && continue
            ncol = min(size(Nb, 2), size(Np, 2))
            # Alg. 1 steps 5–6: id(Y'_B + Z'_B) after block nullification
            Yid = Y[Bidx, :] * view(Nb, :, 1:ncol) + Z[Bidx, :] * view(Np, :, 1:ncol)
            Rloc, Sloc, Tt = _srs_row_id(Yid; rtol = rtol, rank = k)
            isempty(Rloc) && continue
            Ridx = Bidx[Rloc]
            Sidx = Bidx[Sloc]
            rest = vcat(Sidx, Nidx)
            isempty(rest) && continue
            Fdummy = lu(Matrix{T}(I, length(Ridx), length(Ridx)))
            step0 = SRSStep{T}(Ridx, Sidx, Nidx, Tt, Fdummy,
                zeros(T, length(rest), length(Ridx)),
                zeros(T, length(Ridx), length(rest)))
            # Alg. 1 step 7: Ŷ = E Y, Ω̂ = F^{-1} Ω, Ẑ = F' Z, Ψ̂ = E^{-*} Ψ
            _srs_E!(step0, Y)
            _srs_Finv!(step0, Ω)
            _srs_Fadj!(step0, Z)
            _srs_Einvadj!(step0, Ψ)
            # Alg. 1 step 8: X_{R,BN} ≈ Ŷ_R Ω̂_{BN}^†
            XR_BN = _srs_extract(Y[Ridx, :], Ω[BN, :]; rtol = extr_rtol)
            XBN_R = adjoint(_srs_extract(Z[Ridx, :], Ψ[BN, :]; rtol = extr_rtol))
            pos = Dict{Int, Int}(g => i for (i, g) in enumerate(BN))
            rpos = [pos[i] for i in Ridx]
            restpos = [pos[i] for i in rest]
            ARR = Matrix{T}(XR_BN[:, rpos])
            if symmetric && length(rpos) > 0
                ARR .= (ARR + XBN_R[rpos, :]) ./ 2
            end
            X_R_rest = Matrix{T}(XR_BN[:, restpos])
            X_rest_R = Matrix{T}(XBN_R[restpos, :])
            Frr = try
                lu(ARR)
            catch
                lu(ARR + T(1e-12) * I)
            end
            step = SRSStep{T}(Ridx, Sidx, Nidx, Tt, Frr, X_rest_R, X_R_rest)
            # Alg. 1 step 9: Ỹ = L Ŷ, Ω̃ = U^{-1} Ω̂, Ẑ = U* Ẑ, Ψ̃ = L^{-*} Ψ̂
            _srs_L!(step, Y)
            _srs_Uinv!(step, Ω)
            _srs_Uadj!(step, Z)
            _srs_Linvadj!(step, Ψ)
            for i in Ridx
                active[i] = false
            end
            push!(steps, step)
        end
    end

    root_idx = findall(active)
    isempty(root_idx) && (root_idx = Int[1])
    Droot = _srs_extract(Y[root_idx, :], Ω[root_idx, :]; rtol = extr_rtol)
    nr = length(root_idx)
    if symmetric && nr > 1 && size(Droot, 1) == nr && size(Droot, 2) == nr
        Droot .= (Droot + Droot') ./ 2
    end
    if size(Droot) != (nr, nr)
        # underdetermined root (too few samples): ridge least-squares square
        Droot = Matrix{T}(Droot)
        if size(Droot, 1) != nr || size(Droot, 2) != nr
            Droot = Matrix{T}(I, nr, nr)
        end
    end
    Droot = Droot + T(1e-12) * I
    Froot = lu(Droot)
    return SRSFactor{T}(steps, root_idx, Froot, n)
end
