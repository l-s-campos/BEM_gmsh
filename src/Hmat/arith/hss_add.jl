# Nested HSS addition + recompression (hm-toolbox hss_sum / hss_compress).
# Same binary clustering required. Generators are stacked, then QR-proper
# and truncated SVD of sibling B and nested R,W.

function _hss_copy(N::HSSNode{T}) where {T}
    if N.leaf
        return HSSNode{T}(true, N.root, N.m, N.n, copy(N.D), copy(N.U), copy(N.V),
            copy(N.Rl), copy(N.Rr), copy(N.Wl), copy(N.Wr), copy(N.B12), copy(N.B21),
            nothing, nothing, N.nl)
    end
    L = _hss_copy(N.left)
    R = _hss_copy(N.right)
    return HSSNode{T}(false, N.root, N.m, N.n, copy(N.D), copy(N.U), copy(N.V),
        copy(N.Rl), copy(N.Rr), copy(N.Wl), copy(N.Wr), copy(N.B12), copy(N.B21),
        L, R, N.nl)
end

function _hss_same_partition(A::HSSNode, B::HSSNode)
    (A.leaf == B.leaf && A.root == B.root && A.m == B.m && A.n == B.n &&
        A.nl == B.nl) || return false
    A.leaf && return true
    return _hss_same_partition(A.left, B.left) && _hss_same_partition(A.right, B.right)
end

function _hss_blkdiag(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    m1, n1 = size(A)
    m2, n2 = size(B)
    C = zeros(T, m1 + m2, n1 + n2)
    m1 > 0 && n1 > 0 && (C[1:m1, 1:n1] = A)
    m2 > 0 && n2 > 0 && (C[(m1 + 1):end, (n1 + 1):end] = B)
    return C
end

function _hss_sum_rec!(C::HSSNode{T}, A::HSSNode{T}, B::HSSNode{T}) where {T}
    if C.leaf
        C.D = A.D + B.D
        C.U = hcat(A.U, B.U)
        C.V = hcat(A.V, B.V)
        return C
    end
    C.B12 = _hss_blkdiag(A.B12, B.B12)
    C.B21 = _hss_blkdiag(A.B21, B.B21)
    if !C.root
        C.Rl = _hss_blkdiag(A.Rl, B.Rl)
        C.Rr = _hss_blkdiag(A.Rr, B.Rr)
        C.Wl = _hss_blkdiag(A.Wl, B.Wl)
        C.Wr = _hss_blkdiag(A.Wr, B.Wr)
    end
    _hss_sum_rec!(C.left, A.left, B.left)
    _hss_sum_rec!(C.right, A.right, B.right)
    return C
end

function _thin_qr(U::AbstractMatrix{T}) where {T}
    m, k = size(U)
    k == 0 && return zeros(T, m, 0), zeros(T, 0, 0)
    F = qr(U)
    kk = min(m, k)
    Q = F.Q * Matrix{T}(I, m, kk)
    R = F.R[1:kk, :]
    return Q, R
end

function _hss_mulU!(N::HSSNode, X::AbstractMatrix)
    k = N.leaf ? size(N.U, 2) : size(N.Rl, 2)
    size(X, 1) == k || return N
    if N.leaf
        N.U = N.U * X
    else
        N.Rl = N.Rl * X
        N.Rr = N.Rr * X
    end
    return N
end

function _hss_mulV!(N::HSSNode, X::AbstractMatrix)
    k = N.leaf ? size(N.V, 2) : size(N.Wl, 2)
    size(X, 1) == k || return N
    if N.leaf
        N.V = N.V * X
    else
        N.Wl = N.Wl * X
        N.Wr = N.Wr * X
    end
    return N
end

function _hss_hcat_RS(B::AbstractMatrix, R::AbstractMatrix, S::AbstractMatrix)
    size(R, 2) == size(S, 1) && size(S, 1) > 0 || return B
    return hcat(B, R * S)
end

function _hss_qr_child!(ch::HSSNode{T}) where {T}
    if ch.leaf
        Uq, Up = _thin_qr(ch.U)
        Vq, Vp = _thin_qr(ch.V)
        ch.U = Uq
        ch.V = Vq
        return Up, Vp
    end
    Uq, Up = _thin_qr(vcat(ch.Rl, ch.Rr))
    ju = size(ch.Rl, 1)
    ch.Rl = Uq[1:ju, :]
    ch.Rr = Uq[(ju + 1):end, :]
    Vq, Vp = _thin_qr(vcat(ch.Wl, ch.Wr))
    jv = size(ch.Wl, 1)
    ch.Wl = Vq[1:jv, :]
    ch.Wr = Vq[(jv + 1):end, :]
    return Up, Vp
end

function _hss_proper!(N::HSSNode{T}) where {T}
    N.leaf && return N
    _hss_proper!(N.left)
    _hss_proper!(N.right)
    Pl, Ql = _hss_qr_child!(N.left)
    Pr, Qr = _hss_qr_child!(N.right)
    if !isempty(Pl) && !isempty(Qr)
        N.B12 = Pl * N.B12 * Qr'
    elseif !isempty(Pl)
        N.B12 = Pl * N.B12
    elseif !isempty(Qr)
        N.B12 = N.B12 * Qr'
    end
    if !isempty(Pr) && !isempty(Ql)
        N.B21 = Pr * N.B21 * Ql'
    elseif !isempty(Pr)
        N.B21 = Pr * N.B21
    elseif !isempty(Ql)
        N.B21 = N.B21 * Ql'
    end
    if !N.root
        !isempty(Pl) && (N.Rl = Pl * N.Rl)
        !isempty(Ql) && (N.Wl = Ql * N.Wl)
        !isempty(Pr) && (N.Rr = Pr * N.Rr)
        !isempty(Qr) && (N.Wr = Qr * N.Wr)
    end
    return N
end

function _hss_tsvd(A::AbstractMatrix{T}, tol::Real) where {T}
    m, n = size(A)
    (m == 0 || n == 0) && return zeros(T, m, 0), zeros(T, 0, 0), zeros(T, n, 0)
    F = svd(A; full = false)
    isempty(F.S) && return zeros(T, m, 0), zeros(T, 0, 0), zeros(T, n, 0)
    τ = float(tol) * abs(F.S[1])
    r = count(s -> s > τ, F.S)
    r == 0 && return zeros(T, m, 0), zeros(T, 0, 0), zeros(T, n, 0)
    S = diagm(F.S[1:r])
    return F.U[:, 1:r], S, F.V[:, 1:r]
end

function _hss_backward!(N::HSSNode{T}, tol::Real, S::AbstractMatrix{T},
        Tm::AbstractMatrix{T}) where {T}
    N.leaf && return N
    if N.root
        U, Su, V = _hss_tsvd(N.B12, tol)
        N.B12 = Su
        Tl = Su'
        _hss_mulU!(N.left, U)
        _hss_mulV!(N.right, V)
        U2, Sl, V2 = _hss_tsvd(N.B21, tol)
        N.B21 = Sl
        Tu = Sl'
        _hss_mulV!(N.left, V2)
        _hss_mulU!(N.right, U2)
        !N.left.leaf && _hss_backward!(N.left, tol, Su, Tu)
        !N.right.leaf && _hss_backward!(N.right, tol, Sl, Tl)
        return N
    end
    k12 = size(N.B12, 2)
    Su = _hss_hcat_RS(N.B12, N.Rl, S)
    Tl = _hss_hcat_RS(Matrix(N.B12'), N.Wr, Tm)
    Us, Su2, Vs = _hss_tsvd(Su, tol)
    Ut, Tl2, Vt = _hss_tsvd(Tl, tol)
    if k12 > 0 && size(Vs, 1) >= k12 && size(Ut, 2) > 0
        N.B12 = Su2 * Vs[1:k12, :]' * Ut
    else
        N.B12 = Su2
    end
    N.Rl = Us' * N.Rl
    N.Wr = Ut' * N.Wr
    _hss_mulU!(N.left, Us)
    _hss_mulV!(N.right, Ut)
    k21 = size(N.B21, 2)
    Sl = _hss_hcat_RS(N.B21, N.Rr, S)
    Tu = _hss_hcat_RS(Matrix(N.B21'), N.Wl, Tm)
    Us2, Sl2, Vs2 = _hss_tsvd(Sl, tol)
    Ut2, Tu2, Vt2 = _hss_tsvd(Tu, tol)
    if k21 > 0 && size(Vs2, 1) >= k21 && size(Ut2, 2) > 0
        N.B21 = Sl2 * Vs2[1:k21, :]' * Ut2
    else
        N.B21 = Sl2
    end
    N.Rr = Us2' * N.Rr
    N.Wl = Ut2' * N.Wl
    _hss_mulU!(N.right, Us2)
    _hss_mulV!(N.left, Ut2)
    !N.right.leaf && _hss_backward!(N.right, tol, Sl2, Tl2)
    !N.left.leaf && _hss_backward!(N.left, tol, Su2, Tu2)
    return N
end

function _hss_recompress!(N::HSSNode{T}, rtol::Real) where {T}
    N.leaf && return N
    _hss_proper!(N)
    Z = zeros(T, 0, 0)
    _hss_backward!(N, float(rtol), Z, Z)
    return N
end

function _hss_fnorm(N::HSSNode)
    N.leaf && return sqrt(sum(abs2, N.D))
    return sqrt(_hss_fnorm(N.left)^2 + _hss_fnorm(N.right)^2 +
        sum(abs2, N.B12) + sum(abs2, N.B21))
end

function _hss_neg!(N::HSSNode)
    if N.leaf
        N.D .*= -1
    else
        N.B12 .*= -1
        N.B21 .*= -1
        _hss_neg!(N.left)
        _hss_neg!(N.right)
    end
    return N
end

"""
    hss_add(A, B; rtol=1e-8) -> HSSMatrix

Nested sum of two HSS matrices on the same cluster tree, then recompress.
"""
function hss_add(A::HSSMatrix{T}, B::HSSMatrix{T}; rtol = 1e-8) where {T}
    size(A) == size(B) || throw(DimensionMismatch("hss_add size $(size(A)) vs $(size(B))"))
    A.perm == B.perm && A.cperm == B.cperm ||
        throw(ArgumentError("hss_add requires the same row/column clustering (perm)"))
    _hss_same_partition(A.root, B.root) ||
        throw(ArgumentError("hss_add: incompatible HSS partitions"))
    Croot = _hss_copy(A.root)
    _hss_sum_rec!(Croot, A.root, B.root)
    _hss_recompress!(Croot, rtol)
    return HSSMatrix{T}(Croot, A.m, A.n, copy(A.perm), copy(A.iperm),
        copy(A.cperm), copy(A.icperm))
end

Base.:+(A::HSSMatrix, B::HSSMatrix) = hss_add(A, B)

function Base.:-(A::HSSMatrix{T}) where {T}
    R = HSSMatrix{T}(_hss_copy(A.root), A.m, A.n, copy(A.perm), copy(A.iperm),
        copy(A.cperm), copy(A.icperm))
    _hss_neg!(R.root)
    return R
end

Base.:-(A::HSSMatrix, B::HSSMatrix) = A + (-B)
