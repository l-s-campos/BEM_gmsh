# Gujjula–Ambikasaran ACA_only_nodes (NNCA/NNCA2D/ACA.hpp).
# Pivots on K[row_indices, col_indices]; returns LU factors for L2P = (Ac / R) / L.
# Tensor kernels (`SMatrix{p,p}`): point skeletons, Frobenius block pivots,
# Ac is (p N1)×(p r) and R is the skeleton intersection (p r)×(p r).

mutable struct NNCAACAWorkspace{T}
    row::Vector{T}
    col::Vector{T}
    remaining_row::BitVector
    remaining_col::BitVector
    U::Matrix{T}
    V::Matrix{T}
    Ac::Matrix{T}
    Ar::Matrix{T}
    L::Matrix{T}
    R::Matrix{T}
    row_bases::Vector{Int}
    col_bases::Vector{Int}
    x1::Vector{T}
    x2::Vector{T}
    x3::Vector{T}
    y1::Vector{T}
    y2::Vector{T}
    y3::Vector{T}
end

function NNCAACAWorkspace{T}() where {T}
    z = Matrix{T}(undef, 0, 0)
    return NNCAACAWorkspace{T}(T[], T[], BitVector(), BitVector(),
        z, copy(z), copy(z), copy(z), copy(z), copy(z), Int[], Int[],
        T[], T[], T[], T[], T[], T[])
end

function _ensure_len!(v::Vector, n::Int)
    length(v) < n && resize!(v, n)
    return v
end

function _ensure_bit!(v::BitVector, n::Int)
    length(v) < n && resize!(v, n)
    return v
end

function _ensure_mat!(M::Matrix{T}, m::Int, n::Int) where {T}
    (size(M, 1) < m || size(M, 2) < n) && return Matrix{T}(undef, max(size(M, 1), m), max(size(M, 2), n))
    return M
end

function _nnca_aca_ensure!(ws::NNCAACAWorkspace{T}, n1::Int, n2::Int, rmax::Int) where {T}
    _ensure_len!(ws.row, n2)
    _ensure_len!(ws.col, n1)
    _ensure_bit!(ws.remaining_row, n1)
    _ensure_bit!(ws.remaining_col, n2)
    ws.U = _ensure_mat!(ws.U, n1, rmax)
    ws.V = _ensure_mat!(ws.V, n2, rmax)
    ws.Ac = _ensure_mat!(ws.Ac, n1, rmax)
    ws.Ar = _ensure_mat!(ws.Ar, rmax, n2)
    ws.L = _ensure_mat!(ws.L, rmax, rmax)
    ws.R = _ensure_mat!(ws.R, rmax, rmax)
    empty!(ws.row_bases)
    empty!(ws.col_bases)
    return ws
end

function _nnca_soa_ensure!(ws::NNCAACAWorkspace{T}, m::Int, n::Int) where {T}
    _ensure_len!(ws.x1, m); _ensure_len!(ws.x2, m); _ensure_len!(ws.x3, m)
    _ensure_len!(ws.y1, n); _ensure_len!(ws.y2, n); _ensure_len!(ws.y3, n)
    return ws
end

const _NNCA_ACA_LOCK = ReentrantLock()
const _NNCA_ACA_TLS = Any[]

function _nnca_thread_ws(::Type{T}) where {T}
    tid = Threads.threadid()
    tls = _NNCA_ACA_TLS
    if tid <= length(tls)
        ws = tls[tid]
        ws isa NNCAACAWorkspace{T} && return ws
    end
    return _nnca_thread_ws_slow(T, tid)
end

function _nnca_thread_ws_slow(::Type{T}, tid::Int) where {T}
    lock(_NNCA_ACA_LOCK) do
        n = max(tid, Threads.maxthreadid())
        while length(_NNCA_ACA_TLS) < n
            push!(_NNCA_ACA_TLS, nothing)
        end
        ws = _NNCA_ACA_TLS[tid]
        if !(ws isa NNCAACAWorkspace{T})
            ws = NNCAACAWorkspace{T}()
            _NNCA_ACA_TLS[tid] = ws
        end
        return ws::NNCAACAWorkspace{T}
    end
end

function _maxabs_masked(v, mask, n)
    idx = 0
    m = zero(real(eltype(v)))
    @inbounds for i in 1:n
        mask[i] || continue
        a = abs(v[i])
        if idx == 0 || a > m
            m = a
            idx = i
        end
    end
    return idx, m
end

function nnca_aca(K, row_indices::Vector{Int}, col_indices::Vector{Int}, ::Type{T};
        tol::Float64, rank::Int=typemax(Int)) where {T}
    if is_tensor_eltype(eltype(K))
        return nnca_aca_block(K, row_indices, col_indices, T; tol=tol, rank=rank)
    end
    N1 = length(row_indices)
    N2 = length(col_indices)
    if N1 == 0 || N2 == 0
        return Int[], Int[], zeros(T, N1, 0), zeros(T, 0, N2), zeros(T, 0, 0), zeros(T, 0, 0)
    end
    rmax = min(N1, N2, rank)
    ws = _nnca_aca_ensure!(_nnca_thread_ws(T), N1, N2, rmax)
    row = ws.row
    col = ws.col
    remaining_row = ws.remaining_row
    remaining_col = ws.remaining_col
    fill!(view(remaining_row, 1:N1), true)
    fill!(view(remaining_col, 1:N2), true)
    U = ws.U
    V = ws.V
    row_bases = ws.row_bases
    col_bases = ws.col_bases

    computed = 0
    row_index = 0
    col_index = 0
    @inbounds for l in 1:N1
        _kernel_row!(row, K, row_indices[l], col_indices, ws)
        col_index, mx = _maxabs_masked(row, remaining_col, N2)
        if col_index != 0 && mx > T(1e-36)
            row_index = l
            break
        end
    end
    if row_index == 0 || col_index == 0
        Z = zeros(T, N1, 0)
        return Int[], Int[], Z, zeros(T, 0, N2), zeros(T, 0, 0), zeros(T, 0, 0)
    end

    push!(row_bases, row_index)
    push!(col_bases, col_index)
    @inbounds for s in 1:N2
        V[s, 1] = row[s]
        ws.Ar[1, s] = row[s]
    end
    _kernel_col!(col, K, col_indices[col_index], row_indices, ws)
    piv0 = row[col_index]
    @inbounds for s in 1:N1
        U[s, 1] = col[s] / piv0
        ws.Ac[s, 1] = col[s]
    end
    remaining_col[col_index] = false
    remaining_row[row_index] = false
    computed = 1
    normS = zero(real(T))
    row_index, = _maxabs_masked(col, remaining_row, N1)

    while computed < rmax
        uview = view(U, 1:N1, computed)
        vview = view(V, 1:N2, computed)
        if !(norm(uview) * norm(vview) > tol * normS)
            break
        end
        row_index == 0 && break
        push!(row_bases, row_index)
        _kernel_row!(row, K, row_indices[row_index], col_indices, ws)
        @inbounds for l in 1:computed
            ul = U[row_index, l]
            for s in 1:N2
                row[s] -= ul * V[s, l]
            end
        end
        col_index, mx = _maxabs_masked(row, remaining_col, N2)
        if col_index == 0 || mx <= T(1e-36)
            pop!(row_bases)
            break
        end
        push!(col_bases, col_index)
        nxt = computed + 1
        @inbounds for s in 1:N2
            V[s, nxt] = row[s]
            ws.Ar[nxt, s] = row[s]
        end
        _kernel_col!(col, K, col_indices[col_index], row_indices, ws)
        @inbounds for s in 1:N1
            ws.Ac[s, nxt] = col[s]
        end
        @inbounds for l in 1:computed
            vl = V[col_index, l]
            for s in 1:N1
                col[s] -= vl * U[s, l]
            end
        end
        piv = row[col_index]
        @inbounds for s in 1:N1
            U[s, nxt] = col[s] / piv
        end
        nu = zero(real(T))
        nv = zero(real(T))
        @inbounds for s in 1:N1
            nu += abs2(U[s, nxt])
        end
        @inbounds for s in 1:N2
            nv += abs2(V[s, nxt])
        end
        if sqrt(nu) < T(1e-36) || sqrt(nv) < T(1e-36)
            pop!(row_bases)
            pop!(col_bases)
            break
        end
        computed = nxt
        remaining_col[col_index] = false
        remaining_row[row_index] = false
        ns2 = normS * normS + nu * nv
        @inbounds for l in 1:(computed - 1)
            du = zero(T)
            dv = zero(T)
            for s in 1:N1
                du += U[s, l] * U[s, computed]
            end
            for s in 1:N2
                dv += V[s, l] * V[s, computed]
            end
            ns2 += 2 * abs(du * dv)
        end
        normS = sqrt(ns2)
        row_index, = _maxabs_masked(col, remaining_row, N1)
    end

    r = length(row_bases)
    Ac = Matrix{T}(undef, N1, r)
    Ar = Matrix{T}(undef, r, N2)
    L = zeros(T, r, r)
    R = zeros(T, r, r)
    @inbounds for j in 1:r
        for i in 1:N1
            Ac[i, j] = ws.Ac[i, j]
        end
        for i in 1:N2
            Ar[j, i] = ws.Ar[j, i]
        end
        L[j, j] = one(T)
        R[j, j] = V[col_bases[j], j]
        for i in 1:(j - 1)
            L[j, i] = U[row_bases[j], i]
            R[i, j] = V[col_bases[j], i]
        end
    end
    return copy(row_bases), copy(col_bases), Ac, Ar, L, R
end

"""
Block NNCA ACA: each pivot is a point (`p×p` kernel block). Skeletons are `r`
points; `Ac` is `(p N1)×(p r)` and `R` is the original skeleton intersection
(so L2P = `Ac / R`). `L` is identity.
"""
function nnca_aca_block(K, row_indices::Vector{Int}, col_indices::Vector{Int}, ::Type{T};
        tol::Float64, rank::Int=typemax(Int)) where {T}
    p, q = tensor_blocksize(eltype(K))
    p == q || throw(ArgumentError("block NNCA requires square SMatrix{p,p}, got $(eltype(K))"))
    N1 = length(row_indices)
    N2 = length(col_indices)
    Z = zeros(T, p * N1, 0)
    if N1 == 0 || N2 == 0
        return Int[], Int[], Z, zeros(T, 0, p * N2), zeros(T, 0, 0), zeros(T, 0, 0)
    end
    rmax = min(N1, N2, rank)
    remaining_row = trues(N1)
    remaining_col = trues(N2)
    row_bases = Int[]
    col_bases = Int[]
    Uvec = Matrix{T}[]
    Vvec = Matrix{T}[]
    AcVec = Matrix{T}[]
    ArVec = Matrix{T}[]

    row = Matrix{T}(undef, p, p * N2)
    col = Matrix{T}(undef, p * N1, p)
    computed = 0
    row_index = 0
    col_index = 0
    @inbounds for l in 1:N1
        _kernel_block_row!(row, K, row_indices[l], col_indices, p)
        col_index, mx = _maxfrob_row(row, remaining_col, p, N2)
        if col_index != 0 && mx > T(1e-36)
            row_index = l
            break
        end
    end
    if row_index == 0 || col_index == 0
        return Int[], Int[], Z, zeros(T, 0, p * N2), zeros(T, 0, 0), zeros(T, 0, 0)
    end

    δ = _block_at(row, col_index, Val(p))
    if min_svd_vals(δ) <= T(1e-36)
        return Int[], Int[], Z, zeros(T, 0, p * N2), zeros(T, 0, 0), zeros(T, 0, 0)
    end
    v = copy(row)
    push!(row_bases, row_index)
    push!(col_bases, col_index)
    _kernel_block_col!(col, K, col_indices[col_index], row_indices, p)
    u = col * inv(δ)
    push!(Uvec, copy(u))
    push!(Vvec, v)
    push!(AcVec, copy(col))
    push!(ArVec, copy(row))
    remaining_col[col_index] = false
    remaining_row[row_index] = false
    computed = 1
    normS = zero(real(T))
    row_index, = _maxfrob_col(col, remaining_row, p, N1)

    while computed < rmax && norm(u) * norm(Vvec[end]) > tol * normS
        row_index == 0 && break
        push!(row_bases, row_index)
        _kernel_block_row!(row, K, row_indices[row_index], col_indices, p)
        row_temp = copy(row)
        i0 = p * (row_index - 1)
        @inbounds for l in 1:computed
            Ui = view(Uvec[l], (i0 + 1):(i0 + p), :)
            mul!(row, Ui, Vvec[l], -one(T), one(T))
        end
        col_index, mx = _maxfrob_row(row, remaining_col, p, N2)
        if col_index == 0 || mx <= T(1e-36)
            pop!(row_bases)
            break
        end
        push!(col_bases, col_index)
        v = copy(row)
        _kernel_block_col!(col, K, col_indices[col_index], row_indices, p)
        col_temp = copy(col)
        j0 = p * (col_index - 1)
        @inbounds for l in 1:computed
            Vj = view(Vvec[l], :, (j0 + 1):(j0 + p))
            mul!(col, Uvec[l], Vj, -one(T), one(T))
        end
        δ = _block_at(row, col_index, Val(p))
        if min_svd_vals(δ) <= T(1e-36)
            pop!(row_bases)
            pop!(col_bases)
            break
        end
        u = col * inv(δ)
        if norm(u) < T(1e-36) || norm(v) < T(1e-36)
            pop!(row_bases)
            pop!(col_bases)
            break
        end
        push!(Uvec, copy(u))
        push!(Vvec, v)
        push!(AcVec, col_temp)
        push!(ArVec, row_temp)
        computed += 1
        remaining_col[col_index] = false
        remaining_row[row_index] = false
        ns2 = normS * normS + (norm(u) * norm(v))^2
        @inbounds for l in 1:(computed - 1)
            ns2 += 2 * abs(dot(Uvec[l], u) * dot(Vvec[l], v))
        end
        normS = sqrt(ns2)
        row_index, = _maxfrob_col(col, remaining_row, p, N1)
    end

    r = length(row_bases)
    Ac = Matrix{T}(undef, p * N1, p * r)
    Ar = Matrix{T}(undef, p * r, p * N2)
    @inbounds for i in 1:r
        Ac[:, (p * (i - 1) + 1):(p * i)] = AcVec[i]
        Ar[(p * (i - 1) + 1):(p * i), :] = ArVec[i]
    end
    Isk = row_indices[row_bases]
    Jsk = col_indices[col_bases]
    As = Matrix{T}(undef, p * r, p * r)
    _kernel_block!(As, K, Isk, Jsk)
    L = Matrix{T}(I, p * r, p * r)
    return row_bases, col_bases, Ac, Ar, L, As
end

@inline function _block_at(row::AbstractMatrix{T}, j::Int, ::Val{p}) where {T, p}
    off = p * (j - 1)
    return SMatrix{p, p, T, p * p}(ntuple(Val(p * p)) do k
        a = (k - 1) % p + 1
        b = (k - 1) ÷ p + 1
        @inbounds row[a, off + b]
    end)
end

function _maxfrob_row(row::AbstractMatrix{T}, mask, p::Int, N2::Int) where {T}
    idx = 0
    m = zero(real(T))
    @inbounds for j in 1:N2
        mask[j] || continue
        off = p * (j - 1)
        s = zero(real(T))
        for b in 1:p, a in 1:p
            s += abs2(row[a, off + b])
        end
        a = sqrt(s)
        if idx == 0 || a > m
            m = a
            idx = j
        end
    end
    return idx, m
end

function _maxfrob_col(col::AbstractMatrix{T}, mask, p::Int, N1::Int) where {T}
    idx = 0
    m = zero(real(T))
    @inbounds for i in 1:N1
        mask[i] || continue
        off = p * (i - 1)
        s = zero(real(T))
        for b in 1:p, a in 1:p
            s += abs2(col[off + a, b])
        end
        a = sqrt(s)
        if idx == 0 || a > m
            m = a
            idx = i
        end
    end
    return idx, m
end

function _kernel_block_row!(row::AbstractMatrix{T}, K, i::Int, J::Vector{Int}, p::Int) where {T}
    @inbounds for t in eachindex(J)
        Bij = K[i, J[t]]
        off = p * (t - 1)
        for b in 1:p, a in 1:p
            row[a, off + b] = Bij[a, b]
        end
    end
    return row
end

function _kernel_block_col!(col::AbstractMatrix{T}, K, j::Int, I::Vector{Int}, p::Int) where {T}
    @inbounds for t in eachindex(I)
        Bij = K[I[t], j]
        off = p * (t - 1)
        for b in 1:p, a in 1:p
            col[off + a, b] = Bij[a, b]
        end
    end
    return col
end

function _kernel_row!(row::AbstractVector, K, i::Int, J::Vector{Int}, ws=nothing)
    @inbounds for t in eachindex(J)
        row[t] = K[i, J[t]]
    end
    return row
end

function _kernel_row!(row::AbstractVector, K::KernelMatrix, i::Int, J::Vector{Int},
        ws=nothing)
    f = kernel(K)
    X = rowelements(K)
    Y = colelements(K)
    xi = X[i]
    n = length(J)
    if xi isa SVector{2}
        ws === nothing && (ws = _nnca_thread_ws(eltype(row)))
        _nnca_soa_ensure!(ws, 1, n)
        @inbounds for t in 1:n
            yt = Y[J[t]]
            ws.y1[t] = yt[1]
            ws.y2[t] = yt[2]
        end
        x1, x2 = xi[1], xi[2]
        @inbounds @simd for t in 1:n
            row[t] = f(SVector{2}(x1, x2), SVector{2}(ws.y1[t], ws.y2[t]))
        end
    elseif xi isa SVector{3}
        ws === nothing && (ws = _nnca_thread_ws(eltype(row)))
        _nnca_soa_ensure!(ws, 1, n)
        @inbounds for t in 1:n
            yt = Y[J[t]]
            ws.y1[t] = yt[1]
            ws.y2[t] = yt[2]
            ws.y3[t] = yt[3]
        end
        x1, x2, x3 = xi[1], xi[2], xi[3]
        @inbounds @simd for t in 1:n
            row[t] = f(SVector{3}(x1, x2, x3),
                SVector{3}(ws.y1[t], ws.y2[t], ws.y3[t]))
        end
    else
        @inbounds for t in 1:n
            row[t] = f(xi, Y[J[t]])
        end
    end
    return row
end

function _kernel_col!(col::AbstractVector, K, j::Int, I::Vector{Int}, ws=nothing)
    @inbounds for t in eachindex(I)
        col[t] = K[I[t], j]
    end
    return col
end

function _kernel_col!(col::AbstractVector, K::KernelMatrix, j::Int, I::Vector{Int},
        ws=nothing)
    f = kernel(K)
    X = rowelements(K)
    Y = colelements(K)
    yj = Y[j]
    n = length(I)
    if yj isa SVector{2}
        ws === nothing && (ws = _nnca_thread_ws(eltype(col)))
        _nnca_soa_ensure!(ws, n, 1)
        @inbounds for t in 1:n
            xt = X[I[t]]
            ws.x1[t] = xt[1]
            ws.x2[t] = xt[2]
        end
        y1, y2 = yj[1], yj[2]
        @inbounds @simd for t in 1:n
            col[t] = f(SVector{2}(ws.x1[t], ws.x2[t]), SVector{2}(y1, y2))
        end
    elseif yj isa SVector{3}
        ws === nothing && (ws = _nnca_thread_ws(eltype(col)))
        _nnca_soa_ensure!(ws, n, 1)
        @inbounds for t in 1:n
            xt = X[I[t]]
            ws.x1[t] = xt[1]
            ws.x2[t] = xt[2]
            ws.x3[t] = xt[3]
        end
        y1, y2, y3 = yj[1], yj[2], yj[3]
        @inbounds @simd for t in 1:n
            col[t] = f(SVector{3}(ws.x1[t], ws.x2[t], ws.x3[t]),
                SVector{3}(y1, y2, y3))
        end
    else
        @inbounds for t in 1:n
            col[t] = f(X[I[t]], yj)
        end
    end
    return col
end

function _kernel_block!(out::AbstractMatrix{T}, K, I::Vector{Int}, J::Vector{Int}) where {T}
    Te = eltype(K)
    if is_tensor_eltype(Te)
        p, q = tensor_blocksize(Te)
        m, n = length(I), length(J)
        size(out, 1) == p * m && size(out, 2) == q * n || throw(DimensionMismatch(
            "block kernel fill: $(size(out)) vs ($p×$m, $q×$n)"))
        threaded = Threads.nthreads() > 1 && m * n >= 1024
        if threaded
            Threads.@threads for jj in 1:n
                j = J[jj]
                j0 = q * (jj - 1)
                @inbounds for ii in 1:m
                    Bij = K[I[ii], j]
                    i0 = p * (ii - 1)
                    for b in 1:q, a in 1:p
                        out[i0 + a, j0 + b] = Bij[a, b]
                    end
                end
            end
        else
            @inbounds for jj in 1:n
                j0 = q * (jj - 1)
                for ii in 1:m
                    Bij = K[I[ii], J[jj]]
                    i0 = p * (ii - 1)
                    for b in 1:q, a in 1:p
                        out[i0 + a, j0 + b] = Bij[a, b]
                    end
                end
            end
        end
        return out
    end
    m, n = length(I), length(J)
    size(out, 1) == m && size(out, 2) == n || throw(DimensionMismatch())
    _kernel_block_scalar!(out, K, I, J, m, n)
    return out
end

function _kernel_block_scalar!(out, K::KernelMatrix, I, J, m, n)
    (m == 0 || n == 0) && return out
    f = kernel(K)
    X = rowelements(K)
    Y = colelements(K)
    x1 = X[I[1]]
    if x1 isa SVector{2} || x1 isa SVector{3}
        ws = _nnca_thread_ws(eltype(out))
        _nnca_soa_ensure!(ws, m, n)
        if x1 isa SVector{2}
            @inbounds for ii in 1:m
                p = X[I[ii]]
                ws.x1[ii] = p[1]
                ws.x2[ii] = p[2]
            end
            @inbounds for jj in 1:n
                q = Y[J[jj]]
                ws.y1[jj] = q[1]
                ws.y2[jj] = q[2]
            end
            @inbounds for jj in 1:n
                y1 = ws.y1[jj]
                y2 = ws.y2[jj]
                @simd for ii in 1:m
                    out[ii, jj] = f(SVector{2}(ws.x1[ii], ws.x2[ii]),
                        SVector{2}(y1, y2))
                end
            end
        else
            @inbounds for ii in 1:m
                p = X[I[ii]]
                ws.x1[ii] = p[1]
                ws.x2[ii] = p[2]
                ws.x3[ii] = p[3]
            end
            @inbounds for jj in 1:n
                q = Y[J[jj]]
                ws.y1[jj] = q[1]
                ws.y2[jj] = q[2]
                ws.y3[jj] = q[3]
            end
            @inbounds for jj in 1:n
                y1 = ws.y1[jj]
                y2 = ws.y2[jj]
                y3 = ws.y3[jj]
                @simd for ii in 1:m
                    out[ii, jj] = f(SVector{3}(ws.x1[ii], ws.x2[ii], ws.x3[ii]),
                        SVector{3}(y1, y2, y3))
                end
            end
        end
        return out
    end
    @inbounds for jj in 1:n
        yj = Y[J[jj]]
        for ii in 1:m
            out[ii, jj] = f(X[I[ii]], yj)
        end
    end
    return out
end

function _kernel_block_scalar!(out, K, I, J, m, n)
    threaded = Threads.nthreads() > 1 && m * n >= 4096
    if threaded
        Threads.@threads for jj in 1:n
            j = J[jj]
            @inbounds for ii in 1:m
                out[ii, jj] = K[I[ii], j]
            end
        end
    else
        @inbounds for jj in 1:n, ii in 1:m
            out[ii, jj] = K[I[ii], J[jj]]
        end
    end
    return out
end
