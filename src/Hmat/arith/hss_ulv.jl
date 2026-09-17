# ULV factorization of HSS (Chandrasekaran–Gu–Pals, SIMAX 2006).
# Leaf QR of U zeros the off-diagonal; remaining blocks merge with sibling B;
# internal nodes compress the stacked nested generators the same way; root dense
# solve; right orthogonal transforms applied coarse-to-fine.
# Orthogonal factors are stored as packed QR Q (Householder / compact WY), not
# dense m×m matrices.

"""Packed left/right orthogonal factor. `swap=true` is complement-then-range."""
struct ULVQ{T}
    Q::Any
    k::Int
    swap::Bool
    m::Int
end

"""ULV factors of an [`HSSMatrix`](@ref). `ldiv!` applies the inverse."""
struct ULVFactor{T}
    n::Int
    perm::Vector{Int}
    iperm::Vector{Int}
    root::HSSNode{T}
    Us::Vector{ULVQ{T}}
    Ls::Vector{Matrix{T}}
    rights::Vector{Tuple{Vector{Int}, ULVQ{T}}}
    dense::Union{Matrix{T}, Nothing}
end

Base.size(F::ULVFactor) = (F.n, F.n)
Base.size(F::ULVFactor, d::Integer) = d == 1 || d == 2 ? F.n : 1
Base.eltype(::ULVFactor{T}) where {T} = T
function Base.show(io::IO, F::ULVFactor)
    return print(io, "ULVFactor{", eltype(F), "} n=", F.n)
end
Base.show(io::IO, ::MIME"text/plain", F::ULVFactor) = show(io, F)

function _ulv_swap_blocks!(Y::AbstractVecOrMat, ntop::Int)
    m = size(Y, 1)
    ntop <= 0 && return Y
    nbot = m - ntop
    nbot <= 0 && return Y
    if Y isa AbstractVector
        tmp = Y[1:ntop]
        copyto!(view(Y, 1:nbot), view(Y, (ntop + 1):m))
        copyto!(view(Y, (nbot + 1):m), tmp)
    else
        tmp = Y[1:ntop, :]
        copyto!(view(Y, 1:nbot, :), view(Y, (ntop + 1):m, :))
        copyto!(view(Y, (nbot + 1):m, :), tmp)
    end
    return Y
end

"""Left-multiply `Y` by packed `Q` (`adj=false`) or `Q'` (`adj=true`)."""
function _ulv_lmulQ!(Y::AbstractVecOrMat, Qf::ULVQ, adj::Bool)
    Qf.Q isa UniformScaling && return Y
    if !Qf.swap
        adj ? lmul!(Qf.Q', Y) : lmul!(Qf.Q, Y)
        return Y
    end
    k = Qf.k
    m = size(Y, 1)
    mk = m - k
    if adj
        lmul!(Qf.Q', Y)
        _ulv_swap_blocks!(Y, k)
    else
        _ulv_swap_blocks!(Y, mk)
        lmul!(Qf.Q, Y)
    end
    return Y
end

function _ulv_rmulQ!(X::AbstractMatrix, Qf::ULVQ)
    Qf.Q isa UniformScaling && return X
    rmul!(X, Qf.Q)
    return X
end

function _ulv_packQ(U::AbstractMatrix{T}) where {T}
    m, k = size(U)
    (k == 0 || k >= m) && return ULVQ{T}(I, k, false, m)
    return ULVQ{T}(qr!(copy(U)).Q, k, true, m)
end

function _ulv_BV(B::AbstractMatrix{T}, V::AbstractMatrix{T}, rows::Vector{Int}) where {T}
    (isempty(rows) || size(B, 2) == 0) &&
        return zeros(T, size(B, 1), length(rows))
    if length(rows) == size(V, 1)
        contig = true
        @inbounds for i in eachindex(rows)
            rows[i] == i || (contig = false; break)
        end
        contig && return B * transpose(V)
    end
    Vr = V[rows, :]
    return B * transpose(Vr)
end

function _ulv_mergeD(Dl, Dr, Ul, Ur, B12, B21, Vl, Vr, cindl, cindr)
    T = eltype(Dl)
    ncl, ncr = size(Dl, 1), size(Dr, 1)
    D = zeros(T, ncl + ncr, ncl + ncr)
    ncl > 0 && (D[1:ncl, 1:ncl] = Dl)
    ncr > 0 && (D[(ncl + 1):end, (ncl + 1):end] = Dr)
    if ncl > 0 && ncr > 0
        D[1:ncl, (ncl + 1):end] = Ul * _ulv_BV(B12, Vr, cindr)
        D[(ncl + 1):end, 1:ncl] = Ur * _ulv_BV(B21, Vl, cindl)
    end
    return D
end

function _ulv_nestedU(Ul::AbstractMatrix{T}, Ur::AbstractMatrix{T}, A::HSSNode{T}) where {T}
    UlR = size(Ul, 2) == 0 ? zeros(T, size(Ul, 1), size(A.Rl, 2)) : Ul * A.Rl
    UrR = size(Ur, 2) == 0 ? zeros(T, size(Ur, 1), size(A.Rr, 2)) : Ur * A.Rr
    return vcat(UlR, UrR)
end

function _ulv_nestedV(Vl::AbstractMatrix{T}, Vr::AbstractMatrix{T}, A::HSSNode{T}) where {T}
    VlW = size(Vl, 2) == 0 ? zeros(T, size(Vl, 1), size(A.Wl, 2)) : Vl * A.Wl
    VrW = size(Vr, 2) == 0 ? zeros(T, size(Vr, 1), size(A.Wr, 2)) : Vr * A.Wr
    return vcat(VlW, VrW)
end

mutable struct _ULVIdx
    u::Int
    l::Int
end

function _ulv_compress!(F::ULVFactor{T}, D::Matrix{T}, U::Matrix{T}, V::Matrix{T},
        vrows::Vector{Int}, curQ::Vector{Int}) where {T}
    s, k = size(U)
    Qf = _ulv_packQ(U)
    push!(F.Us, Qf)
    _ulv_lmulQ!(U, Qf, true)
    U2 = U[(s - k + 1):s, :]
    _ulv_lmulQ!(D, Qf, true)
    mk = s - k
    Dt = Matrix{T}(undef, s, mk)
    @inbounds for j in 1:mk, i in 1:s
        Dt[i, j] = D[j, i]
    end
    qrF = qr!(Dt)
    Qtf = ULVQ{T}(qrF.Q, 0, false, s)
    push!(F.Ls, transpose(copy(qrF.R)))
    if !isempty(vrows)
        Vblk = V[vrows, :]
        _ulv_lmulQ!(Vblk, Qtf, true)
        V[vrows, :] = Vblk
    end
    push!(F.rights, (copy(curQ), Qtf))
    D2 = D[(mk + 1):s, :]
    _ulv_rmulQ!(D2, Qtf)
    push!(F.Ls, D2[:, 1:mk])
    Dkeep = D2[:, (mk + 1):s]
    return Dkeep, U2, V, collect(1:mk), collect((mk + 1):s)
end

function _ulv_fact_rec!(F::ULVFactor{T}, A::HSSNode{T}, cur::Vector{Int}) where {T}
    if A.leaf
        k = size(A.U, 2)
        m = A.m
        D = copy(A.D)
        U = copy(A.U)
        V = copy(A.V)
        if k >= m
            return D, U, V, Int[], collect(1:m)
        end
        return _ulv_compress!(F, D, U, V, collect(1:m), cur)
    end
    Dl, Ul, Vl, indl, cindl = _ulv_fact_rec!(F, A.left, cur[1:A.nl])
    Dr, Ur, Vr, indr, cindr = _ulv_fact_rec!(F, A.right, cur[(A.nl + 1):end])
    push!(F.Ls, Ul)
    push!(F.Ls, _ulv_BV(A.B12, Vr, indr))
    push!(F.Ls, Ur)
    push!(F.Ls, _ulv_BV(A.B21, Vl, indl))
    D = _ulv_mergeD(Dl, Dr, Ul, Ur, A.B12, A.B21, Vl, Vr, cindl, cindr)
    if A.root
        push!(F.Ls, D)
        reverse!(F.rights)
        return D, zeros(T, 0, 0), zeros(T, 0, 0),
            vcat(indl, A.nl .+ indr), vcat(cindl, A.nl .+ cindr)
    end
    U = _ulv_nestedU(Ul, Ur, A)
    V = _ulv_nestedV(Vl, Vr, A)
    k = size(U, 2)
    s = size(U, 1)
    if k >= s
        return D, U, V, vcat(indl, A.nl .+ indr), vcat(cindl, A.nl .+ cindr)
    end
    cmerge = vcat(cindl, A.nl .+ cindr)
    D, U, V, _, _ = _ulv_compress!(F, D, U, V, cmerge, cur[cmerge])
    nz = s - k
    n_cindl = length(cindl)
    if nz < n_cindl
        indl = vcat(indl, cindl[1:nz])
        cindl = cindl[(nz + 1):end]
    else
        indl = collect(1:A.left.m)
        indr = vcat(indr, cindr[1:(nz - n_cindl)])
        cindr = cindr[(nz - n_cindl + 1):end]
        cindl = Int[]
    end
    return D, U, V, vcat(indl, A.nl .+ indr), vcat(cindl, A.nl .+ cindr)
end

"""
    ulv(A::HSSMatrix) -> ULVFactor

ULV factorization of an HSS matrix (weak nested off-diagonals).
"""
function ulv(A::HSSMatrix{T}) where {T}
    (A.m == A.n && A.perm == A.cperm) ||
        throw(ArgumentError("ulv requires square HSS assembled from a single tree"))
    if A.root.leaf
        return ULVFactor{T}(A.n, copy(A.perm), copy(A.iperm), A.root,
            ULVQ{T}[], Matrix{T}[], Tuple{Vector{Int}, ULVQ{T}}[], copy(A.root.D))
    end
    F = ULVFactor{T}(A.n, copy(A.perm), copy(A.iperm), A.root,
        ULVQ{T}[], Matrix{T}[], Tuple{Vector{Int}, ULVQ{T}}[], nothing)
    _ulv_fact_rec!(F, A.root, collect(1:A.n))
    return F
end

function LinearAlgebra.ldiv!(F::ULVFactor{T}, x::AbstractVector{T};
        global_index = use_global_index()) where {T}
    length(x) == F.n || throw(DimensionMismatch())
    xl = global_index ? x[F.perm] : copy(x)
    y = zeros(T, F.n)
    if F.dense !== nothing
        y .= F.dense \ xl
    else
        _ulv_solve_rec!(F, F.root, collect(1:F.n), xl, y, _ULVIdx(1, 1))
        for (ind, Qf) in F.rights
            yb = y[ind]
            _ulv_lmulQ!(yb, Qf, false)
            y[ind] = yb
        end
    end
    if global_index
        x[F.perm] = y
    else
        copyto!(x, y)
    end
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, F::ULVFactor, x::AbstractVector;
        global_index = use_global_index())
    y === x || copyto!(y, x)
    return ldiv!(F, y; global_index = global_index)
end

function Base.:\(F::ULVFactor, b::AbstractVector)
    x = copy(b)
    ldiv!(F, x)
    return x
end

function _ulv_solve_rec!(F::ULVFactor{T}, A::HSSNode{T}, cur::Vector{Int},
        b::AbstractVector{T}, x::AbstractVector{T}, idx::_ULVIdx) where {T}
    if A.leaf
        k = size(A.U, 2)
        m = A.m
        if k >= m
            return Int[], collect(1:m), b
        end
        Qf = F.Us[idx.u]
        idx.u += 1
        b2 = copy(b)
        _ulv_lmulQ!(b2, Qf, true)
        Lelim = F.Ls[idx.l]
        idx.l += 1
        mk = size(Lelim, 1)
        z = Lelim \ b2[1:mk]
        x[1:mk] = z
        Lcoup = F.Ls[idx.l]
        idx.l += 1
        brem = b2[(mk + 1):end] .- Lcoup * z
        return collect(1:mk), collect((mk + 1):m), brem
    end
    nl = A.nl
    indl, cindl, bl = _ulv_solve_rec!(F, A.left, cur[1:nl], b[1:nl],
        view(x, 1:nl), idx)
    indr, cindr, br = _ulv_solve_rec!(F, A.right, cur[(nl + 1):end],
        b[(nl + 1):end], view(x, (nl + 1):A.m), idx)
    Ul = F.Ls[idx.l]
    idx.l += 1
    B12ind = F.Ls[idx.l]
    idx.l += 1
    Ur = F.Ls[idx.l]
    idx.l += 1
    B21ind = F.Ls[idx.l]
    idx.l += 1
    xr = view(x, (nl + 1):A.m)
    xl = view(x, 1:nl)
    if !isempty(bl) && size(B12ind, 2) == length(indr) && size(Ul, 1) == length(bl)
        bl = bl .- Ul * (B12ind * xr[indr])
    end
    if !isempty(br) && size(B21ind, 2) == length(indl) && size(Ur, 1) == length(br)
        br = br .- Ur * (B21ind * xl[indl])
    end
    b2 = vcat(bl, br)
    if A.root
        D = F.Ls[idx.l]
        idx.l += 1
        cind = vcat(cindl, nl .+ cindr)
        if !isempty(b2)
            x[cind] = D \ b2
        end
        return Int[], Int[], T[]
    end
    Ucols = size(A.Rl, 2)
    s = length(cindl) + length(cindr)
    if Ucols >= s
        return vcat(indl, nl .+ indr), vcat(cindl, nl .+ cindr), b2
    end
    Qf = F.Us[idx.u]
    idx.u += 1
    _ulv_lmulQ!(b2, Qf, true)
    Lelim = F.Ls[idx.l]
    idx.l += 1
    mk = size(Lelim, 1)
    z = Lelim \ b2[1:mk]
    nz = length(z)
    n_cindl = length(cindl)
    if nz < n_cindl
        x[cindl[1:nz]] = z
        indl = vcat(indl, cindl[1:nz])
        cindl = cindl[(nz + 1):end]
    else
        n_cindl > 0 && (x[cindl] = z[1:n_cindl])
        nkeep = nz - n_cindl
        nkeep > 0 && (x[nl .+ cindr[1:nkeep]] = z[(n_cindl + 1):end])
        indl = collect(1:A.left.m)
        indr = vcat(indr, cindr[1:nkeep])
        cindr = cindr[(nkeep + 1):end]
        cindl = Int[]
    end
    Lcoup = F.Ls[idx.l]
    idx.l += 1
    brem = b2[(mk + 1):end] .- Lcoup * z
    return vcat(indl, nl .+ indr), vcat(cindl, nl .+ cindr), brem
end
