# GPU apply for packed H-matrix / NNCA (KernelAbstractions).
# Assembly stays on the host. `device=:cuda` needs `using CUDA`; `:cpu` is the
# KA CPU backend (tests). Default `device=:host` is the existing CPU apply.
# If `x`/`y` live on `A.backend` (e.g. `CuArray`), permute and GEMV stay there.

const _HMAT_CUDA_PKGID = Base.PkgId(
    Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")

function _hmat_ka_backend(device::Symbol)
    device === :cpu && return CPU()
    device === :cuda || throw(ArgumentError(
        "device must be :cuda, :cpu, or :host; got $device"))
    CUDA = _hmat_require_cuda()
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl reports no functional GPU. Pass device=:cpu for the KA host backend."))
    isdefined(CUDA, :CUDABackend) || throw(ArgumentError(
        "CUDA.CUDABackend is missing; KernelAbstractions must be loaded."))
    return CUDA.CUDABackend()
end

function _hmat_require_cuda()
    haskey(Base.loaded_modules, _HMAT_CUDA_PKGID) || throw(ArgumentError(
        "GPU H-matrix / H² apply needs CUDA.jl loaded first (`using CUDA`). " *
        "Pass device=:cpu to run the same kernels on the host."))
    return Base.loaded_modules[_HMAT_CUDA_PKGID]
end

function _hmat_to_backend(backend, x::AbstractArray)
    y = KernelAbstractions.allocate(backend, eltype(x), size(x)...)
    copyto!(y, x)
    return y
end

_hmat_on_backend(backend, x::AbstractArray) =
    KernelAbstractions.get_backend(x) == backend

_hmat_max0(v::AbstractVector) = isempty(v) ? 0 : Int(maximum(v))

@kernel function _gpu_gather1!(dst, src, perm, a)
    t = @index(Global, Linear)
    @inbounds dst[t] = a * convert(typeof(a), src[perm[t]])
end

@kernel function _gpu_gatherp!(dst, src, perm, a, p)
    t = @index(Global, Linear)
    i0 = p * (t - Int32(1))
    g0 = p * (perm[t] - Int32(1))
    @inbounds for k in Int32(1):p
        dst[i0 + k] = a * convert(typeof(a), src[g0 + k])
    end
end

@kernel function _gpu_scatter1!(y, yt, perm, b, replace)
    t = @index(Global, Linear)
    i = perm[t]
    if replace != Int32(0)
        @inbounds y[i] = yt[t]
    else
        @inbounds y[i] = b * y[i] + yt[t]
    end
end

@kernel function _gpu_scatterp!(y, yt, perm, b, p, replace)
    t = @index(Global, Linear)
    i0 = p * (t - Int32(1))
    g0 = p * (perm[t] - Int32(1))
    if replace != Int32(0)
        @inbounds for k in Int32(1):p
            y[g0 + k] = yt[i0 + k]
        end
    else
        @inbounds for k in Int32(1):p
            y[g0 + k] = b * y[g0 + k] + yt[i0 + k]
        end
    end
end

@kernel function _gpu_scale_copy!(dst, src, a)
    t = @index(Global, Linear)
    @inbounds dst[t] = a * convert(typeof(a), src[t])
end

@kernel function _gpu_axpy!(y, yt, b, replace)
    t = @index(Global, Linear)
    if replace != Int32(0)
        @inbounds y[t] = yt[t]
    else
        @inbounds y[t] = b * y[t] + yt[t]
    end
end

function _gpu_permute_in!(dst, src, perm_d, a, p::Int)
    backend = KernelAbstractions.get_backend(dst)
    n = length(perm_d)
    n == 0 && return
    if p == 1
        _gpu_gather1!(backend)(dst, src, perm_d, a; ndrange = n)
    else
        _gpu_gatherp!(backend)(dst, src, perm_d, a, Int32(p); ndrange = n)
    end
    KernelAbstractions.synchronize(backend)
    return
end

function _gpu_permute_out!(y, yt, perm_d, b, p::Int)
    backend = KernelAbstractions.get_backend(y)
    n = length(perm_d)
    n == 0 && return
    replace = Int32(iszero(b) ? 1 : 0)
    if p == 1
        _gpu_scatter1!(backend)(y, yt, perm_d, b, replace; ndrange = n)
    else
        _gpu_scatterp!(backend)(y, yt, perm_d, b, Int32(p), replace; ndrange = n)
    end
    KernelAbstractions.synchronize(backend)
    return
end

function _gpu_copy_scale!(dst, src, a)
    backend = KernelAbstractions.get_backend(dst)
    n = length(dst)
    n == 0 && return
    _gpu_scale_copy!(backend)(dst, src, a; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return
end

function _gpu_axpy_out!(y, yt, b)
    backend = KernelAbstractions.get_backend(y)
    n = length(yt)
    n == 0 && return
    replace = Int32(iszero(b) ? 1 : 0)
    _gpu_axpy!(backend)(y, yt, b, replace; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return
end

function _gpu_stage_x!(xloc, x, a, global_index, perm_d, host_perm, p, ::Type{T}) where {T}
    backend = KernelAbstractions.get_backend(xloc)
    aa = T(a)
    if _hmat_on_backend(backend, x)
        if global_index
            _gpu_permute_in!(xloc, x, perm_d, aa, p)
        else
            _gpu_copy_scale!(xloc, x, aa)
        end
    else
        xt = global_index ? _nnca_permute_in(x, host_perm, p) : (a == 1 ? x : a .* x)
        if global_index && a != 1
            xt = a .* xt
        end
        copyto!(xloc, T.(xt))
    end
    return
end

function _gpu_unstage_y!(y, ysrc, b, global_index, perm_d, host_perm, p, ::Type{T}) where {T}
    backend = KernelAbstractions.get_backend(ysrc)
    bb = T(b)
    if _hmat_on_backend(backend, y)
        if global_index
            _gpu_permute_out!(y, ysrc, perm_d, bb, p)
        else
            _gpu_axpy_out!(y, ysrc, bb)
        end
    else
        yt = Vector{T}(undef, length(ysrc))
        copyto!(yt, ysrc)
        if global_index
            _nnca_permute_out!(y, yt, host_perm, p, b)
        elseif iszero(b)
            copyto!(y, yt)
        else
            y .= T(b) .* y .+ yt
        end
    end
    return
end

"""
    gpu_wrap(A, device)

`device === :host` returns `A`. Otherwise [`gpu`](@ref)`(A; device)`.
"""
gpu_wrap(A, device::Symbol) = device === :host ? A : gpu(A; device=device)

function gpu_wrap(A::NNCAMatrix, device::Symbol)
    device === :host && return A
    size(A, 1) == size(A, 2) || return A
    return gpu(A; device=device)
end

# =============================================================================
# Flatten SMatrix leaves to scalar blocks
# =============================================================================

_hmat_scalar_type(::Type{T}) where {T <: Number} = T
_hmat_scalar_type(::Type{S}) where {S <: SMatrix} = eltype(S)
_hmat_block_pq(::Type{T}) where {T <: Number} = (1, 1)
_hmat_block_pq(::Type{S}) where {p, q, S <: SMatrix{p, q}} = (p, q)

function _flatten_blocks(M::Matrix{T}) where {T <: Number}
    return M
end

function _flatten_blocks(M::Matrix{S}) where {T, p, q, S <: SMatrix{p, q, T}}
    m, n = size(M)
    out = Matrix{T}(undef, p * m, q * n)
    @inbounds for j in 1:n, i in 1:m
        Bij = M[i, j]
        i0 = p * (i - 1)
        j0 = q * (j - 1)
        for b in 1:q, a in 1:p
            out[i0 + a, j0 + b] = Bij[a, b]
        end
    end
    return out
end

function _flatten_rk(R::RkMatrix{T}) where {T <: Number}
    return R.A, R.B
end

function _flatten_rk(R::RkMatrix{S}) where {T, p, q, S <: SMatrix{p, q, T}}
    return _flatten_blocks(R.A), _flatten_blocks(R.B)
end

# =============================================================================
# Packed H-matrix
# =============================================================================

"""
    GPUHMatrix{T}

Packed H-matrix for GPU (or KA-CPU) matvec. Scalar size `(nrows, ncols)`;
tensor H-matrices are flattened to `(p n)×(q n)`. `mul!` with `x`/`y` on
`backend` (e.g. `CuArray`) permutes and applies in place on the device.
"""
mutable struct GPUHMatrix{T, VD, VI, B} <: AbstractStructuredMatrix{T}
    nrows::Int
    ncols::Int
    p::Int
    q::Int
    rowperm::Vector{Int}
    colperm::Vector{Int}
    backend::B
    xloc::VD
    yloc::VD
    tloc::VD
    # dense leaves
    dense_row0::VI
    dense_col0::VI
    dense_m::VI
    dense_n::VI
    dense_ptr::VI
    dense_prefix::VI
    dense_data::VD
    n_dense::Int
    n_dense_rows::Int
    # Rk leaves
    rk_row0::VI
    rk_col0::VI
    rk_m::VI
    rk_n::VI
    rk_r::VI
    rk_Aptr::VI
    rk_Bptr::VI
    rk_tptr::VI
    rk_rprefix::VI
    rk_mprefix::VI
    rk_A::VD
    rk_B::VD
    n_rk::Int
    n_rk_rows::Int
    n_rk_rank::Int
    dense_colors::Vector{Vector{Int32}}
    rk_colors::Vector{Vector{Int32}}
    rowperm_d::VI
    colperm_d::VI
    dense_color_ids::Vector{VI}
    rk_color_ids::Vector{VI}
    max_dense_m::Int
    max_rk_m::Int
    max_rk_r::Int
end

Base.size(A::GPUHMatrix) = (A.nrows, A.ncols)
Base.eltype(::GPUHMatrix{T}) where {T} = T
rowperm(A::GPUHMatrix) = A.rowperm
colperm(A::GPUHMatrix) = A.colperm

function Base.show(io::IO, A::GPUHMatrix)
    return print(io, "GPUHMatrix{", eltype(A), "} $(A.nrows)×$(A.ncols) ",
        "dense=$(A.n_dense) rk=$(A.n_rk)")
end
Base.show(io::IO, ::MIME"text/plain", A::GPUHMatrix) = show(io, A)

function compression_ratio(A::GPUHMatrix)
    ns = Base.summarysize(A)
    return (length(A) * sizeof(eltype(A))) / ns
end

"""
    gpu(H::HMatrix; device=:cuda, T=nothing)
    gpu(A::NNCAMatrix; device=:cuda, T=nothing)

Upload a packed apply copy. `device=:cuda` needs `using CUDA`; `:cpu` uses the
KernelAbstractions CPU backend. `T` defaults to the scalar eltype (`Float32`
is allowed for faster device GEMV). Host `x`/`y` permute on the CPU and copy
once per apply; `CuArray` (or other `A.backend`) vectors stay on the device.
"""
function gpu(H::HMatrix; device::Symbol = :cuda, T = nothing)
    Te = eltype(H)
    S = something(T, _hmat_scalar_type(Te))
    S === Float32 || S === Float64 || throw(ArgumentError(
        "gpu T must be Float32 or Float64; got $S"))
    backend = _hmat_ka_backend(device)
    return _gpu_hmatrix(H, S, backend)
end

gpu(H::Hermitian{<:Any, <:HMatrix}; kwargs...) = gpu(parent(H); kwargs...)
gpu(A::GPUHMatrix; kwargs...) = A
gpu_wrap(A::GPUHMatrix, ::Symbol) = A

function _gpu_hmatrix(H::HMatrix, ::Type{S}, backend) where {S}
    Te = eltype(H)
    p, q = _hmat_block_pq(Te)
    nrows = p * size(H, 1)
    ncols = q * size(H, 2)
    rperm = p == 1 ? collect(rowperm(H)) : _expand_perm(collect(rowperm(H)), p)
    cperm = q == 1 ? collect(colperm(H)) : _expand_perm(collect(colperm(H)), q)
    ls = leaves(H)
    d_row0 = Int32[]
    d_col0 = Int32[]
    d_m = Int32[]
    d_n = Int32[]
    d_ptr = Int32[]
    d_data = S[]
    r_row0 = Int32[]
    r_col0 = Int32[]
    r_m = Int32[]
    r_n = Int32[]
    r_r = Int32[]
    r_Aptr = Int32[]
    r_Bptr = Int32[]
    r_tptr = Int32[]
    r_A = S[]
    r_B = S[]
    tacc = 0
    for leaf in ls
        hasdata(leaf) || continue
        d = data(leaf)
        ir = expand_range(rowrange(leaf), p)
        jr = expand_range(colrange(leaf), q)
        if d isa RkMatrix
            Af, Bf = _flatten_rk(d)
            AfS = S.(Af)
            BfS = S.(Bf)
            push!(r_row0, Int32(first(ir)))
            push!(r_col0, Int32(first(jr)))
            push!(r_m, Int32(size(AfS, 1)))
            push!(r_n, Int32(size(BfS, 1)))
            push!(r_r, Int32(size(AfS, 2)))
            push!(r_Aptr, Int32(length(r_A) + 1))
            push!(r_Bptr, Int32(length(r_B) + 1))
            push!(r_tptr, Int32(tacc + 1))
            append!(r_A, vec(AfS))
            append!(r_B, vec(BfS))
            tacc += size(AfS, 2)
        else
            M = _flatten_blocks(d isa AbstractMatrix ? Matrix(d) : d)
            MS = S.(M)
            push!(d_row0, Int32(first(ir)))
            push!(d_col0, Int32(first(jr)))
            push!(d_m, Int32(size(MS, 1)))
            push!(d_n, Int32(size(MS, 2)))
            push!(d_ptr, Int32(length(d_data) + 1))
            append!(d_data, vec(MS))
        end
    end
    n_dense = length(d_m)
    n_rk = length(r_m)
    d_prefix = ones(Int32, n_dense + 1)
    @inbounds for i in 1:n_dense
        d_prefix[i + 1] = d_prefix[i] + d_m[i]
    end
    r_rprefix = ones(Int32, n_rk + 1)
    r_mprefix = ones(Int32, n_rk + 1)
    @inbounds for i in 1:n_rk
        r_rprefix[i + 1] = r_rprefix[i] + r_r[i]
        r_mprefix[i + 1] = r_mprefix[i] + r_m[i]
    end
    n_dense_rows = Int(d_prefix[end] - 1)
    n_rk_rows = Int(r_mprefix[end] - 1)
    n_rk_rank = Int(r_rprefix[end] - 1)
    d_colors = _hmat_row_colors(d_row0, d_m)
    r_colors = _hmat_row_colors(r_row0, r_m)
    empty32 = Int32[]
    emptyS = S[]
    to(v) = _hmat_to_backend(backend, v)
    xloc = KernelAbstractions.allocate(backend, S, ncols)
    yloc = KernelAbstractions.allocate(backend, S, nrows)
    tloc = KernelAbstractions.allocate(backend, S, max(n_rk_rank, 1))
    rowperm_d = to(Int32.(rperm))
    colperm_d = to(Int32.(cperm))
    d_color_ids = typeof(rowperm_d)[to(ids) for ids in d_colors if !isempty(ids)]
    r_color_ids = typeof(rowperm_d)[to(ids) for ids in r_colors if !isempty(ids)]
    return GPUHMatrix{S, typeof(xloc), typeof(to(d_prefix)), typeof(backend)}(
        nrows, ncols, p, q, rperm, cperm, backend,
        xloc, yloc, tloc,
        to(n_dense == 0 ? empty32 : d_row0),
        to(n_dense == 0 ? empty32 : d_col0),
        to(n_dense == 0 ? empty32 : d_m),
        to(n_dense == 0 ? empty32 : d_n),
        to(n_dense == 0 ? empty32 : d_ptr),
        to(d_prefix),
        to(isempty(d_data) ? emptyS : d_data),
        n_dense, n_dense_rows,
        to(n_rk == 0 ? empty32 : r_row0),
        to(n_rk == 0 ? empty32 : r_col0),
        to(n_rk == 0 ? empty32 : r_m),
        to(n_rk == 0 ? empty32 : r_n),
        to(n_rk == 0 ? empty32 : r_r),
        to(n_rk == 0 ? empty32 : r_Aptr),
        to(n_rk == 0 ? empty32 : r_Bptr),
        to(n_rk == 0 ? empty32 : r_tptr),
        to(r_rprefix),
        to(r_mprefix),
        to(isempty(r_A) ? emptyS : r_A),
        to(isempty(r_B) ? emptyS : r_B),
        n_rk, n_rk_rows, n_rk_rank,
        d_colors, r_colors,
        rowperm_d, colperm_d, d_color_ids, r_color_ids,
        _hmat_max0(d_m), _hmat_max0(r_m), _hmat_max0(r_r),
    )
end

# Interval coloring: leaves whose row ranges overlap cannot share a color.
# Equal-`row0` only is not enough — a coarse block [1,64) overlaps a fine
# child [33,64).
function _hmat_row_colors(row0::Vector{Int32}, m::Vector{Int32})
    n = length(row0)
    n == 0 && return Vector{Vector{Int32}}()
    order = sortperm(row0)
    ends = Int32[]
    colors = Vector{Vector{Int32}}()
    @inbounds for idx in order
        r0 = row0[idx]
        r1 = r0 + m[idx]
        cfound = 0
        for c in eachindex(ends)
            if ends[c] <= r0
                cfound = c
                break
            end
        end
        if cfound == 0
            push!(ends, r1)
            push!(colors, Int32[Int32(idx)])
        else
            ends[cfound] = r1
            push!(colors[cfound], Int32(idx))
        end
    end
    return colors
end

# One thread per output row of a leaf (ids = leaves in one row-color).
@kernel function _gpu_h_dense_row!(y, x, row0, col0, m, n, ptr, data, ids)
    i, t = @index(Global, NTuple)
    ℓ = ids[t]
    mm = m[ℓ]
    ii = Int32(i)
    if ii <= mm
        nn = n[ℓ]
        p = ptr[ℓ]
        r0 = row0[ℓ]
        c0 = col0[ℓ]
        i0 = ii - Int32(1)
        acc = zero(eltype(y))
        @inbounds for j in Int32(0):(nn - Int32(1))
            acc += data[p + i0 + j * mm] * x[c0 + j]
        end
        @inbounds y[r0 + i0] += acc
    end
end

# One thread per (leaf, rank) — writes disjoint t slots, no coloring.
@kernel function _gpu_h_rk1!(tbuf, x, col0, n, r, Bptr, tptr, B)
    k, ℓ = @index(Global, NTuple)
    kk = Int32(k)
    ll = Int32(ℓ)
    rr = r[ll]
    if kk <= rr
        nn = n[ll]
        bp = Bptr[ll]
        c0 = col0[ll]
        k0 = kk - Int32(1)
        acc = zero(eltype(tbuf))
        @inbounds for j in Int32(0):(nn - Int32(1))
            acc += B[bp + j + k0 * nn] * x[c0 + j]
        end
        @inbounds tbuf[tptr[ll] + k0] = acc
    end
end

@kernel function _gpu_h_rk2_row!(y, tbuf, row0, m, r, Aptr, tptr, A, ids)
    i, g = @index(Global, NTuple)
    ℓ = ids[g]
    mm = m[ℓ]
    ii = Int32(i)
    if ii <= mm
        rr = r[ℓ]
        ap = Aptr[ℓ]
        tp = tptr[ℓ]
        r0 = row0[ℓ]
        i0 = ii - Int32(1)
        acc = zero(eltype(y))
        @inbounds for k in Int32(0):(rr - Int32(1))
            acc += A[ap + i0 + k * mm] * tbuf[tp + k]
        end
        @inbounds y[r0 + i0] += acc
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::GPUHMatrix{T}, x::AbstractVector,
        a::Number = 1, b::Number = 0; global_index = use_global_index()) where {T}
    length(x) == A.ncols && length(y) == A.nrows || throw(DimensionMismatch())
    _gpu_stage_x!(A.xloc, x, a, global_index, A.colperm_d, A.colperm, 1, T)
    fill!(A.yloc, zero(T))
    backend = A.backend
    if A.n_dense > 0 && A.max_dense_m > 0
        kern = _gpu_h_dense_row!(backend)
        for ids in A.dense_color_ids
            isempty(ids) && continue
            kern(A.yloc, A.xloc, A.dense_row0, A.dense_col0, A.dense_m, A.dense_n,
                A.dense_ptr, A.dense_data, ids; ndrange = (A.max_dense_m, length(ids)))
            KernelAbstractions.synchronize(backend)
        end
    end
    if A.n_rk > 0 && A.max_rk_r > 0
        k1 = _gpu_h_rk1!(backend)
        k1(A.tloc, A.xloc, A.rk_col0, A.rk_n, A.rk_r, A.rk_Bptr, A.rk_tptr,
            A.rk_B; ndrange = (A.max_rk_r, A.n_rk))
        KernelAbstractions.synchronize(backend)
        if A.max_rk_m > 0
            k2 = _gpu_h_rk2_row!(backend)
            for ids in A.rk_color_ids
                isempty(ids) && continue
                k2(A.yloc, A.tloc, A.rk_row0, A.rk_m, A.rk_r, A.rk_Aptr, A.rk_tptr,
                    A.rk_A, ids; ndrange = (A.max_rk_m, length(ids)))
                KernelAbstractions.synchronize(backend)
            end
        end
    end
    KernelAbstractions.synchronize(backend)
    _gpu_unstage_y!(y, A.yloc, b, global_index, A.rowperm_d, A.rowperm, 1, T)
    return y
end

function Base.:*(A::GPUHMatrix{T}, x::AbstractVector) where {T}
    return mul!(similar(x, T, A.nrows), A, x)
end

# =============================================================================
# Packed NNCA
# =============================================================================

"""
    GPUNNCAMatrix{T}

Packed NNCA H² for GPU (or KA-CPU) apply. Size `(p n)×(p n)`. Device-resident
`x`/`y` skip the host permute/copy.
"""
mutable struct GPUNNCAMatrix{T, VD, VI, B} <: AbstractStructuredMatrix{T}
    n::Int
    p::Int
    perm::Vector{Int}
    backend::B
    xloc::VD
    yloc::VD
    outgoing::VD
    incoming::VD
    potential::VD
    l2p_data::VD
    l2p_ptr::VI
    l2p_m::VI
    l2p_n::VI
    out_ptr::VI
    out_len::VI
    in_ptr::VI
    isleaf::VI
    row0::VI
    ch_ptr::VI
    ch_ids::VI
    m2l_dsptr::VI
    m2l_src::VI
    m2l_ptr::VI
    m2l_m::VI
    m2l_n::VI
    m2l_data::VD
    near_dsptr::VI
    near_src::VI
    near_ptr::VI
    near_m::VI
    near_n::VI
    near_data::VD
    self_ptr::VI
    self_m::VI
    self_data::VD
    levels::Vector{Vector{Int32}}
    leaf_ids::Vector{Int32}
    nn::Int
    perm_d::VI
    level_ids_d::Vector{VI}
    leaf_ids_d::VI
    max_l2p_m::Int
    max_l2p_n::Int
end

Base.size(A::GPUNNCAMatrix) = (A.p * A.n, A.p * A.n)
Base.eltype(::GPUNNCAMatrix{T}) where {T} = T
rowperm(A::GPUNNCAMatrix) = A.p == 1 ? A.perm : _expand_perm(A.perm, A.p)
colperm(A::GPUNNCAMatrix) = rowperm(A)

function Base.show(io::IO, A::GPUNNCAMatrix)
    sz = size(A)
    return print(io, "GPUNNCAMatrix{", eltype(A), "} $(sz[1])×$(sz[2]) p=$(A.p)")
end
Base.show(io::IO, ::MIME"text/plain", A::GPUNNCAMatrix) = show(io, A)

gpu(A::GPUNNCAMatrix; kwargs...) = A
gpu_wrap(A::GPUNNCAMatrix, ::Symbol) = A

function gpu(A::NNCAMatrix{TT}; device::Symbol = :cuda, T = nothing) where {TT}
    S = something(T, TT)
    S === Float32 || S === Float64 || throw(ArgumentError(
        "gpu T must be Float32 or Float64; got $S"))
    backend = _hmat_ka_backend(device)
    return _gpu_nnca(A, S, backend)
end

function _gpu_nnca(A::NNCAMatrix, ::Type{S}, backend) where {S}
    nn = length(A.boxes)
    p = A.p
    l2p_m = zeros(Int32, nn)
    l2p_n = zeros(Int32, nn)
    l2p_ptr = ones(Int32, nn)
    l2p_data = S[]
    out_ptr = ones(Int32, nn)
    out_len = zeros(Int32, nn)
    in_ptr = ones(Int32, nn)
    isleafv = zeros(Int32, nn)
    row0 = zeros(Int32, nn)
    ch_ptr = ones(Int32, nn + 1)
    ch_ids = Int32[]
    out_acc = 1
    in_acc = 1
    id2 = A.id2node
    for id in 1:nn
        b = A.boxes[id]
        node = id2[id]
        isleafv[id] = isleaf(node) ? Int32(1) : Int32(0)
        if isleaf(node)
            ir = expand_range(index_range(node), p)
            row0[id] = Int32(first(ir))
        end
        Lm, Ln = size(b.L2P)
        l2p_m[id] = Int32(Lm)
        l2p_n[id] = Int32(Ln)
        l2p_ptr[id] = Int32(length(l2p_data) + 1)
        if Lm > 0 && Ln > 0
            append!(l2p_data, S.(vec(b.L2P)))
        end
        out_len[id] = Int32(length(b.outgoing))
        out_ptr[id] = Int32(out_acc)
        out_acc += max(length(b.outgoing), 0)
        in_ptr[id] = Int32(in_acc)
        in_acc += max(length(b.incoming), 0)
        ch_ptr[id] = Int32(length(ch_ids) + 1)
        if !isleaf(node)
            for c in children(node)
                push!(ch_ids, Int32(node_id(c)))
            end
        end
    end
    ch_ptr[nn + 1] = Int32(length(ch_ids) + 1)

    m2l_dsptr = Int32.(A.m2l_dsptr)
    m2l_src = Int32.(A.m2l_src)
    m2l_ptr = Int32.(A.m2l_ptr)
    m2l_m = Int32.(A.m2l_m)
    m2l_n = Int32.(A.m2l_n)
    m2l_data = S.(A.m2l_data)

    near_dsptr = Int32.(A.near_dsptr)
    near_src = Int32.(A.near_src)
    near_ptr = Int32.(A.near_ptr)
    near_m = Int32.(A.near_m)
    near_n = Int32.(A.near_n)
    near_data = S.(A.near_data)
    self_ptr = zeros(Int32, nn)
    self_m = zeros(Int32, nn)
    self_data = S[]
    for id in 1:nn
        b = A.boxes[id]
        isleafv[id] == 1 || continue
        if !isempty(b.self)
            self_ptr[id] = Int32(length(self_data) + 1)
            self_m[id] = Int32(size(b.self, 1))
            append!(self_data, S.(vec(b.self)))
        end
    end

    nscal = p * A.n
    to(v) = _hmat_to_backend(backend, v)
    empty32 = Int32[]
    emptyS = S[]
    xloc = KernelAbstractions.allocate(backend, S, nscal)
    yloc = KernelAbstractions.allocate(backend, S, nscal)
    outgoing = KernelAbstractions.allocate(backend, S, max(out_acc - 1, 1))
    incoming = KernelAbstractions.allocate(backend, S, max(in_acc - 1, 1))
    potential = KernelAbstractions.allocate(backend, S, nscal)
    levels = [Int32.(lev) for lev in A.levels]
    leaf_ids = Int32.(A.leaf_ids)
    perm_d = to(Int32.(A.perm))
    level_ids_d = typeof(perm_d)[to(lev) for lev in levels if !isempty(lev)]
    leaf_ids_d = to(isempty(leaf_ids) ? empty32 : leaf_ids)
    return GPUNNCAMatrix{S, typeof(xloc), typeof(to(l2p_ptr)), typeof(backend)}(
        A.n, p, copy(A.perm), backend,
        xloc, yloc, outgoing, incoming, potential,
        to(isempty(l2p_data) ? emptyS : l2p_data),
        to(l2p_ptr), to(l2p_m), to(l2p_n),
        to(out_ptr), to(out_len), to(in_ptr), to(isleafv), to(row0),
        to(ch_ptr), to(isempty(ch_ids) ? empty32 : ch_ids),
        to(m2l_dsptr),
        to(isempty(m2l_src) ? empty32 : m2l_src),
        to(isempty(m2l_ptr) ? empty32 : m2l_ptr),
        to(isempty(m2l_m) ? empty32 : m2l_m),
        to(isempty(m2l_n) ? empty32 : m2l_n),
        to(isempty(m2l_data) ? emptyS : m2l_data),
        to(near_dsptr),
        to(isempty(near_src) ? empty32 : near_src),
        to(isempty(near_ptr) ? empty32 : near_ptr),
        to(isempty(near_m) ? empty32 : near_m),
        to(isempty(near_n) ? empty32 : near_n),
        to(isempty(near_data) ? emptyS : near_data),
        to(self_ptr), to(self_m),
        to(isempty(self_data) ? emptyS : self_data),
        levels, leaf_ids, nn,
        perm_d, level_ids_d, leaf_ids_d,
        _hmat_max0(l2p_m), _hmat_max0(l2p_n),
    )
end

@kernel function _gpu_nnca_m2m!(out, x, l2p, lptr, lm, ln, isleaf, row0,
        ch_ptr, ch_ids, optr, olen, ids)
    k, t = @index(Global, NTuple)
    id = ids[t]
    m = lm[id]
    n = ln[id]
    kk = Int32(k)
    if kk <= n && m != 0
        lp = lptr[id]
        op = optr[id]
        k0 = kk - Int32(1)
        acc = zero(eltype(out))
        if isleaf[id] == Int32(1)
            r0 = row0[id]
            @inbounds for i in Int32(0):(m - Int32(1))
                acc += l2p[lp + i + k0 * m] * x[r0 + i]
            end
        else
            i = Int32(0)
            c0 = ch_ptr[id]
            c1 = ch_ptr[id + Int32(1)]
            while c0 < c1
                cid = ch_ids[c0]
                clen = olen[cid]
                cp = optr[cid]
                j = Int32(0)
                while j < clen && i < m
                    acc += l2p[lp + i + k0 * m] * out[cp + j]
                    i += Int32(1)
                    j += Int32(1)
                end
                c0 += Int32(1)
            end
        end
        @inbounds out[op + k0] = acc
    end
end

@kernel function _gpu_nnca_m2l!(inc, out, dsptr, src, ptr, mm, nn, data, iptr, optr, ln, ids)
    i, t = @index(Global, NTuple)
    id = ids[t]
    n_in = ln[id]
    ii = Int32(i)
    if ii <= n_in
        i0 = ii - Int32(1)
        dp = iptr[id]
        acc = zero(eltype(inc))
        a = dsptr[id]
        b = dsptr[id + Int32(1)]
        while a < b
            m = mm[a]
            if ii <= m
                s = src[a]
                p = ptr[a]
                n = nn[a]
                sp = optr[s]
                @inbounds for j in Int32(0):(n - Int32(1))
                    acc += data[p + i0 + j * m] * out[sp + j]
                end
            end
            a += Int32(1)
        end
        @inbounds inc[dp + i0] = acc
    end
end

@kernel function _gpu_nnca_l2l!(pot, inc, l2p, lptr, lm, ln, isleaf, row0,
        ch_ptr, ch_ids, iptr, ids)
    i, t = @index(Global, NTuple)
    id = ids[t]
    m = lm[id]
    n = ln[id]
    ii = Int32(i)
    if ii <= m && n != 0
        lp = lptr[id]
        ip = iptr[id]
        i0 = ii - Int32(1)
        if isleaf[id] == Int32(1)
            r0 = row0[id]
            acc = zero(eltype(pot))
            @inbounds for k in Int32(0):(n - Int32(1))
                acc += l2p[lp + i0 + k * m] * inc[ip + k]
            end
            @inbounds pot[r0 + i0] = acc
        else
            remain = i0
            c0 = ch_ptr[id]
            c1 = ch_ptr[id + Int32(1)]
            cid = Int32(0)
            localj = Int32(0)
            hit = Int32(0)
            while c0 < c1
                cid = ch_ids[c0]
                clen = ln[cid]
                if remain < clen
                    localj = remain
                    hit = Int32(1)
                    c0 = c1
                else
                    remain -= clen
                    c0 += Int32(1)
                end
            end
            if hit != 0
                acc = zero(eltype(inc))
                @inbounds for k in Int32(0):(n - Int32(1))
                    acc += l2p[lp + i0 + k * m] * inc[ip + k]
                end
                @inbounds inc[iptr[cid] + localj] += acc
            end
        end
    end
end

@kernel function _gpu_nnca_near!(pot, x, dsptr, src, ptr, mm, nn, data,
        sptr, sm, sdata, row0, lm, ids)
    i, t = @index(Global, NTuple)
    id = ids[t]
    nrows = lm[id]
    ii = Int32(i)
    if ii <= nrows
        i0 = ii - Int32(1)
        r0 = row0[id]
        acc = pot[r0 + i0]
        ms = sm[id]
        if ms > 0 && ii <= ms
            p = sptr[id]
            @inbounds for j in Int32(0):(ms - Int32(1))
                acc += sdata[p + i0 + j * ms] * x[r0 + j]
            end
        end
        a = dsptr[id]
        b = dsptr[id + Int32(1)]
        while a < b
            m = mm[a]
            if ii <= m
                s = src[a]
                p = ptr[a]
                n = nn[a]
                c0 = row0[s]
                @inbounds for j in Int32(0):(n - Int32(1))
                    acc += data[p + i0 + j * m] * x[c0 + j]
                end
            end
            a += Int32(1)
        end
        @inbounds pot[r0 + i0] = acc
    end
end

function _gpu_launch2(backend, kern, n1, n2, args...)
    (n1 < 1 || n2 < 1) && return
    kern(args...; ndrange = (n1, n2))
    KernelAbstractions.synchronize(backend)
    return
end

function LinearAlgebra.mul!(y::AbstractVector, A::GPUNNCAMatrix{T}, x::AbstractVector,
        a::Number = 1, b::Number = 0; global_index = use_global_index()) where {T}
    nscal = A.p * A.n
    length(x) == nscal && length(y) == nscal || throw(DimensionMismatch())
    _gpu_stage_x!(A.xloc, x, a, global_index, A.perm_d, A.perm, A.p, T)
    fill!(A.outgoing, zero(T))
    fill!(A.incoming, zero(T))
    fill!(A.potential, zero(T))
    backend = A.backend
    m2m = _gpu_nnca_m2m!(backend)
    m2l = _gpu_nnca_m2l!(backend)
    l2l = _gpu_nnca_l2l!(backend)
    near = _gpu_nnca_near!(backend)
    for ids in Iterators.reverse(A.level_ids_d)
        isempty(ids) && continue
        _gpu_launch2(backend, m2m, A.max_l2p_n, length(ids),
            A.outgoing, A.xloc, A.l2p_data, A.l2p_ptr, A.l2p_m, A.l2p_n,
            A.isleaf, A.row0, A.ch_ptr, A.ch_ids, A.out_ptr, A.out_len, ids)
    end
    for ids in A.level_ids_d
        isempty(ids) && continue
        _gpu_launch2(backend, m2l, A.max_l2p_n, length(ids),
            A.incoming, A.outgoing, A.m2l_dsptr, A.m2l_src, A.m2l_ptr,
            A.m2l_m, A.m2l_n, A.m2l_data, A.in_ptr, A.out_ptr, A.l2p_n, ids)
    end
    for ids in A.level_ids_d
        isempty(ids) && continue
        _gpu_launch2(backend, l2l, A.max_l2p_m, length(ids),
            A.potential, A.incoming, A.l2p_data, A.l2p_ptr, A.l2p_m, A.l2p_n,
            A.isleaf, A.row0, A.ch_ptr, A.ch_ids, A.in_ptr, ids)
    end
    if !isempty(A.leaf_ids_d)
        _gpu_launch2(backend, near, A.max_l2p_m, length(A.leaf_ids_d),
            A.potential, A.xloc, A.near_dsptr, A.near_src, A.near_ptr,
            A.near_m, A.near_n, A.near_data, A.self_ptr, A.self_m, A.self_data,
            A.row0, A.l2p_m, A.leaf_ids_d)
    end
    _gpu_unstage_y!(y, A.potential, b, global_index, A.perm_d, A.perm, A.p, T)
    return y
end

function Base.:*(A::GPUNNCAMatrix{T}, x::AbstractVector) where {T}
    return mul!(similar(x, T, size(A, 1)), A, x)
end
