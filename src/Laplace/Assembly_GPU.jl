# 2-D Laplace dense H, G on a GPU via KernelAbstractions (Julia kernels, not CUDA-C).
# Far field matches `_far_nodal_scalar!`. Near / singular stays on the CPU
# (`integrate_element`) unless `near=:gpu`. Solve is host LinearSolve.

export H_G_gpu, gpu_float_support

H_G_gpu(dad::BEMdata; kwargs...) = throw(ArgumentError(
    "method=:gpu supports 2-D Laplace and 2-D isotropic Elasticity; got $(typeof(dad.properties))"))

using KernelAbstractions

const _CUDA_PKGID = Base.PkgId(
    Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")

"""
    gpu_float_support()

Floating-point formats this GPU can run. Needs `using CUDA`. Host `Float32` /
`Float64` are always listed; device flags come from compute capability.
"""
function gpu_float_support()
    if !haskey(Base.loaded_modules, _CUDA_PKGID)
        return (
            available = false,
            name = nothing,
            capability = nothing,
            Float16 = missing,
            Float32 = true,
            Float64 = true,
            BFloat16 = missing,
            TensorFloat32 = missing,
            Float8 = missing,
            fp64_fp32_ratio = nothing,
            recommended = Float32,
            memory_bytes = nothing,
            message = "Load CUDA.jl (`using CUDA`) to query the device.",
        )
    end
    CUDA = Base.loaded_modules[_CUDA_PKGID]
    if !isdefined(CUDA, :functional) || !CUDA.functional()
        return (
            available = false,
            name = nothing,
            capability = nothing,
            Float16 = missing,
            Float32 = true,
            Float64 = true,
            BFloat16 = missing,
            TensorFloat32 = missing,
            Float8 = missing,
            fp64_fp32_ratio = nothing,
            recommended = Float32,
            memory_bytes = nothing,
            message = "CUDA.jl is loaded but reports no functional GPU.",
        )
    end
    dev = CUDA.device()
    cap = CUDA.capability(dev)
    major = Int(cap.major)
    minor = Int(cap.minor)
    ratio = _cuda_fp64_ratio(CUDA, dev)
    mem = try
        Int(CUDA.totalmem(dev))
    catch
        try
            Int(CUDA.total_memory())
        catch
            nothing
        end
    end
    return (
        available = true,
        name = CUDA.name(dev),
        capability = cap,
        Float16 = major > 5 || (major == 5 && minor >= 3),
        Float32 = true,
        Float64 = true,
        BFloat16 = major >= 8,
        TensorFloat32 = major >= 8,
        Float8 = major > 8 || (major == 8 && minor >= 9),
        fp64_fp32_ratio = ratio,
        recommended = Float32,
        memory_bytes = mem,
        message = nothing,
    )
end

function _cuda_fp64_ratio(CUDA, dev)
    for attr in (
            :DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
            :CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
        )
        isdefined(CUDA, attr) || continue
        try
            r = Int(CUDA.attribute(dev, getproperty(CUDA, attr)))
            r > 0 && return 1 // r
        catch
        end
    end
    # Turing GeForce (SM 7.5) is 1/32 of FP32.
    cap = CUDA.capability(dev)
    return Int(cap.major) >= 8 ? (1 // 64) : (1 // 32)
end

"""
    H_G_gpu(dad::BEMdata{<:Laplace}; T=Float32, npg=20, near=:cpu,
            near_factor=1.5, device=:cuda, threaded=true)

Dense 2-D Laplace `H` and `G` with a KernelAbstractions far-field kernel.

| keyword | meaning |
|---------|---------|
| `T` | `Float32` (default, native GPU rate) or `Float64` (~1/32 on GeForce) |
| `near` | `:cpu` (default) overwrites near/singular with [`integrate_element`](@ref); `:gpu` leaves far lumping |
| `device` | `:cuda` (default) or `:cpu` (same kernel on the KA CPU backend) |

`using CUDA` is required for `device=:cuda`. The linear solve stays on the host
(`solve` / LinearSolve.jl).
"""
function H_G_gpu(
        dad::BEMdata{<:Laplace};
        T::Type{<:AbstractFloat} = Float32,
        npg::Integer = 20,
        near::Symbol = :cpu,
        near_factor::Real = 1.5,
        device::Symbol = :cuda,
        threaded::Bool = true,
    )
    dad.dimension == 2 || throw(ArgumentError("H_G_gpu supports 2-D Laplace only"))
    (near === :cpu || near === :gpu) ||
        throw(ArgumentError("near must be :cpu or :gpu; got $near"))
    (device === :cuda || device === :cpu) ||
        throw(ArgumentError("device must be :cuda or :cpu; got $device"))
    T === Float32 || T === Float64 ||
        throw(ArgumentError("H_G_gpu T must be Float32 or Float64; got $T"))

    _init_quadrature!(dad, npg)
    packed = _pack_laplace2d(dad, T)
    skip_ie = (near === :cpu) ? _near_skip_mask(dad; near_factor = float(near_factor)) :
        zeros(UInt8, packed.nt, packed.ne)
    backend = _ka_backend(device)
    H, G = _far_assemble_ka(backend, packed, T; skip_ie = skip_ie)
    if near === :cpu
        _correct_near_cpu!(H, G, dad; near_factor = float(near_factor),
            threaded = threaded)
    end
    _rowsum_diag!(H)
    set_cache!(dad; H, G)
    return H, G
end

function _ka_backend(device::Symbol)
    device === :cpu && return CPU()
    CUDA = _require_cuda()
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl reports no functional GPU. Pass device=:cpu to run the same kernel on the host."))
    isdefined(CUDA, :CUDABackend) || throw(ArgumentError(
        "CUDA.CUDABackend is missing; KernelAbstractions must be loaded (it is a BEM dependency)."))
    return CUDA.CUDABackend()
end

function _require_cuda()
    haskey(Base.loaded_modules, _CUDA_PKGID) || throw(ArgumentError(
        "GPU assembly needs CUDA.jl loaded first (`using CUDA`). " *
        "Pass device=:cpu to exercise the KernelAbstractions kernel on the host."))
    return Base.loaded_modules[_CUDA_PKGID]
end

function _pack_boundary2d(dad::BEMdata, ::Type{T}) where {T<:AbstractFloat}
    dad.dimension == 2 || throw(ArgumentError("GPU assembly supports 2-D only"))
    nt = Int(dad.nt)
    n = Int(dad.n)
    n == 0 && throw(ArgumentError("H_G_gpu: no boundary nodes"))
    pts = dad.collocation
    length(pts) == nt || throw(DimensionMismatch("collocation length"))
    first(pts) isa SVector{2} || throw(ArgumentError("GPU assembly: 2-D points required"))

    px = Vector{T}(undef, nt)
    py = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        p = pts[i]
        px[i] = T(p[1])
        py[i] = T(p[2])
    end
    nx = Vector{T}(undef, n)
    ny = Vector{T}(undef, n)
    @inbounds for j in 1:n
        nj = dad.Normal[j]
        nx[j] = T(nj[1])
        ny[j] = T(nj[2])
    end

    elems = dad.elements
    ne = length(elems)
    wloc = dad.elem_weight

    nodecount = zeros(Int32, n)
    @inbounds for el in elems
        for node in el.index
            nodecount[node] += Int32(1)
        end
    end
    inc_ptr = Vector{Int32}(undef, n + 1)
    inc_ptr[1] = 1
    @inbounds for j in 1:n
        inc_ptr[j + 1] = inc_ptr[j] + nodecount[j]
    end
    nnz = Int(inc_ptr[end] - 1)
    inc_e = Vector{Int32}(undef, nnz)
    inc_w = Vector{T}(undef, nnz)
    fill!(nodecount, 0)
    @inbounds for e in 1:ne
        el = elems[e]
        for k in eachindex(el.index)
            node = Int(el.index[k])
            p = inc_ptr[node] + nodecount[node]
            nodecount[node] += Int32(1)
            inc_e[p] = Int32(e)
            inc_w[p] = T(el.Jacobian[k] * wloc[k])
        end
    end

    return (
        px = px, py = py, nx = nx, ny = ny,
        inc_ptr = inc_ptr, inc_e = inc_e, inc_w = inc_w,
        n = n, nt = nt, ne = ne,
    )
end

function _pack_laplace2d(dad::BEMdata{<:Laplace}, ::Type{T}) where {T<:AbstractFloat}
    p = _pack_boundary2d(dad, T)
    return (; p..., kcond = T(dad.properties.k))
end

"""Host Float64 near test (same as `_accumulate_element_scalar!`), packed as `UInt8[nt, ne]`."""
function _near_skip_mask(dad::BEMdata; near_factor::Float64)
    nt = Int(dad.nt)
    elems = dad.elements
    ne = length(elems)
    skip = zeros(UInt8, nt, ne)
    nf = near_factor
    @inbounds for e in 1:ne
        el = elems[e]
        xj = dad.Nodes[el.index]
        for i in 1:nt
            pf = point(dad, i)
            if _near_element(pf, xj, el; factor = nf) || _source_on_element(el, i)
                skip[i, e] = UInt8(1)
            end
        end
    end
    return skip
end

function _to_backend(backend, x::AbstractArray)
    y = KernelAbstractions.allocate(backend, eltype(x), size(x)...)
    copyto!(y, x)
    return y
end

function _far_assemble_ka(backend, packed, ::Type{T}; skip_ie) where {T<:AbstractFloat}
    nt = packed.nt
    n = packed.n
    ne = packed.ne
    H = KernelAbstractions.allocate(backend, T, nt, nt)
    G = KernelAbstractions.allocate(backend, T, nt, n)
    fill!(H, zero(T))
    fill!(G, zero(T))
    if nt == 0 || n == 0 || ne == 0
        return Array(H), Array(G)
    end
    px = _to_backend(backend, packed.px)
    py = _to_backend(backend, packed.py)
    nx = _to_backend(backend, packed.nx)
    ny = _to_backend(backend, packed.ny)
    inc_ptr = _to_backend(backend, packed.inc_ptr)
    inc_e = _to_backend(backend, packed.inc_e)
    inc_w = _to_backend(backend, packed.inc_w)
    skip = _to_backend(backend, skip_ie)
    eps2 = eps(T)^2
    kern = _laplace_far_kernel!(backend)
    kern(H, G, px, py, nx, ny, inc_ptr, inc_e, inc_w, skip, packed.kcond, eps2;
        ndrange = (nt, n))
    KernelAbstractions.synchronize(backend)
    return Array(H), Array(G)
end

@kernel function _laplace_far_kernel!(
        H, G, px, py, nx, ny,
        inc_ptr, inc_e, inc_w, skip_ie,
        kcond, eps2)
    i, j = @index(Global, NTuple)
    TT = eltype(H)
    pxi = px[i]
    pyi = py[i]
    rx = px[j] - pxi
    ry = py[j] - pyi
    R2 = rx * rx + ry * ry
    if i != j && R2 > eps2
        R = sqrt(R2)
        inv2π = one(TT) / (TT(2) * TT(π))
        U = -log(R) * inv2π / kcond
        Tn = (rx * nx[j] + ry * ny[j]) * inv2π / R2
        accH = zero(TT)
        accG = zero(TT)
        p = inc_ptr[j]
        b = inc_ptr[j + 1]
        while p < b
            e = inc_e[p]
            w = inc_w[p]
            p += one(typeof(p))
            if skip_ie[i, e] == UInt8(0)
                accH += Tn * w
                accG += U * w
            end
        end
        H[i, j] = accH
        G[i, j] = accG
    end
end

function _correct_near_cpu!(H, G, dad; near_factor::Float64, threaded::Bool)
    nf = near_factor
    elems = dad.elements
    _collocation_loop!(threaded, dad.nt) do i
        pf = point(dad, i)
        @inbounds for el in elems
            xj = dad.Nodes[el.index]
            if _near_element(pf, xj, el; factor = nf) || _source_on_element(el, i)
                nn = length(el)
                hloc = zeros(eltype(H), nn)
                gloc = zeros(eltype(G), nn)
                integrate_element(dad, el, xj, pf, hloc, gloc; source = i)
                for (k, j) in enumerate(el.index)
                    H[i, j] += hloc[k]
                    G[i, j] += gloc[k]
                end
            end
        end
    end
    return nothing
end
