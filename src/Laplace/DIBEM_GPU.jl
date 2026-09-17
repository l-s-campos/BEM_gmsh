# 2-D Laplace dense DIBEM on a GPU via KernelAbstractions (Julia kernels).
# Pairwise F, D and far RIM (IF, ID). Near Gauss stays on the host.

export DIBEM_gpu

"""
    DIBEM_gpu(dad::BEMdata{<:Laplace}; rbf=PHS(), T=Float64, npg=16,
              near_factor=1.5, device=:cuda, threaded=true)

Dense collocation DIBEM (`centers=:collocation`, lumped RIM) with
KernelAbstractions kernels for:

- ``F_{ij}=φ(‖x_i-x_j‖)`` and ``D_{ij}=u*(x_i,x_j)``
- far IF / ID nodal RIM (same skip mask as [`H_G_gpu`](@ref))

Near elements reuse [`_dibem_near_gauss!`](@ref). The CPD solve ``Fc=IF`` and
``M=D\\mathrm{diag}(c)`` stay on the host (Float64).

`rbf` must be a PHS (`PHS1`…`PHS7`) or [`FundamentalRBF`](@ref). `using CUDA`
is required for `device=:cuda`.
"""
function DIBEM_gpu(
        dad::BEMdata{<:Laplace};
        rbf = PHS(),
        T::Type{<:AbstractFloat} = Float64,
        npg::Integer = 16,
        near_factor::Real = 1.5,
        device::Symbol = :cuda,
        threaded::Bool = true,
        centers::Symbol = :collocation,
        rim::Symbol = :lumped,
        kwargs...,
    )
    dad.dimension == 2 || throw(ArgumentError("DIBEM_gpu supports 2-D Laplace only"))
    centers === :collocation || throw(ArgumentError(
        "DIBEM_gpu centers must be :collocation (got $centers)"))
    rim === :lumped || throw(ArgumentError(
        "DIBEM_gpu rim must be :lumped (got $rim)"))
    (device === :cuda || device === :cpu) ||
        throw(ArgumentError("device must be :cuda or :cpu; got $device"))
    T === Float32 || T === Float64 ||
        throw(ArgumentError("DIBEM_gpu T must be Float32 or Float64; got $T"))
    T === Float32 && @warn "DIBEM_gpu T=Float32: PHS F-solve is ill-conditioned; prefer T=Float64" maxlog=1
    kind = _gpu_rbf_kind(rbf)

    _init_quadrature!(dad, npg)
    packed = _pack_laplace2d(dad, T)
    skip_ie = _near_skip_mask(dad; near_factor = float(near_factor))
    backend = _ka_backend(device)
    epsa = T(AVOID_INF)
    F, D = _dibem_FD_ka(backend, packed, T, kind, epsa)
    IF, ID = _dibem_IFID_ka(backend, packed, T, kind, epsa, skip_ie)
    _dibem_correct_near_cpu!(IF, ID, dad, rbf; near_factor = float(near_factor),
        threaded = threaded)

    Fh = Float64.(F)
    Dh = Float64.(D)
    IFh = Float64.(IF)
    IDh = Float64.(ID)
    pts = all_points(dad)

    if rbf isa FundamentalRBF
        if dad.dimension == 2 && dad.ni == 0
            sd = assemble_sbm!(dad)
            Fh = Matrix{Float64}(sd.G)
        else
            ε = 1e-10
            @inbounds for i in 1:dad.nt
                Fh[i, i] += ε
            end
        end
        A = copy(Fh)
        @inbounds for i in 1:dad.nt
            A[i, i] = 1e-10
        end
        c = A \ IDh
        M = copy(Fh)
        @inbounds for i in 1:dad.nt
            M[i, i] = 0.0
        end
        M .*= c'
        @inbounds for i in 1:dad.nt
            M[i, i] = -sum(view(M, i, :)) + IDh[i]
        end
        set_cache!(dad; M, dibem_F=Fh, dibem_c=c, dibem_ID=IDh, dibem_rbf=rbf,
            dibem_method=:gpu_fs, dibem_D=Dh)
        return M
    end

    _dibem_ridge_F!(Fh)
    IP = _dibem_monomial_IP(dad, rbf)
    c = _dibem_poly_c(Fh, IFh, pts, rbf; IP=IP)
    M = Dh .* c'
    @inbounds for i in 1:dad.nt
        M[i, i] = 0
        M[i, i] = -sum(view(M, i, :)) + IDh[i]
    end
    set_cache!(dad; M, dibem_F=Fh, dibem_c=c, dibem_ID=IDh, dibem_D=Dh, dibem_IF=IFh,
        dibem_IP=IP, dibem_rbf=rbf, dibem_method=:gpu, dibem_centers=:collocation)
    return M
end

function _gpu_rbf_kind(rbf)
    rbf isa PHS1 && return Int32(1)
    rbf isa PHS2 && return Int32(2)
    rbf isa PHS3 && return Int32(3)
    rbf isa PHS4 && return Int32(4)
    rbf isa PHS5 && return Int32(5)
    rbf isa PHS6 && return Int32(6)
    rbf isa PHS7 && return Int32(7)
    rbf isa FundamentalRBF && return Int32(10)
    throw(ArgumentError(
        "DIBEM_gpu rbf must be PHS1–PHS7 or FundamentalRBF; got $(typeof(rbf))"))
end

@inline function _dibem_phi(kind::Int32, R, kcond, epsa)
    if kind == Int32(1)
        return R
    elseif kind == Int32(2)
        return (R * R) * log(R + epsa)
    elseif kind == Int32(3)
        return R * R * R
    elseif kind == Int32(4)
        R2 = R * R
        return (R2 * R2) * log(R + epsa)
    elseif kind == Int32(5)
        R2 = R * R
        return R2 * R2 * R
    elseif kind == Int32(6)
        R2 = R * R
        R4 = R2 * R2
        return (R4 * R2) * log(R + epsa)
    elseif kind == Int32(7)
        R2 = R * R
        return R2 * R2 * R2 * R
    else
        return -log(R) / (typeof(R)(2) * typeof(R)(π) * kcond)
    end
end

@inline function _dibem_phi_int2d(kind::Int32, R, kcond, epsa)
    T = typeof(R)
    if kind == Int32(1)
        return (R * R * R) / T(3)
    elseif kind == Int32(2)
        R4 = (R * R) * (R * R)
        return (T(4) * R4 * log(R + epsa) - R4) / T(16)
    elseif kind == Int32(3)
        R2 = R * R
        return (R2 * R2 * R) / T(5)
    elseif kind == Int32(4)
        R2 = R * R
        R6 = R2 * R2 * R2
        return R6 * log(R + epsa) / T(6) - R6 / T(36)
    elseif kind == Int32(5)
        R2 = R * R
        R4 = R2 * R2
        return (R4 * R2 * R) / T(7)
    elseif kind == Int32(6)
        R2 = R * R
        R8 = (R2 * R2) * (R2 * R2)
        return R8 * log(R + epsa) / T(8) - R8 / T(64)
    elseif kind == Int32(7)
        R2 = R * R
        R4 = R2 * R2
        return (R4 * R4 * R) / T(9)
    else
        R2 = R * R
        return -(T(2) * R2 * log(R) - R2) / (T(8) * T(π) * kcond)
    end
end

@inline function _dibem_rad_id2d(R, kcond)
    T = typeof(R)
    R2 = R * R
    return -(T(2) * R2 * log(R) - R2) / (T(8) * T(π) * kcond)
end

function _dibem_FD_ka(backend, packed, ::Type{T}, kind::Int32, epsa::T) where {T<:AbstractFloat}
    nt = packed.nt
    F = KernelAbstractions.allocate(backend, T, nt, nt)
    D = KernelAbstractions.allocate(backend, T, nt, nt)
    fill!(F, zero(T))
    fill!(D, zero(T))
    px = _to_backend(backend, packed.px)
    py = _to_backend(backend, packed.py)
    kern = _dibem_FD_kernel!(backend)
    kern(F, D, px, py, packed.kcond, kind, epsa, eps(T)^2;
        ndrange = (nt, nt))
    KernelAbstractions.synchronize(backend)
    return Array(F), Array(D)
end

@kernel function _dibem_FD_kernel!(F, D, px, py, kcond, kind, epsa, eps2)
    i, j = @index(Global, NTuple)
    if i != j
        rx = px[j] - px[i]
        ry = py[j] - py[i]
        R2 = rx * rx + ry * ry
        if R2 > eps2
            R = sqrt(R2)
            TT = eltype(D)
            inv2π = one(TT) / (TT(2) * TT(π))
            D[i, j] = -log(R) * inv2π / kcond
            F[i, j] = _dibem_phi(kind, R, kcond, epsa)
        end
    end
end

function _dibem_IFID_ka(backend, packed, ::Type{T}, kind::Int32, epsa::T,
        skip_ie) where {T<:AbstractFloat}
    nt = packed.nt
    n = packed.n
    IF = KernelAbstractions.allocate(backend, T, nt)
    ID = KernelAbstractions.allocate(backend, T, nt)
    fill!(IF, zero(T))
    fill!(ID, zero(T))
    px = _to_backend(backend, packed.px)
    py = _to_backend(backend, packed.py)
    nx = _to_backend(backend, packed.nx)
    ny = _to_backend(backend, packed.ny)
    inc_ptr = _to_backend(backend, packed.inc_ptr)
    inc_e = _to_backend(backend, packed.inc_e)
    inc_w = _to_backend(backend, packed.inc_w)
    skip = _to_backend(backend, skip_ie)
    kern = _dibem_IFID_kernel!(backend)
    kern(IF, ID, px, py, nx, ny, inc_ptr, inc_e, inc_w, skip,
        packed.kcond, kind, epsa, T(1e-10), Int32(n);
        ndrange = nt)
    KernelAbstractions.synchronize(backend)
    return Array(IF), Array(ID)
end

@kernel function _dibem_IFID_kernel!(
        IF, ID, px, py, nx, ny,
        inc_ptr, inc_e, inc_w, skip_ie,
        kcond, kind, epsa, rmin, n)
    i = @index(Global)
    pxi = px[i]
    pyi = py[i]
    accIF = zero(eltype(IF))
    accID = zero(eltype(ID))
    j = one(typeof(n))
    while j <= n
        rx = px[j] - pxi
        ry = py[j] - pyi
        R2 = rx * rx + ry * ry
        R = sqrt(R2)
        if R > rmin
            rn = rx * nx[j] + ry * ny[j]
            phiI = _dibem_phi_int2d(kind, R, kcond, epsa)
            radI = _dibem_rad_id2d(R, kcond)
            p = inc_ptr[j]
            b = inc_ptr[j + 1]
            while p < b
                e = inc_e[p]
                w = inc_w[p]
                p += one(typeof(p))
                if skip_ie[i, e] == UInt8(0)
                    wJn = w * rn / R2
                    accIF += phiI * wJn
                    accID += radI * wJn
                end
            end
        end
        j += one(typeof(j))
    end
    IF[i] = accIF
    ID[i] = accID
end

function _dibem_correct_near_cpu!(IF, ID, dad, rbf; near_factor::Float64, threaded::Bool)
    geos = _rim_build_elements(dad)
    props = dad.properties
    dimv = Val(Int(dad.dimension))
    nf = near_factor
    _collocation_loop!(threaded, dad.nt) do i
        x = point(dad, i)
        accF = 0.0
        accD = 0.0
        @inbounds for g in geos
            if _near_element(x, g.nodes, g.el; factor=nf) || _source_on_element(g.el, i)
                for q in eachindex(g.wJ)
                    wJ = g.wJ[q]
                    wJ == 0 && continue
                    y = g.y[q]
                    r = y - x
                    R = norm(r)
                    R < 1e-14 && continue
                    wJn = wJ * _rim_factor(g.n[q], r, R, dimv)
                    accF += int(rbf, x, y) * wJn
                    accD += radial_integral(props, R, dimv) * wJn
                end
            end
        end
        IF[i] += accF
        ID[i] += accD
    end
    return nothing
end
