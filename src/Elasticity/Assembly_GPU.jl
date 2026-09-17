# 2-D isotropic elasticity (Kelvin) dense H, G and DIBEM on GPU via
# KernelAbstractions. Far field matches `_far_nodal_vec!`. Near / singular
# stays on the CPU (`integrate_element`). DIBEM: pairwise F and Kelvin D;
# IF / ID stay on the host (`_dibem_elast_IF_ID`).

"""
    H_G_gpu(dad::BEMdata{<:Elasticity}; T=Float32, npg=20, near=:cpu,
            near_factor=1.5, device=:cuda, threaded=true, singular=:guiggiani)

Dense 2-D Kelvin `H` and `G` with a KernelAbstractions far-field kernel.
Layout is node-major `(2 nt)×(2 nt)` / `(2 nt)×(2 n)`, same as
[`H_G_full_direct`](@ref) for `Vectorial`. Rigid-body row-sum is applied
on the host. `using CUDA` is required for `device=:cuda`.
"""
function H_G_gpu(
        dad::BEMdata{<:Elasticity};
        T::Type{<:AbstractFloat} = Float32,
        npg::Integer = 20,
        near::Symbol = :cpu,
        near_factor::Real = 1.5,
        device::Symbol = :cuda,
        threaded::Bool = true,
        singular::Symbol = :guiggiani,
    )
    dad.dimension == 2 || throw(ArgumentError("H_G_gpu elasticity supports 2-D only"))
    (near === :cpu || near === :gpu) ||
        throw(ArgumentError("near must be :cpu or :gpu; got $near"))
    (device === :cuda || device === :cpu) ||
        throw(ArgumentError("device must be :cuda or :cpu; got $device"))
    T === Float32 || T === Float64 ||
        throw(ArgumentError("H_G_gpu T must be Float32 or Float64; got $T"))
    (singular === :guiggiani || singular === :telles) ||
        throw(ArgumentError("singular must be :guiggiani or :telles; got $singular"))

    _init_quadrature!(dad, npg)
    set_cache!(dad; singular=singular)
    packed = _pack_boundary2d(dad, T)
    skip_ie = (near === :cpu) ? _near_skip_mask(dad; near_factor = float(near_factor)) :
        zeros(UInt8, packed.nt, packed.ne)
    μ = T(dad.properties.mu)
    ν = T(effective_nu(dad.properties))
    backend = _ka_backend(device)
    H, G = _elast_far_assemble_ka(backend, packed, T, μ, ν; skip_ie = skip_ie)
    if near === :cpu
        _correct_near_cpu_vec!(H, G, dad; near_factor = float(near_factor),
            threaded = threaded)
    end
    _rowsum_diag_vec!(H, 2, packed.nt)
    set_cache!(dad; H, G, singular=singular)
    return H, G
end

function _elast_far_assemble_ka(backend, packed, ::Type{T}, μ::T, ν::T;
        skip_ie) where {T<:AbstractFloat}
    nt = packed.nt
    n = packed.n
    ndof_t = 2 * nt
    ndof_n = 2 * n
    H = KernelAbstractions.allocate(backend, T, ndof_t, ndof_t)
    G = KernelAbstractions.allocate(backend, T, ndof_t, ndof_n)
    fill!(H, zero(T))
    fill!(G, zero(T))
    if nt == 0 || n == 0
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
    kern = _kelvin_far_kernel!(backend)
    kern(H, G, px, py, nx, ny, inc_ptr, inc_e, inc_w, skip, μ, ν, eps(T)^2;
        ndrange = (nt, n))
    KernelAbstractions.synchronize(backend)
    return Array(H), Array(G)
end

@inline function _kelvin2d_UT(rx, ry, nx, ny, R, μ, ν)
    T = typeof(R)
    dr1 = rx / R
    dr2 = ry / R
    drdn = dr1 * nx + dr2 * ny
    omν = one(T) - ν
    prod1 = T(4) * T(π) * omν
    prod2 = (T(3) - T(4) * ν) * (-log(R))
    base = T(2) * prod1 * μ
    u11 = (prod2 + dr1 * dr1) / base
    u12 = (dr1 * dr2) / base
    u22 = (prod2 + dr2 * dr2) / base
    fat = one(T) / (prod1 * R)
    fat2 = one(T) - T(2) * ν
    t11 = -drdn * (fat2 + T(2) * dr1 * dr1) * fat
    t22 = -drdn * (fat2 + T(2) * dr2 * dr2) * fat
    cross12 = dr1 * ny - dr2 * nx
    t12 = -(drdn * T(2) * dr1 * dr2 - fat2 * cross12) * fat
    t21 = -(drdn * T(2) * dr1 * dr2 - fat2 * (-cross12)) * fat
    return u11, u12, u22, t11, t12, t21, t22
end

@kernel function _kelvin_far_kernel!(
        H, G, px, py, nx, ny,
        inc_ptr, inc_e, inc_w, skip_ie,
        μ, ν, eps2)
    i, j = @index(Global, NTuple)
    rx = px[j] - px[i]
    ry = py[j] - py[i]
    R2 = rx * rx + ry * ry
    if i != j && R2 > eps2
        R = sqrt(R2)
        u11, u12, u22, t11, t12, t21, t22 = _kelvin2d_UT(
            rx, ry, nx[j], ny[j], R, μ, ν)
        acc = zero(eltype(H))
        p = inc_ptr[j]
        b = inc_ptr[j + 1]
        while p < b
            e = inc_e[p]
            w = inc_w[p]
            p += one(typeof(p))
            if skip_ie[i, e] == UInt8(0)
                acc += w
            end
        end
        i0 = 2 * i - 1
        j0 = 2 * j - 1
        G[i0, j0] = u11 * acc
        G[i0, j0 + 1] = u12 * acc
        G[i0 + 1, j0] = u12 * acc
        G[i0 + 1, j0 + 1] = u22 * acc
        H[i0, j0] = t11 * acc
        H[i0, j0 + 1] = t12 * acc
        H[i0 + 1, j0] = t21 * acc
        H[i0 + 1, j0 + 1] = t22 * acc
    end
end

function _correct_near_cpu_vec!(H, G, dad; near_factor::Float64, threaded::Bool)
    dim = dad.dimension
    nf = near_factor
    elems = dad.elements
    _collocation_loop!(threaded, dad.nt) do i
        pf = point(dad, i)
        ii = expand(i, dim)
        @inbounds for el in elems
            xj = dad.Nodes[el.index]
            if _near_element(pf, xj, el; factor = nf) || _source_on_element(el, i)
                jj = expand(el.index, dim)
                hloc = zeros(eltype(H), dim, length(jj))
                gloc = zeros(eltype(G), dim, length(jj))
                integrate_element(dad, el, xj, pf, hloc, gloc; source = i)
                H[ii, jj] .+= hloc
                G[ii, jj] .+= gloc
            end
        end
    end
    return nothing
end

function _rowsum_diag_vec!(H::AbstractMatrix, dim::Integer, nt::Integer)
    @views for i in 1:nt
        ii = expand(i, dim)
        H[ii, ii] .= 0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end
    return H
end

"""
    DIBEM_gpu(dad::BEMdata{<:Elasticity}; rbf=PHS(), T=Float64, npg=12,
              device=:cuda, threaded=true)

Dense 2-D Kelvin DIBEM: pairwise `F` and `D=U*` on the GPU, RIM `IF`/`ID`
and the CPD solve on the host. Default `T=Float64` (PHS `F`-solve).
"""
function DIBEM_gpu(
        dad::BEMdata{<:Elasticity};
        rbf = PHS(),
        T::Type{<:AbstractFloat} = Float64,
        npg::Integer = 12,
        device::Symbol = :cuda,
        threaded::Bool = true,
        centers::Symbol = :collocation,
        kwargs...,
    )
    dad.dimension == 2 || throw(ArgumentError("DIBEM_gpu elasticity supports 2-D only"))
    centers === :collocation || throw(ArgumentError(
        "DIBEM_gpu centers must be :collocation (got $centers)"))
    (device === :cuda || device === :cpu) ||
        throw(ArgumentError("device must be :cuda or :cpu; got $device"))
    T === Float32 || T === Float64 ||
        throw(ArgumentError("DIBEM_gpu T must be Float32 or Float64; got $T"))
    T === Float32 && @warn "DIBEM_gpu T=Float32: PHS F-solve is ill-conditioned; prefer T=Float64" maxlog=1
    kind = _gpu_rbf_kind(rbf)

    packed = _pack_boundary2d(dad, T)
    μ = T(dad.properties.mu)
    ν = T(effective_nu(dad.properties))
    backend = _ka_backend(device)
    F, D = _elast_dibem_FD_ka(backend, packed, T, μ, ν, kind, T(AVOID_INF))
    Fh = Float64.(F)
    Dh = Float64.(D)
    IF, ID, pts = _dibem_elast_IF_ID(dad, rbf; npg=Int(npg), threaded=threaded)
    _dibem_ridge_F!(Fh)
    IP = _dibem_monomial_IP(dad, rbf)
    c = _dibem_poly_c(Fh, IF, pts, rbf; IP=IP)
    dim = 2
    ndof = dim * dad.nt
    M = zeros(ndof, ndof)
    @inbounds for j in 1:dad.nt
        a = c[j]
        cols = expand(j, dim)
        for d in 1:dim
            M[:, cols[d]] .= a .* Dh[:, cols[d]]
        end
    end
    @inbounds for i in 1:dad.nt
        rows = expand(i, dim)
        M[rows, rows] .= 0
        S = zeros(dim, dim)
        for d in 1:dim
            S[:, d] = vec(sum(view(M, rows, d:dim:ndof); dims=2))
        end
        M[rows, rows] .= .-S .+ ID[rows, :]
    end
    M .*= dad.properties.rho
    set_cache!(dad; M, dibem_F=Fh, dibem_rbf=rbf, dibem_method=:gpu,
        dibem_c=c, dibem_ID=ID, dibem_D=Dh, dibem_centers=:collocation)
    return M
end

function _elast_dibem_FD_ka(backend, packed, ::Type{T}, μ::T, ν::T,
        kind::Int32, epsa::T) where {T<:AbstractFloat}
    nt = packed.nt
    ndof = 2 * nt
    F = KernelAbstractions.allocate(backend, T, nt, nt)
    D = KernelAbstractions.allocate(backend, T, ndof, ndof)
    fill!(F, zero(T))
    fill!(D, zero(T))
    px = _to_backend(backend, packed.px)
    py = _to_backend(backend, packed.py)
    kern = _elast_dibem_FD_kernel!(backend)
    dummyk = one(T)
    kern(F, D, px, py, μ, ν, kind, dummyk, epsa, eps(T)^2;
        ndrange = (nt, nt))
    KernelAbstractions.synchronize(backend)
    return Array(F), Array(D)
end

@kernel function _elast_dibem_FD_kernel!(F, D, px, py, μ, ν, kind, dummyk, epsa, eps2)
    i, j = @index(Global, NTuple)
    if i != j
        rx = px[j] - px[i]
        ry = py[j] - py[i]
        R2 = rx * rx + ry * ry
        if R2 > eps2
            R = sqrt(R2)
            F[i, j] = _dibem_phi(kind, R, dummyk, epsa)
            u11, u12, u22, _, _, _, _ = _kelvin2d_UT(
                rx, ry, zero(R), one(R), R, μ, ν)
            i0 = 2 * i - 1
            j0 = 2 * j - 1
            D[i0, j0] = u11
            D[i0, j0 + 1] = u12
            D[i0 + 1, j0] = u12
            D[i0 + 1, j0 + 1] = u22
        end
    end
end
