# 3D anisotropic Kelvin–Somigliana kernels (Ting–Lee / Barnett–Lothe).
# U_ij = H_ij(n̂) / (4π R),  H = (1/2π) ∮ Γ^{-1}(z) dψ  on the plane ⟂ n̂.
# Traction T_ij = n_p C_piqk U_qj,k  (field derivatives of U).

export AnisotropicElasticity3D
export aniso3d_isotropic, aniso3d_cubic, aniso3d_hcp, aniso3d_trigonal, aniso3d_full
export voigt_to_tensor, rotate_stiffness, tensor_to_voigt
export barnett_lothe_H

const _VOIGT_IJ = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

"""Voigt 6×6 stiffness → ``C_{ijkl}`` with minor + major symmetries."""
function voigt_to_tensor(Cv::AbstractMatrix{T}) where {T}
    size(Cv) == (6, 6) || throw(ArgumentError("expected 6×6 Voigt stiffness"))
    C = zeros(T, 3, 3, 3, 3)
    @inbounds for a in 1:6, b in 1:6
        i, j = _VOIGT_IJ[a]
        k, l = _VOIGT_IJ[b]
        v = Cv[a, b]
        C[i, j, k, l] = v
        C[j, i, k, l] = v
        C[i, j, l, k] = v
        C[j, i, l, k] = v
        C[k, l, i, j] = v
        C[l, k, i, j] = v
        C[k, l, j, i] = v
        C[l, k, j, i] = v
    end
    return C
end

function _smatrix66(C::AbstractMatrix{T}) where {T}
    size(C) == (6, 6) || throw(ArgumentError("expected 6×6 Voigt matrix"))
    return SMatrix{6,6,T}(NTuple{36,T}(T(C[i]) for i in 1:36))
end

function AnisotropicElasticity3D(C::AbstractMatrix; rho::Real=1.0, nψ::Int=32)
    T = float(promote_type(eltype(C), typeof(rho)))
    Cv = _smatrix66(Matrix{T}(C))
    return AnisotropicElasticity3D{T}(Cv, voigt_to_tensor(Cv), T(rho), nψ)
end

"""
    aniso3d_isotropic(E, ν; rho=1.0, nψ=32)

Isotropic 3D stiffness from ``E, ν`` (same Lamé pair as 3-D Kelvin).
"""
function aniso3d_isotropic(E::Real, ν::Real; rho::Real=1.0, nψ::Int=32)
    T = float(promote_type(typeof(E), typeof(ν), typeof(rho)))
    μ = T(E) / (2 * (1 + T(ν)))
    λ = T(E) * T(ν) / ((1 + T(ν)) * (1 - 2 * T(ν)))
    c11 = λ + 2μ
    C = zeros(T, 6, 6)
    C[1, 1] = C[2, 2] = C[3, 3] = c11
    C[1, 2] = C[2, 1] = C[1, 3] = C[3, 1] = C[2, 3] = C[3, 2] = λ
    C[4, 4] = C[5, 5] = C[6, 6] = μ
    return AnisotropicElasticity3D(C; rho=rho, nψ=nψ)
end

"""Cubic crystal ``C_{11}, C_{12}, C_{44}``. `theta` Euler degrees; `zxz=true` for z–x–z."""
function aniso3d_cubic(C11::Real, C12::Real, C44::Real; theta=(0.0, 0.0, 0.0),
        zxz::Bool=false, rho::Real=1.0, nψ::Int=32)
    T = float(promote_type(typeof(C11), typeof(C12), typeof(C44)))
    C = zeros(T, 6, 6)
    C[1, 1] = C[2, 2] = C[3, 3] = T(C11)
    C[1, 2] = C[2, 1] = C[1, 3] = C[3, 1] = C[2, 3] = C[3, 2] = T(C12)
    C[4, 4] = C[5, 5] = C[6, 6] = T(C44)
    C = rotate_stiffness(C, theta; zxz=zxz)
    return AnisotropicElasticity3D(C; rho=rho, nψ=nψ)
end

function aniso3d_hcp(C11::Real, C12::Real, C13::Real, C33::Real, C44::Real;
        theta=(0.0, 0.0, 0.0), zxz::Bool=false, rho::Real=1.0, nψ::Int=32)
    T = float(promote_type(typeof(C11), typeof(C12), typeof(C13), typeof(C33), typeof(C44)))
    C66 = (T(C11) - T(C12)) / 2
    C = zeros(T, 6, 6)
    C[1, 1] = C[2, 2] = T(C11)
    C[3, 3] = T(C33)
    C[1, 2] = C[2, 1] = T(C12)
    C[1, 3] = C[3, 1] = C[2, 3] = C[3, 2] = T(C13)
    C[4, 4] = C[5, 5] = T(C44)
    C[6, 6] = C66
    C = rotate_stiffness(C, theta; zxz=zxz)
    return AnisotropicElasticity3D(C; rho=rho, nψ=nψ)
end

function aniso3d_trigonal(C11::Real, C12::Real, C13::Real, C14::Real, C33::Real, C44::Real;
        theta=(0.0, 0.0, 0.0), zxz::Bool=false, rho::Real=1.0, nψ::Int=32)
    T = float(promote_type(typeof(C11), typeof(C14)))
    C66 = (T(C11) - T(C12)) / 2
    C = zeros(T, 6, 6)
    C[1, 1] = C[2, 2] = T(C11)
    C[3, 3] = T(C33)
    C[1, 2] = C[2, 1] = T(C12)
    C[1, 3] = C[3, 1] = C[2, 3] = C[3, 2] = T(C13)
    C[1, 4] = C[4, 1] = T(C14)
    C[2, 4] = C[4, 2] = -T(C14)
    C[5, 6] = C[6, 5] = T(C14)
    C[4, 4] = C[5, 5] = T(C44)
    C[6, 6] = C66
    C = rotate_stiffness(C, theta; zxz=zxz)
    return AnisotropicElasticity3D(C; rho=rho, nψ=nψ)
end

function aniso3d_full(C::AbstractMatrix; theta=(0.0, 0.0, 0.0), zxz::Bool=false, rho::Real=1.0,
        nψ::Int=32)
    Cr = rotate_stiffness(C, theta; zxz=zxz)
    return AnisotropicElasticity3D(Cr; rho=rho, nψ=nψ)
end

_Rx(a) = @SMatrix [1.0 0.0 0.0; 0.0 cos(a) -sin(a); 0.0 sin(a) cos(a)]
_Ry(a) = @SMatrix [cos(a) 0.0 sin(a); 0.0 1.0 0.0; -sin(a) 0.0 cos(a)]
_Rz(a) = @SMatrix [cos(a) -sin(a) 0.0; sin(a) cos(a) 0.0; 0.0 0.0 1.0]

"""Rotate Voigt stiffness. `theta` in degrees `(θx,θy,θz)` (x–y–z) or z–x–z if `zxz`."""
function rotate_stiffness(C::AbstractMatrix, theta; zxz::Bool=false)
    θ = Tuple(float(x) for x in theta)
    all(iszero, θ) && return Matrix{Float64}(C)
    deg = π / 180
    Q = if zxz
        length(θ) >= 2 || throw(ArgumentError("z-x-z needs at least two angles"))
        θz1 = θ[1] * deg
        θx = θ[2] * deg
        θz2 = length(θ) >= 3 ? θ[3] * deg : 0.0
        _Rz(θz2) * _Rx(θx) * _Rz(θz1)
    else
        θx = θ[1] * deg
        θy = length(θ) >= 2 ? θ[2] * deg : 0.0
        θz = length(θ) >= 3 ? θ[3] * deg : 0.0
        _Rz(θz) * _Ry(θy) * _Rx(θx)
    end
    C4 = voigt_to_tensor(Matrix{Float64}(C))
    Cp = zeros(Float64, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        s = 0.0
        for p in 1:3, q in 1:3, r in 1:3, t in 1:3
            s += Q[i, p] * Q[j, q] * Q[k, r] * Q[l, t] * C4[p, q, r, t]
        end
        Cp[i, j, k, l] = s
    end
    return tensor_to_voigt(Cp)
end

function tensor_to_voigt(C4::Array{T,4}) where {T}
    Cv = zeros(T, 6, 6)
    @inbounds for a in 1:6, b in 1:6
        i, j = _VOIGT_IJ[a]
        k, l = _VOIGT_IJ[b]
        Cv[a, b] = C4[i, j, k, l]
    end
    return Cv
end

function christoffel(C4::Array{T,4}, n::SVector{3}) where {T}
    Γ = MMatrix{3,3,T}(undef)
    @inbounds for i in 1:3, k in 1:3
        s = zero(T)
        for j in 1:3, l in 1:3
            s += C4[i, j, k, l] * n[j] * n[l]
        end
        Γ[i, k] = s
    end
    return SMatrix(Γ)
end

function _plane_basis(n::SVector{3,Float64})
    ax = abs(n[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
    m = n × ax
    m = m / norm(m)
    p = n × m
    return m, p
end

"""Barnett–Lothe tensor ``H(n̂)`` by trapezoid on the unit circle ⟂ ``n̂``."""
function barnett_lothe_H(C4::Array{Float64,4}, nhat::SVector{3,Float64}; nψ::Int=48)
    m, p = _plane_basis(nhat)
    H = @SMatrix zeros(3, 3)
    invN = 1 / nψ
    @inbounds for k in 0:nψ-1
        ψ = 2π * k * invN
        z = cos(ψ) * m + sin(ψ) * p
        H = H + inv(christoffel(C4, z))
    end
    return H * invN
end

function _U_aniso3d(C4::Array{Float64,4}, rvec::SVector{3,Float64}; nψ::Int=48)
    R = norm(rvec)
    R < 1e-30 && return @SMatrix zeros(3, 3)
    nhat = rvec / R
    H = barnett_lothe_H(C4, nhat; nψ=nψ)
    return H / (4π * R)
end

function _dU_aniso3d(C4::Array{Float64,4}, rvec::SVector{3,Float64}; nψ::Int=48)
    R = norm(rvec)
    h = 1e-7 * max(R, 1e-3)
    dU = MArray{Tuple{3,3,3},Float64}(undef)
    @inbounds for s in 1:3
        e = SVector{3,Float64}(ntuple(k -> k == s ? 1.0 : 0.0, 3))
        Up = _U_aniso3d(C4, rvec + h * e; nψ=nψ)
        Um = _U_aniso3d(C4, rvec - h * e; nψ=nψ)
        dU[:, :, s] .= (Up - Um) / (2h)
    end
    return SArray(dU)
end

function _T_aniso3d(C4::Array{Float64,4}, rvec::SVector{3,Float64}, n̂::SVector{3,Float64};
        nψ::Int=48)
    dU = _dU_aniso3d(C4, rvec; nψ=nψ)
    T = MMatrix{3,3,Float64}(undef)
    @inbounds for i in 1:3, j in 1:3
        s = 0.0
        for p in 1:3, q in 1:3, k in 1:3
            s += n̂[p] * C4[p, i, q, k] * dU[q, j, k]
        end
        T[i, j] = s
    end
    # Match Kelvin T_ij (row = force dir, col = traction dir in this codebase).
    return SMatrix(T)'
end

function fundamental(props::AnisotropicElasticity3D, r::SVector{3}, n::SVector{3}; nψ::Int=props.nψ)
    rvec = SVector{3,Float64}(r[1], r[2], r[3])
    n̂ = SVector{3,Float64}(n[1], n[2], n[3])
    R = norm(rvec)
    if R < 1e-30
        Z = zero(Mat{3,3,Float64})
        return KernelPair(Z, Z)
    end
    U = _U_aniso3d(props.C4, rvec; nψ=nψ)
    Tker = _T_aniso3d(props.C4, rvec, n̂; nψ=nψ)
    return KernelPair(Mat{3,3}(Tuple(U)), Mat{3,3}(Tuple(Tker)))
end

function fundamental(dad::BEMdata{<:AnisotropicElasticity3D}, r::Point3D, n::Point3D)
    kp = fundamental(dad.properties, r, n)
    return _to_smat(kp.U), _to_smat(kp.T)
end
