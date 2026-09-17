# Elasticity fundamental solutions (Kelvin + Lekhnitskii)
using Tensorial: Mat, Tensor
using StaticArrays: MMatrix, MArray

# Isotropic elasticity — Kelvin (Tensorial)
# =============================================================================

"""
    fundamental(props::Elasticity, r::SVector{2}, n) -> KernelPair{Mat{2,2}, Mat{2,2}}

2D Kelvin fundamental solution (plane strain / mapped plane stress).

Uses `Tensorial.Mat` for the displacement (`U`) and traction (`T`)
tensors. Legacy: `calsolfund` for `elastico`.
"""
function fundamental_U(props::Elasticity, r::SVector{2})
    ν = effective_nu(props)
    μ = props.mu
    R2 = _R2(r)
    R = sqrt(R2)
    dr = _to_vec(r) / R
    prod1 = 4π * (1 - ν)
    prod2 = (3 - 4ν) * log(1 / R)
    base = 2 * prod1 * μ
    return (prod2 * _I2 + _otimes(dr, dr)) / base
end

function fundamental(props::Elasticity, r::SVector{2}, n::SVector{2})
    ν = effective_nu(props)
    μ = props.mu
    R2 = _R2(r)
    R = sqrt(R2)
    dr = _to_vec(r) / R
    n̂ = _to_vec(n)
    drdn = _dot(dr, n̂)

    # match legacy form:
    prod1 = 4π * (1 - ν)
    prod2 = (3 - 4ν) * log(1 / R)
    base = 2 * prod1 * μ

    δ = _I2
    U = ((prod2) * δ + _otimes(dr, dr)) / base

    # T_ij
    fat = 1 / (prod1 * R)
    term_sym = (1 - 2ν) * δ + 2 * (_otimes(dr, dr))
    Tmat = -drdn * term_sym * fat
    # antisymmetric part from (1-2ν)(dr_i n_j - dr_j n_i)
    # T12 gets + (1-2ν)(dr1 n2 - dr2 n1), T21 opposite
    cross = (1 - 2ν) * (dr[1] * n̂[2] - dr[2] * n̂[1])
    # rebuild with the cross terms (legacy formula)
    t11 = -drdn * ((1 - 2ν) + 2 * dr[1]^2) * fat
    t22 = -drdn * ((1 - 2ν) + 2 * dr[2]^2) * fat
    t12 = -(drdn * 2 * dr[1] * dr[2] - (1 - 2ν) * (dr[1] * n̂[2] - dr[2] * n̂[1])) * fat
    t21 = -(drdn * 2 * dr[1] * dr[2] - (1 - 2ν) * (dr[2] * n̂[1] - dr[1] * n̂[2])) * fat
    Tker = @Mat [t11 t12; t21 t22]

    return KernelPair(U, Tker)
end

"""
3D Kelvin fundamental solution.
"""
function fundamental_U(props::Elasticity, r::SVector{3})
    ν = props.nu
    μ = props.mu
    R = _R(r)
    dr = _to_vec(r) / R
    cU = 1 / (16π * μ * (1 - ν) * R)
    return cU * ((3 - 4ν) * _I3 + _otimes(dr, dr))
end

function fundamental(props::Elasticity, r::SVector{3}, n::SVector{3})
    ν = props.nu
    μ = props.mu
    R2 = _R2(r)
    R = sqrt(R2)
    dr = _to_vec(r) / R
    n̂ = _to_vec(n)
    drdn = _dot(dr, n̂)

    cU = 1 / (16π * μ * (1 - ν) * R)
    δ = _I3
    U = cU * ((3 - 4ν) * δ + _otimes(dr, dr))

    cT = 1 / (8π * (1 - ν) * R^2)
    t = MMatrix{3,3,Float64}(undef)
    @inbounds for i in 1:3, j in 1:3
        δij = ifelse(i == j, 1.0, 0.0)
        t[i, j] = -cT * (
            drdn * ((1 - 2ν) * δij + 3 * dr[i] * dr[j]) -
            (1 - 2ν) * (dr[i] * n̂[j] - dr[j] * n̂[i])
        )
    end
    return KernelPair(U, Mat{3,3}(Tuple(t)))
end

# BEMdata convenience — return SMatrix for assembly compatibility
function fundamental(dad::BEMdata{<:Elasticity}, r::Point2D, n::Point2D)
    kp = fundamental(dad.properties, r, n)
    return _to_smat(kp.U), _to_smat(kp.T)
end
function fundamental(dad::BEMdata{<:Elasticity}, r::Point3D, n::Point3D)
    kp = fundamental(dad.properties, r, n)
    return _to_smat(kp.U), _to_smat(kp.T)
end

# =============================================================================
# Stress kernels D, S  (interior stress recovery)
# =============================================================================

"""
    fundamental_stress(props::Elasticity, r, n) -> StressKernels

Third-order Kelvin tensors `D` and `S` for 2D plane strain (legacy `caldsolfund`):

```math
σ_{ij}(x) = D_{kij}(r,n)\\,t_k - S_{kij}(r,n)\\,u_k
```
"""
function fundamental_stress(props::Elasticity, r::SVector{2}, n::SVector{2})
    ν = effective_nu(props)
    μ = props.mu
    R = _R(r)
    dr = _to_vec(r) / R
    n̂ = _to_vec(n)
    drdn = _dot(dr, n̂)
    fat1 = 4π * (1 - ν)
    fat2 = 1 - 2ν
    # D[k,i,j] and S[k,i,j]
    D = MArray{Tuple{2,2,2},Float64}(undef)
    S = MArray{Tuple{2,2,2},Float64}(undef)
    @inbounds for k in 1:2, i in 1:2, j in 1:2
        δki = i == k ? 1.0 : 0.0
        δkj = j == k ? 1.0 : 0.0
        δij = i == j ? 1.0 : 0.0
        d1 = fat2 * (δki * dr[j] + δkj * dr[i] - δij * dr[k])
        d2 = 2 * dr[i] * dr[j] * dr[k]
        D[k, i, j] = (d1 + d2) / (fat1 * R)

        t1 = 2 * drdn * (fat2 * δij * dr[k] + ν * (δki * dr[j] + δkj * dr[i]) -
                         4 * dr[i] * dr[j] * dr[k])
        t2 = 2ν * (n̂[i] * dr[j] * dr[k] + n̂[j] * dr[i] * dr[k])
        t3 = fat2 * (2 * n̂[k] * dr[i] * dr[j] + n̂[j] * δki + n̂[i] * δkj) -
             (1 - 4ν) * n̂[k] * δij
        S[k, i, j] = (t1 + t2 + t3) * 2μ / (fat1 * R^2)
    end
    # enforce minor symmetry on last two indices
    @inbounds for k in 1:2
        D[k, 2, 1] = D[k, 1, 2]
        S[k, 2, 1] = S[k, 1, 2]
    end
    DT = Tensor{Tuple{2,2,2},Float64}(Tuple(D))
    ST = Tensor{Tuple{2,2,2},Float64}(Tuple(S))
    return StressKernels(DT, ST)
end

"""
3-D Kelvin stress kernels `D`, `S` (Aliabadi / Kane):

```math
σ_{ij}(x) = D_{kij}(r,n)\\,t_k - S_{kij}(r,n)\\,u_k
```
"""
function fundamental_stress(props::Elasticity, r::SVector{3}, n::SVector{3})
    ν = props.nu
    μ = props.mu
    R = _R(r)
    dr = _to_vec(r) / R
    n̂ = _to_vec(n)
    drdn = _dot(dr, n̂)
    fat1 = 8π * (1 - ν)
    fat2 = 1 - 2ν
    D = MArray{Tuple{3,3,3},Float64}(undef)
    S = MArray{Tuple{3,3,3},Float64}(undef)
    @inbounds for k in 1:3, i in 1:3, j in 1:3
        δki = i == k ? 1.0 : 0.0
        δkj = j == k ? 1.0 : 0.0
        δij = i == j ? 1.0 : 0.0
        d1 = fat2 * (δki * dr[j] + δkj * dr[i] - δij * dr[k])
        d2 = 3 * dr[i] * dr[j] * dr[k]
        D[k, i, j] = (d1 + d2) / (fat1 * R^2)

        t1 = 3 * drdn * (fat2 * δij * dr[k] + ν * (δki * dr[j] + δkj * dr[i]) -
                         5 * dr[i] * dr[j] * dr[k])
        t2 = 3ν * (n̂[i] * dr[j] * dr[k] + n̂[j] * dr[i] * dr[k])
        t3 = fat2 * (3 * n̂[k] * dr[i] * dr[j] + n̂[j] * δki + n̂[i] * δkj) -
             (1 - 4ν) * n̂[k] * δij
        S[k, i, j] = (t1 + t2 + t3) * μ / (4π * (1 - ν) * R^3)
    end
    @inbounds for k in 1:3
        D[k, 2, 1] = D[k, 1, 2]
        D[k, 3, 1] = D[k, 1, 3]
        D[k, 3, 2] = D[k, 2, 3]
        S[k, 2, 1] = S[k, 1, 2]
        S[k, 3, 1] = S[k, 1, 3]
        S[k, 3, 2] = S[k, 2, 3]
    end
    DT = Tensor{Tuple{3,3,3},Float64}(Tuple(D))
    ST = Tensor{Tuple{3,3,3},Float64}(Tuple(S))
    return StressKernels(DT, ST)
end

fundamental_stress(dad::BEMdata{<:Elasticity}, r, n) =
    fundamental_stress(dad.properties, r, n)

"""
    fundamental_hyper(props::Elasticity, r, n, nf) -> KernelPair

2D Kelvin traction BIE kernels: contract the stress tensors `D,S` with the
**collocation** normal `nf`.

```math
U^h_{ik} = n^ξ_j D_{kij},\\qquad T^h_{ik} = n^ξ_j S_{kij}
```

Same contraction as Dual BEM (`kelvin_DS` → `n_ξ · {D,S}`). Guiggiani orders
for this pair are `(-1, -2)`.
"""
function fundamental_hyper(props::Elasticity, r::SVector{2}, n::SVector{2}, nf::SVector{2})
    sk = fundamental_stress(props, r, n)
    D, S = sk.D, sk.S
    n1, n2 = nf[1], nf[2]
    u11 = n1 * D[1, 1, 1] + n2 * D[1, 1, 2]
    u12 = n1 * D[2, 1, 1] + n2 * D[2, 1, 2]
    u21 = n1 * D[1, 2, 1] + n2 * D[1, 2, 2]
    u22 = n1 * D[2, 2, 1] + n2 * D[2, 2, 2]
    t11 = n1 * S[1, 1, 1] + n2 * S[1, 1, 2]
    t12 = n1 * S[2, 1, 1] + n2 * S[2, 1, 2]
    t21 = n1 * S[1, 2, 1] + n2 * S[1, 2, 2]
    t22 = n1 * S[2, 2, 1] + n2 * S[2, 2, 2]
    return KernelPair(@Mat([u11 u12; u21 u22]), @Mat([t11 t12; t21 t22]))
end

function fundamental_hyper(dad::BEMdata{<:Elasticity}, r, n, nf)
    kp = fundamental_hyper(dad.properties, r, n, nf)
    return _to_smat(kp.U), _to_smat(kp.T)
end

"""3-D Kelvin traction BIE: ``U^h_{ik}=n^ξ_j D_{kij}``, ``T^h_{ik}=n^ξ_j S_{kij}``."""
function fundamental_hyper(props::Elasticity, r::SVector{3}, n::SVector{3}, nf::SVector{3})
    sk = fundamental_stress(props, r, n)
    D, S = sk.D, sk.S
    Uh = zeros(MMatrix{3,3,Float64})
    Th = zeros(MMatrix{3,3,Float64})
    @inbounds for i in 1:3, k in 1:3
        u = 0.0
        t = 0.0
        for j in 1:3
            u += nf[j] * D[k, i, j]
            t += nf[j] * S[k, i, j]
        end
        Uh[i, k] = u
        Th[i, k] = t
    end
    return KernelPair(Mat{3,3}(Tuple(Uh)), Mat{3,3}(Tuple(Th)))
end

# =============================================================================
# Gradients of U and T  (∂/∂x, ∂/∂y of the Kelvin kernels)
# =============================================================================

"""
    fundamental_grad(props::Elasticity, r, n) -> (Ux, Tx, Uy, Ty)

Cartesian derivatives of the 2D Kelvin kernels (legacy `calc_deriv_solfund`).
Each entry is a `Mat{2,2}`.
"""
function fundamental_grad(props::Elasticity, r::SVector{2}, n::SVector{2})
    ν = effective_nu(props)
    μ = props.mu
    r1, r2 = r[1], r[2]
    R = _R(r)
    nx, ny = n[1], n[2]
    R2 = R^2

    C1 = 1 / (8π * (1 - ν) * μ)
    C2 = -1 / (4π * (1 - ν))
    C3 = (r1 * nx + r2 * ny) / R

    # ∂U/∂x
    u11x = (C1 * (2r1 - (2r1^3) / R2 + r1 * (4ν - 3))) / R2
    u12x = (C1 * (r2 - (2r1^2 * r2) / R2)) / R2
    u22x = (C1 * (r1 * (4ν - 3) - (2r1 * r2^2) / R2)) / R2
    Ux = @Mat [u11x u12x; u12x u22x]

    t11x = (C2 * ((2nx * r1^2) / R2 - nx * (2ν - 1) +
            (2C3 * (2r1 - (4r1^3) / R2 + r1 * (2ν - 1))) / R)) / R2
    t12x = (C2 * (ny * (2ν - 1) -
            (2 * ((2ν - 1) * (ny * r1^2 - nx * r2 * r1) - nx * r1 * r2)) / R2 +
            (2C3 * (r2 - (4r1^2 * r2) / R2)) / R)) / R2
    t21x = (C2 * ((2 * ((2ν - 1) * (ny * r1^2 - nx * r2 * r1) + nx * r1 * r2)) / R2 -
            ny * (2ν - 1) + (2C3 * (r2 - (4r1^2 * r2) / R2)) / R)) / R2
    t22x = (C2 * ((2C3 * (r1 * (2ν - 1) - (4r1 * r2^2) / R2)) / R -
            nx * (2ν - 1) + (2nx * r2^2) / R2)) / R2
    Tx = @Mat [t11x t12x; t21x t22x]

    # ∂U/∂y
    u11y = (C1 * (r2 * (4ν - 3) - (2r1^2 * r2) / R2)) / R2
    u12y = (C1 * (r1 - (2r1 * r2^2) / R2)) / R2
    u22y = (C1 * (2r2 - (2r2^3) / R2 + r2 * (4ν - 3))) / R2
    Uy = @Mat [u11y u12y; u12y u22y]

    t11y = (C2 * ((2C3 * (r2 * (2ν - 1) - (4r1^2 * r2) / R2)) / R -
            ny * (2ν - 1) + (2ny * r1^2) / R2)) / R2
    t12y = (C2 * ((2 * ((2ν - 1) * (nx * r2^2 - ny * r1 * r2) + ny * r1 * r2)) / R2 -
            nx * (2ν - 1) + (2C3 * (r1 - (4r1 * r2^2) / R2)) / R)) / R2
    t21y = (C2 * (nx * (2ν - 1) -
            (2 * ((2ν - 1) * (nx * r2^2 - ny * r1 * r2) - ny * r1 * r2)) / R2 +
            (2C3 * (r1 - (4r1 * r2^2) / R2)) / R)) / R2
    t22y = (C2 * ((2ny * r2^2) / R2 - ny * (2ν - 1) +
            (2C3 * (2r2 - (4r2^3) / R2 + r2 * (2ν - 1))) / R)) / R2
    Ty = @Mat [t11y t12y; t21y t22y]

    return Ux, Tx, Uy, Ty
end

fundamental_grad(dad::BEMdata{<:Elasticity}, r, n) =
    fundamental_grad(dad.properties, r, n)

# =============================================================================
# Anisotropic elasticity — Lekhnitskii
# =============================================================================

"""
    lekhnitskii_engineering(E1, E2, G12, ν12; η12_1=0, η12_2=0, plane_strain=false, kwargs...)

Full 2-D anisotropic compliance from engineering constants (Ting / Lekhnitskii):

```math
D_{11}=1/E_1,\\; D_{22}=1/E_2,\\; D_{12}=-ν_{12}/E_1,\\;
D_{66}=1/G_{12},\\; D_{16}=η_{12,1}/E_1,\\; D_{26}=η_{12,2}/E_2.
```

`ν12` is ``-ε_2/ε_1`` under uniaxial ``σ_1`` (the large Poisson). Cordeiro
2015 writes it as ``ν_{yx}=0.344``; EABE 2020 Table 1 as ``ν_{21}=0.334``.
``D_{12}=-ν_{12}/E_1``. Literal ``-ν_{yx}/E_y`` is not positive definite.

Plane strain (`plane_strain=true`) uses ``D^*_{ij}=D_{ij}-D_{i3}D_{j3}/D_{33}``
with optional `E3` (default `E2`). Listed `ν31`,`ν32` are taken as
``ν_{13},ν_{23}`` so ``D_{13}=-ν_{31}/E_1``, ``D_{23}=-ν_{32}/E_2``
(Cordeiro EPD: ``ν_{zx}=0.40``, ``ν_{zy}=0.25``, ``η_{xy,z}=0.50``).
"""
function lekhnitskii_engineering(
        E1::Real, E2::Real, G12::Real, ν12::Real;
        η12_1::Real=0, η12_2::Real=0,
        plane_strain::Bool=false,
        E3::Real=E2, ν31::Real=0, ν32::Real=0, η12_3::Real=0,
    )
    T = float(promote_type(typeof(E1), typeof(E2), typeof(G12), typeof(ν12)))
    D11 = T(1) / T(E1)
    D22 = T(1) / T(E2)
    D12 = -T(ν12) / T(E1)
    D66 = T(1) / T(G12)
    D16 = T(η12_1) / T(E1)
    D26 = T(η12_2) / T(E2)
    D = @SMatrix [
        D11 D12 D16
        D12 D22 D26
        D16 D26 D66
    ]
    if plane_strain
        # D13=-ν13/E1=-ν31/E3. Cordeiro lists ν31,ν32 without E3; using
        # −ν31/E3 with E3=E2 makes D* indefinite (νzx=0.40, Ey=10 GPa).
        # Take the listed values as ν13, ν23 (Maxwell on Ex, Ey).
        D33 = T(1) / T(E3)
        D13 = -T(ν31) / T(E1)
        D23 = -T(ν32) / T(E2)
        D36 = T(η12_3) / T(E3)
        v3 = SVector{3,T}(D13, D23, D36)
        D = D - (v3 * v3') / D33
    end
    return lekhnitskii_params(inv(D))
end

"""
    lekhnitskii_params(E1, E2, G12, ν12; θ_deg=0) -> LekhnitskiiParams

Build Lekhnitskii complex parameters from orthotropic engineering constants
(optionally rotated by `θ_deg`). Legacy: `Compute_Material`.
"""
function lekhnitskii_params(E1::Real, E2::Real, G12::Real, ν12::Real; θ_deg=0.0)
    T = float(promote_type(typeof(E1), typeof(E2), typeof(G12), typeof(ν12)))
    ν21 = ν12 * E2 / E1
    # reduced stiffness Q (plane stress orthotropic)
    den = 1 - ν12 * ν21
    Q = @SMatrix [
        E1/den          ν12*E2/den      0
        ν12*E2/den      E2/den          0
        0               0               G12
    ]
    # rotate
    θ = T(θ_deg) * T(π) / 180
    m, n = cos(θ), sin(θ)
    Rot = @SMatrix [
        m^2   n^2    2m*n
        n^2   m^2   -2m*n
       -m*n   m*n    m^2-n^2
    ]
    C = inv(Rot) * Q * inv(Rot')
    return lekhnitskii_params(C)
end

"""
    lekhnitskii_params(C::AbstractMatrix) -> LekhnitskiiParams

From a full ``3×3`` reduced stiffness (Voigt order 11, 22, 12).
"""
function lekhnitskii_params(C::AbstractMatrix)
    T = float(eltype(C))
    C = SMatrix{3,3,T}(C)
    compliance = inv(C)
    a11, a12, a16 = compliance[1, 1], compliance[1, 2], compliance[1, 3]
    a22, a26, a66 = compliance[2, 2], compliance[2, 3], compliance[3, 3]

    # Lekhnitskii / Cordeiro eq. (10), z = x₁ + μ x₂:
    #   a11 μ⁴ − 2 a16 μ³ + (2 a12 + a66) μ² − 2 a26 μ + a22 = 0
    # The swapped quartic a22 μ⁴ + ⋯ + a11 is the reciprocal polynomial
    # (roots 1/μ). It coincides with this one only when a11 = a22 (isotropy).
    b = -2a16
    c = 2a12 + a66
    d = -2a26
    e = a22
    a = a11
    companion = @SMatrix [
        0        0        0       -e/a
        1        0        0       -d/a
        0        1        0       -c/a
        0        0        1       -b/a
    ]
    rts = eigvals(Matrix(companion))  # Complex
    aux = Complex{T}[]
    for r in rts
        if imag(r) > 0
            push!(aux, Complex{T}(r))
        end
    end
    length(aux) < 2 && error("Lekhnitskii: expected 2 roots with Im>0, got $(rts)")
    # order by real part
    if real(aux[1]) > real(aux[2])
        mi = SVector{2,Complex{T}}(aux[2], aux[1])
    else
        mi = SVector{2,Complex{T}}(aux[1], aux[2])
    end

    q = @SMatrix [
        a11*mi[1]^2+a12-a16*mi[1]    a11*mi[2]^2+a12-a16*mi[2]
        a12*mi[1]+a22/mi[1]-a26      a12*mi[2]+a22/mi[2]-a26
    ]
    Matriz = Complex{T}[
        1            -1             1             -1
        mi[1]        -conj(mi[1])   mi[2]         -conj(mi[2])
        q[1, 1]      -conj(q[1, 1]) q[1, 2]       -conj(q[1, 2])
        q[2, 1]      -conj(q[2, 1]) q[2, 2]       -conj(q[2, 2])
    ]
    im1 = one(Complex{T}) * im
    b1 = Complex{T}[0, -1 / (2π * im1), 0, 0]
    b2 = Complex{T}[1 / (2π * im1), 0, 0, 0]
    A1 = Matriz \ b1
    A2 = Matriz \ b2
    A = @SMatrix [
        A1[1] A1[3]
        A2[1] A2[3]
    ]
    g = @SMatrix [
        mi[1]  mi[2]
        -1     -1
    ]
    return LekhnitskiiParams{T}(mi, A, q, g, C)
end

"""
    lekhnitskii_rotate(p::LekhnitskiiParams, α) -> LekhnitskiiParams

Rebuild Lekhnitskii parameters in axes rotated by `α` (new ``x_1`` at angle
`α` from the current ``x_1``). Roots satisfy
``μ'=(μ\\cosα-\\sinα)/(\\cosα+μ\\sinα)``.
"""
function lekhnitskii_rotate(p::LekhnitskiiParams, α::Real)
    abs(α) < 1e-15 && return p
    m, n = cos(α), sin(α)
    Rot = @SMatrix [
        m^2   n^2    2m*n
        n^2   m^2   -2m*n
       -m*n   m*n    m^2-n^2
    ]
    Ctip = Rot * p.C * transpose(Rot)
    Ctip = (Ctip + transpose(Ctip)) / 2
    return lekhnitskii_params(Ctip)
end

"""
    fundamental(props::AnisotropicElasticity, y, x, n) -> KernelPair{Mat{2,2},Mat{2,2}}

2D Lekhnitskii fundamental solution at field `y`, source `x`, normal `n` at `y`.
Legacy: `calsolfund` for `elastico_aniso`.
"""
function fundamental(props::AnisotropicElasticity, y::SVector{2}, x::SVector{2}, n::SVector{2})
    p = props.params
    mi, A, q, g = p.mi, p.A, p.q, p.g
    dx = y[1] - x[1]
    dy = y[2] - x[2]
    z1 = dx + mi[1] * dy
    z2 = dx + mi[2] * dy

    lns = @SMatrix [log(z1) 0; 0 log(z2)]
    U = 2 * real(A * lns * conj(q)')

    mi_n_z = @SMatrix [
        (mi[1]*n[1]-n[2])/z1    0
        0                       (mi[2]*n[1]-n[2])/z2
    ]
    Tker = 2 * real(A * mi_n_z * conj(g)')
    return KernelPair(Mat{2,2}(Tuple(U)), Mat{2,2}(Tuple(Tker)))
end

# BEMdata: r = y - x  (same as isotropic convention)
function fundamental(dad::BEMdata{<:AnisotropicElasticity}, r::Point2D, n::Point2D)
    # need absolute positions — anisotropic kernel is not translation-only in the
    # same r-only form when written with pg, pf. With r = y-x it is fine:
    y = r          # treat r as y-x, source at origin equivalent
    x = zero(r)
    kp = fundamental(dad.properties, y, x, n)
    return _to_smat(kp.U), _to_smat(kp.T)
end

"""
    fundamental_stress(props::AnisotropicElasticity, y, x, n) -> StressKernels

Anisotropic D, S tensors (legacy `caldsolfund` for `elastico_aniso`).
"""
function fundamental_stress(
    props::AnisotropicElasticity,
    y::SVector{2},
    x::SVector{2},
    n::SVector{2},
)
    p = props.params
    mu, Afs, A3, q, g = p.mi, p.A, p.C, p.q, p.g
    dx = y[1] - x[1]
    dy = y[2] - x[2]
    z1 = dx + mu[1] * dy
    z2 = dx + mu[2] * dy

    invz = @SMatrix [1/z1 0; 0 1/z2]
    u_x = -2 * real(Afs * invz * conj(q)')
    R2 = @SMatrix [mu[1]/z1 0; 0 mu[2]/z2]
    u_y = -2 * real(Afs * R2 * conj(q)')

    mu_n_z = @SMatrix [
        (mu[1]*n[1]-n[2])/z1^2  0
        0                       (mu[2]*n[1]-n[2])/z2^2
    ]
    mu2_n_z = @SMatrix [
        mu[1]*(mu[1]*n[1]-n[2])/z1^2  0
        0                             mu[2]*(mu[2]*n[1]-n[2])/z2^2
    ]
    p_x = 2 * real(Afs * mu_n_z * conj(g)')
    p_y = 2 * real(Afs * mu2_n_z * conj(g)')

    # Voigt → tensor via stiffness A3
    D1 = A3 * SVector(u_x[1, 1], u_y[2, 1], u_y[1, 1] + u_x[2, 1])
    D2 = A3 * SVector(u_x[1, 2], u_y[2, 2], u_y[1, 2] + u_x[2, 2])
    S1 = A3 * SVector(p_x[1, 1], p_y[2, 1], p_y[1, 1] + p_x[2, 1])
    S2 = A3 * SVector(p_x[1, 2], p_y[2, 2], p_y[1, 2] + p_x[2, 2])

    D = zeros(MArray{Tuple{2,2,2},Float64})
    S = zeros(MArray{Tuple{2,2,2},Float64})
    D[1, 1, 1] = D1[1]; D[1, 2, 2] = D1[2]; D[1, 1, 2] = D1[3]; D[1, 2, 1] = D1[3]
    D[2, 1, 1] = D2[1]; D[2, 2, 2] = D2[2]; D[2, 1, 2] = D2[3]; D[2, 2, 1] = D2[3]
    S[1, 1, 1] = S1[1]; S[1, 2, 2] = S1[2]; S[1, 1, 2] = S1[3]; S[1, 2, 1] = S1[3]
    S[2, 1, 1] = S2[1]; S[2, 2, 2] = S2[2]; S[2, 1, 2] = S2[3]; S[2, 2, 1] = S2[3]

    return StressKernels(
        Tensor{Tuple{2,2,2},Float64}(Tuple(D)),
        Tensor{Tuple{2,2,2},Float64}(Tuple(S)),
    )
end

"""
    fundamental_hyper(props::AnisotropicElasticity, y, x, n, nf) -> KernelPair

Traction BIE kernels: ``U^h_{ik}=n^ξ_j D_{kij}``, ``T^h_{ik}=n^ξ_j S_{kij}``
(Cordeiro & Leonel 2020). Guiggiani orders `(-1, -2)`.
"""
function fundamental_hyper(
        props::AnisotropicElasticity,
        y::SVector{2}, x::SVector{2}, n::SVector{2}, nf::SVector{2},
    )
    sk = fundamental_stress(props, y, x, n)
    D, S = sk.D, sk.S
    n1, n2 = nf[1], nf[2]
    u11 = n1 * D[1, 1, 1] + n2 * D[1, 1, 2]
    u12 = n1 * D[2, 1, 1] + n2 * D[2, 1, 2]
    u21 = n1 * D[1, 2, 1] + n2 * D[1, 2, 2]
    u22 = n1 * D[2, 2, 1] + n2 * D[2, 2, 2]
    t11 = n1 * S[1, 1, 1] + n2 * S[1, 1, 2]
    t12 = n1 * S[2, 1, 1] + n2 * S[2, 1, 2]
    t21 = n1 * S[1, 2, 1] + n2 * S[1, 2, 2]
    t22 = n1 * S[2, 2, 1] + n2 * S[2, 2, 2]
    return KernelPair(@Mat([u11 u12; u21 u22]), @Mat([t11 t12; t21 t22]))
end

function fundamental_hyper(dad::BEMdata{<:AnisotropicElasticity}, r, n, nf)
    kp = fundamental_hyper(dad.properties, r, zero(r), n, nf)
    return _to_smat(kp.U), _to_smat(kp.T)
end

# =============================================================================
