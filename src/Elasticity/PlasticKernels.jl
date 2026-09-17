# 2-D Kelvin initial-stress kernels for constant-cell elastoplasticity.
# Gao & Davies (2002) / Telles (1983); Voigt in-plane (σ11, σ22, σ12).

export initial_strain_kernel, initial_stress_kernel, initial_stress_free_term

"""
    initial_strain_kernel(props, r) -> SMatrix{2,3}

Displacement kernel ``E_{ijk}`` contracted with in-plane Voigt
``(σ_{11},σ_{22},σ_{12})``. Thesis (4.17), field derivatives of Kelvin ``U``:

```math
E_{ijk}=\\frac{U_{ij,k}+U_{ik,j}}{2}
```

so ``Δu_i = E_{i,α}\\,σ^p_α``. ``r = y - x`` (field minus source).
"""
function initial_strain_kernel(props::Elasticity, r::SVector{2})
    R = norm(r)
    R < 1e-16 && return zero(SMatrix{2,3,Float64,6})
    n0 = SVector(1.0, 0.0)
    Ux, _, Uy, _ = fundamental_grad(props, r, n0)
    # E_i11 = ∂U_i1/∂y1, E_i22 = ∂U_i2/∂y2,
    # Voigt shear: E_i12+E_i21 = ∂U_i1/∂y2 + ∂U_i2/∂y1
    return @SMatrix [
        Ux[1, 1]  Uy[1, 2]  (Uy[1, 1] + Ux[1, 2])
        Ux[2, 1]  Uy[2, 2]  (Uy[2, 1] + Ux[2, 2])
    ]
end

"""
    initial_stress_kernel(props, r) -> SMatrix{3,3}

Strongly singular ``E_{ijkl}`` in in-plane Voigt (thesis 4.28, plane strain
with ``α=1,β=2,γ=4``). Multiplies ``(σ_{11},σ_{22},σ_{12})``; shear column
already includes the ``σ_{12}+σ_{21}`` factor. Plane stress uses
[`effective_nu`](@ref).
"""
function initial_stress_kernel(props::Elasticity, r::SVector{2})
    ν = effective_nu(props)
    R = norm(r)
    R < 1e-16 && return zero(SMatrix{3,3,Float64,9})
    e1, e2 = r[1] / R, r[2] / R
    fac = 1 / (4π * (1 - ν) * R * R)
    f2 = 1 - 2ν
    function E4(i, j, k, l)
        ei = i == 1 ? e1 : e2
        ej = j == 1 ? e1 : e2
        ek = k == 1 ? e1 : e2
        el = l == 1 ? e1 : e2
        δij = i == j ? 1.0 : 0.0
        δik = i == k ? 1.0 : 0.0
        δil = i == l ? 1.0 : 0.0
        δjk = j == k ? 1.0 : 0.0
        δjl = j == l ? 1.0 : 0.0
        δkl = k == l ? 1.0 : 0.0
        t = f2 * (δik * δjl + δjk * δil - δij * δkl + 2 * δij * ek * el)
        t += 2ν * (δil * ej * ek + δjk * el * ei + δik * el * ej + δjl * ei * ek)
        t += 2 * δkl * ei * ej
        t -= 8 * ei * ej * ek * el
        return fac * t
    end
    return @SMatrix [
        E4(1, 1, 1, 1)  E4(1, 1, 2, 2)  2 * E4(1, 1, 1, 2)
        E4(2, 2, 1, 1)  E4(2, 2, 2, 2)  2 * E4(2, 2, 1, 2)
        E4(1, 2, 1, 1)  E4(1, 2, 2, 2)  2 * E4(1, 2, 1, 2)
    ]
end

"""
    initial_stress_free_term(props) -> SMatrix{3,3}

Jump ``F_{ijkl}`` in in-plane Voigt (thesis 4.31). Added only when the
source sits inside the cell (centroid of a constant cell).
"""
function initial_stress_free_term(props::Elasticity)
    ν = effective_nu(props)
    c = -1 / (8 * (1 - ν))
    # F_ijkl = c * [(δ_ik δ_jl + δ_il δ_jk) + (1-4ν) δ_ij δ_kl]
    F1111 = c * (2 + (1 - 4ν))
    F1122 = c * (1 - 4ν)
    F2211 = F1122
    F2222 = F1111
    # σ12 += F_1212 σ12 + F_1221 σ21 = 2 c σ12  →  F_v[3,3] = 2c = -1/(4(1-ν))
    F1212v = 2c
    return @SMatrix [
        F1111   F1122   0.0
        F2211   F2222   0.0
        0.0     0.0     F1212v
    ]
end
