export DIBEM_dense, dibem_elasticity!, dibem_matrix

"""
    DIBEM_dense(dad::BEMdata{<:Elasticity}; rbf=PHS())

Dense **Direct Interpolation BEM** operator `M` for 2D isotropic elasticity:

```
∫_Ω U*(ξ,X) · b(X) dΩ  ≈  (M b)(ξ)
```

`M` is `(2·nt) × (2·nt)`, stored in `dad.cache.M`.

Same discrete structure as the Laplace [`DIBEM_dense`](@ref) in `Laplace/Domain.jl`:

```
F c = IF ,   M-block_ij = c_j U*(x_i,x_j)  (i≠j) ,
M-block_ii = ID_i − ∑_{j≠i} M-block_ij
```

where `IF` / `ID` come from the radial integration identity
`∫_Ω f dΩ = ∫_Γ (∫_0^R f ρ dρ)(n·r/R²) dΓ` applied to the RBF and to the
Kelvin single layer `U*`.

For compressed backends see [`DIBEM`](@ref) with `method=:hmatrix` etc.
([`Domain_fast.jl`](@ref)).
"""
function DIBEM_dense(dad::BEMdata{<:Elasticity}; rbf=PHS())
    dim = dad.dimension
    dim == 2 || error("DIBEM_dense(Elasticity) is 2D only (got dimension=$dim)")
    nt = dad.nt
    props = dad.properties
    n0 = dad.Normal[1]  # dummy normal — only U from Kelvin is used in D

    F = zeros(nt, nt)
    D = zeros(2nt, 2nt)          # U* blocks between collocation poles
    IF = zeros(nt)               # ∫ φ̂ (n·r/R²) dΓ
    ID = zeros(2nt, 2)           # ∫ Û* (n·r/R²) dΓ   (2×2 per pole, row-major)

    @showprogress "Assembling F and D (elasticity DIBEM)" for j in 1:nt, i in 1:nt
        x = point(dad, i)
        xj = point(dad, j)
        r2 = sqeuclidean(x, xj)
        F[i, j] = rbf(r2)
        if r2 > 0
            rvec = xj - x
            U, _ = fundamental(dad, rvec, n0)
            D[2i-1:2i, 2j-1:2j] .= U
        end
    end

    @showprogress "Integrating RBF / Kelvin particular (elasticity)" for i in 1:nt
        x = point(dad, i)
        for elem in dad.elements
            for j in eachindex(elem.index)
                ind = elem.index[j]
                xj = dad.Nodes[ind]
                r = xj - x
                R = norm(r)
                R < 1e-10 && continue
                wJn = dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
                IF[i] += int(rbf, x, xj) * wJn
                e = r / R
                ID[2i-1:2i, :] .+= _galerkin_Ustar(props, R, e) * wJn
            end
        end
    end

    # c' = IF' / F  →  M-block_ij = c_j U*_ij
    c = vec(IF' / F)             # length nt
    M = zeros(2nt, 2nt)
    @inbounds for j in 1:nt
        a = c[j]
        M[:, 2j-1] .= a .* D[:, 2j-1]
        M[:, 2j]   .= a .* D[:, 2j]
    end
    # diagonal blocks from constant-body-force identity
    @inbounds for i in 1:nt
        rows = 2i-1:2i
        M[rows, rows] .= 0
        # row-sum of off-diagonal blocks acting on e₁, e₂
        s1 = sum(view(M, rows, 1:2:2nt); dims=2)
        s2 = sum(view(M, rows, 2:2:2nt); dims=2)
        M[rows, rows] .= .-hcat(s1, s2) .+ ID[rows, :]
    end

    set_cache!(dad; M, dibem_F=F, dibem_rbf=rbf, dibem_method=:dense,
        dibem_c=c, dibem_ID=ID)
    return M
end

"""
    dibem_elasticity!(dad; rbf=PHS(), kwargs...)

Alias for [`DIBEM_dense`](@ref)`(dad::BEMdata{<:Elasticity})`. Kept for
callers in thermoelasticity / plates.
"""
function dibem_elasticity!(dad::BEMdata{<:Elasticity}; rbf=PHS(), kwargs...)
    return DIBEM_dense(dad; rbf=rbf)
end

"""
    dibem_matrix(dad::BEMdata{<:Elasticity}; rbf=PHS(), rebuild=false, method=:dense)

Return the elasticity DIBEM operator `M`. Rebuilds via [`DIBEM`](@ref) if
missing or `rebuild=true`.
"""
function dibem_matrix(dad::BEMdata{<:Elasticity}; rbf=PHS(), rebuild::Bool=false,
        method::Symbol=:dense, kwargs...)
    if rebuild || !has_cache(dad, :M)
        return DIBEM(dad; method=method, rbf=rbf, kwargs...)
    end
    return dad.M
end

"""
    _galerkin_Ustar(props, R, e) → SMatrix{2,2}

Radial particular of the 2D Kelvin single layer along direction `e = r/R`:

```
Û*(R,e) = ∫_0^R U*(ρ e) ρ dρ
```

so that `∫_Ω U* dΩ = ∫_Γ Û* (n·r/R²) dΓ`.

With
`U* = [(3-4ν) log(1/ρ) I + e⊗e] / (8π μ (1-ν))`:

```
Û* = [ (3-4ν)(R²/4 − R²/2 log R) I + (R²/2) e⊗e ] / (8π μ (1-ν))
```
"""
function _galerkin_Ustar(props::Elasticity, R::Real, e::SVector{2})
    ν = effective_nu(props)
    μ = shear_modulus(props)
    base = 8 * π * μ * (1 - ν)
    R2 = R * R
    logR = log(R)
    c_iso = (3 - 4ν) * (R2 / 4 - R2 / 2 * logR)
    c_dir = R2 / 2
    # (c_iso * I + c_dir * e⊗e) / base
    e1, e2 = e[1], e[2]
    return @SMatrix [
        (c_iso + c_dir * e1 * e1)/base    (c_dir * e1 * e2)/base
        (c_dir * e2 * e1)/base            (c_iso + c_dir * e2 * e2)/base
    ]
end
