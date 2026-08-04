export DIBEM_dense, dibem!

"""
    DIBEM_dense(dad; rbf=PHS())

Dense **Direct Interpolation BEM** operator `M`:

```
∫_Ω β(X) u*(ξ,X) dΩ  ≈  (M β)(ξ)
```

Stores `M` in `dad.cache.M` and returns it.

For large problems see [`DIBEM`](@ref) with `method=:hmatrix` or `:fmm`
([`DIBEM_Hmat`](@ref), [`DIBEM_FMM`](@ref) in `Domain_fast.jl`).
"""
function DIBEM_dense(dad::BEMdata{<:Laplace}; rbf=PHS())

    npoly = binomial(dad.dimension + rbf.poly_deg, rbf.poly_deg)
    mon = MonomialBasis(dad.dimension, rbf.poly_deg)
    F = zeros(dad.nt, dad.nt)
    D = zeros(dad.nt, dad.nt)
    IF = zeros(dad.nt)
    ID = zeros(dad.nt)
    P = zeros(npoly, dad.nt)
    IP = zeros(npoly)
    k = dad.properties.k


    props = dad.properties
    n0 = dad.Normal[1]  # dummy normal — only U from fundamental is used in D
    @showprogress "Assembling F and D" for j in 1:dad.nt, i in 1:dad.nt
        x = point(dad, i)
        xj = point(dad, j)
        r2 = sqeuclidean(x, xj)
        F[i, j] = rbf(r2)
        if r2 > 0
            # single-layer u* from Fundamental.jl (same as H,G assembly)
            D[i, j] = fundamental(props, xj - x, n0).U
        end
    end

    @showprogress "Integrating fundamental solutions and radial basis functions" for i in 1:dad.nt
        x = point(dad, i)
        P[:, i] = mon(x)
        for elem in dad.elements
            for j in eachindex(elem.index)
                ind = elem.index[j]
                xj = ind <= dad.n ? dad.Nodes[ind] : dad.internalNodes[ind-dad.n]
                r = xj - x
                R = norm(r)
                if R < 1e-10
                    continue
                    # R = 1e-16
                end
                # @infiltrate
                if i == 1
                    IP += int(mon, x, xj) * dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
                end
                wJn = dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
                IF[i] += int(rbf, x, xj) * wJn
                # Galerkin tensor remainder ∫ n·∇G* dΓ (primitive of u* = fundamental.U)
                ID[i] += _galerkin_n_dot_gradG(props, R) * wJn
            end

        end
    end
    # @infiltrate
    # @show F[1:5, 1:5], D[1:5, 1:5], IF[1:5], ID[1:5]

    # Z = zeros(npoly, npoly)
    # aux = [F P'; P Z]
    # # @infiltrate
    # M = ([IF; IP]'/aux)[1:end-npoly] .* D

    M = IF' / F .* D
    for i = 1:dad.nt #Laço dos pontos radiais
        M[i, i] = 0
        M[i, i] = -sum(M[i, :]) + ID[i]
    end
    set_cache!(dad; M, dibem_F=F, dibem_rbf=rbf, dibem_method=:dense)
    return M
end

"""
    dibem_matrix(dad; rbf=PHS(), rebuild=false, method=:dense) -> M

Return the DIBEM operator `M`. Rebuilds via [`DIBEM`](@ref) if missing or
`rebuild=true`.
"""
function dibem_matrix(dad::BEMdata{<:Laplace}; rbf=PHS(), rebuild::Bool=false,
        method::Symbol=:dense, kwargs...)
    if rebuild || !has_cache(dad, :M)
        return DIBEM(dad; method=method, rbf=rbf, kwargs...)
    end
    return dad.M
end

export dibem_matrix

"""
    _galerkin_n_dot_gradG(props, R) → scalar

Radial factor in `n·∇G*` for the Galerkin tensor of the Laplace single layer,
such that `∫ (n·∇G*) dΓ = ∫ _galerkin_n_dot_gradG(R) (n·r/R²) dΓ`.

For 2D: G* = -(2 R² ln R − R²)/(8π k) with ∇²G* = u* = fundamental.U.
"""
function _galerkin_n_dot_gradG(props::Laplace, R::Real)
    k = float(props.k)
    # matches legacy: -(2 R² log R − R²) / (8π k)
    return -(2 * R^2 * log(R) - R^2) / (8 * π * k)
end
