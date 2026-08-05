# =============================================================================
# Local (n, t) reference system for 2D elasticity BEM
# Leonardo Bernardo e Silva (2026) §4.7
#
# Global displacements/tractions relate to nodal normal–tangent components by
#
#   [u₁]   [ n₁  -n₂ ] [u_n]          [t₁]   [ n₁  -n₂ ] [t_n]
#   [u₂] = [ n₂   n₁ ] [u_t]    ,     [t₂] = [ n₂   n₁ ] [t_t]
#
# i.e. u_g = R u_ℓ with columns of R = (n̂, t̂), t̂ = (−n₂, n₁).
#
# Assembled operators in the global frame (H, G) become
#
#   Ĥ = H R_block ,   Ĝ = G R_block
#
# so that  Ĥ û = Ĝ t̂ + p , with û, t̂ written in (n, t) at every boundary node.
# BCs are then imposed on (u_n, u_t, t_n, t_t) by the usual column exchange.
# Rigid-body diagonal terms of H must be formed in (x₁, x₂) *before* this map
# (already done in H_G_full_direct).
# =============================================================================

export local_basis2d, node_rotation2d, rotation_block2d
export transform_HG_local, transform_HG_local!
export applyBC_local, applyBC_local!
export global_to_local_field, local_to_global_field
export solve_local, bc_global_to_local!

"""
    local_basis2d(n) -> (n̂, t̂)

Unit outward normal and right-handed tangent `t̂ = (−n̂₂, n̂₁)`.
"""
function local_basis2d(n)
    nlen = hypot(n[1], n[2])
    nlen < eps(Float64) && throw(ArgumentError("zero normal"))
    n̂ = Point2D(n[1] / nlen, n[2] / nlen)
    t̂ = Point2D(-n̂[2], n̂[1])
    return n̂, t̂
end

"""
    node_rotation2d(n) -> SMatrix{2,2}

Rotation `R` with `u_global = R * u_local`, columns `(n̂, t̂)`.
Eq. (4.63) of Leonardo (2026).
"""
function node_rotation2d(n)
    n̂, t̂ = local_basis2d(n)
    return @SMatrix [n̂[1] t̂[1]; n̂[2] t̂[2]]
end

"""
    rotation_block2d(dad) -> Matrix

Block-diagonal `R_block` (`2n × 2n`) stacking `node_rotation2d` at each
boundary node. Internal collocation columns are not included (only boundary).
"""
function rotation_block2d(dad::BEMdata{<:Vectorial})
    dad.dimension == 2 || throw(ArgumentError("local frame is 2D only"))
    n = dad.n
    R = zeros(2n, 2n)
    @inbounds for j in 1:n
        Rj = node_rotation2d(dad.Normal[j])
        cols = 2(j - 1) + 1:2j
        R[cols, cols] .= Rj
    end
    return R
end

"""
    transform_HG_local(H, G, dad) -> (Ĥ, Ĝ)

Right-multiply boundary columns of `H` and all columns of `G` by the nodal
rotations (Leonardo Eqs. 4.64–4.66):

```math
\\hat H = H\\,R_{\\mathrm{block}},\\qquad \\hat G = G\\,R_{\\mathrm{block}}.
```

Internal-node columns of `H` (if `nt > n`) are left unchanged.
"""
function transform_HG_local(H::AbstractMatrix, G::AbstractMatrix, dad::BEMdata{<:Vectorial})
    Hhat = copy(H)
    Ghat = copy(G)
    transform_HG_local!(Hhat, Ghat, H, G, dad)
    return Hhat, Ghat
end

function transform_HG_local!(
        Hhat::AbstractMatrix,
        Ghat::AbstractMatrix,
        H::AbstractMatrix,
        G::AbstractMatrix,
        dad::BEMdata{<:Vectorial},
    )
    dad.dimension == 2 || throw(ArgumentError("local frame is 2D only"))
    n = dad.n
    size(G, 2) == 2n || throw(DimensionMismatch("G must have 2n columns"))
    # start from global operators
    Hhat === H || (Hhat .= H)
    Ghat === G || (Ghat .= G)
    @inbounds for j in 1:n
        Rj = Matrix(node_rotation2d(dad.Normal[j]))
        cols = 2(j - 1) + 1:2j
        # Ĥ[:,cols] = H[:,cols] * Rⱼ  (and same for G)
        Hhat[:, cols] = H[:, cols] * Rj
        Ghat[:, cols] = G[:, cols] * Rj
    end
    return Hhat, Ghat
end

"""
    global_to_local_field(dad, v_global) -> v_local

Map a 2n boundary field (displacements or tractions) from (x, y) to (n, t):
`v_ℓ = Rᵀ v_g` per node.
"""
function global_to_local_field(dad::BEMdata{<:Vectorial}, v_global::AbstractVector)
    dad.dimension == 2 || throw(ArgumentError("local frame is 2D only"))
    n = dad.n
    length(v_global) >= 2n || throw(DimensionMismatch("field length < 2n"))
    v_local = similar(v_global, 2n)
    @inbounds for j in 1:n
        Rj = node_rotation2d(dad.Normal[j])
        vg = SVector(v_global[2j-1], v_global[2j])
        vl = Rj' * vg
        v_local[2j-1] = vl[1]
        v_local[2j] = vl[2]
    end
    return v_local
end

"""
    local_to_global_field(dad, v_local) -> v_global

Map a 2n boundary field from (n, t) to (x, y): `v_g = R v_ℓ` per node.
"""
function local_to_global_field(dad::BEMdata{<:Vectorial}, v_local::AbstractVector)
    dad.dimension == 2 || throw(ArgumentError("local frame is 2D only"))
    n = dad.n
    length(v_local) >= 2n || throw(DimensionMismatch("field length < 2n"))
    v_global = similar(v_local, 2n)
    @inbounds for j in 1:n
        Rj = node_rotation2d(dad.Normal[j])
        vl = SVector(v_local[2j-1], v_local[2j])
        vg = Rj * vl
        v_global[2j-1] = vg[1]
        v_global[2j] = vg[2]
    end
    return v_global
end

"""
    applyBC_local!(dad; p=nothing)

Build the mixed system in the **local** (n, t) frame (Leonardo §4.7):

1. `Ĥ, Ĝ ← transform_HG_local(H, G, dad)`
2. Column exchange from `dad.BC` / `dad.BV`, now interpreted as
   - dof `2i-1` → normal (`u_n` / `t_n`)
   - dof `2i`   → tangent (`u_t` / `t_t`)
3. Optional body-force boundary contribution `p` (`Ĥ û = Ĝ t̂ + p`).

Result cached as `dad.A`, `dad.B`, `dad.b`, plus `H_local`, `G_local`.
"""
function applyBC_local!(dad::BEMdata{<:Vectorial}; p::Union{Nothing,AbstractVector}=nothing)
    dad.dimension == 2 || throw(ArgumentError("applyBC_local! is 2D only"))
    has_cache(dad, :H) && has_cache(dad, :G) ||
        error("assemble H, G first (H_G_full_direct)")

    H, G = dad.H, dad.G
    ndof = 2 * dad.n
    nrows = size(H, 1)

    Hhat, Ghat = transform_HG_local(H, G, dad)

    A = Matrix(Hhat[1:ndof, 1:ndof])   # boundary rows/cols (collocation on boundary)
    # If internal collocation exists, keep full rows for residual — standard BEM
    # uses boundary collocation only for the square system:
    if nrows != ndof
        # use only boundary collocation rows
        A = Matrix(Hhat[1:ndof, 1:ndof])
        Ghat_b = Matrix(Ghat[1:ndof, 1:ndof])
    else
        Ghat_b = Matrix(Ghat)
    end
    Bmat = copy(Ghat_b)
    b = zeros(eltype(A), ndof)

    BC = dad.BC
    BV = dad.BV
    length(BC) >= ndof || error("BC length $(length(BC)) < 2n = $ndof")

    fill!(b, 0)
    @inbounds for dof in 1:ndof
        if BC[dof] == 0
            # Dirichlet in local direction: unknown is local traction
            A[:, dof] .= .-view(Ghat_b, :, dof)
            b .-= view(Hhat, 1:ndof, dof) .* BV[dof]
        else
            # Neumann in local direction: known local traction
            b .+= view(Ghat_b, :, dof) .* BV[dof]
        end
    end

    if p !== nothing
        length(p) >= ndof || throw(DimensionMismatch("p length"))
        b .+= view(p, 1:ndof)
    end

    set_cache!(dad; A, B=Bmat, b, H_local=Hhat, G_local=Ghat)
    return nothing
end

"""Alias kept for naming symmetry with [`applyBC`](@ref)."""
applyBC_local(dad::BEMdata{<:Vectorial}; kwargs...) = applyBC_local!(dad; kwargs...)

"""
    solve_local(dad; p=nothing) -> u_global

Solve 2D elasticity with BCs in the nodal (n, t) frame (Leonardo §4.7).

# Boundary-condition convention
`dad.BC` / `dad.BV` components at node `i`:
| slot | direction | BC=0 (Dirichlet) | BC=1 (Neumann) |
|------|-----------|------------------|----------------|
| `2i-1` | normal `n̂` | prescribed `u_n` | prescribed `t_n` |
| `2i`   | tangent `t̂` | prescribed `u_t` | prescribed `t_t` |

After the solve, global fields are stored in `dad.u` / `dad.traction`, and local
fields in `dad.u_local` / `dad.traction_local`.
"""
function solve_local(dad::BEMdata{<:Vectorial}; p::Union{Nothing,AbstractVector}=nothing)
    applyBC_local!(dad; p=p)
    x = bem_linsolve(dad.A, dad.b)
    ndof = 2 * dad.n
    u_loc = zeros(eltype(x), ndof)
    t_loc = zeros(eltype(x), ndof)
    _split_sol_local!(dad, x, u_loc, t_loc)
    u_glb = local_to_global_field(dad, u_loc)
    t_glb = local_to_global_field(dad, t_loc)
    set_cache!(dad; u=u_glb, traction=t_glb, T=u_glb,
               u_local=u_loc, traction_local=t_loc)
    return u_glb
end

function _split_sol_local!(dad, x, u_loc, t_loc)
    BC = dad.BC
    BV = dad.BV
    @inbounds for dof in eachindex(BC)
        dof > length(u_loc) && break
        if BC[dof] == 0
            t_loc[dof] = x[dof]
            u_loc[dof] = BV[dof]
        else
            t_loc[dof] = BV[dof]
            u_loc[dof] = x[dof]
        end
    end
    return nothing
end

"""
    bc_global_to_local!(dad)

Rewrite `dad.BC` / `dad.BV` from global (x, y) components into local (n, t)
components **when both DOFs at a node share the same BC type** (both Dirichlet
or both Neumann). Mixed (x-Dirichlet / y-Neumann) nodes are left unchanged and
reported via `@warn` — those cannot be rotated without changing the constraint
set.

Useful to compare `solve(dad; frame=:global)` vs `solve(dad; frame=:local)` on
the same mesh after conversion.
"""
function bc_global_to_local!(dad::BEMdata{<:Vectorial})
    dad.dimension == 2 || throw(ArgumentError("bc_global_to_local! is 2D only"))
    n_mixed = 0
    @inbounds for j in 1:dad.n
        d1, d2 = 2j - 1, 2j
        if dad.BC[d1] != dad.BC[d2]
            n_mixed += 1
            continue
        end
        Rj = node_rotation2d(dad.Normal[j])
        vg = SVector(dad.BV[d1], dad.BV[d2])
        vl = Rj' * vg
        dad.BV[d1] = vl[1]
        dad.BV[d2] = vl[2]
        # BC types unchanged (both already equal)
    end
    n_mixed > 0 && @warn "bc_global_to_local!: left $n_mixed mixed-type nodes in global frame"
    return dad
end
