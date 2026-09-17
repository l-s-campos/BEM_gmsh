export applyBC, applyBC!
export applyBC_blocks

# =============================================================================
# Laplace — dense
# =============================================================================

"""
    applyBC(dad; blocks=false, M=nothing, κ2=0)

Build the mixed BC linear system on `dad`.

# Keywords
- `blocks=false` — classical column-swap on dense `H,G` (legacy).
- `blocks=true`  — pack unknowns as `x = [T_u; q_q]` from BC blocks
  [`build_block_mixed_system`](@ref). Prefer for hierarchical / FMM `H,G`
  (whole subblocks exchanged). Optional domain mass `M` with shift `κ2`
  builds `(H+κ²M)`-style blocks without densifying.
"""
function applyBC(dad::BEMdata{<:LaplaceLike};
                 blocks::Bool=false, M=nothing, κ2::Real=0.0)
    H = dad.H
    G = dad.G

    # structured operators always use the block (or legacy matrix-free) path
    structured = H isa HMatrices.HMatrix || H isa ColWeightedOp ||
                 H isa BlockMixedOperator
    if blocks || structured
        return applyBC_blocks(dad; M=M, κ2=κ2)
    end

    if has_cache(dad, :A) && has_cache(dad, :B) && has_cache(dad, :b)
        A = dad.A
        B = dad.B
        b = dad.b
        A .= H
        if size(B) == size(G)
            B .= G
        else
            B = deepcopy(G)
            dad.cache.B = B
        end
        fill!(b, 0)
    else
        A = deepcopy(H)
        B = deepcopy(G)
        b = zeros(eltype(H), size(H, 1))
        set_cache!(dad; A, B, b)
    end
    applyBC(dad, A, B, b)
    return nothing
end

function applyBC(dad::BEMdata{<:LaplaceLike}, A, B, b)
    n = dad.n
    for bc in 1:n
        if dad.BC[bc] == 0  # Dirichlet: unknown is q
            A[:, bc] .= .-view(dad.G, :, bc)
            if size(B, 2) >= bc
                B[:, bc] .= .-view(dad.H, :, bc)
            end
        end
    end
    fill!(b, 0)
    H = dad.H
    G = dad.G
    for j in 1:n
        if dad.BC[j] == 0
            b .-= view(H, :, j) .* dad.BV[j]
        else
            b .+= view(G, :, j) .* dad.BV[j]
        end
    end
    return nothing
end

# =============================================================================
# Laplace — block-partitioned mixed BC (Hmat / FMM / optional dense)
# =============================================================================

"""
    applyBC_blocks(dad; M=nothing, κ2=0)

Partition `H,G` (and optional `M`) into BC blocks and build

```
A = [Huu+κ²Muu  −Guq;  Hqu+κ²Mqu  −Gqq] ,   x = [T_u; q_q]
```

Caches `A`, `b`, `bc_idx`, `hg_blocks` on `dad`.
"""
function applyBC_blocks(dad::BEMdata{<:LaplaceLike};
                        M=nothing, κ2::Real=0.0)
    H = dad.H
    G = dad.G
    Mop = M === nothing ? (has_cache(dad, :M) && κ2 != 0 ? dad.M : nothing) : M
    sys = build_block_mixed_system(H, G, dad; M=Mop, κ2=κ2)
    set_cache!(dad; A=sys.A, b=sys.b, bc_idx=sys.idx, hg_blocks=sys.blocks)
    return nothing
end

applyBC!(dad::BEMdata{<:LaplaceLike}; kwargs...) = applyBC(dad; kwargs...)

# =============================================================================
# Elasticity — dense
# =============================================================================

"""
    applyBC(dad::BEMdata{<:Elasticity}; frame=:global, p=nothing)

Build mixed BC system for elasticity.

- `frame=:global` — classical (x, y) DOFs
- `frame=:local` — nodal (n, t) DOFs (Leonardo §4.7); see [`applyBC_local!`](@ref)
"""
function applyBC(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity,AnisotropicElasticity3D}};
        frame::Symbol=:global, p=nothing)
    if frame === :local
        return applyBC_local!(dad; p=p)
    elseif frame !== :global
        throw(ArgumentError("frame must be :global or :local (got $frame)"))
    end
    H = dad.H
    G = dad.G
    ndof = size(H, 1)

    if has_cache(dad, :A) && has_cache(dad, :b)
        A = dad.A
        Bmat = has_cache(dad, :B) ? dad.B : deepcopy(G)
        b = dad.b
        A .= H
        if size(Bmat) == size(G)
            Bmat .= G
        else
            Bmat = deepcopy(G)
        end
        fill!(b, 0)
        set_cache!(dad; B=Bmat)
    else
        A = deepcopy(H)
        Bmat = deepcopy(G)
        b = zeros(eltype(H), ndof)
        set_cache!(dad; A, B=Bmat, b)
    end
    applyBC(dad, A, dad.B, b)
    return nothing
end

applyBC!(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity,AnisotropicElasticity3D}}; kwargs...) =
    applyBC(dad; kwargs...)

function applyBC(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity,AnisotropicElasticity3D}}, A, B, b)
    dim = dad.dimension
    n = dad.n
    ndof_b = dim * n
    BC = dad.BC
    BV = dad.BV
    H = dad.H
    G = dad.G

    length(BC) >= ndof_b ||
        error("Elasticity BC length $(length(BC)) < dimension*n = $ndof_b")

    A .= H
    fill!(b, 0)
    for dof in 1:ndof_b
        if BC[dof] == 0
            A[:, dof] .= .-view(G, :, dof)
            b .-= view(H, :, dof) .* BV[dof]
        else
            b .+= view(G, :, dof) .* BV[dof]
        end
    end
    return nothing
end
