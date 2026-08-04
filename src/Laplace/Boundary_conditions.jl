export applyBC

# =============================================================================
# Laplace — dense
# =============================================================================

function applyBC(dad::BEMdata{<:Union{Laplace,OrthotropicLaplace}})
    H = dad.H
    G = dad.G

    # hierarchical or factored (ColWeightedOp) — matrix-free mixed BC
    if H isa HMatrices.HMatrix || H isa ColWeightedOp
        return applyBC_Hmat(dad)
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

function applyBC(dad::BEMdata{<:Union{Laplace,OrthotropicLaplace}}, A, B, b)
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
# Laplace — H-matrix (matrix-free mixed operator)
# =============================================================================

function applyBC_Hmat(dad::BEMdata{<:Laplace})
    H = dad.H
    G = dad.G
    A = MixedBCOperator(H, G, dad.BC, dad.n, dad.nt)
    b = mixed_bc_rhs(H, G, dad)
    set_cache!(dad; A, b)
    return nothing
end

# =============================================================================
# Elasticity — dense
# =============================================================================

function applyBC(dad::BEMdata{<:Elasticity})
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

function applyBC(dad::BEMdata{<:Elasticity}, A, B, b)
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
