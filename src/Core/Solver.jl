# Steady BEM solvers (Laplace + Elasticity). Transient drivers: Laplace/Solver.jl

export solve, solve_Hmat, split_sol!, split_sol
# solve_local exported from Elasticity/LocalFrame.jl
# reduced_heat_system / reduced_wave_system live in Laplace/Solver.jl

# =============================================================================
# Steady Laplace
# =============================================================================

"""
    solve(dad::BEMdata; kwargs...)

Apply mixed BCs (`applyBC`) and solve ``A x = b``.

For Laplace, stores the full field in `dad.T` / `dad.q`. Dense systems use
`bem_linsolve`; hierarchical / mixed operators use GMRES (`solve_Hmat`).
`factor=:ulv` uses HODLR LU of the mixed 2×2 with HSS on all four tiles
and ULV on `Huu` and the Schur of `-Gqq`.

For elasticity, `frame=:global` (default) uses Cartesian BCs; `frame=:local`
uses nodal ``(n,t)`` components ([`solve_local`](@ref)).
"""
function solve(dad::BEMdata{<:LaplaceLike};
               blocks::Bool=false, M=nothing, κ2::Real=0.0,
               factor::Symbol=:auto)
    applyBC(dad; blocks=blocks, M=M, κ2=κ2)
    A = dad.A
    b = dad.b

    # BC blocks → HSS+ULV on square diagonals, or one-level H-LU
    if A isa BlockMixedOperator && factor === :ulv
        F = factor_block_ulv(A, dad.collocation)
        x = F \ b
        Tfull, qfull = scatter_block_sol!(dad, x, dad.bc_idx)
        set_cache!(dad; T=Tfull, q=qfull, block_ulv=F)
        return dad.T
    end
    if A isa BlockMixedOperator && factor !== :gmres
        try
            F = factor_block_hlu(A)
            x = F \ b
            Tfull, qfull = scatter_block_sol!(dad, x, dad.bc_idx)
            set_cache!(dad; T=Tfull, q=qfull, block_lu=F)
            return dad.T
        catch e
            e isa ArgumentError || rethrow()
            # fall through to GMRES
        end
    end

    # Hierarchical / matrix-free → iterative
    if A isa BlockMixedOperator || A isa HMatrices.HMatrix
        return solve_Hmat(dad)
    end

    x = bem_linsolve(A, b)
    Tfull = zeros(eltype(x), dad.nt)
    qfull = zeros(eltype(x), dad.n)
    Tfull[1:length(x)] .= x
    split_sol!(dad, Tfull, qfull)
    set_cache!(dad; T=Tfull[1:dad.nt], q=qfull)
    return dad.T
end

function solve_Hmat(dad::BEMdata{<:LaplaceLike}; Pl=nothing, atol=1e-10, rtol=1e-8, itmax=0)
    A = dad.A
    b = dad.b
    itm = itmax > 0 ? Int(itmax) : max(4 * size(A, 1), 200)
    if Pl !== nothing || isdefined(HMatrices, :gmres_h)
        x, stats = HMatrices.gmres_h(A, b; Pl=Pl, atol=atol, rtol=rtol, itmax=itm)
    else
        x, stats = Krylov.gmres(A, b; atol=atol, rtol=rtol, itmax=itm)
    end

    if A isa BlockMixedOperator && has_cache(dad, :bc_idx)
        Tfull, qfull = scatter_block_sol!(dad, x, dad.bc_idx)
    else
        Tfull = zeros(eltype(x), dad.nt)
        qfull = zeros(eltype(x), dad.n)
        Tfull[1:length(x)] .= x
        split_sol!(dad, Tfull, qfull)
    end
    set_cache!(dad; T=Tfull, q=qfull, gmres_stats=stats)
    return dad.T
end

function split_sol!(dad::BEMdata{<:LaplaceLike}, T, q)
    @inbounds for bc in eachindex(dad.BC)
        if dad.BC[bc] == 0
            q[bc] = T[bc]
            T[bc] = dad.BV[bc]
        else
            q[bc] = dad.BV[bc]
        end
    end
    return nothing
end
split_sol(dad, T, q) = split_sol!(dad, T, q)

# =============================================================================
# Steady Elasticity
# =============================================================================

"""
    solve(dad::BEMdata{<:Elasticity}; frame=:global, p=nothing)

Steady elasticity solve.

- `frame = :global` (default) — BCs in (x, y) components
- `frame = :local` — BCs in nodal (n, t) components (Leonardo 2026 §4.7);
  see [`solve_local`](@ref)
"""
function solve(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity,AnisotropicElasticity3D}};
        frame::Symbol=:global, p=nothing)
    if frame === :local
        return solve_local(dad; p=p)
    elseif frame !== :global
        throw(ArgumentError("frame must be :global or :local (got $frame)"))
    end
    p === nothing || @warn "body-force vector p is only applied for frame=:local; ignored" 
    applyBC(dad)
    dim = dad.dimension
    ndof_b = dim * dad.n
    x = bem_linsolve(dad.A, dad.b)
    u = zeros(eltype(x), ndof_b)
    traction = zeros(eltype(x), ndof_b)
    split_sol!(dad, x, u, traction)
    set_cache!(dad; u=u, traction=traction, T=u)
    return u
end

function split_sol!(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity,AnisotropicElasticity3D}}, x, u, traction)
    BC = dad.BC
    BV = dad.BV
    @inbounds for dof in eachindex(BC)
        if BC[dof] == 0
            traction[dof] = x[dof]
            u[dof] = BV[dof]
        else
            traction[dof] = BV[dof]
            u[dof] = x[dof]
        end
    end
    return nothing
end

