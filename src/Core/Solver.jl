# Steady BEM solvers (Laplace + Elasticity). Transient drivers: Laplace/Solver.jl

export solve, solve_Hmat, split_sol!, split_sol
export reduced_heat_system, heat_rhs, heat_rhs!
export reduced_wave_system, wave_rhs, wave_rhs!
# solve_local exported from Elasticity/LocalFrame.jl

# =============================================================================
# Steady Laplace
# =============================================================================

function solve(dad::BEMdata{<:Union{Laplace,OrthotropicLaplace}})
    applyBC(dad)
    A = dad.A
    b = dad.b
    if A isa MixedBCOperator || A isa HMatrices.HMatrix
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

function solve_Hmat(dad::BEMdata{<:Laplace})
    A = dad.A
    b = dad.b
    x, stats = Krylov.gmres(A, b; atol=1e-10, rtol=1e-8, itmax=max(4 * size(A, 1), 200))
    Tfull = zeros(dad.nt)
    qfull = zeros(dad.n)
    Tfull[1:length(x)] .= x
    split_sol!(dad, Tfull, qfull)
    set_cache!(dad; T=Tfull, q=qfull, gmres_stats=stats)
    return dad.T
end

function split_sol!(dad::BEMdata{<:Union{Laplace,OrthotropicLaplace}}, T, q)
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
function solve(dad::BEMdata{<:Elasticity}; frame::Symbol=:global, p=nothing)
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

function split_sol!(dad::BEMdata{<:Elasticity}, x, u, traction)
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

# =============================================================================
# AD-compatible reduced systems (pure linear algebra, Dual-friendly)
# =============================================================================

"""
    reduced_heat_system(A, M, b, BC, ni) -> (; B, f, unknown, known)

Build the first-order reduced ODE
``\\dot u = B u + f`` after static condensation of Dirichlet dofs.
"""
function reduced_heat_system(A, M, b, BC::AbstractVector{<:Integer}, ni::Integer)
    BCT = vcat(BC, ones(eltype(BC), ni))
    unknown = BCT .== 1
    known = BCT .== 0
    A00 = A[known, known]
    A01 = A[known, unknown]
    A10 = A[unknown, known]
    A11 = A[unknown, unknown]
    M01 = M[known, unknown]
    M11 = M[unknown, unknown]
    A1 = A10 / A00
    Mred = M11 - A1 * M01
    B = bem_linsolve(Mred, A11 - A1 * A01)
    f = bem_linsolve(Mred, b[unknown] - A1 * b[known])
    return (; B, f, unknown, known, BCT)
end

"""Out-of-place RHS for AD / DiffEq (first-order heat)."""
heat_rhs(u, p, t) = p.B * u .+ p.f

"""In-place RHS (non-AD solvers)."""
function heat_rhs!(du, u, p, t)
    mul!(du, p.B, u)
    du .+= p.f
    return nothing
end

"""
    reduced_wave_system(A, M, b, BC, ni) -> (; B, f, unknown, known)

Second-order reduced form ``\\ddot u = B u + f``.
"""
function reduced_wave_system(A, M, b, BC::AbstractVector{<:Integer}, ni::Integer)
    BCT = vcat(BC, ones(eltype(BC), ni))
    unknown = BCT .== 1
    known = BCT .== 0
    A00 = A[known, known]
    A01 = A[known, unknown]
    A10 = A[unknown, known]
    A11 = A[unknown, unknown]
    M01 = M[known, unknown]
    M11 = M[unknown, unknown]
    A1 = A10 / A00
    Mred = M11 - A1 * M01
    B = bem_linsolve(Mred, -(A11 - A1 * A01))
    f = bem_linsolve(Mred, b[unknown] - A1 * b[known])
    return (; B, f, unknown, known, BCT)
end

wave_rhs(u, p, t) = p.B * u .+ p.f
function wave_rhs!(ddu, du, u, p, t)
    if hasproperty(p, :B)
        mul!(ddu, p.B, u)
        ddu .+= p.f
    else
        wave_full_rhs!(ddu, du, u, p, t)
    end
    return nothing
end
