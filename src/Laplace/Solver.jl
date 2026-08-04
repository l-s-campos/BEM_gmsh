export solve, solve_Houbolt, solve_Houbolt_heat, solve_transient, solve_transient_o2
export reduced_heat_system, heat_rhs, heat_rhs!
export reduced_wave_system, wave_rhs, wave_rhs!, wave_full_rhs!, wave_static_operators
export build_heat_ode, build_wave_ode
# MMM / MMC modal transient (thesis §4.4–4.5) — symbols exported from ModalModified.jl

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
    x = A \ b
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

function solve(dad::BEMdata{<:Elasticity})
    applyBC(dad)
    dim = dad.dimension
    ndof_b = dim * dad.n
    x = dad.A \ dad.b
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

All arguments may carry `ForwardDiff.Dual` entries; no mutation of inputs.
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
    # A1 = A10 / A00
    A1 = A10 / A00
    Mred = M11 - A1 * M01
    B = Mred \ (A11 - A1 * A01)
    f = Mred \ (b[unknown] - A1 * b[known])
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
    B = Mred \ (-(A11 - A1 * A01))
    f = Mred \ (b[unknown] - A1 * b[known])
    return (; B, f, unknown, known, BCT)
end

"""
    build_heat_ode(dad; u0=nothing) -> (ODEProblem, sys)

Construct an `ODEProblem` with **out-of-place** RHS `heat_rhs` so that
`ForwardDiff` / SciML sensitivity can differentiate through the dynamics.
"""
function build_heat_ode(dad::BEMdata{<:Laplace}; u0=nothing, tspan=(0.0, 1.0))
    has_cache(dad, :M) || error("call DIBEM(dad) first")
    applyBC(dad)
    sys = reduced_heat_system(dad.A, dad.M, dad.b, dad.BC, dad.ni)
    nu = count(sys.unknown)
    u₀ = u0 === nothing ? zeros(nu) : u0
    p = (B=sys.B, f=sys.f)
    prob = ODEProblem{false}(heat_rhs, u₀, tspan, p)
    return prob, sys
end

"""
    wave_static_operators(dad) -> (; A, b, M)

Apply BCs to the **spatial** operators `H,G` only (no mass).
Returns copies of the BC-modified `A,b` and the DIBEM mass `M`.
Does not leave `dad.H` modified.
"""
function wave_static_operators(dad::BEMdata{<:Laplace})
    has_cache(dad, :M) || error("call DIBEM(dad) first")
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    if has_cache(dad, :A)
        dad.cache.A = nothing
    end
    applyBC(dad)
    return (; A=copy(dad.A), b=copy(dad.b), M=dad.M)
end

"""
    build_wave_ode(dad; u0, du0, tspan)

Full-system second-order ODE (no unknown reduction), matching the Houbolt form

```
M ü + A u = b
```

where `A,b` come from `applyBC(H,G)` and `M` is the DIBEM mass
(same operators as [`solve_Houbolt`](@ref)).

RHS: solve M*uee = b - A*u on the full collocation vector (no reduced unknowns).
"""
function build_wave_ode(dad::BEMdata{<:Laplace}; u0=nothing, du0=nothing, tspan=(0.0, 1.0))
    ops = wave_static_operators(dad)
    n = dad.nt
    u₀ = u0 === nothing ? zeros(n) : Float64.(u0)
    du₀ = du0 === nothing ? zeros(n) : Float64.(du0)
    length(u₀) == n || throw(DimensionMismatch("u0 length $(length(u₀)) ≠ nt=$n"))
    length(du₀) == n || throw(DimensionMismatch("du0 length $(length(du₀)) ≠ nt=$n"))
    # pin Dirichlet DOFs in IC
    @inbounds for i in 1:dad.n
        if dad.BC[i] == 0
            u₀[i] = dad.BV[i]
            du₀[i] = 0.0
        end
    end
    p = (A=ops.A, b=ops.b, M=ops.M, BC=dad.BC, BV=dad.BV, n_b=dad.n)
    prob = SecondOrderODEProblem(wave_full_rhs!, du₀, u₀, tspan, p)
    return prob, p
end

"""Full-system wave residual: M ü = b - A u  (Dirichlet rows forced to ü=0)."""
function wave_full_rhs!(ddu, du, u, p, t)
    # r = b - A*u
    mul!(ddu, p.A, u)
    @. ddu = p.b - ddu
    # ü = M \ r
    ddu .= p.M \ ddu
    # Dirichlet: keep u fixed ⇒ ü = 0
    @inbounds for i in 1:p.n_b
        if p.BC[i] == 0
            ddu[i] = 0.0
        end
    end
    return nothing
end

# keep reduced helpers available for experiments / heat
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

# =============================================================================
# Transient drivers
# =============================================================================

"""
    solve_Houbolt(dad, Δt, tf; u0=nothing)

Second-order **Houbolt** on the **full** BEM system (Loeffler / elastico style):

```
A = applyBC(H + 2 M/Δt², G)
xⁿ⁺¹ = A \\ (b + (M/Δt²) (5 uⁿ - 4 uⁿ⁻¹ + uⁿ⁻²))
```

with physical field `u` recovered by [`split_sol!`](@ref). Does not permanently
modify `dad.H`.

> Heat (first-order): [`solve_Houbolt_heat`](@ref) / [`solve_transient`](@ref).
"""
function solve_Houbolt(dad::BEMdata{<:Laplace}, Δt, tf; u0=nothing)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    Δt > 0 || throw(ArgumentError("Δt must be > 0"))

    t = collect(0:Δt:tf)
    nT = length(t)
    T = zeros(dad.nt, nT)
    q = zeros(dad.n, nT)
    if u0 !== nothing
        length(u0) == dad.nt || throw(DimensionMismatch("u0"))
        T[:, 1] .= u0
    end
    # enforce Dirichlet in IC
    @inbounds for i in 1:dad.n
        dad.BC[i] == 0 && (T[i, 1] = dad.BV[i])
    end
    set_cache!(dad; T, q, time=t)

    H0 = dad.H
    Mt = dad.M ./ (Δt^2)
    try
        set_cache!(dad; H = H0 + 2 .* Mt)
        if has_cache(dad, :A)
            dad.cache.A = nothing
        end
        applyBC(dad)
        FA = lu(dad.A)
        b = copy(dad.b)

        @showprogress "Time stepping (Houbolt wave)" for i in 4:nT
            # rhs = b + Mt * (5 u^{i-1} - 4 u^{i-2} + u^{i-3})
            @views rhs = b .+ Mt * (5 .* T[:, i-1] .- 4 .* T[:, i-2] .+ T[:, i-3])
            x = FA \ rhs
            T[:, i] .= x
            split_sol!(dad, view(T, :, i), view(q, :, i))
        end
    finally
        set_cache!(dad; H=H0)
        if has_cache(dad, :A)
            dad.cache.A = nothing
        end
    end
    return dad.T
end

"""
    solve_Houbolt_heat(dad, Δt, tf; u0=nothing)

First-order Houbolt for diffusion (legacy `potencial_transiente` scheme).
"""
function solve_Houbolt_heat(dad::BEMdata{<:Laplace}, Δt, tf; u0=nothing)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    t = collect(0:Δt:tf)
    nT = length(t)
    T = zeros(dad.nt, nT)
    q = zeros(dad.n, nT)
    if u0 !== nothing
        T[:, 1] .= u0
    end
    set_cache!(dad; T, q, time=t)

    H0 = dad.H
    try
        set_cache!(dad; H = H0 - 11 * dad.M / (6 * Δt))
        if has_cache(dad, :A)
            dad.cache.A = nothing
        end
        applyBC(dad)
        FA = lu(dad.A)
        @showprogress "Time stepping (Houbolt heat)" for i in 4:nT
            rhs = dad.b .+ dad.M * (-18 .* T[:, i-1] .+ 9 .* T[:, i-2] .- 2 .* T[:, i-3]) / (6 * Δt)
            x = FA \ rhs
            T[:, i] .= x
            split_sol!(dad, view(T, :, i), view(q, :, i))
        end
    finally
        set_cache!(dad; H=H0)
        if has_cache(dad, :A)
            dad.cache.A = nothing
        end
    end
    return dad.T
end

function solve_transient(dad::BEMdata{<:Laplace}, Δt, tf;
    abstol=1e-6, reltol=1e-6, alg=Tsit5())
    prob, sys = build_heat_ode(dad; tspan=(0.0, tf))
    # prefer in-place for speed when not differentiating
    p = prob.p
    ff = ODEFunction(heat_rhs!; jac=(J, u, pp, t) -> (J .= pp.B))
    prob_ip = ODEProblem(ff, prob.u0, prob.tspan, p)
    sol = DifferentialEquations.solve(prob_ip, alg; abstol=abstol, reltol=reltol, progress=true)

    tgrid = 0:Δt:tf
    nsteps = length(tgrid)
    T = zeros(dad.nt, nsteps)
    unknown, known = sys.unknown, sys.known
    for (i, ti) in enumerate(tgrid)
        T[unknown, i] .= sol(ti)
        T[known, i] .= dad.BV[dad.BC .== 0]
    end
    set_cache!(dad; T, time=tgrid, ode_sol=sol)
    return sol
end

"""
    solve_transient_o2(dad, Δt, tf; u0, du0, abstol, reltol, alg)

Wave integration with **DifferentialEquations.jl** on the **full** system
``M ü + A u = b`` (same `A,b,M` as Houbolt — no reduced unknowns).

Samples the solution on `0:Δt:tf` into `dad.T`.
"""
function solve_transient_o2(dad::BEMdata{<:Laplace}, Δt, tf;
    u0=nothing, du0=nothing,
    abstol=1e-6, reltol=1e-6, alg=nothing, progress=false)
    prob, p = build_wave_ode(dad; u0=u0, du0=du0, tspan=(0.0, tf))
    sol = if alg === nothing
        DifferentialEquations.solve(prob; abstol=abstol, reltol=reltol, progress=progress)
    else
        DifferentialEquations.solve(prob, alg; abstol=abstol, reltol=reltol, progress=progress)
    end
    tgrid = collect(0:Δt:tf)
    nsteps = length(tgrid)
    T = zeros(dad.nt, nsteps)
    q = zeros(dad.n, nsteps)
    @inbounds for (i, ti) in enumerate(tgrid)
        ui = _position_at(sol, ti)
        T[:, i] .= ui
        # pin Dirichlet
        for j in 1:dad.n
            if dad.BC[j] == 0
                T[j, i] = dad.BV[j]
            else
                q[j, i] = dad.BV[j]
            end
        end
    end
    set_cache!(dad; T, q, time=tgrid, ode_sol=sol)
    return sol
end

function _position_at(sol, t)
    val = sol(t)
    if hasproperty(val, :x)
        return val.x[2]          # ArrayPartition (du, u) → u
    elseif val isa Tuple
        return val[2]
    else
        n = length(val) ÷ 2
        return val[n+1:end]
    end
end
