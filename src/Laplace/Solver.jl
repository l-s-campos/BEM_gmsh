# Transient Laplace drivers (steady solve lives in Core/Solver.jl)

export solve_Houbolt, solve_Houbolt_heat, solve_dq_heat, solve_transient, solve_transient_o2
export solve_Newmark
export build_heat_ode, build_wave_ode, wave_static_operators, wave_full_rhs!
export stable_wave_subspace, WaveParam
export reduced_heat_system, heat_rhs, heat_rhs!
export reduced_wave_system, wave_rhs, wave_rhs!

"""
    reduced_heat_system(A, M, b, BC, ni) -> (; B, f, unknown, known)

First-order ODE ``\\dot u_F = B u_F + f`` on free temperatures
(Neumann + interiors). Dirichlet slots of `A` are the flux unknowns
after [`applyBC`](@ref); ``\\dot u_D = 0``.
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
    Ared = A11 - A1 * A01
    bred = b[unknown] - A1 * b[known]
    Mred = M11 - A1 * M01
    B = bem_linsolve(Mred, Ared)
    f = bem_linsolve(Mred, -bred)
    return (; B, f, unknown, known, BCT)
end

"""Same condensation as [`reduced_heat_system`](@ref) for ``A x - b = M \\ddot u``."""
const reduced_wave_system = reduced_heat_system

heat_rhs(u, p, t) = p.B * u .+ p.f
function heat_rhs!(du, u, p, t)
    mul!(du, p.B, u)
    du .+= p.f
    return nothing
end
wave_rhs(u, p, t) = heat_rhs(u, p, t)
function wave_rhs!(ddu, du, u, p, t)
    return heat_rhs!(ddu, u, p, t)
end


"""
    build_heat_ode(dad; u0=nothing) -> (ODEProblem, sys)

Out-of-place `heat_rhs` so ForwardDiff can differentiate the reduced heat ODE
``\\dot u_F = B u_F + f``.
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

Apply BCs to spatial `H,G` only. Does not modify `H`.
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
    stable_wave_subspace(B; atol=0, ωmax=Inf)

Real invariant subspace with ``Re λ ≤ atol`` and ``√|λ| ≤ ωmax``.
Prefer [`solve_mmm!`](@ref) for production modal filtering.
"""
function stable_wave_subspace(B::AbstractMatrix; atol::Real=0.0, ωmax::Real=Inf)
    n = size(B, 1)
    size(B, 2) == n || throw(DimensionMismatch("B must be square"))
    F = schur(Matrix{Float64}(B))
    ω = sqrt.(abs.(F.values))
    sel = (real.(F.values) .<= atol) .& (ω .<= ωmax)
    k = count(sel)
    k == 0 && error("stable_wave_subspace: no eigenvalue with Re λ ≤ $atol")
    k < n && (F = ordschur(F, sel))
    return (; V=F.Z[:, 1:k], Bs=F.T[1:k, 1:k], n_keep=k, n_drop=n - k, values=F.values)
end

"""Parameters for the first-order wave ODE ``y=(v,u)``."""
struct WaveParam{TB,Tf,TV,TL}
    B::TB
    f::Tf
    load::TL
    n::Int
    unknown::BitVector
    known::BitVector
    BV::Vector{Float64}
    BC::Vector{Int}
    n_b::Int
    nt::Int
    V::TV
    n_full::Int
    n_drop::Int
    ν::Float64
end

"""
    build_wave_ode(dad; u0, du0, tspan, load, filter_unstable, ωmax, ν)

Reduced wave ``\\ddot u_F + ν \\dot u_F = B u_F + load(t)\\, f``
from ``H u - G q = M \\ddot u`` after condensing Dirichlet fluxes.
"""
function build_wave_ode(dad::BEMdata{<:Laplace}; u0=nothing, du0=nothing,
                       tspan=(0.0, 1.0), load=nothing, filter_unstable::Bool=false,
                       ωmax::Union{Real,Nothing}=nothing, ν::Real=0.0)
    ops = wave_static_operators(dad)
    sys = reduced_wave_system(ops.A, ops.M, ops.b, dad.BC, dad.ni)
    unknown = sys.unknown
    nfull = count(unknown)
    ufull = zeros(dad.nt)
    dufull = zeros(dad.nt)
    if u0 !== nothing
        length(u0) == dad.nt || length(u0) == nfull || throw(DimensionMismatch("u0"))
        length(u0) == dad.nt ? (ufull .= u0) : (ufull[unknown] .= u0)
    end
    if du0 !== nothing
        length(du0) == dad.nt || length(du0) == nfull || throw(DimensionMismatch("du0"))
        length(du0) == dad.nt ? (dufull .= du0) : (dufull[unknown] .= du0)
    end
    @inbounds for i in 1:dad.n
        if dad.BC[i] == 0
            ufull[i] = dad.BV[i]
            dufull[i] = 0.0
        end
    end
    u₀ = ufull[unknown]
    du₀ = dufull[unknown]
    B = sys.B
    f = sys.f
    V = Matrix{Float64}(I, nfull, nfull)
    n_drop = 0
    if filter_unstable || ωmax !== nothing
        ωm = ωmax === nothing ? Inf : float(ωmax)
        sub = stable_wave_subspace(B; ωmax=ωm)
        V = sub.V
        B = sub.Bs
        f = V' * f
        u₀ = V' * u₀
        du₀ = V' * du₀
        n_drop = sub.n_drop
    end
    nu = size(B, 1)
    p = WaveParam(B, f, load, nu, unknown, sys.known, copy(dad.BV), copy(dad.BC),
        dad.n, dad.nt, V, nfull, n_drop, float(ν))
    ff = ODEFunction{true}(wave_fo_rhs!; jac=wave_fo_jac!)
    return ODEProblem{true}(ff, vcat(du₀, u₀), tspan, p), p
end

"""First-order residual: ẏ = (B u − ν v + load(t) f, v)."""
function wave_fo_rhs!(dy, y, p, t)
    n = p.n
    v = view(y, 1:n)
    u = view(y, (n + 1):(2n))
    s = p.load === nothing ? 1.0 : float(p.load(t))
    mul!(view(dy, 1:n), p.B, u)
    @inbounds for i in 1:n
        dy[i] += s * p.f[i] - p.ν * v[i]
        dy[n + i] = v[i]
    end
    return nothing
end

function wave_fo_jac!(J, y, p, t)
    n = p.n
    fill!(J, 0)
    @inbounds J[1:n, (n + 1):(2n)] .= p.B
    @inbounds for i in 1:n
        J[i, i] = -p.ν
        J[n + i, i] = 1
    end
    return J
end
wave_fo_jac!(J::AbstractMatrix{<:Number}, p, t) = wave_fo_jac!(J, nothing, p, t)

"""Reduced residual ``\\ddot u = B u + load(t) f - ν \\dot u`` (tests)."""
function wave_full_rhs!(ddu, du, u, p, t)
    s = p.load === nothing ? 1.0 : float(p.load(t))
    mul!(ddu, p.B, u)
    @. ddu = ddu + s * p.f - p.ν * du
    return nothing
end

# ---------------------------------------------------------------------------
# Shared history / Houbolt factor (H restored before the loop)
# ---------------------------------------------------------------------------

function _pin_dirichlet!(Tcol, dad)
    @inbounds for i in 1:dad.n
        dad.BC[i] == 0 && (Tcol[i] = dad.BV[i])
    end
    return Tcol
end

"""
Nodal wave source ``f`` in ``H u - G q = M(\\ddot u - f)``.

`force` is `nothing`, a length-`nt` vector, a scalar, `force(t)` returning
either of those, or `force(x, t)` / `force(x)` on collocation points.
"""
function _wave_bodyforce(dad, force, t)
    force === nothing && return nothing
    if force isa AbstractVector
        length(force) == dad.nt || throw(DimensionMismatch("force"))
        return force
    elseif force isa Number
        return fill(float(force), dad.nt)
    elseif force isa Function
        if hasmethod(force, Tuple{typeof(t)}) || hasmethod(force, Tuple{Float64}) ||
                hasmethod(force, Tuple{Real})
            v = force(t)
            if v isa AbstractVector
                length(v) == dad.nt || throw(DimensionMismatch("force(t)"))
                return v
            elseif v isa Number
                return fill(float(v), dad.nt)
            end
        end
        pts = all_points(dad)
        try
            return Float64[float(force(p, t)) for p in pts]
        catch
            return Float64[float(force(p)) for p in pts]
        end
    end
    throw(ArgumentError("force must be nothing, a vector, a number, or a function"))
end

function _apply_wave_force!(rhs, dad, force, t, Mf)
    fv = _wave_bodyforce(dad, force, t)
    fv === nothing && return rhs
    mul!(Mf, dad.M, fv)
    rhs .-= Mf
    return rhs
end

function _init_history(dad, Δt, tf; u0=nothing)
    Δt > 0 || throw(ArgumentError("Δt must be > 0"))
    t = collect(0:Δt:tf)
    T = zeros(dad.nt, length(t))
    q = zeros(dad.n, length(t))
    if u0 !== nothing
        length(u0) == dad.nt || throw(DimensionMismatch("u0"))
        T[:, 1] .= u0
    end
    _pin_dirichlet!(view(T, :, 1), dad)
    set_cache!(dad; T, q, time=t)
    return t, T, q
end

"""Form `A,b` from a temporary `H + α M` without leaving `dad.H` modified."""
function _factor_shifted_H(dad, Hshift)
    H0 = dad.H
    try
        set_cache!(dad; H=Hshift)
        has_cache(dad, :A) && (dad.cache.A = nothing)
        applyBC(dad)
        A = dad.A
        b = copy(dad.b)
        return bem_linfactor(A, b), b
    finally
        set_cache!(dad; H=H0)
        has_cache(dad, :A) && (dad.cache.A = nothing)
    end
end

# ---------------------------------------------------------------------------
# Houbolt
# ---------------------------------------------------------------------------

"""
    solve_Houbolt(dad, Δt, tf; u0=nothing, du0=nothing, force=nothing)

Second-order Houbolt on ``H u - G q = M(\\ddot u - f)``.

`force` is the wave-equation source ``f`` in ``\\ddot u = Δu + f``
(after any scaling of `M` by ``1/c²``): a nodal vector, a scalar,
`force(t)`, or `force(x, t)`.

Steps 2–3: implicit Euler
``(H - M/Δt²) u^n = G q + (M/Δt²)(u^{n-1} + Δt v^{n-1}) - M f^n``,
``a^n = (u^n - u^{n-1} - Δt v^{n-1})/Δt²``, ``v^n = v^{n-1} + Δt a^n``.
From step 4: classical Houbolt. `dad.H` is not modified after return.
"""
function solve_Houbolt(dad::BEMdata{<:Laplace}, Δt, tf; u0=nothing, du0=nothing,
        force=nothing)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    t, T, q = _init_history(dad, Δt, tf; u0=u0)
    nT = length(t)
    v = zeros(dad.nt)
    du0 !== nothing && (v .= du0)
    Mt = dad.M ./ (Δt^2)
    Δt2 = Δt^2
    Mf = zeros(dad.nt)
    if nT >= 2
        lse, be = _factor_shifted_H(dad, dad.H .- Mt)
        for i in 2:min(3, nT)
            Tprev = T[:, i - 1]
            rhs = be .- Mt * (Tprev .+ Δt .* v)
            _apply_wave_force!(rhs, dad, force, t[i], Mf)
            T[:, i] .= bem_linsolve!(lse, rhs)
            split_sol!(dad, view(T, :, i), view(q, :, i))
            a = (T[:, i] .- Tprev .- Δt .* v) ./ Δt2
            v .+= Δt .* a
        end
    end
    nT >= 4 || return dad.T
    lscache, b = _factor_shifted_H(dad, dad.H .- 2 .* Mt)
    @showprogress "Time stepping (Houbolt wave)" for i in 4:nT
        @views rhs = b .- Mt * (5 .* T[:, i - 1] .- 4 .* T[:, i - 2] .+ T[:, i - 3])
        _apply_wave_force!(rhs, dad, force, t[i], Mf)
        T[:, i] .= bem_linsolve!(lscache, rhs)
        split_sol!(dad, view(T, :, i), view(q, :, i))
    end
    return dad.T
end

"""
    solve_Newmark(dad::BEMdata{<:Laplace}, Δt, tf; u0=nothing, β=1/4, γ=1/2, force=nothing)

Newmark on ``H u - G q = M(\\ddot u - f)``. `force` is the wave source ``f``
in ``\\ddot u = Δu + f`` (see [`solve_Houbolt`](@ref)).
"""
function solve_Newmark(dad::BEMdata{<:Laplace}, Δt, tf;
        u0=nothing, du0=nothing, β::Real=1 / 4, γ::Real=1 / 2, force=nothing)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    t, T, q = _init_history(dad, Δt, tf; u0=u0)
    nT = length(t)
    nT >= 2 || return dad.T
    v = du0 === nothing ? zeros(dad.nt) : collect(Float64, du0)
    length(v) == dad.nt || throw(DimensionMismatch("du0"))
    a = zeros(dad.nt)
    a0 = 1 / (β * Δt^2)
    a2 = 1 / (β * Δt)
    a3 = 1 / (2β) - 1
    a6 = Δt * (1 - γ)
    a7 = γ * Δt
    Mt = dad.M
    Mf = zeros(dad.nt)
    lscache, b = _factor_shifted_H(dad, dad.H .- a0 .* Mt)
    @showprogress "Time stepping (Newmark wave)" for i in 2:nT
        pred = a0 .* T[:, i - 1] .+ a2 .* v .+ a3 .* a
        rhs = b .- Mt * pred
        _apply_wave_force!(rhs, dad, force, t[i], Mf)
        T[:, i] .= bem_linsolve!(lscache, rhs)
        _pin_dirichlet!(view(T, :, i), dad)
        an = a0 .* (T[:, i] .- T[:, i - 1]) .- a2 .* v .- a3 .* a
        v = v .+ a6 .* a .+ a7 .* an
        a = an
        split_sol!(dad, view(T, :, i), view(q, :, i))
    end
    return dad.T
end

"""
    solve_Houbolt_heat(dad, Δt, tf; u0=nothing)

First-order Houbolt for ``H u - G q = M ú``.
Steps 2–3 are backward Euler so the 3-level stencil is not started from zeros.
`dad.H` is not modified after return.
"""
function solve_Houbolt_heat(dad::BEMdata{<:Laplace}, Δt, tf; u0=nothing)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    t, T, q = _init_history(dad, Δt, tf; u0=u0)
    nT = length(t)
    nT >= 2 || return dad.T
    _euler_heat_steps!(dad, T, q, Δt, min(3, nT))
    nT >= 4 || return dad.T
    lscache, b = _factor_shifted_H(dad, dad.H .- 11 .* dad.M ./ (6 * Δt))
    c = 1 / (6 * Δt)
    @showprogress "Time stepping (Houbolt heat)" for i in 4:nT
        rhs = b .+ dad.M * ((-18) .* T[:, i - 1] .+ 9 .* T[:, i - 2] .- 2 .* T[:, i - 3]) .* c
        T[:, i] .= bem_linsolve!(lscache, rhs)
        split_sol!(dad, view(T, :, i), view(q, :, i))
    end
    return dad.T
end

"""Backward Euler on ``(H - M/Δt) u^{n+1} - G q = -(M/Δt) u^n`` for columns `2:ilast`."""
function _euler_heat_steps!(dad, T, q, Δt, ilast::Integer)
    lscache, b = _factor_shifted_H(dad, dad.H .- dad.M ./ Δt)
    Minvdt = dad.M ./ Δt
    @inbounds for i in 2:ilast
        rhs = b .- Minvdt * view(T, :, i - 1)
        T[:, i] .= bem_linsolve!(lscache, rhs)
        split_sol!(dad, view(T, :, i), view(q, :, i))
    end
    return nothing
end


# ---------------------------------------------------------------------------
# Differential quadrature time (Wang §1.9, Grid V: ends + Legendre roots)
# ---------------------------------------------------------------------------

"""Wang Grid V on [-1,1]: endpoints plus roots of P_{N-2}."""
function _gridV_nodes(N::Integer)
    N >= 2 || throw(ArgumentError("DQ time needs N ≥ 2 nodes"))
    if N == 2
        return [-1.0, 1.0]
    end
    ξ, _ = gausslegendre(N - 2)
    return vcat(-1.0, ξ, 1.0)
end

function _dq_At(N::Integer, ΔT::Real)
    ξ = _gridV_nodes(N)
    poly = ArbitraryPolynomial(ξ)
    Aξ = diff_matrix(poly.weights, poly.nodes, degree(poly))
    return (2 / ΔT) .* Aξ
end

"""
    solve_dq_heat(dad, tf; u0, nτ=8, nblocks=1)

Wang §1.9 DQ time integration of ``H u - G q = M ú`` after BC reduction
to ``ú_F = B u_F + f``. Each block uses Grid V (Legendre roots + ends).
`dad.H` is not modified.
"""
function solve_dq_heat(dad::BEMdata{<:Laplace}, tf::Real; u0=nothing,
        nτ::Integer=8, nblocks::Integer=1)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    nτ >= 2 || throw(ArgumentError("nτ ≥ 2"))
    nblocks >= 1 || throw(ArgumentError("nblocks ≥ 1"))
    applyBC(dad)
    sys = reduced_heat_system(dad.A, dad.M, dad.b, dad.BC, dad.ni)
    B, f = sys.B, sys.f
    unknown, known = sys.unknown, sys.known
    nf = count(unknown)
    nf >= 1 || error("solve_dq_heat: no free temperatures")

    u0_full = u0 === nothing ? zeros(dad.nt) : collect(Float64, u0)
    length(u0_full) == dad.nt || throw(DimensionMismatch("u0"))
    uF = u0_full[unknown]
    uD = dad.BV[dad.BC .== 0]

    N = Int(nτ)
    nT = nblocks * (N - 1) + 1
    T = zeros(dad.nt, nT)
    q = zeros(dad.n, nT)
    t = zeros(nT)
    T[unknown, 1] .= uF
    T[known, 1] .= uD
    t[1] = 0.0
    col = 1
    ΔT = float(tf) / nblocks
    At = _dq_At(N, ΔT)
    Att = At[2:N, 2:N]
    a1 = At[2:N, 1]
    Iτ = Matrix{Float64}(I, N - 1, N - 1)
    Inf_ = Matrix{Float64}(I, nf, nf)
    K = kron(Att, Inf_) .- kron(Iτ, B)
    F = lu(K)
    rhs = zeros(nf * (N - 1))
    ones_τ = ones(N - 1)

    for blk in 1:nblocks
        rhs .= kron(ones_τ, f) .- kron(a1, uF)
        x = F \ rhs
        Uunk = reshape(x, nf, N - 1)
        t0 = (blk - 1) * ΔT
        ξ = _gridV_nodes(N)
        @inbounds for k in 1:(N - 1)
            col += 1
            T[unknown, col] .= view(Uunk, :, k)
            T[known, col] .= uD
            t[col] = t0 + (ξ[k + 1] + 1) * ΔT / 2
        end
        uF = Uunk[:, end]
    end
    set_cache!(dad; T, q, time=t)
    return dad.T
end


# ---------------------------------------------------------------------------
# DiffEq
# ---------------------------------------------------------------------------

"""
    solve_transient(dad, Δt, tf; abstol=1e-6, reltol=1e-6, alg=Rodas5P()) -> sol

First-order heat ODE ``ú_F = B u_F + f`` via OrdinaryDiffEq (default
Rodas5P). Samples the solution on `0:Δt:tf` into `dad.T` / `dad.q`.
Call [`dibem!`](@ref) first so `M` exists.
"""
function solve_transient(dad::BEMdata{<:Laplace}, Δt, tf;
    abstol=1e-6, reltol=1e-6, alg=Rodas5P(), progress=false)
    prob, sys = build_heat_ode(dad; tspan=(0.0, tf))
    ff = ODEFunction(heat_rhs!; jac=(J, u, pp, t) -> (J .= pp.B))
    sol = OrdinaryDiffEq.solve(ODEProblem(ff, prob.u0, prob.tspan, prob.p),
        alg; abstol=abstol, reltol=reltol, adaptive=true, progress=progress)
    tgrid = collect(0:Δt:tf)
    T = zeros(dad.nt, length(tgrid))
    q = zeros(dad.n, length(tgrid))
    unknown, known = sys.unknown, sys.known
    for (i, ti) in enumerate(tgrid)
        T[unknown, i] .= sol(ti)
        T[known, i] .= dad.BV[dad.BC .== 0]
    end
    set_cache!(dad; T, q, time=tgrid, ode_sol=sol)
    return sol
end

"""
    solve_transient_o2(dad, Δt, tf; ν=0, filter_unstable=false, ...)

Wave ``\\ddot u + ν \\dot u = B u + load(t)\\, f`` as first-order ``y=(v,u)``.
Default integrator: Rodas5P.
"""
function solve_transient_o2(dad::BEMdata{<:Laplace}, Δt, tf;
    u0=nothing, du0=nothing,
    abstol=1e-6, reltol=1e-6, alg=Rodas5P(), progress=false, load=nothing,
    filter_unstable::Bool=false, ωmax::Union{Real,Nothing}=nothing, ν::Real=0.0)
    prob, p = build_wave_ode(dad; u0=u0, du0=du0, tspan=(0.0, tf), load=load,
                            filter_unstable=filter_unstable, ωmax=ωmax, ν=ν)
    sol = OrdinaryDiffEq.solve(prob, alg; abstol=abstol, reltol=reltol,
                                      adaptive=true, progress=progress)
    tgrid = collect(0:Δt:tf)
    T = zeros(dad.nt, length(tgrid))
    q = zeros(dad.n, length(tgrid))
    n = p.n
    uF = Vector{Float64}(undef, p.n_full)
    @inbounds for (i, ti) in enumerate(tgrid)
        y = sol(ti)
        mul!(uF, p.V, view(y, (n + 1):(2n)))
        T[p.unknown, i] .= uF
        for j in 1:dad.n
            if dad.BC[j] == 0
                T[j, i] = dad.BV[j]
            else
                q[j, i] = dad.BV[j]
            end
        end
    end
    set_cache!(dad; T, q, time=tgrid, ode_sol=sol, wave_n_drop=p.n_drop)
    return sol
end
