# Elastodynamic Houbolt with DIBEM M from Monta_M_RIMd:
#   H u + M ü = G t
# matching elastico_transiente.jl (aplicaCDC(H + 2 M/Δt²)).

"""
    solve_Houbolt(dad::BEMdata{<:Elasticity}, Δt, tf; u0=nothing)

Second-order Houbolt on ``H u + M ü = G t`` (`M` from [`DIBEM`](@ref),
same sign as `Monta_M_RIMd`). Steps 2–3 are BDF1 Euler so the 4-level
stencil is not started from zeros. `dad.H` is restored after return.
Stores `u` (`2 n_t × nT`) and `traction` (`2 n × nT`).
"""
function solve_Houbolt(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity3D}}, Δt, tf;
        u0=nothing)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    ndof, nb = _neq(dad)
    Δt > 0 || throw(ArgumentError("Δt must be > 0"))
    t = collect(0:Δt:tf)
    U = zeros(ndof, length(t))
    q = zeros(nb, length(t))
    if u0 !== nothing
        length(u0) == ndof || throw(DimensionMismatch("u0"))
        U[:, 1] .= u0
    end
    _pin_known!(view(U, :, 1), dad)
    nT = length(t)
    if nT >= 2
        # BDF1 Euler: (H + M/Δt²) u^n = G t + (M/Δt²) u^{n−1}
        Me = dad.M ./ (Δt^2)
        lse, be = _factor_shifted_H(dad, dad.H .+ Me)
        @inbounds for i in 2:min(3, nT)
            @views rhs = be .+ Me * U[:, i - 1]
            U[:, i] .= bem_linsolve!(lse, rhs)
            _scatter_step!(dad, view(U, :, i), view(q, :, i))
        end
    end
    if nT >= 4
        Mt = dad.M ./ (Δt^2)
        lscache, b = _factor_shifted_H(dad, dad.H .+ 2 .* Mt)
        @showprogress "Time stepping (Houbolt elasticity)" for i in 4:nT
            @views rhs = b .+ Mt * (5 .* U[:, i - 1] .- 4 .* U[:, i - 2] .+ U[:, i - 3])
            U[:, i] .= bem_linsolve!(lscache, rhs)
            _scatter_step!(dad, view(U, :, i), view(q, :, i))
        end
    end
    set_cache!(dad; u=U, traction=q, T=U, q=q, time=t)
    return dad.u
end

"""
    solve_Newmark(dad::BEMdata{<:Elasticity}, Δt, tf; u0=nothing, β=1/4, γ=1/2)

Average-acceleration Newmark on ``H u + M ü = G t``.
``β=1/4, γ=1/2`` is unconditionally stable (no numerical damping);
``β=1/6`` is linear acceleration.
"""
function solve_Newmark(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity3D}}, Δt, tf;
        u0=nothing, du0=nothing, β::Real=1 / 4, γ::Real=1 / 2)
    has_cache(dad, :M) || error("Mass matrix M missing — call DIBEM(dad) first.")
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    ndof, nb = _neq(dad)
    Δt > 0 || throw(ArgumentError("Δt must be > 0"))
    t = collect(0:Δt:tf)
    nT = length(t)
    U = zeros(ndof, nT)
    q = zeros(nb, nT)
    if u0 !== nothing
        length(u0) == ndof || throw(DimensionMismatch("u0"))
        U[:, 1] .= u0
    end
    _pin_known!(view(U, :, 1), dad)
    v = du0 === nothing ? zeros(ndof) : collect(Float64, du0)
    length(v) == ndof || throw(DimensionMismatch("du0"))
    a = zeros(ndof)
    nT >= 2 || (set_cache!(dad; u=U, traction=q, T=U, q=q, time=t); return dad.u)

    a0 = 1 / (β * Δt^2)
    a2 = 1 / (β * Δt)
    a3 = 1 / (2β) - 1
    a6 = Δt * (1 - γ)
    a7 = γ * Δt
    Mt = dad.M
    lscache, b = _factor_shifted_H(dad, dad.H .+ a0 .* Mt)
    @showprogress "Time stepping (Newmark elasticity)" for i in 2:nT
        pred = a0 .* U[:, i - 1] .+ a2 .* v .+ a3 .* a
        rhs = b .+ Mt * pred
        U[:, i] .= bem_linsolve!(lscache, rhs)
        _pin_known!(view(U, :, i), dad)
        an = a0 .* (U[:, i] .- U[:, i - 1]) .- a2 .* v .- a3 .* a
        v = v .+ a6 .* a .+ a7 .* an
        a = an
        _scatter_step!(dad, view(U, :, i), view(q, :, i))
    end
    set_cache!(dad; u=U, traction=q, T=U, q=q, time=t)
    return dad.u
end
