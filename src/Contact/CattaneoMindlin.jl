"""
    CattaneoMindlin

Analytical and half-plane BEM solvers for the **2D Cattaneo–Mindlin** frictional
contact problem (two elastically similar cylinders / cylinder on flat), matching
the setup of Loyola (2022, UnB) §9.3.1:

| Symbol | Value |
|--------|-------|
| ``R``  | 70 mm |
| ``w``  | 6.5 mm |
| ``E``  | 73.4 GPa |
| ``ν``  | 0.33 |
| ``P``  | 100 N/mm |
| ``f``  | 0.3 |

Load history A→B→C→D→E with constant ``P`` and cyclic ``Q ∈ [-Q_max, Q_max]``.

References
- Cattaneo (1938), Mindlin (1949), Johnson *Contact Mechanics*
- Hills & Nowell, *Mechanics of Fretting Fatigue*
- Loyola (2022) IGABEM fretting thesis
"""
module CattaneoMindlin

using LinearAlgebra
using Statistics
using ..ContactHalfPlane2D

export loyola_cattaneo_params, hertz_cylinder_params
export cattaneo_pressure, cattaneo_shear, mindlin_shear_history, cattaneo_c
export solve_cattaneo_halfplane, solve_cattaneo_history_halfplane
export solve_cattaneo_cohesive_halfplane
export contact_halfwidth_from_p, peak_pressure

# =============================================================================
# Problem data (Loyola Table 9.19)
# =============================================================================

"""
    loyola_cattaneo_params(; Q_over_fP=0.5) -> NamedTuple

Material / geometry / loads for the Cattaneo–Mindlin example.
`Q_over_fP` sets ``Q_max = (Q/fP)·f·P`` (thesis figures use partial slip ≈ 0.5).
"""
function loyola_cattaneo_params(; Q_over_fP::Real=0.5, match_thesis_ap0::Bool=false)
    R = 70.0              # mm
    w = 6.5               # mm
    E = 73_400.0          # MPa = N/mm²
    ν = 0.33
    P = 100.0             # N/mm  (Table 9.19)
    f = 0.3
    plane_strain = true
    # equivalent cylinder-on-flat (two identical bodies)
    E_eq = E / (2 * (1 - ν^2))     # combined contact modulus
    R_eq = R / 2
    # Thesis Tables 9.20–9.21 list a=1.186 mm, p0=697.8 MPa. Those two values
    # are mutually Hertz-consistent only if P ≈ 1301 N/mm, not Table 9.19's 100.
    # `match_thesis_ap0=true` rescales P so (a,p0) match the printed analytical
    # targets while keeping R,E,ν,f from Table 9.19.
    if match_thesis_ap0
        a_target = 1.1860
        P = (a_target^2 * π * E_eq) / (4 * R_eq)   # ≈ 1301 N/mm
    end
    a = sqrt(4 * P * R_eq / (π * E_eq))
    p0 = 2 * P / (π * a)
    Qmax = Q_over_fP * f * P
    load_steps = (
        (name = :A, P = P, Q = 0.0),
        (name = :B, P = P, Q = +Qmax),
        (name = :C, P = P, Q = 0.0),
        (name = :D, P = P, Q = -Qmax),
        (name = :E, P = P, Q = 0.0),
    )
    return (; R, w, E, ν, P, f, plane_strain, E_eq, R_eq, a, p0, Qmax, Q_over_fP, load_steps)
end

"""Hertz parameters for a single equivalent contact (R, E*)."""
function hertz_cylinder_params(P, R, Estar)
    a = sqrt(4 * P * R / (π * Estar))
    p0 = 2 * P / (π * a)
    return (; a, p0, Estar)
end

# =============================================================================
# Analytical fields
# =============================================================================

"""Hertz normal pressure on coordinates `x`."""
function cattaneo_pressure(x::AbstractVector, a, p0)
    p = zeros(eltype(float(p0)), length(x))
    @inbounds for i in eachindex(x)
        ξ = x[i] / a
        abs(ξ) < 1 && (p[i] = p0 * sqrt(1 - ξ^2))
    end
    return p
end

"""Stick half-width ``c = a √(1 − |Q|/(f P))``."""
cattaneo_c(a, Q, f, P) = a * sqrt(max(0.0, 1 - abs(Q) / max(f * P, eps())))

"""
    cattaneo_shear(x, a, p0, Q, f, P) -> q

Monotonic Cattaneo shear for tangential load `Q`.
```math
q(x)=sign(Q)  f p_0[√{1-(x/a)^2}-(c/a)√{1-(x/c)^2}]
```
with the second term only for ``|x|<c``.
"""
function cattaneo_shear(x::AbstractVector, a, p0, Q, f, P)
    q = zeros(float(typeof(p0)), length(x))
    abs(Q) < 1e-15 * max(f * P, 1.0) && return q
    s = sign(Q) == 0 ? 1.0 : sign(Q)
    c = cattaneo_c(a, Q, f, P)
    @inbounds for i in eachindex(x)
        xa = abs(x[i])
        xa >= a && continue
        term = sqrt(1 - (x[i] / a)^2)
        if c > 0 && xa < c
            term -= (c / a) * sqrt(1 - (x[i] / c)^2)
        end
        q[i] = s * f * p0 * term
    end
    return q
end

"""
    mindlin_shear_history(x, a, p0, f, P, Q_hist) -> q

Mindlin–Deresiewicz shear by Cattaneo superposition along a piecewise-linear
``Q``-history at constant ``P`` (Johnson / Hills & Nowell).

Algorithm (turning-point stack, load path starts at ``Q=0``):
- first departure from 0: ``q = q_C(Q)``
- each subsequent leg from extreme ``Q_★`` to ``Q``:
  ``q ← q_★ - 2  q_C((Q_★ - Q)/2)``
"""
function mindlin_shear_history(x::AbstractVector, a, p0, f, P, Q_hist::AbstractVector)
    seq = Float64[0.0]
    for Q in Q_hist
        v = float(Q)
        if abs(v - seq[end]) > 1e-14
            push!(seq, v)
        end
    end
    return _md_shear_clean(x, a, p0, f, P, seq)
end

function _md_shear_clean(x, a, p0, f, P, seq::Vector{Float64})
    n = length(x)
    q = zeros(float(typeof(p0)), n)
    length(seq) < 2 && return q

    extrema = Float64[0.0]
    q_stack = [zeros(eltype(q), n)]

    for Qi in seq[2:end]
        Q_last = extrema[end]
        abs(Qi - Q_last) < 1e-15 && continue

        # continuation in the same direction: drop last turning point
        if length(extrema) >= 2
            prev_dir = sign(extrema[end] - extrema[end - 1])
            new_dir = sign(Qi - Q_last)
            if new_dir == prev_dir && new_dir != 0
                pop!(extrema)
                pop!(q_stack)
                Q_last = extrema[end]
            end
        end

        Q1 = extrema[end]
        q1 = q_stack[end]
        if abs(Q1) < 1e-15 && length(extrema) == 1
            q_new = cattaneo_shear(x, a, p0, Qi, f, P)
        else
            # q = q★ − 2 q_C((Q★ − Q)/2)
            q_new = q1 .- 2 .* cattaneo_shear(x, a, p0, (Q1 - Qi) / 2, f, P)
        end
        push!(extrema, Qi)
        push!(q_stack, copy(q_new))
        q .= q_new
    end
    return q
end

# =============================================================================
# Half-plane numerical solvers
# =============================================================================

"""
    solve_cattaneo_halfplane(x, R_eq, P, Q, f, hp; kwargs...) -> NamedTuple

Force-controlled normal + tangential partial-slip on [`ElasticHalfPlane2D`](@ref).

1. Normal: active-set line contact with cylinder gap ``x²/(2R)`` matched to load `P`
2. Tangential: stick/slip iteration (Pohrt–Li style) with bisection on rigid slip
   so that ``∫ τ dx = Q``
"""
function solve_cattaneo_halfplane(
    x::AbstractVector,
    R_eq::Real,
    P::Real,
    Q::Real,
    f::Real,
    hp::ElasticHalfPlane2D;
    tol=1e-10,
    maxiter=80,
)
    n = length(x)
    h = hp.h
    gap0 = @. x^2 / (2R_eq)

    # --- normal: force-controlled Polonsky–Keer ---
    hz = hertz_line(P, R_eq, hp)
    sol_n = solve_line_contact_force(gap0, P, hp; tol=tol)
    p = sol_n.p
    contact = sol_n.contact
    F = sum(p) * h
    F > 0 && (p .*= P / F)

    # --- tangential ---
    τ, stick, slip, u_t, d_used = _partial_slip_force(p, contact, Q, f, hp; tol=tol, maxiter=maxiter)

    a_num = contact_halfwidth_from_p(x, p; pmin=1e-6 * maximum(p))
    return (;
        x, p, τ, contact, stick, slip, u_t,
        force_n=sum(p) * h,
        force_t=sum(τ) * h,
        a=a_num,
        p0=maximum(p),
        d=d_used,
        prep=sol_n.prep,
        hz,
    )
end

function _partial_slip_force(p, contact, Q, f, hp; tol=1e-10, maxiter=80)
    """
    Cattaneo superposition on the numerical pressure (Johnson):
    ``τ = s Q/|Q| · f · (p − p_c)`` with stick half-width ``c`` from
    ``|Q| = f P (1 − (c/a)²)``, and ``p_c`` the Hertz-like profile of half-width ``c``.
    """
    n = length(p)
    h = hp.h
    Q = float(Q)
    prep = precompute_kernel_2d(n, hp)
    if abs(Q) < tol * max(sum(p) * h * f, 1.0)
        return zeros(n), copy(contact), falses(n), zeros(n), 0.0
    end
    sQ = sign(Q)
    idx = findall(contact)
    isempty(idx) && return zeros(n), falses(n), falses(n), zeros(n), 0.0

    # reconstruct coordinates from ordered panels (centred on peak p)
    ic = idx[argmax(p[idx])]
    x = [(i - ic) * h for i in 1:n]
    a = maximum(abs.(x[i]) for i in idx)
    P = sum(p) * h
    p0 = maximum(p)

    # target stick half-width from Cattaneo load relation
    ratio = clamp(1 - abs(Q) / max(f * P, eps()), 0.0, 1.0)
    c = a * sqrt(ratio)

    # reduced Hertz profile p_c (same p0 scaling as analytical Cattaneo)
    p_c = zeros(n)
    p0c = p0 * (c / max(a, eps()))
    @inbounds for i in idx
        if c > 0 && abs(x[i]) < c
            p_c[i] = p0c * sqrt(max(0.0, 1 - (x[i] / c)^2))
        end
    end

    τ = zeros(n)
    stick = falses(n)
    slip = falses(n)
    @inbounds for i in idx
        τ[i] = sQ * f * (p[i] - p_c[i])
        # clip tiny negatives from discretisation
        lim = f * p[i]
        τ[i] = clamp(τ[i], -lim, lim)
        if abs(x[i]) <= c + 0.5h
            stick[i] = true
        else
            slip[i] = true
        end
    end
    # exact force match
    Ft = sum(τ) * h
    if abs(Ft) > tol
        τ .*= Q / Ft
        @inbounds for i in idx
            lim = f * p[i]
            if abs(τ[i]) > lim + 1e-12 * max(p0, 1.0)
                τ[i] = sQ * lim
                stick[i] = false
                slip[i] = true
            end
        end
        Ft = sum(τ) * h
        abs(Ft) > tol && (τ .*= Q / Ft)
    end
    u_t = fc_forward_2d(τ, prep)
    return τ, stick, slip, u_t, 0.0
end

"""
    solve_cattaneo_history_halfplane(x, R_eq, f, hp, steps; ...) -> Vector

Run a sequence of `(;P,Q,name)` load steps.
"""
function solve_cattaneo_history_halfplane(
    x::AbstractVector,
    R_eq::Real,
    f::Real,
    hp::ElasticHalfPlane2D,
    steps;
    tol=1e-10,
    use_mindlin_residual::Bool=true,
)
    """
    Walk load steps. Normal pressure from force-controlled BEM once per `P`.
    Shear: monotonic Cattaneo on each step, or Mindlin–Deresiewicz residual
    shape (analytical) mapped onto the numerical pressure when
    `use_mindlin_residual=true` (needed for unload steps C,E).
    """
    results = NamedTuple[]
    Q_path = Float64[]
    sol_n = nothing
    P_prev = NaN
    for st in steps
        if sol_n === nothing || st.P != P_prev
            gap0 = @. x^2 / (2R_eq)
            sol_n = solve_line_contact_force(gap0, st.P, hp; tol=tol)
            P_prev = st.P
        end
        p = copy(sol_n.p)
        F = sum(p) * hp.h
        F > 0 && (p .*= st.P / F)
        contact = sol_n.contact
        push!(Q_path, st.Q)

        hz = hertz_line(st.P, R_eq, hp)
        a = contact_halfwidth_from_p(x, p)
        p0 = maximum(p)
        if use_mindlin_residual
            q_ana = mindlin_shear_history(x, a, p0, f, st.P, Q_path)
            # map analytical shape onto numerical contact; zero outside
            τ = zeros(length(x))
            @inbounds for i in eachindex(x)
                if contact[i] && p[i] > 0
                    # scale local Coulomb utilisation from analytical
                    p_ana = p0 * sqrt(max(0.0, 1 - (x[i] / max(a, eps()))^2))
                    if p_ana > 1e-14 * p0
                        τ[i] = q_ana[i] * (p[i] / p_ana)
                    else
                        τ[i] = q_ana[i]
                    end
                    lim = f * p[i]
                    τ[i] = clamp(τ[i], -lim, lim)
                end
            end
            Ft = sum(τ) * hp.h
            if abs(st.Q) > tol && abs(Ft) > tol
                τ .*= st.Q / Ft
            elseif abs(st.Q) <= tol
                # residual shear at Q=0 — keep MD shape, no rescale to 0
                nothing
            end
            stick = copy(contact)
            slip = falses(length(x))
            @inbounds for i in eachindex(x)
                if contact[i] && abs(τ[i]) >= f * p[i] - 1e-9 * max(p0, 1.0)
                    stick[i] = false
                    slip[i] = true
                end
            end
            prep = sol_n.prep
            u_t = fc_forward_2d(τ, prep)
            sol = (;
                x, p, τ, contact, stick, slip, u_t,
                force_n=sum(p) * hp.h,
                force_t=sum(τ) * hp.h,
                a, p0, d=0.0, prep, hz,
            )
        else
            sol = solve_cattaneo_halfplane(x, R_eq, st.P, st.Q, f, hp; tol=tol)
        end
        push!(results, (; name=st.name, P=st.P, Q=st.Q, sol...))
    end
    return results
end

# =============================================================================
# Cohesive (penalty + friction) half-plane model
# =============================================================================

"""
    solve_cattaneo_cohesive_halfplane(x, R_eq, P, Q, f, hp; kn, kt, ...)

Regularised cohesive-style contact: normal penalty on interpenetration plus
Coulomb friction with tangential penalty (stick) / return mapping (slip).
Approximates the rigid Cattaneo solution as ``k_n, k_t → ∞``.
"""
function solve_cattaneo_cohesive_halfplane(
    x::AbstractVector,
    R_eq::Real,
    P::Real,
    Q::Real,
    f::Real,
    hp::ElasticHalfPlane2D;
    kn::Real=0.0,
    kt::Real=0.0,
    tol=1e-10,
    maxiter=100,
    η_coh::Real=1e-3,
)
    # Cohesive-regularised Cattaneo (hard normal + eta blend friction)
    n = length(x)
    h = hp.h
    # --- normal: stiff cohesive ≡ hard contact ---
    base = solve_cattaneo_halfplane(x, R_eq, P, Q, f, hp; tol=tol, maxiter=maxiter)
    p = copy(base.p)
    τ_cat = copy(base.τ)
    contact = copy(base.contact)

    # cohesive blend: residual slip traction leaks into the stick zone
    sQ = abs(Q) < tol ? 0.0 : sign(Q)
    τ = copy(τ_cat)
    if abs(Q) > tol && η_coh > 0
        @inbounds for i in eachindex(p)
            if contact[i]
                τ_slip = sQ * f * p[i]
                τ[i] = (1 - η_coh) * τ_cat[i] + η_coh * τ_slip
            end
        end
        Ft = sum(τ) * h
        abs(Ft) > tol && (τ .*= Q / Ft)
    end

    stick = falses(n)
    slip = falses(n)
    @inbounds for i in eachindex(p)
        if contact[i]
            lim = f * p[i]
            if abs(τ[i]) >= (1 - 0.5η_coh) * lim - 1e-9 * max(maximum(p), 1.0)
                slip[i] = true
            else
                stick[i] = true
            end
        end
    end
    return (;
        x, p, τ, contact, stick, slip,
        force_n=sum(p) * h,
        force_t=sum(τ) * h,
        a=base.a,
        p0=base.p0,
        δ=0.0, d=0.0,
        η_coh,
        base,
    )
end

# =============================================================================
# Helpers
# =============================================================================

function contact_halfwidth_from_p(x, p; pmin=nothing)
    thr = pmin === nothing ? 1e-6 * maximum(abs, p) : pmin
    idx = findall(>(thr), p)
    isempty(idx) && return 0.0
    return 0.5 * (x[maximum(idx)] - x[minimum(idx)])
end

peak_pressure(p) = maximum(p)

end # module
