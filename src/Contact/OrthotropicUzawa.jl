"""
    OrthotropicUzawa

Juliá Lerma (2025) Ch. 2.1 contact: coupled half-space operator, elliptic Coulomb
friction, orthotropic Archard wear, Uzawa / Alart–Curnier projections.

Gap convention (opening, Signorini ``g_n ≥ 0``, ``p_n ≥ 0``):

```
g_n = g_geom − δ + w + u_z
p_n ← max(0, p_n − r_n g_n)
```

(`u_z` into the solid, Pohrt sign). The thesis typesets ``p_n^* = p_n + r_n g_n``;
the minus sign is the standard projection for a non-negative gap.
"""
module OrthotropicUzawa

using LinearAlgebra
using Printf
using ..ContactHalfSpace

export OrthotropicLaw, HalfSpaceGrid, ContactState
export isotropic_law, sphere_gap, flat_punch_gap
export make_grid, init_state, solve_contact_step!, commit_tangential_ref!
export contact_resultants, contact_patch_stats, set_approach_for_load!, match_load!
export wear_split, argatov_worn_radius, argatov2011, hegadekatte_wear_depth
export sneddon_pressure
export sliding_wear_steps!, sliding_wear_force_steps!, radial_fretting_cycles!
export project_tangential, orthotropic_slip_norm
export rotate_to_tribological, rotate_from_tribological, uzawa_tangential
export winkler_seed!

"""Orthotropic friction / wear in tribological axes rotated by `β` (radians, CCW)."""
struct OrthotropicLaw{T}
    μ1::T
    μ2::T
    i1::T
    i2::T
    β::T
end

function OrthotropicLaw(μ1::Real, μ2::Real, i1::Real, i2::Real, β::Real=0)
    T = float(promote_type(typeof(μ1), typeof(μ2), typeof(i1), typeof(i2), typeof(β)))
    return OrthotropicLaw{T}(T(μ1), T(μ2), T(i1), T(i2), T(β))
end

isotropic_law(μ::Real, i::Real=0.0) = OrthotropicLaw(μ, μ, i, i, 0)

Base.iszero(law::OrthotropicLaw) = law.μ1 == 0 && law.μ2 == 0

struct HalfSpaceGrid{T}
    x::Vector{T}
    y::Vector{T}
    gap_geom::Matrix{T}
    hs::ElasticHalfSpace{T}
end

function make_grid(x::AbstractVector, y::AbstractVector, hs::ElasticHalfSpace, gap_geom::AbstractMatrix)
    T = typeof(hs.G)
    return HalfSpaceGrid{T}(collect(T, x), collect(T, y), Matrix{T}(gap_geom), hs)
end

function make_grid(nx::Int, ny::Int, Lx::Real, Ly::Real, hs::ElasticHalfSpace, gap_geom)
    x = collect(range(-Lx / 2 + hs.hx / 2, stop=Lx / 2 - hs.hx / 2, length=nx))
    y = collect(range(-Ly / 2 + hs.hy / 2, stop=Ly / 2 - hs.hy / 2, length=ny))
    return make_grid(x, y, hs, gap_geom)
end

sphere_gap(x, y, R) = [(xi^2 + yj^2) / (2R) for xi in x, yj in y]

"""Rigid flat punch of radius `a0`: zero gap inside, large outside."""
function flat_punch_gap(x, y, a0; out=1e3)
    g = fill(float(out), length(x), length(y))
    a2 = a0^2
    @inbounds for j in eachindex(y), i in eachindex(x)
        if x[i]^2 + y[j]^2 <= a2
            g[i, j] = 0
        end
    end
    return g
end

mutable struct ContactState{T}
    pn::Matrix{T}
    ptx::Matrix{T}
    pty::Matrix{T}
    ux::Matrix{T}
    uy::Matrix{T}
    uz::Matrix{T}
    w::Matrix{T}
    gn::Matrix{T}
    gtx::Matrix{T}
    gty::Matrix{T}
    gtx_ref::Matrix{T}
    gty_ref::Matrix{T}
end

function init_state(grid::HalfSpaceGrid{T}) where {T}
    nx, ny = length(grid.x), length(grid.y)
    z() = zeros(T, nx, ny)
    return ContactState(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z())
end

function contact_resultants(st::ContactState, hs::ElasticHalfSpace)
    dA = hs.hx * hs.hy
    P = sum(st.pn) * dA
    Qx = sum(st.ptx) * dA
    Qy = sum(st.pty) * dA
    return P, Qx, Qy
end

"""Contact area, equivalent radius ``a=√(A/π)``, and mean/max pressure."""
function contact_patch_stats(st::ContactState, hs::ElasticHalfSpace; rel=1e-3)
    dA = hs.hx * hs.hy
    pmax = maximum(st.pn)
    thresh = rel * max(pmax, eps())
    n = count(>(thresh), st.pn)
    P = sum(st.pn) * dA
    A = n * dA
    a = A > 0 ? sqrt(A / π) : 0.0
    pmean = A > 0 ? P / A : 0.0
    return (; A, a, a_outer=a, P, pmax, pmean, n)
end

function contact_patch_stats(st::ContactState, grid::HalfSpaceGrid; rel=1e-3)
    stats = contact_patch_stats(st, grid.hs; rel=rel)
    thresh = rel * max(stats.pmax, eps())
    a2_out = 0.0
    @inbounds for j in eachindex(grid.y), i in eachindex(grid.x)
        if st.pn[i, j] > thresh
            r2 = grid.x[i]^2 + grid.y[j]^2
            r2 > a2_out && (a2_out = r2)
        end
    end
    return merge(stats, (; a_outer=sqrt(a2_out)))
end

@inline function rotate_to_tribological(px, py, cβ, sβ)
    pe1 = cβ * px + sβ * py
    pe2 = -sβ * px + cβ * py
    return pe1, pe2
end

@inline function rotate_from_tribological(pe1, pe2, cβ, sβ)
    px = cβ * pe1 - sβ * pe2
    py = sβ * pe1 + cβ * pe2
    return px, py
end

const _rotate_to_e = rotate_to_tribological
const _rotate_from_e = rotate_from_tribological

"""Elliptic Coulomb projection onto the disk ``‖p‖_μ ≤ ρ`` (thesis eq. 2.14)."""
function project_tangential(px, py, ρ, law::OrthotropicLaw, cβ, sβ)
    μ1, μ2 = law.μ1, law.μ2
    if μ1 <= 0 && μ2 <= 0
        return zero(px), zero(py)
    end
    pe1, pe2 = _rotate_to_e(px, py, cβ, sβ)
    nμ = hypot(pe1 / μ1, pe2 / μ2)
    if nμ > ρ && nμ > 0
        s = ρ / nμ
        pe1 *= s
        pe2 *= s
    end
    return _rotate_from_e(pe1, pe2, cβ, sβ)
end

function orthotropic_slip_norm(dgx, dgy, i1, i2, cβ, sβ)
    e1, e2 = rotate_to_tribological(dgx, dgy, cβ, sβ)
    return hypot(i1 * e1, i2 * e2)
end

"""Relaxed Alart–Curnier tangential step. Returns `(px, py, ΔΨt²)`."""
function uzawa_tangential(ptx, pty, sx, sy, pn, law::OrthotropicLaw, cβ, sβ, rt, relax)
    if iszero(law) || iszero(pn)
        return zero(ptx), zero(pty), ptx^2 + pty^2
    end
    e1, e2 = rotate_to_tribological(sx, sy, cβ, sβ)
    m2x, m2y = rotate_from_tribological(law.μ1 * law.μ1 * e1, law.μ2 * law.μ2 * e2, cβ, sβ)
    px_p, py_p = project_tangential(ptx - rt * m2x, pty - rt * m2y, pn, law, cβ, sβ)
    px_new = (1 - relax) * ptx + relax * px_p
    py_new = (1 - relax) * pty + relax * py_p
    return px_new, py_new, (px_new - ptx)^2 + (py_new - pty)^2
end

"""Winkler seed for ``p_n`` when the field is still zero."""
function winkler_seed!(st::ContactState, grid::HalfSpaceGrid, δ, Azz0; P_seed=nothing)
    iszero(st.pn) || return st
    nx, ny = size(st.pn)
    @inbounds for j in 1:ny, i in 1:nx
        pen = δ - grid.gap_geom[i, j] - st.w[i, j]
        st.pn[i, j] = max(pen, 0.0) / max(Azz0, eps())
    end
    if P_seed !== nothing
        Pnow = sum(st.pn) * grid.hs.hx * grid.hs.hy
        Pnow > 0 && (st.pn .*= P_seed / Pnow)
    end
    return st
end

function _penalties(hs, rn, rt)
    rn_d, rt_d = default_penalties(hs)
    return rn === nothing ? rn_d : float(rn), rt === nothing ? rt_d : float(rt)
end

"""
    solve_contact_step!(state, grid, prep, law, δ, gx_o, gy_o; kwargs...) -> niter

One load / wear increment (thesis 2.1.7). `gx_o, gy_o` are the current rigid
tangential slips (absolute). `Δg_t` is measured from `state.gtx_ref`.
Wear increment is multiplied by `wear_jump` (cycle / distance jumping).
"""
function solve_contact_step!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    δ::Real,
    gx_o::Real,
    gy_o::Real;
    rn::Union{Nothing,Real}=nothing,
    rt::Union{Nothing,Real}=nothing,
    tol=1e-8,
    maxiter=400,
    wear_jump=1.0,
    frictionless=iszero(law),
    relax=0.35,
    P_seed=nothing,
    kwargs...,
)
    hs = grid.hs
    nx, ny = length(grid.x), length(grid.y)
    Azz0 = influence_coeff(Kzz, 0, 0, hs)
    rn_, rt_ = _penalties(hs, rn, rt)
    cβ, sβ = cos(law.β), sin(law.β)
    tmp = zeros(Float64, nx, ny)
    w0 = copy(st.w)
    do_wear = law.i1 != 0 || law.i2 != 0
    winkler_seed!(st, grid, δ, Azz0; P_seed=P_seed)

    niter = 0
    Ψ = Inf
    for it in 1:maxiter
        niter = it
        fc_displacements!(st.ux, st.uy, st.uz, st.ptx, st.pty, st.pn, prep; tmp=tmp)

        Ψn = 0.0
        Ψt = 0.0
        @inbounds for j in 1:ny, i in 1:nx
            gn = grid.gap_geom[i, j] - δ + st.w[i, j] + st.uz[i, j]
            st.gn[i, j] = gn
            gtx = gx_o + st.ux[i, j]
            gty = gy_o + st.uy[i, j]
            st.gtx[i, j] = gtx
            st.gty[i, j] = gty

            pn_old = st.pn[i, j]
            pn_proj = max(0.0, pn_old - rn_ * gn)
            pn_new = (1 - relax) * pn_old + relax * pn_proj
            Ψn += (pn_new - pn_old)^2
            st.pn[i, j] = pn_new

            dgx = gtx - st.gtx_ref[i, j]
            dgy = gty - st.gty_ref[i, j]

            pn_t = (frictionless || pn_new == 0) ? zero(pn_new) : pn_new
            px_new, py_new, dΨt = uzawa_tangential(st.ptx[i, j], st.pty[i, j],
                                                   dgx, dgy, pn_t, law, cβ, sβ, rt_, relax)
            Ψt += dΨt
            st.ptx[i, j] = px_new
            st.pty[i, j] = py_new

            if do_wear
                st.w[i, j] = w0[i, j] + wear_jump * abs(pn_new) *
                             orthotropic_slip_norm(dgx, dgy, law.i1, law.i2, cβ, sβ)
            end
        end
        Ψ = sqrt(Ψn) + sqrt(Ψt)
        Ψ <= tol && break
    end
    return niter, Ψ
end

function commit_tangential_ref!(st::ContactState)
    st.gtx_ref .= st.gtx
    st.gty_ref .= st.gty
    return st
end

"""
Adjust rigid approach `δ` so the normal resultant equals `P_target`.

Uses Hertz stiffness ``dP/d(extra) = 1.5 P / extra`` (sphere-like). Prefer
[`match_load!`](@ref) after wear, when the scar is punch-like (Sneddon
``dP/dδ ≈ 2 E^* a``).
"""
function set_approach_for_load!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    P_target::Real,
    gx_o::Real,
    gy_o::Real;
    δ0=nothing,
    rtol=1e-4,
    maxouter=40,
    wear_jump=1.0,
    kwargs...,
)
    Estar = contact_modulus(grid.hs)
    wmax = maximum(st.w)
    δ = δ0 === nothing ? wmax + P_target / (Estar * max(grid.hs.hx, 1e-12)) : float(δ0)
    extra = max(δ - wmax, 10 * eps(Float64))
    niter = 0
    Ψ = Inf
    for _ in 1:maxouter
        wmax = maximum(st.w)
        δ = wmax + extra
        niter, Ψ = solve_contact_step!(st, grid, prep, law, δ, gx_o, gy_o; wear_jump=0, P_seed=P_target, kwargs...)
        P, _, _ = contact_resultants(st, grid.hs)
        err = P - P_target
        abs(err) <= rtol * max(abs(P_target), eps()) && break
        if P < 0.05 * max(P_target, 1.0)
            extra *= 2
            continue
        end
        # Hertz-like: P ∝ extra^{3/2} ⇒ dP/d(extra) = 1.5 P / extra
        slope = 1.5 * P / extra
        extra = max(extra - err / slope, 10 * eps(Float64))
    end
    wmax = maximum(st.w)
    δ = wmax + extra
    niter, Ψ = solve_contact_step!(st, grid, prep, law, δ, gx_o, gy_o; wear_jump=wear_jump, P_seed=P_target, kwargs...)
    return δ, niter, Ψ
end

"""Split total wear depth by hardness (thesis 2.29)."""
function wear_split(w, H_A, H_B)
    s = H_A + H_B
    return (H_B / s) .* w, (H_A / s) .* w
end

"""Sneddon pressure of a frictionless circular flat punch, ``r < a0``."""
sneddon_pressure(r, a0, P) = r < a0 ? P / (2π * a0 * sqrt(a0^2 - r^2)) : zero(r)

"""Argatov worn contact radius for a sliding sphere, ``a^4 = a0^4 + 4 i P s R / π``."""
function argatov_worn_radius(a0, i_w, P, s, R)
    return (a0^4 + 4 * i_w * P * s * R / π)^(1 / 4)
end

"""
Argatov (2011, Wear 271:1147) axisymmetric sliding wear, simplified §11.

Volume-consistent (spherical cap + Archard), used in Juliá Lerma / Paper 1 Fig. 6:

```
a⁴ = a0⁴ + 4 k F s R / π
H  = a² / (2R)                 # total wear depth at the centre
H0 = H − a0²/R                 # online-measured depth, eq. (52)
```

`s` may be a scalar or vector.
"""
function argatov2011(s, a0, R, k, F)
    a = @. (a0^4 + 4 * k * F * s * R / π)^(1 / 4)
    H = @. a^2 / (2R)
    δ0 = a0^2 / R
    H0 = @. H - δ0
    W = @. k * F * s
    return (; a, H, H0, W, δ0)
end

"""Hegadekatte spherical-cap inversion of ``W = k F s = π w² (3R − w)/3``."""
function hegadekatte_wear_depth(s, R, k, F; niter=25)
    W = k * F * s
    W <= 0 && return 0.0
    w = sqrt(W / (π * R))
    for _ in 1:niter
        f = π * w^2 * (3R - w) / 3 - W
        df = π * w * (2R - w)
        abs(df) < eps() && break
        w = max(w - f / df, 0.0)
    end
    return w
end

"""
Adjust rigid approach `δ` until the normal load equals `P_target`.

Warm-starts from the current pressure (does **not** zero or scale `p`). After
wear the scar is punch-like, so the Newton step uses Sneddon stiffness
``dP/dδ ≈ 2 E^* a`` with ``a = √(A/π)``.
"""
function _load_control!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    P_target::Real,
    δ::Real;
    rtol=1e-3,
    maxouter=12,
    kwargs...,
)
    Estar = contact_modulus(grid.hs)
    hx = grid.hs.hx
    niter = 0
    Ψ = Inf
    stats = contact_patch_stats(st, grid)
    for _ in 1:maxouter
        niter, Ψ = solve_contact_step!(st, grid, prep, law, δ, 0.0, 0.0;
                                       wear_jump=0, kwargs...)
        stats = contact_patch_stats(st, grid)
        err = stats.P - P_target
        abs(err) <= rtol * max(abs(P_target), eps()) && return δ, niter, Ψ, stats
        a = max(stats.a, stats.a_outer, hx)
        kn = max(2 * Estar * a, Estar * hx)
        δ = max(δ - 0.75 * err / kn, 1e-12)
    end
    stats = contact_patch_stats(st, grid)
    return δ, niter, Ψ, stats
end

"""Alias: adjust ``δ`` until the normal load equals `P_target` (Sneddon stiffness)."""
const match_load! = _load_control!

"""
Radial-fretting (load–unload) wear of a complete-contact punch.

Matches thesis (2.20): ``Δw = |p_n| ‖Δg_t‖_i`` on every contact node (no
stick mask). A load-step Uzawa with `wear_jump = 2ΔN` covers both strokes
and redistributes `p` as the gap wears, so a jumped block does not carve a
frozen-slip trench. `max_jump` caps `ΔN` so the stick/slip set can recede.
"""
function radial_fretting_cycles!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    P_target::Real;
    δ0=nothing,
    n_end::Int=10^5,
    hist_N::Vector{Int}=[1, 2, 5, 10, 20, 50, 100, 200, 500,
                         10^3, 2 * 10^3, 5 * 10^3, 10^4, 2 * 10^4, 5 * 10^4, 10^5],
    field_N::Vector{Int}=[10^3, 10^4, 10^5],
    rtol=5e-3,
    cap_frac=0.03,
    max_jump::Int=2500,
    kwargs...,
)
    Estar = contact_modulus(grid.hs)
    a0 = begin
        r2 = 0.0
        @inbounds for j in eachindex(grid.y), i in eachindex(grid.x)
            grid.gap_geom[i, j] == 0 || continue
            r2 = max(r2, grid.x[i]^2 + grid.y[j]^2)
        end
        sqrt(r2)
    end
    δ = δ0 === nothing ? P_target / (2 * Estar * max(a0, grid.hs.hx)) : float(δ0)
    dA = grid.hs.hx * grid.hs.hy
    hx = grid.hs.hx
    cβ, sβ = cos(law.β), sin(law.β)
    frictionless = law.μ1 <= 1e-12 && law.μ2 <= 1e-12

    hist_N = sort!(unique!(vcat(hist_N, field_N, [n_end])))
    filter!(n -> 0 < n <= n_end, hist_N)
    fields = Dict{Int,NamedTuple}()
    Nrec = Int[]
    wmax = Float64[]
    vol = Float64[]
    pmax = Float64[]
    Pload = Float64[]
    nslip = Int[]

    function _record!(Ncy, δk, stats)
        push!(Nrec, Ncy)
        push!(wmax, maximum(st.w))
        push!(vol, sum(st.w) * dA)
        push!(pmax, stats.pmax)
        push!(Pload, stats.P)
        sl = 0
        pth = 1e-3 * max(stats.pmax, eps())
        @inbounds for i in eachindex(st.pn)
            st.pn[i] <= pth && continue
            pe1 =  cβ * st.ptx[i] + sβ * st.pty[i]
            pe2 = -sβ * st.ptx[i] + cβ * st.pty[i]
            nμ = hypot(pe1 / max(law.μ1, 1e-16), pe2 / max(law.μ2, 1e-16))
            (frictionless || nμ >= 0.95 * st.pn[i]) && (sl += 1)
        end
        push!(nslip, sl)
        if Ncy in field_N || Ncy == n_end
            fields[Ncy] = (; N=Ncy, δ=δk, P=stats.P, pmax=stats.pmax,
                             pn=copy(st.pn), ptx=copy(st.ptx), pty=copy(st.pty),
                             w=copy(st.w), ux=copy(st.ux), uy=copy(st.uy))
        end
        return nothing
    end

    fill!(st.gtx_ref, 0)
    fill!(st.gty_ref, 0)
    δ, _, _, stats = _load_control!(st, grid, prep, law, P_target, δ;
                                    rtol=rtol, wear_jump=0, kwargs...)
    _record!(0, δ, stats)

    Ndone = 0
    w_last = maximum(st.w)
    dw_cycle = 0.0
    while Ndone < n_end
        target = hist_N[searchsortedfirst(hist_N, Ndone + 1)]
        if Ndone < 4
            ΔN = 1
        else
            cap = cap_frac * hx
            jump = dw_cycle > 0 ? max(1, floor(Int, cap / dw_cycle)) : target - Ndone
            ΔN = min(jump, max_jump, target - Ndone)
        end

        w_before = maximum(st.w)
        # Loaded half-cycle + return stroke. Wear is inside Uzawa (thesis 2.20)
        # so p redistributes as the gap grows — no frozen-slip trench.
        solve_contact_step!(st, grid, prep, law, δ, 0.0, 0.0;
                            wear_jump=2 * ΔN, kwargs...)
        δ, _, _, stats = _load_control!(st, grid, prep, law, P_target, δ;
                                        rtol=rtol, wear_jump=0, kwargs...)
        commit_tangential_ref!(st)
        solve_contact_step!(st, grid, prep, law, 0.0, 0.0, 0.0;
                            wear_jump=0, kwargs...)
        commit_tangential_ref!(st)
        δ, _, _, stats = _load_control!(st, grid, prep, law, P_target, δ;
                                        rtol=rtol, wear_jump=0, kwargs...)
        Ndone += ΔN
        dw_cycle = max(maximum(st.w) - w_before, 0.0) / ΔN
        w_last = maximum(st.w)
        if Ndone in hist_N || Ndone == n_end
            _record!(Ndone, δ, stats)
            get(ENV, "WEAR_VERBOSE", "false") == "true" &&
                @printf "  fret N=%d  ΔN=%d  wmax=%.3e mm  vol=%.3e  P=%.1f  pmax=%.1f  nslip=%d\n" Ndone ΔN w_last (sum(st.w) * dA) stats.P stats.pmax nslip[end]
        end
    end
    return (; N=Nrec, wmax, vol, pmax, P=Pload, nslip, fields, δ)
end

"""
Force-controlled gross-sliding wear (Argatov 2011: constant load `P_target`).

Each increment (i) solves the contact at frozen wear until the load matches
`P_target`, then (ii) applies Archard ``Δw = i_w |p_n| Δs`` with the imposed
disc travel `Δs` (not the elastic tangential gap).

The pressure is warm-started and **not** rescaled: scaling a Hertzian peak
preserves a non-uniform `p` and, with a large `Δs`, excavates a crater so
`w_max` plateaus and `p_max` never approaches the uniform ``P/(π a^2)`` of
Paper 1 Fig. 6. Large sliding increments are split so the centre wear per
substep stays below a fraction of the elastic approach.

`snapshot_s` is an optional list of sliding distances (mm) at which the
centreline ``(w, p_n, g)`` is stored in the returned `snaps` (first step
with ``s ≥`` each target).
"""
function sliding_wear_force_steps!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    P_target::Real,
    Δs::Real,
    nsteps::Int;
    δ0=nothing,
    rtol=1e-3,
    maxouter=12,
    snapshot_s=Float64[],
    kwargs...,
)
    Estar = contact_modulus(grid.hs)
    δ = δ0 === nothing ? P_target / (Estar * max(grid.hs.hx, 1e-12)) : float(δ0)
    extra_ref = float(δ)
    nsamp = nsteps + 1
    hist_w = zeros(nsamp)
    hist_pmax = zeros(nsamp)
    hist_pmean = zeros(nsamp)
    hist_a = zeros(nsamp)
    hist_P = zeros(nsamp)
    hist_s = zeros(nsamp)
    hist_δ = zeros(nsamp)
    cβ, sβ = cos(law.β), sin(law.β)
    fill!(st.gtx_ref, 0)
    fill!(st.gty_ref, 0)

    ic = (length(grid.x) + 1) ÷ 2
    jc = (length(grid.y) + 1) ÷ 2
    hx = grid.hs.hx

    snap_s = sort!(Float64.(collect(snapshot_s)))
    snaps = NamedTuple[]
    isnap = 1
    function _maybe_snap!(s)
        while isnap <= length(snap_s) && s + 1e-9 >= snap_s[isnap]
            push!(snaps, (; s=s,
                            x=copy(grid.x),
                            y=copy(grid.y),
                            w=st.w[:, jc],
                            pn=st.pn[:, jc],
                            pn2d=copy(st.pn),
                            gap=grid.gap_geom[:, jc]))
            isnap += 1
        end
        return nothing
    end

    function _record!(k, s, δk, stats)
        hist_s[k] = s
        hist_w[k] = maximum(st.w)
        hist_pmax[k] = stats.pmax
        hist_pmean[k] = stats.pmean
        hist_a[k] = stats.a
        hist_P[k] = stats.P
        hist_δ[k] = δk
        _maybe_snap!(s)
        return nothing
    end

    # A ring (centre open, load carried at the rim) still matches P, so load
    # control will not raise δ. Reseed Winkler and push the approach until the
    # node of max wear is back in contact, then restore the load.
    function _solve_filled!(δ_in)
        δk, niter_k, Ψk, stats_k = _load_control!(st, grid, prep, law, P_target, δ_in;
                                                  rtol=rtol, maxouter=maxouter, kwargs...)
        pc = st.pn[ic, jc]
        if pc < 0.05 * max(stats_k.pmean, eps()) && stats_k.n > 4
            extra = max(P_target / (2 * Estar * max(stats_k.a_outer, hx)), 0.05 * extra_ref)
            fill!(st.pn, 0)
            fill!(st.ptx, 0)
            fill!(st.pty, 0)
            δk, niter_k, Ψk, stats_k = _load_control!(st, grid, prep, law, P_target,
                                                      δk + extra;
                                                      rtol=rtol, maxouter=maxouter,
                                                      P_seed=P_target, kwargs...)
        end
        return δk, niter_k, Ψk, stats_k
    end

    δ, niter, Ψ, stats = _solve_filled!(δ)
    _record!(1, 0.0, δ, stats)
    get(ENV, "WEAR_VERBOSE", "false") == "true" &&
        @printf "  wear step 0  s=0.0 mm  wmax=%.4f um  P=%.3f N  pmax=%.2f  pmean=%.2f  a=%.3f mm  δ=%.4f um  niter=%d\n" 1e3 * hist_w[1] stats.P stats.pmax stats.pmean stats.a 1e3 * δ niter

    for k in 1:nsteps
        slip = orthotropic_slip_norm(Δs, 0.0, law.i1, law.i2, cβ, sβ)
        extra = max(P_target / (2 * Estar * max(stats.a, hx)), 0.05 * extra_ref)
        nsub = max(1, ceil(Int, stats.pmax * slip / (0.1 * extra)))
        nsub = min(nsub, 10)
        Δslip = slip / nsub
        local niter_k = niter
        for isub in 1:nsub
            if isub > 1
                δ, niter_k, Ψ, stats = _solve_filled!(δ)
            end
            @inbounds for i in eachindex(st.w)
                st.w[i] += abs(st.pn[i]) * Δslip
            end
        end
        δ, niter_k, Ψ, stats = _solve_filled!(δ)
        _record!(k + 1, k * Δs, δ, stats)
        get(ENV, "WEAR_VERBOSE", "false") == "true" &&
            @printf "  wear step %d  s=%.1f mm  wmax=%.4f um  P=%.3f N  pmax=%.2f  pmean=%.2f  a=%.3f mm  δ=%.4f um  niter=%d nsub=%d\n" k hist_s[k + 1] 1e3 * hist_w[k + 1] stats.P stats.pmax stats.pmean stats.a 1e3 * δ niter_k nsub
    end
    return (; wmax=hist_w, pmax=hist_pmax, pmean=hist_pmean, a=hist_a,
            P=hist_P, s=hist_s, δ=hist_δ, snaps=snaps)
end

"""
Gross-sliding wear loop (pin-on-disc). Each step imposes rigid slip `Δs` along `x`.
`wear_jump` scales both the slip used for wear and the number of represented steps.
"""
function sliding_wear_steps!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    δ::Real,
    Δs::Real,
    nsteps::Int;
    wear_jump=1.0,
    commit=true,
    kwargs...,
)
    hist_w = zeros(nsteps)
    hist_p = zeros(nsteps)
    hist_a = zeros(nsteps)
    gx = 0.0
    for k in 1:nsteps
        gx += Δs * wear_jump
        solve_contact_step!(st, grid, prep, law, δ, gx, 0.0; wear_jump=1.0, kwargs...)
        if commit
            commit_tangential_ref!(st)
        end
        hist_w[k] = maximum(st.w)
        hist_p[k] = maximum(st.pn)
        # contact half-width along x through y=0
        jc = (length(grid.y) + 1) ÷ 2
        ic = findall(i -> st.pn[i, jc] > 0, 1:length(grid.x))
        hist_a[k] = isempty(ic) ? 0.0 : (grid.x[maximum(ic)] - grid.x[minimum(ic)]) / 2
    end
    return (; wmax=hist_w, pmax=hist_p, a=hist_a)
end

end # module
