"""
    RollingContact

Juliá Lerma (2025) Ch. 2.2 / Paper 3 steady rolling on the same half-space grid:
Kalker slip, elliptic Coulomb, circumferential Archard groove (Paper 3 eqs. 16, 33).
"""
module RollingContact

using LinearAlgebra
using ..ContactHalfSpace
using ..OrthotropicUzawa

export RollingKinematics, solve_rolling_step!, rolling_match_load!
export creepage_velocity, upwind_slip!, rolling_pass_wear!, apply_groove_wear!

"""Steady rolling kinematics: creepage `(ξx, ξy, φ)` and rolling speed `V`."""
struct RollingKinematics{T}
    V::T
    ξx::T
    ξy::T
    φ::T
end

RollingKinematics(V, ξx, ξy=0, φ=0) =
    RollingKinematics(promote(float(V), float(ξx), float(ξy), float(φ))...)

function creepage_velocity(kin::RollingKinematics, x, y)
    sx = kin.V * (kin.ξx - kin.φ * y)
    sy = kin.V * (kin.ξy + kin.φ * x)
    return sx, sy
end

"""First-order upwind `Dr ut` on each `y`-row (thesis 2.62). `x` increases with `i`."""
function upwind_slip!(sx, sy, ux, uy, x, y, kin::RollingKinematics)
    nx, ny = size(ux)
    @inbounds for j in 1:ny
        Δx = nx > 1 ? x[2] - x[1] : 1.0
        for i in 1:nx
            cx, cy = creepage_velocity(kin, x[i], y[j])
            if i < nx
                dux = (ux[i + 1, j] - ux[i, j]) / Δx
                duy = (uy[i + 1, j] - uy[i, j]) / Δx
            else
                dux = 0.0
                duy = 0.0
            end
            sx[i, j] = cx - kin.V * dux
            sy[i, j] = cy - kin.V * duy
        end
    end
    return sx, sy
end

"""
    rolling_pass_wear!(Ipass, sx, sy, st, grid, law, kin) -> Ipass

Per-revolution Archard integral along each `y`-row (Paper 3 eqs. 16, 19, 33):

```
Ipass(y) = ∫ |p_n| ‖s‖_i dx / V
```

A twin-disc material ring at station `y` crosses the patch once per revolution
and picks up `Ipass(y)` uniformly around the circumference (a groove, not an
Eulerian wedge). `sx, sy` are overwritten with the current Kalker slip.
"""
function rolling_pass_wear!(Ipass, sx, sy, st::ContactState, grid::HalfSpaceGrid,
                           law::OrthotropicLaw, kin::RollingKinematics)
    ny = length(grid.y)
    Δx = grid.hs.hx
    cβ, sβ = cos(law.β), sin(law.β)
    upwind_slip!(sx, sy, st.ux, st.uy, grid.x, grid.y, kin)
    V = max(kin.V, eps())
    @inbounds for j in 1:ny
        acc = 0.0
        for i in 1:length(grid.x)
            s_i = OrthotropicUzawa.orthotropic_slip_norm(sx[i, j], sy[i, j],
                                                         law.i1, law.i2, cβ, sβ)
            acc += abs(st.pn[i, j]) * s_i * Δx / V
        end
        Ipass[j] = acc
    end
    return Ipass
end

"""Add `wear_jump * Ipass[j]` to every `x` on row `j` (circumferential groove)."""
function apply_groove_wear!(st::ContactState, Ipass, wear_jump=1.0)
    nx, ny = size(st.w)
    @inbounds for j in 1:ny
        dw = wear_jump * Ipass[j]
        iszero(dw) && continue
        for i in 1:nx
            st.w[i, j] += dw
        end
    end
    return st
end

"""
    solve_rolling_step!(state, grid, prep, law, kin, δ; kwargs...)

One revolution of steady rolling (thesis 2.2.7). Tangential projection uses
Kalker creepage ``w = s/V = ξ − φ×r − ∂u/∂x`` (CONTACT `cksi`, dimensionless).
`upwind_slip!` still returns the slip velocity `s = V w` for Archard wear.
Wear is applied **after** the contact residual has converged so a cycle jump
does not flatten `p_n` inside the same Uzawa loop.
"""
function solve_rolling_step!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    kin::RollingKinematics,
    δ::Real;
    rn=nothing,
    rt=nothing,
    tol=1e-8,
    maxiter=400,
    wear=true,
    wear_jump=1.0,
    relax=0.35,
    johnson=false,
)
    hs = grid.hs
    nx, ny = length(grid.x), length(grid.y)
    Azz0 = influence_coeff(Kzz, 0, 0, hs)
    rn_d, rt_d = default_penalties(hs)
    rn_ = rn === nothing ? rn_d : float(rn)
    rt_ = rt === nothing ? rt_d : float(rt)
    cβ, sβ = cos(law.β), sin(law.β)
    tmp = zeros(Float64, nx, ny)
    sx = zeros(Float64, nx, ny)
    sy = zeros(Float64, nx, ny)
    frictionless = iszero(law)
    winkler_seed!(st, grid, δ, Azz0)
    V = max(kin.V, eps())

    niter = 0
    Ψ = Inf
    # `johnson`: CONTACT MAXOUT=1 — frictionless normal, then T with frozen p_n.
    do_t = !(johnson && !frictionless)  # coupled (or μ=0): update T in the first loop
    for phase in 1:(johnson && !frictionless ? 2 : 1)
        freeze_n = phase == 2
        update_t = freeze_n || do_t
        if phase == 2
            fill!(st.ptx, 0)
            fill!(st.pty, 0)
        end
        Ψ = Inf
        for _ in 1:maxiter
            niter += 1
            fc_displacements!(st.ux, st.uy, st.uz, st.ptx, st.pty, st.pn, prep; tmp=tmp)
            upwind_slip!(sx, sy, st.ux, st.uy, grid.x, grid.y, kin)
            Ψn = 0.0
            Ψt = 0.0
            @inbounds for j in 1:ny, i in 1:nx
                gn = grid.gap_geom[i, j] - δ + st.w[i, j] + st.uz[i, j]
                st.gn[i, j] = gn
                if !freeze_n
                    pn_old = st.pn[i, j]
                    pn_proj = max(0.0, pn_old - rn_ * gn)
                    pn_new = (1 - relax) * pn_old + relax * pn_proj
                    Ψn += (pn_new - pn_old)^2
                    st.pn[i, j] = pn_new
                else
                    pn_new = st.pn[i, j]
                end
                if update_t && !frictionless
                    pn_t = pn_new == 0 ? zero(pn_new) : pn_new
                    px_new, py_new, dΨt = uzawa_tangential(st.ptx[i, j], st.pty[i, j],
                                                           sx[i, j] / V, sy[i, j] / V,
                                                           pn_t, law, cβ, sβ, rt_, relax)
                    Ψt += dΨt
                    st.ptx[i, j] = px_new
                    st.pty[i, j] = py_new
                else
                    st.ptx[i, j] = 0
                    st.pty[i, j] = 0
                end
            end
            Ψ = sqrt(Ψn) + sqrt(Ψt)
            Ψ <= tol && break
        end
    end
    if wear && (law.i1 != 0 || law.i2 != 0) && wear_jump != 0
        Ipass = zeros(Float64, ny)
        rolling_pass_wear!(Ipass, sx, sy, st, grid, law, kin)
        apply_groove_wear!(st, Ipass, wear_jump)
    end
    return niter, Ψ
end

"""Adjust rigid approach `δ` so the rolling contact load equals `P_target`.

Fresh spherical contact uses Hertz ``P ∝ extra^{3/2}``. Once wear is a
non-trivial fraction of the approach the scar is punch-like and the Newton
step uses Sneddon ``dP/dδ ≈ 2 E^* a`` (same switch as [`match_load!`](@ref)).
Hertz slope on a flattened groove overshoots `δ` and seeds a checkerboard `p_n`.
"""
function rolling_match_load!(
    st::ContactState,
    grid::HalfSpaceGrid,
    prep,
    law::OrthotropicLaw,
    kin::RollingKinematics,
    P_target::Real,
    δ::Real;
    rtol=5e-3,
    maxouter=20,
    kwargs...,
)
    niter, Ψ = 0, Inf
    Estar = contact_modulus(grid.hs)
    hx = grid.hs.hx
    for _ in 1:maxouter
        niter, Ψ = solve_rolling_step!(st, grid, prep, law, kin, δ; wear=false, kwargs...)
        stats = contact_patch_stats(st, grid)
        err = stats.P - P_target
        abs(err) <= rtol * max(abs(P_target), eps()) && return δ, niter, Ψ
        wmax = maximum(st.w)
        extra = max(δ - wmax, 10 * eps(Float64))
        if wmax > 0.2 * extra
            a = max(stats.a, stats.a_outer, hx)
            kn = max(2 * Estar * a, Estar * hx)
            δ = max(δ - 0.7 * err / kn, wmax + 1e-12)
        else
            slope = 1.5 * max(stats.P, 0.05 * abs(P_target)) / extra
            extra = max(extra - 0.6 * err / slope, 10 * eps(Float64))
            δ = wmax + extra
        end
    end
    niter, Ψ = solve_rolling_step!(st, grid, prep, law, kin, δ; wear=false, kwargs...)
    return δ, niter, Ψ
end

end # module
