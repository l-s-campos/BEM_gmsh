"""
    JuliaLermaCases

Shared drivers for Juliá Lerma (2025) Ch. 3 examples. Units: N, mm, MPa.
"""
module JuliaLermaCases

using LinearAlgebra
using ..ContactHalfSpace
using ..OrthotropicUzawa
using ..SubsurfaceStress
using ..RollingContact

export pin_hertz, pin_wear, pin_friction, pin_orthotropic
export spherical_fretting, flat_punch_static, flat_punch_cycle
export rolling_spheres, twin_discs
export mossakovskii_ratio, square_mesh, rect_mesh
export PIN, FRET, PUNCH, ROLL, DISC

const PIN = (
    E = 2.10e5, ν = 0.3, R = 50.0, δ = 4.5e-4, L = 1.6,
    i = 1.33e-7, i2 = 2.66e-7, μ1 = 0.25, μ2 = 0.50,
)
const FRET = (
    E = 2.10e5, ν = 0.3, R = 50.0, δ = 8e-3, amp = 4e-3, L = 1.6,
    μ1 = 0.10, μ2 = 0.65, i1 = 1.330e-7, i2 = 8.645e-7,
)
const PUNCH = (
    E_A = 2.00e5, E_B = 2.00e7, ν = 0.3, a0 = 1.8, P = 750.0, L = 4.5,
    i = 1.33e-7, μ1 = 0.1, μ2 = 0.4, i2 = 5.32e-7,
)
# Paper 3 prints L, Lx, Ly as half-widths (computational domain ±L).
# `square_mesh` / `rect_mesh` take the full width, so drivers use `2L`.
const ROLL = (
    G = 1.0, ν = 0.28, R = 337.5, P = 0.4705, L = 4.08,
    μ = 0.4013, μ1 = 0.40, μ2 = 0.20, ξx = -0.0031,
)
const DISC = (
    E = 2.08e5, ν = 0.3, P = 300.0,
    RAx = 32.5, RAy = 32.5, RBx = 32.3, RBy = Inf,
    Lx = 0.35, Ly = 1.4, ξx = -0.005,
    μ1 = 0.60, μ2 = 0.30, i1 = 2.0e-6, i2 = 1.0e-6,
    ωrpm = 300.0,
)

function square_mesh(N, L, G_A, ν_A, G_B, ν_B)
    hx = L / N
    hs = combined_halfspace(G_A, ν_A, G_B, ν_B; hx=hx, hy=hx)
    x = collect(range(-L / 2 + hx / 2, stop=L / 2 - hx / 2, length=N))
    return x, hs
end

function rect_mesh(nx, ny, Lx, Ly, G_A, ν_A, G_B, ν_B)
    hx, hy = Lx / nx, Ly / ny
    hs = combined_halfspace(G_A, ν_A, G_B, ν_B; hx=hx, hy=hy)
    x = collect(range(-Lx / 2 + hx / 2, stop=Lx / 2 - hx / 2, length=nx))
    y = collect(range(-Ly / 2 + hy / 2, stop=Ly / 2 - hy / 2, length=ny))
    return x, y, hs
end

# Popov, Handbook of Contact Mechanics, eq. (2.141):
# kn_adhesion / kn_frictionless = (1−ν) ln(3−4ν) / (1−2ν)
mossakovskii_ratio(ν) = (1 - ν) * log(3 - 4ν) / (1 - 2ν)

function _pin_hs(N; L=PIN.L)
    G = G_from_E(PIN.E, PIN.ν)
    return square_mesh(N, L, G, PIN.ν, G, PIN.ν)
end

"""Frictionless pin-on-disc Hertz (thesis 3.1, s = 0)."""
function pin_hertz(N; L=0.6, tol=1e-8, maxiter=400)
    x, hs = _pin_hs(N; L=L)
    grid = make_grid(x, x, hs, sphere_gap(x, x, PIN.R))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    niter, Ψ = solve_contact_step!(st, grid, prep, isotropic_law(0.0), PIN.δ, 0.0, 0.0;
                                   tol=tol, maxiter=maxiter)
    hz = hertz_sphere(PIN.R, PIN.δ, contact_modulus(hs))
    P, _, _ = contact_resultants(st, hs)
    z = 0.48 * hz.a
    _, _, σVM_ana = hertz_axis_stress(z, hz.a, hz.p0, PIN.ν)
    σ = subsurface_stress(0.0, 0.0, z, st.ptx, st.pty, st.pn, x, x, hs, PIN.ν)
    return (; N, P, pmax=maximum(st.pn), σVM=σ.VM, hz, σVM_ana, niter, Ψ,
            errP=abs(P - hz.P) / hz.P,
            errp=abs(maximum(st.pn) - hz.p0) / hz.p0,
            errVM=abs(σ.VM / hz.p0 - σVM_ana / hz.p0),
            st, grid, prep, x, hs)
end

"""Isotropic sliding wear vs Argatov."""
function pin_wear(N; L=PIN.L, Δs=5.0, nsteps=4, μ=0.0, i=PIN.i, tol=1e-6)
    x, hs = _pin_hs(N; L=L)
    grid = make_grid(x, x, hs, sphere_gap(x, x, PIN.R))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    hz = hertz_sphere(PIN.R, PIN.δ, contact_modulus(hs))
    hist = sliding_wear_steps!(st, grid, prep, isotropic_law(μ, i), PIN.δ, Δs, nsteps;
                               tol=tol, maxiter=250)
    s = nsteps * Δs
    a_arg = argatov_worn_radius(hz.a, i, hz.P, s, PIN.R)
    w_arg = a_arg^2 / (2PIN.R)
    w = hist.wmax[end]
    return (; N, w, w_arg, a=hist.a[end], a_arg, s, hz,
            errw=abs(w - w_arg) / max(w_arg, eps()), st, hist)
end

"""Isotropic frictional pin, no wear (μ > 0)."""
function pin_friction(N; μ=0.25, L=0.6, tol=1e-7)
    x, hs = _pin_hs(N; L=L)
    grid = make_grid(x, x, hs, sphere_gap(x, x, PIN.R))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    law = isotropic_law(μ, 0.0)
    niter, Ψ = solve_contact_step!(st, grid, prep, law, PIN.δ, 0.0, 0.0; tol=tol)
    # small tangential shift so friction is activated
    commit_tangential_ref!(st)
    solve_contact_step!(st, grid, prep, law, PIN.δ, 1e-4, 0.0; tol=tol)
    P, Qx, Qy = contact_resultants(st, hs)
    hz = hertz_sphere(PIN.R, PIN.δ, contact_modulus(hs))
    z = 0.48 * hz.a
    σ = subsurface_stress(0.0, 0.0, z, st.ptx, st.pty, st.pn, x, x, hs, PIN.ν)
    return (; N, P, Qx, Qy, pmax=maximum(st.pn), σVM=σ.VM, hz, niter, Ψ, st)
end

"""Orthotropic pin sliding (θ = −β)."""
function pin_orthotropic(N; β=π/4, L=PIN.L, Δs=2.0, nsteps=3, tol=1e-6)
    x, hs = _pin_hs(N; L=L)
    grid = make_grid(x, x, hs, sphere_gap(x, x, PIN.R))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    law = OrthotropicLaw(PIN.μ1, PIN.μ2, PIN.i, PIN.i2, β)
    hist = sliding_wear_steps!(st, grid, prep, law, PIN.δ, Δs, nsteps; tol=tol, maxiter=250)
    P, Qx, Qy = contact_resultants(st, hs)
    return (; N, P, Qx, Qy, w=hist.wmax[end], β, st)
end

"""Spherical punch fretting: normal indent then one tangential half-cycle."""
function spherical_fretting(N; β=0.0, L=FRET.L, wear_jump=1.0, tol=1e-6)
    G = G_from_E(FRET.E, FRET.ν)
    x, hs = square_mesh(N, L, G, FRET.ν, G, FRET.ν)
    grid = make_grid(x, x, hs, sphere_gap(x, x, FRET.R))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    law = OrthotropicLaw(FRET.μ1, FRET.μ2, FRET.i1, FRET.i2, β)
    solve_contact_step!(st, grid, prep, law, FRET.δ, 0.0, 0.0; tol=tol, wear_jump=0)
    commit_tangential_ref!(st)
    niter, Ψ = solve_contact_step!(st, grid, prep, law, FRET.δ, FRET.amp, 0.0;
                                   tol=tol, wear_jump=wear_jump)
    P, Qx, Qy = contact_resultants(st, hs)
    n_contact = count(>(0), st.pn)
    cβ, sβ = cos(β), sin(β)
    n_slip = count(eachindex(st.pn)) do i
        st.pn[i] <= 0 && return false
        pe1, pe2 = rotate_to_tribological(st.ptx[i], st.pty[i], cβ, sβ)
        return hypot(pe1 / FRET.μ1, pe2 / FRET.μ2) > 0.95 * st.pn[i]
    end
    return (; N, P, Qx, Qy, w=maximum(st.w), n_contact, n_slip, niter, Ψ, β, st)
end

"""Frictionless (or frictional) flat punch under load P (Sneddon / Mossakovskii)."""
function flat_punch_static(N; μ=0.0, L=PUNCH.L, tol=1e-6)
    GA = G_from_E(PUNCH.E_A, PUNCH.ν)
    GB = G_from_E(PUNCH.E_B, PUNCH.ν)
    x, hs = square_mesh(N, L, GA, PUNCH.ν, GB, PUNCH.ν)
    grid = make_grid(x, x, hs, flat_punch_gap(x, x, PUNCH.a0; out=10.0))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    Estar = contact_modulus(hs)
    δ0 = PUNCH.P / (2 * Estar * PUNCH.a0)
    law = isotropic_law(μ, 0.0)
    δ, niter, Ψ, _ = match_load!(st, grid, prep, law, PUNCH.P, δ0;
                                 tol=tol, rtol=5e-3, wear_jump=0, maxouter=30)
    P, Qx, Qy = contact_resultants(st, hs)
    kn = P / max(δ, eps())
    kn0 = 2 * Estar * PUNCH.a0
    jc = (N + 1) ÷ 2
    return (; N, P, δ, kn, kn0, kn_ratio=kn / kn0, pmax=maximum(st.pn),
            pcen=st.pn[jc, jc], niter, Ψ, moss=mossakovskii_ratio(PUNCH.ν), st)
end

"""One load–unload cycle of the flat punch (radial fretting wear)."""
function flat_punch_cycle(N; μ=0.2, L=PUNCH.L, tol=1e-6)
    GA = G_from_E(PUNCH.E_A, PUNCH.ν)
    GB = G_from_E(PUNCH.E_B, PUNCH.ν)
    x, hs = square_mesh(N, L, GA, PUNCH.ν, GB, PUNCH.ν)
    grid = make_grid(x, x, hs, flat_punch_gap(x, x, PUNCH.a0; out=10.0))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    Estar = contact_modulus(hs)
    δ0 = PUNCH.P / (2 * Estar * PUNCH.a0)
    law = isotropic_law(μ, PUNCH.i)
    δ, _, _ = set_approach_for_load!(st, grid, prep, law, PUNCH.P, 0.0, 0.0;
                                     δ0=δ0, tol=tol, rtol=5e-3, wear_jump=1.0, maxouter=30)
    commit_tangential_ref!(st)
    solve_contact_step!(st, grid, prep, law, 0.0, 0.0, 0.0; tol=tol, wear_jump=1.0)
    P, _, _ = contact_resultants(st, hs)
    return (; N, w=maximum(st.w), P_unload=P, δ, st)
end

"""Two identical spheres, tractive rolling, no wear."""
function rolling_spheres(N; β=0.0, μiso=ROLL.μ, L=2 * ROLL.L, tol=1e-6)
    x, hs = square_mesh(N, L, ROLL.G, ROLL.ν, ROLL.G, ROLL.ν)
    grid = make_grid(x, x, hs, sphere_gap(x, x, ROLL.R / 2))
    prep = precompute_kernels(N, N, hs)
    hz = hertz_sphere_load(ROLL.R / 2, ROLL.P, contact_modulus(hs))
    law = β == 0 && μiso !== nothing ? isotropic_law(μiso, 0.0) :
          OrthotropicLaw(ROLL.μ1, ROLL.μ2, 0.0, 0.0, β)
    st = init_state(grid)
    δ, _, _ = set_approach_for_load!(st, grid, prep, law, ROLL.P, 0.0, 0.0;
                                     δ0=hz.δ, tol=tol, wear_jump=0, maxouter=25)
    kin = RollingKinematics(1.0, ROLL.ξx, 0.0, 0.0)
    niter, Ψ = solve_rolling_step!(st, grid, prep, law, kin, δ; wear=false, tol=tol)
    P, Qx, Qy = contact_resultants(st, hs)
    return (; N, P, Qx, Qy, δ, hz, niter, Ψ,
            errP=abs(P - ROLL.P) / ROLL.P, Qx_over_μP=Qx / (ROLL.μ1 * P), st)
end

"""Twin-discs, one (or `nrev`) revolutions with wear."""
function twin_discs(N; β=0.0, nrev=1, nx=N, ny=3N, tol=1e-6,
                    Lx=2 * DISC.Lx, Ly=2 * DISC.Ly)
    G = G_from_E(DISC.E, DISC.ν)
    x, y, hs = rect_mesh(nx, ny, Lx, Ly, G, DISC.ν, G, DISC.ν)
    Rx = 1 / (1 / DISC.RAx + 1 / DISC.RBx)
    Ry = DISC.RAy   # B is cylindrical, RBy = ∞
    gap = [(xi^2 / (2Rx) + yj^2 / (2Ry)) for xi in x, yj in y]
    grid = make_grid(x, y, hs, gap)
    prep = precompute_kernels(nx, ny, hs)
    law = OrthotropicLaw(DISC.μ1, DISC.μ2, DISC.i1, DISC.i2, β)
    st = init_state(grid)
    hz = hertz_sphere_load(Rx, DISC.P, contact_modulus(hs))  # order-of-magnitude δ seed
    δ, _, _ = set_approach_for_load!(st, grid, prep, law, DISC.P, 0.0, 0.0;
                                     δ0=max(hz.δ, 1e-4), tol=tol, wear_jump=0, maxouter=20)
    V = DISC.ωrpm * 2π / 60 * DISC.RAx
    kin = RollingKinematics(V, DISC.ξx, 0.0, 0.0)
    niter, Ψ = 0, Inf
    for _ in 1:nrev
        niter, Ψ = solve_rolling_step!(st, grid, prep, law, kin, δ; wear=true, tol=tol)
    end
    P, Qx, Qy = contact_resultants(st, hs)
    return (; N=nx, ny, P, Qx, Qy, w=maximum(st.w), niter, Ψ,
            errP=abs(P - DISC.P) / DISC.P, st)
end

end # module
