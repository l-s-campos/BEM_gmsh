# =============================================================================
# Mass-conserving Reynolds (Elrod–Adams p–θ / JFO) on structured grids
# =============================================================================
# Profito, Giacopini, Zachariadis & Dini, Tribol. Lett. 60:18 (2015) §4.1.
#
#   ∇ · (Γ ∇p) = ∇ · (θ ρ h 𝐯) + ρ ∂(θ h)/∂t ,
#   Γ = ρ h³ / (12 μ) ,   (p − p_cav)(1 − θ) = 0 .
#
# Full-film incompressible Reynolds (Guiggiani / DIBEM) is recovered when
# θ ≡ 1 and μ, ρ constant. Cavitation, compressibility and piezoviscosity
# make the problem a complementarity convection–diffusion system; it is
# discretized here with a conservative finite-volume / Ausas sweep rather
# than Laplace FS + DIBEM.

export DowsonHigginson, Barus, EyringCouette, ConstRheology, eval_rho_mu
export ElrodOptions, solve_elrod_1d, solve_elrod_1d!, solve_elrod_1d_periodic!
export solve_elrod_2d!, solve_elrod_radial!
export film_parabolic_slider, film_double_parabolic, film_journal, film_pocket
export profito_single_slider, profito_double_slider, profito_journal
export profito_squeeze, profito_pocket

# ---------------------------------------------------------------------------
# Rheology
# ---------------------------------------------------------------------------

"""Dowson–Higginson `ρ/ρ0 = (C1 + C2 p) / (C1 + p)` (`C1` in Pa, `C2` dimless)."""
struct DowsonHigginson
    ρ0::Float64
    C1::Float64
    C2::Float64
end
DowsonHigginson(; ρ0=850.0, C1=2.22e9, C2=1.66) = DowsonHigginson(ρ0, C1, C2)

"""Barus `μ = μ0 exp(α p)` (`α` in 1/Pa)."""
struct Barus
    μ0::Float64
    α::Float64
end
Barus(; μ0=0.01, α=0.0) = Barus(μ0, α)

"""Local Eyring correction on Couette shear `U/h`: `μ ← τ0 asinh(μ U/(h τ0)) * h / U`."""
struct EyringCouette
    τ0::Float64
end

struct ConstRheology
    ρ::Float64
    μ::Float64
end
ConstRheology(; ρ=850.0, μ=0.01) = ConstRheology(ρ, μ)

function eval_rho_mu(p::Real, h::Real, U::Real, rheo::ConstRheology)
    return rheo.ρ, rheo.μ
end
function eval_rho_mu(p::Real, h::Real, U::Real, rheo::NamedTuple)
    ρ = rheo.ρ0 * (rheo.C1 + rheo.C2 * p) / (rheo.C1 + p)
    μ = rheo.μ0 * exp(rheo.α * p)
    if haskey(rheo, :τ0) && rheo.τ0 > 0 && h > 0 && abs(U) > 0
        γ = abs(U) / h
        z = μ * γ / rheo.τ0
        if z > 1e-8
            μ = rheo.τ0 * asinh(z) / γ
        end
    end
    return ρ, μ
end

_default_rheo(; ρ0=850.0, μ0=0.01, C1=Inf, C2=1.66, α=0.0, τ0=0.0) =
    (ρ0=ρ0, μ0=μ0, C1=C1, C2=C2, α=α, τ0=τ0)

function _rho_mu(p, h, U, rheo)
    if rheo isa ConstRheology
        return rheo.ρ, rheo.μ
    elseif rheo isa NamedTuple
        C1 = rheo.C1
        if !isfinite(C1)
            ρ = rheo.ρ0
        else
            ρ = rheo.ρ0 * (C1 + rheo.C2 * p) / (C1 + p)
        end
        μ = rheo.μ0 * exp(rheo.α * p)
        τ0 = get(rheo, :τ0, 0.0)
        if τ0 > 0 && h > 0 && abs(U) > 0
            γ = abs(U) / h
            z = μ * γ / τ0
            z > 1e-8 && (μ = τ0 * asinh(z) / γ)
        end
        return ρ, μ
    else
        return eval_rho_mu(p, h, U, rheo)
    end
end

# ---------------------------------------------------------------------------
# Geometry helpers (Profito §4.1)
# ---------------------------------------------------------------------------

"""Single convergent–divergent parabola: `h(0)=h(L)=hmax`, `h(L/2)=hmin`."""
film_parabolic_slider(x, L, hmax, hmin) = hmin + (hmax - hmin) * (2x / L - 1)^2

"""Two parabolas in series on `[0, L]` (Profito double slider)."""
function film_double_parabolic(x, L, hmax, hmin)
    half = 0.5 * L
    ξ = x < half ? x : x - half
    return film_parabolic_slider(ξ, half, hmax, hmin)
end

"""Journal film `h = c (1 + ε cos(x/R))`, `x = R θ`, θ = 0 at maximum gap."""
film_journal(x, R, c, ε) = c * (1 + ε * cos(x / R))

"""Tapered slider with a rectangular pocket (Profito Table 4)."""
function film_pocket(x, a, c, Lp, hmax, hmin, Hp)
    hland = hmax + (hmin - hmax) * (x / a)
    return (c <= x <= c + Lp) ? hland + Hp : hland
end

# ---------------------------------------------------------------------------
# 1-D Elrod–Adams (infinite / long bearing)
# ---------------------------------------------------------------------------

Base.@kwdef struct ElrodOptions
    pcav::Float64 = 0.0
    ωp::Float64 = 0.9
    ωθ::Float64 = 0.6
    maxiter::Int = 20_000
    tol::Float64 = 1e-8
    verbose::Bool = false
end

"""
    solve_elrod_1d!(p, θ, h, dx; U, rheo, opt, hold=nothing, dt=Inf)

Node-centered 1-D FVM, Dirichlet `p[1]`, `p[end]` already set.
`U > 0` (flow to the right); Couette is first-order upwind in `θ`.
"""
function solve_elrod_1d!(p::AbstractVector, θ::AbstractVector, h::AbstractVector, dx::Real;
        U::Real, rheo, opt::ElrodOptions=ElrodOptions(),
        hold=nothing, θold=nothing, dt::Real=Inf)
    n = length(p)
    n == length(θ) == length(h) || throw(DimensionMismatch("p, θ, h length"))
    n >= 3 || throw(ArgumentError("need ≥ 3 nodes"))
    dx = float(dx)
    squeeze = isfinite(dt) && hold !== nothing
    pcav = opt.pcav
    cav = falses(n)
    ptry = copy(p)
    dpmax = 0.0
    it = 0
    @inbounds for iter in 1:opt.maxiter
        it = iter
        # unknown interior nodes (not cavitated)
        idx = Int[]
        for i in 2:(n - 1)
            cav[i] || push!(idx, i)
        end
        m = length(idx)
        if m == 0
            break
        end
        loc = fill(0, n)
        for (k, i) in enumerate(idx)
            loc[i] = k
        end
        A = zeros(m, m)
        rhs = zeros(m)
        for (k, i) in enumerate(idx)
            hf = 0.5 * (h[i] + h[i + 1])
            hb = 0.5 * (h[i] + h[i - 1])
            pf = 0.5 * (p[i] + p[i + 1])
            pb = 0.5 * (p[i] + p[i - 1])
            ρf, μf = _rho_mu(pf, hf, U, rheo)
            ρb, μb = _rho_mu(pb, hb, U, rheo)
            Γf = ρf * hf^3 / (12 * μf)
            Γb = ρb * hb^3 / (12 * μb)
            ρc, _ = _rho_mu(p[i], h[i], U, rheo)
            θf = 1.0
            θb = cav[i - 1] ? θ[i - 1] : 1.0
            coup = (ρf * U / 2) * dx * (θf * hf) - (ρb * U / 2) * dx * (θb * hb)
            sq = 0.0
            if squeeze
                ho = hold[i]
                to = θold === nothing ? θ[i] : θold[i]
                sq = ρc * dx^2 * (1.0 * h[i] - to * ho) / dt
            end
            A[k, k] = Γf + Γb
            rhs[k] = -coup - sq
            # neighbour i-1
            if i - 1 == 1 || cav[i - 1]
                rhs[k] += Γb * (cav[i - 1] ? pcav : p[1])
            else
                A[k, loc[i - 1]] -= Γb
            end
            # neighbour i+1
            if i + 1 == n || cav[i + 1]
                rhs[k] += Γf * (cav[i + 1] ? pcav : p[n])
            else
                A[k, loc[i + 1]] -= Γf
            end
        end
        @inbounds for k in 1:m
            A[k, k] += 1e-14 * abs(A[k, k])
        end
        sol = A \ rhs
        ptry .= p
        for (k, i) in enumerate(idx)
            ptry[i] = sol[k]
        end
        for i in 2:(n - 1)
            cav[i] && (ptry[i] = pcav)
        end
        dpmax = maximum(abs, ptry .- p)
        p .= ptry
        # complementarity update
        changed = false
        for i in 2:(n - 1)
            if cav[i]
                hf = 0.5 * (h[i] + h[i + 1])
                hb = 0.5 * (h[i] + h[i - 1])
                ρf, μf = _rho_mu(pcav, hf, U, rheo)
                ρb, μb = _rho_mu(pcav, hb, U, rheo)
                Γf = ρf * hf^3 / (12 * μf)
                Γb = ρb * hb^3 / (12 * μb)
                ρc, _ = _rho_mu(pcav, h[i], U, rheo)
                coef_θ = (ρf * U / 2) * dx * hf
                coup_wo = -(ρb * U / 2) * dx * ((cav[i - 1] ? θ[i - 1] : 1.0) * hb)
                sq_wo = 0.0
                if squeeze
                    ho = hold[i]
                    to = θold === nothing ? θ[i] : θold[i]
                    sq_wo = ρc * dx^2 * (-to * ho) / dt
                    coef_θ += ρc * dx^2 * h[i] / dt
                end
                lhs = Γf * (p[i + 1] - pcav) - Γb * (pcav - p[i - 1])
                θstar = abs(coef_θ) > 1e-30 ? (lhs - coup_wo - sq_wo) / coef_θ : θ[i]
                if θstar >= 1.0
                    cav[i] = false
                    θ[i] = 1.0
                    changed = true
                else
                    θ[i] = clamp(θstar, 0.0, 1.0)
                end
            elseif p[i] < pcav
                cav[i] = true
                p[i] = pcav
                θ[i] = 1.0
                changed = true
            else
                θ[i] = 1.0
            end
        end
        θ[1] = 1.0
        if !changed && dpmax < opt.tol * max(maximum(abs, p), 1.0)
            break
        end
    end
    opt.verbose && println("  Elrod 1D  iter=", it, "  Δp=", dpmax, "  n_cav=", count(cav))
    return p, θ
end

"""Periodic 1-D (infinitely long journal). `p` is fixed by cavitation, not a Dirichlet value."""
function solve_elrod_1d_periodic!(p::AbstractVector, θ::AbstractVector, h::AbstractVector, dx::Real;
        U::Real, rheo, opt::ElrodOptions=ElrodOptions())
    n = length(p)
    n == length(θ) == length(h) || throw(DimensionMismatch("p, θ, h"))
    wrap(i) = mod(i - 1, n) + 1
    cav = falses(n)
    pcav = opt.pcav
    dpmax = 0.0
    it = 0
    ptry = copy(p)
    @inbounds for iter in 1:opt.maxiter
        it = iter
        idx = [i for i in 1:n if !cav[i]]
        m = length(idx)
        m == 0 && break
        loc = fill(0, n)
        for (k, i) in enumerate(idx)
            loc[i] = k
        end
        A = zeros(m, m)
        rhs = zeros(m)
        for (k, i) in enumerate(idx)
            ip = wrap(i + 1); im = wrap(i - 1)
            hf = 0.5 * (h[i] + h[ip]); hb = 0.5 * (h[i] + h[im])
            pf = 0.5 * (p[i] + p[ip]); pb = 0.5 * (p[i] + p[im])
            ρf, μf = _rho_mu(pf, hf, U, rheo)
            ρb, μb = _rho_mu(pb, hb, U, rheo)
            Γf = ρf * hf^3 / (12 * μf)
            Γb = ρb * hb^3 / (12 * μb)
            θb = cav[im] ? θ[im] : 1.0
            coup = (ρf * U / 2) * dx * hf - (ρb * U / 2) * dx * (θb * hb)
            A[k, k] = Γf + Γb
            rhs[k] = -coup
            if cav[im]
                rhs[k] += Γb * pcav
            else
                A[k, loc[im]] -= Γb
            end
            if cav[ip]
                rhs[k] += Γf * pcav
            else
                A[k, loc[ip]] -= Γf
            end
        end
        @inbounds for k in 1:m
            A[k, k] += 1e-14 * abs(A[k, k])
        end
        sol = try
            A \ rhs
        catch
            @inbounds for k in 1:m
                A[k, k] += 1e-8 * abs(A[k, k])
            end
            A \ rhs
        end
        ptry .= p
        for (k, i) in enumerate(idx)
            ptry[i] = sol[k]
        end
        for i in 1:n
            cav[i] && (ptry[i] = pcav)
        end
        dpmax = maximum(abs, ptry .- p)
        p .= ptry
        changed = false
        for i in 1:n
            ip = wrap(i + 1); im = wrap(i - 1)
            if cav[i]
                hf = 0.5 * (h[i] + h[ip]); hb = 0.5 * (h[i] + h[im])
                ρf, μf = _rho_mu(pcav, hf, U, rheo)
                ρb, μb = _rho_mu(pcav, hb, U, rheo)
                Γf = ρf * hf^3 / (12 * μf)
                Γb = ρb * hb^3 / (12 * μb)
                coef_θ = (ρf * U / 2) * dx * hf
                coup_wo = -(ρb * U / 2) * dx * ((cav[im] ? θ[im] : 1.0) * hb)
                lhs = Γf * (p[ip] - pcav) - Γb * (pcav - p[im])
                θstar = abs(coef_θ) > 1e-30 ? (lhs - coup_wo) / coef_θ : θ[i]
                if θstar >= 1
                    cav[i] = false; θ[i] = 1.0; changed = true
                else
                    θ[i] = clamp(θstar, 0.0, 1.0)
                end
            elseif p[i] < pcav
                cav[i] = true; p[i] = pcav; θ[i] = 1.0; changed = true
            else
                θ[i] = 1.0
            end
        end
        if !changed && dpmax < opt.tol * max(maximum(abs, p), 1.0)
            break
        end
    end
    opt.verbose && println("  Elrod 1D periodic  iter=", it, "  Δp=", dpmax, "  n_cav=", count(cav))
    return p, θ
end

function solve_elrod_1d(h::AbstractVector, dx::Real; U, rheo, pleft, pright,
        opt::ElrodOptions=ElrodOptions(), kwargs...)
    n = length(h)
    p = fill(max(pleft, pright, opt.pcav), n)
    p[1] = pleft
    p[n] = pright
    θ = ones(n)
    solve_elrod_1d!(p, θ, h, dx; U=U, rheo=rheo, opt=opt, kwargs...)
    return p, θ
end

# ---------------------------------------------------------------------------
# 2-D Cartesian Elrod–Adams
# ---------------------------------------------------------------------------

"""
    solve_elrod_2d!(p, θ, h, dx, dy; U, rheo, opt, bc, periodic_x=false, fixed=nothing)

Cell-node grid `(nx, ny)`. `bc` is a NamedTuple of Dirichlet pressures
`(:left, :right, :bottom, :top)` (`nothing` = no-flux). `periodic_x`
wraps the `i` index (journal). `fixed` is an optional `Bool` mask of
cells held at their current `p` (oil-supply patch).
"""
function solve_elrod_2d!(p::AbstractMatrix, θ::AbstractMatrix, h::AbstractMatrix,
        dx::Real, dy::Real;
        U::Real, rheo, opt::ElrodOptions=ElrodOptions(),
        bc=(left=0.0, right=0.0, bottom=nothing, top=nothing),
        periodic_x::Bool=false,
        fixed=nothing,
        hold=nothing, θold=nothing, dt::Real=Inf)
    nx, ny = size(p)
    size(θ) == (nx, ny) && size(h) == (nx, ny) || throw(DimensionMismatch("p, θ, h"))
    dx = float(dx); dy = float(dy)
    pcav = opt.pcav
    squeeze = isfinite(dt) && hold !== nothing
    function _apply_bc!()
        if !periodic_x
            bc.left !== nothing && (p[1, :] .= bc.left; θ[1, :] .= 1.0)
            bc.right !== nothing && (p[nx, :] .= bc.right)
        end
        bc.bottom !== nothing && (p[:, 1] .= bc.bottom; θ[:, 1] .= 1.0)
        bc.top !== nothing && (p[:, ny] .= bc.top; θ[:, ny] .= 1.0)
        return nothing
    end
    _apply_bc!()
    wrap(i) = periodic_x ? (mod(i - 1, nx) + 1) : i
    i0 = periodic_x ? 1 : 2
    i1 = periodic_x ? nx : nx - 1
    j0 = bc.bottom === nothing ? 1 : 2
    j1 = bc.top === nothing ? ny : ny - 1
    dpmax = 0.0
    it = 0
    @inbounds for iter in 1:opt.maxiter
        it = iter
        dpmax = 0.0
        for j in j0:j1, i in i0:i1
            fixed !== nothing && fixed[i, j] && continue
            ie = wrap(i + 1)
            iw = wrap(i - 1)
            jn = min(j + 1, ny)
            js = max(j - 1, 1)
            hf = 0.5 * (h[i, j] + h[ie, j])
            hb = 0.5 * (h[i, j] + h[iw, j])
            hn = 0.5 * (h[i, j] + h[i, jn])
            hs = 0.5 * (h[i, j] + h[i, js])
            pf = 0.5 * (p[i, j] + p[ie, j])
            pb = 0.5 * (p[i, j] + p[iw, j])
            pn = 0.5 * (p[i, j] + p[i, jn])
            ps = 0.5 * (p[i, j] + p[i, js])
            ρf, μf = _rho_mu(pf, hf, U, rheo)
            ρb, μb = _rho_mu(pb, hb, U, rheo)
            ρn, μn = _rho_mu(pn, hn, 0.0, rheo)
            ρs, μs = _rho_mu(ps, hs, 0.0, rheo)
            Γf = ρf * hf^3 / (12 * μf)
            Γb = ρb * hb^3 / (12 * μb)
            Γn = ρn * hn^3 / (12 * μn)
            Γs = ρs * hs^3 / (12 * μs)
            # no-flux: drop the missing face
            if !periodic_x && i == 1
                Γb = 0.0
            end
            if !periodic_x && i == nx
                Γf = 0.0
            end
            if j == 1 && bc.bottom === nothing
                Γs = 0.0
            end
            if j == ny && bc.top === nothing
                Γn = 0.0
            end
            ax = dy / dx
            ay = dx / dy
            ke = Γf * ax; kw = Γb * ax
            kn = Γn * ay; ks = Γs * ay
            den = ke + kw + kn + ks
            den < 1e-30 && continue
            θw = θ[iw, j]
            ρc, _ = _rho_mu(p[i, j], h[i, j], U, rheo)
            to = (squeeze && θold !== nothing) ? θold[i, j] : θ[i, j]
            ho = squeeze ? hold[i, j] : h[i, j]
            coup1 = (ρf * U / 2) * dy * hf - (ρb * U / 2) * dy * (θw * hb)
            sq1 = squeeze ? ρc * dx * dy * (h[i, j] - to * ho) / dt : 0.0
            pstar = (ke * p[ie, j] + kw * p[iw, j] + kn * p[i, jn] + ks * p[i, js] -
                     coup1 - sq1) / den
            pold = p[i, j]
            if pstar >= pcav
                p[i, j] = (1 - opt.ωp) * p[i, j] + opt.ωp * pstar
                θ[i, j] = 1.0
            else
                p[i, j] = pcav
                coup_wo = -(ρb * U / 2) * dy * (θw * hb)
                coef_θ = (ρf * U / 2) * dy * hf
                sq_wo = 0.0
                if squeeze
                    ho = hold[i, j]
                    to = θold === nothing ? θ[i, j] : θold[i, j]
                    sq_wo = ρc * dx * dy * (-to * ho) / dt
                    coef_θ += ρc * dx * dy * h[i, j] / dt
                end
                lhs = ke * (p[ie, j] - pcav) + kn * (p[i, jn] - pcav) +
                      ks * (p[i, js] - pcav) - kw * (pcav - p[iw, j])
                if abs(coef_θ) > 1e-30
                    θstar = (lhs - coup_wo - sq_wo) / coef_θ
                else
                    θstar = θ[i, j]
                end
                θ[i, j] = clamp((1 - opt.ωθ) * θ[i, j] + opt.ωθ * θstar, 0.0, 1.0)
            end
            dpmax = max(dpmax, abs(p[i, j] - pold))
        end
        _apply_bc!()
        if dpmax < opt.tol * max(maximum(abs, p), 1.0)
            break
        end
    end
    opt.verbose && println("  Elrod 2D  iter=", it, "  Δp=", dpmax)
    return p, θ
end

# ---------------------------------------------------------------------------
# Axisymmetric squeeze (circular plates)
# ---------------------------------------------------------------------------

"""
    solve_elrod_radial!(p, θ, r, h; μ, ρ, pcav, hold, θold, dt, pouter)

Radial FVM, `r[1]≈0` (symmetry), Dirichlet `p[end] = pouter`.
Pure squeeze: no Couette.
"""
function solve_elrod_radial!(p::AbstractVector, θ::AbstractVector, r::AbstractVector, h::Real;
        μ::Real, ρ::Real, opt::ElrodOptions=ElrodOptions(),
        hold::Real, θold::AbstractVector, dt::Real, pouter::Real)
    n = length(p)
    n == length(θ) == length(r) || throw(DimensionMismatch("p, θ, r"))
    p[n] = pouter
    θ[n] = 1.0
    pcav = opt.pcav
    Γ = ρ * h^3 / (12 * μ)
    dpmax = 0.0
    @inbounds for iter in 1:opt.maxiter
        dpmax = 0.0
        for i in 1:(n - 1)
            if i == 1
                rf = 0.5 * (r[1] + r[2])
                drf = r[2] - r[1]
                # symmetry: no inner flux; area ~ π r_f^2
                Ae = 2π * rf
                vol = π * rf^2
                ke = Γ * Ae / drf
                lhs_p = ke * p[2]
                den = ke
                sq_coef = ρ * vol * h / dt
                sq_wo = ρ * vol * (-θold[1] * hold) / dt
            else
                rf = 0.5 * (r[i] + r[i + 1])
                rb = 0.5 * (r[i] + r[i - 1])
                drf = r[i + 1] - r[i]
                drb = r[i] - r[i - 1]
                Ae = 2π * rf
                Ab = 2π * rb
                vol = π * (rf^2 - rb^2)
                ke = Γ * Ae / drf
                kw = Γ * Ab / drb
                lhs_p = ke * p[i + 1] + kw * p[i - 1]
                den = ke + kw
                sq_coef = ρ * vol * h / dt
                sq_wo = ρ * vol * (-θold[i] * hold) / dt
            end
            # ∇·(Γ∇p) = ρ ∂(θ h)/∂t   (no Couette)
            # ke (p+ − p) − kw (p − p−) = sq_coef * θ + sq_wo
            pstar = (lhs_p - sq_coef * θ[i] - sq_wo) / den
            pold = p[i]
            if pstar >= pcav
                p[i] = (1 - opt.ωp) * p[i] + opt.ωp * pstar
                θ[i] = 1.0
            else
                p[i] = pcav
                # ke (p+ − pcav) − kw (pcav − p−) = sq_coef θ + sq_wo
                if i == 1
                    lhs = ke * (p[2] - pcav)
                else
                    rf = 0.5 * (r[i] + r[i + 1])
                    rb = 0.5 * (r[i] + r[i - 1])
                    drf = r[i + 1] - r[i]
                    drb = r[i] - r[i - 1]
                    ke = Γ * (2π * rf) / drf
                    kw = Γ * (2π * rb) / drb
                    lhs = ke * (p[i + 1] - pcav) - kw * (pcav - p[i - 1])
                end
                if abs(sq_coef) > 1e-30
                    θstar = (lhs - sq_wo) / sq_coef
                else
                    θstar = θ[i]
                end
                θ[i] = clamp((1 - opt.ωθ) * θ[i] + opt.ωθ * θstar, 0.0, 1.0)
            end
            dpmax = max(dpmax, abs(p[i] - pold))
        end
        p[n] = pouter
        θ[n] = 1.0
        if dpmax < opt.tol * max(maximum(abs, p), 1.0)
            break
        end
    end
    return p, θ
end

# ---------------------------------------------------------------------------
# Profito §4.1 case constructors
# ---------------------------------------------------------------------------

function profito_single_slider(; n::Int=301, compressible::Bool=true)
    L = 76.2e-3
    hmax = 8e-6
    hmin = 4e-6
    U = 4.57
    x = collect(range(0.0, L; length=n))
    h = film_parabolic_slider.(x, L, hmax, hmin)
    rheo = compressible ?
        (ρ0=580.0, μ0=39e-3, C1=2.22e9, C2=1.66, α=0.0, τ0=0.0) :
        ConstRheology(; ρ=580.0, μ=39e-3)
    pleft = 3.36414e3
    pright = 0.0
    opt = ElrodOptions(; pcav=0.0, maxiter=25_000, tol=1e-9)
    return (; x, h, L, U, rheo, pleft, pright, opt, name="single slider")
end

function profito_double_slider(; n::Int=401, compressible::Bool=true)
    L = 76.2e-3
    hmax = 50.8e-6
    hmin = 25.4e-6
    U = 4.57
    x = collect(range(0.0, L; length=n))
    h = film_double_parabolic.(x, L, hmax, hmin)
    rheo = compressible ?
        (ρ0=580.0, μ0=39e-3, C1=2.22e9, C2=1.66, α=0.0, τ0=0.0) :
        ConstRheology(; ρ=580.0, μ=39e-3)
    opt = ElrodOptions(; pcav=0.0, maxiter=25_000, tol=1e-9)
    return (; x, h, L, U, rheo, pleft=0.0, pright=0.0, opt, name="double slider")
end

function profito_journal(; ε::Real=0.93, nθ::Int=601, ny::Int=13, piezoviscous::Bool=true)
    R = 31.29e-3
    Lb = 625.8e-3
    c = 40.0e-6
    ω = 250.0
    U = ω * R
    x = collect(range(0.0, 2π * R; length=nθ + 1)[1:nθ])  # periodic, last = first
    y = collect(range(0.0, Lb; length=ny))
    h = [film_journal(xi, R, c, ε) for xi in x, _yj in y]
    rheo = piezoviscous ?
        (ρ0=850.0, μ0=5.7e-3, C1=Inf, C2=1.66, α=11.2e-9, τ0=0.0) :
        ConstRheology(; ρ=850.0, μ=5.7e-3)
    opt = ElrodOptions(; pcav=-1e5, maxiter=40_000, tol=1e-8, ωp=0.8, ωθ=0.5)
    return (; x, y, h, R, Lb, c, ε, U, rheo, opt, p0=0.0, name="journal ε=$ε")
end

function profito_squeeze(; nr::Int=80)
    R = 5e-3
    hmin = 9.14e-6
    ha = 320.8e-6
    ω = 99.74
    T = 2π / ω
    μ = 5e-3
    ρ = 850.0
    p0 = 1e5
    pcav = 0.0
    r = collect(range(0.0, R; length=nr))
    r[1] = 0.25 * (r[2] - r[1])  # tiny finite centre radius
    return (; r, R, hmin, ha, ω, T, μ, ρ, p0, pcav,
        ncycle=3, nt=576, name="squeeze plates")
end

function profito_pocket(; ny::Int=29, infinite::Bool=false)
    a = 20e-3
    b = infinite ? 300e-3 : 10e-3
    c = 4e-3
    Lp = 6e-3
    hmax = 1.1e-6
    hmin = 1.0e-6
    Hp = 0.4e-6
    U = 1.0
    nx = 126
    x = collect(range(0.0, a; length=nx))
    y = collect(range(0.0, b; length=ny))
    h = [film_pocket(xi, a, c, Lp, hmax, hmin, Hp) for xi in x, _yj in y]
    rheo = (ρ0=850.0, μ0=10e-3, C1=2.22e9, C2=1.66, α=12e-9, τ0=5e6)
    opt = ElrodOptions(; pcav=0.0, maxiter=40_000, tol=1e-8, ωp=0.8, ωθ=0.5)
    pside = 1e5
    return (; x, y, h, a, b, U, rheo, opt, pside, name=infinite ? "pocket infinite" : "pocket finite")
end
