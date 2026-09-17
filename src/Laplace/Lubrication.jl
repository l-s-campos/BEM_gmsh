# =============================================================================
# Hydrodynamic lubrication (Reynolds) via Laplace FS + DIBEM
# =============================================================================
# Guiggiani, EABE 119:183–188 (2020):  𝒫 = p h^{3/2}  turns Reynolds into
#
#     ∇²𝒫 + f 𝒫 = g ,    f = −∇²(h^{3/2}) / h^{3/2} ,
#                         g = 6μU h^{−3/2} ∂h/∂x .
#
# Special films make f = k constant (Helmholtz / Poisson / Klein–Gordon).
# Here the Laplace fundamental solution plus DIBEM mass treats the domain
# terms for any of those films (and the linear wedge, where f is not constant):
#
#     (H + M Diag(f)) 𝒫 − G q = M g .
#
# Pad example: p = 0 on Γ ⇒ 𝒫 = 0 on Γ.
#
# Without cavitation, the original Reynolds equation is the heterogeneous
# conductivity problem already treated by DIBEM (Barcelos–Loeffler):
#
#     ∇ · (h³ ∇p) = 6μU ∂h/∂x
#
# i.e. ∇ · (K ∇p) = f with K = h³. That path does not need the 𝒫 map.

export FilmProfile, film_h1, film_h2, film_h3, film_h4, film_h5, film_linear
export guiggiani_films, film_ratio, film_b
export infinite_bearing_pressure, linear_wedge_pressure
export mesh_guiggiani_pad, apply_ambient_pressure!
export solve_reynolds_dibem!, solve_reynolds_particular!, solve_reynolds_het!
export solve_reynolds_cfp!, characteristic_theta, film_parabolic, film_journal_unfolded
export film_journal_misaligned, film_h, film_hx
export periodic_x_pairs, mark_periodic_x!
export reynolds_pressure, interior_laplace, eval_reynolds_particular
export particular_h2

"""Guiggiani film `h = h(x)` on the pad `x ∈ [0, L]` (leading → trailing)."""
struct FilmProfile
    name::String
    h::Function
    hx::Function
    f::Function
    a::Float64
    hi::Float64
    ho::Float64
    L::Float64
    k::Float64
    λ::Float64
    φ::Float64
end

film_ratio(film::FilmProfile) = film.a
film_b(a::Real) = (1 / float(a))^(3 / 2)
film_b(film::FilmProfile) = film_b(film.a)

_xi(x, L) = x / L

"""λ bounds for monotone convergent films, Guiggiani §4 (`a = hi/ho`)."""
function film_lambda_limits(a::Real)
    b = film_b(a)
    return (;
        λ̄ = acos(b),
        λstar = log(1 / b),
        λhat = acosh(1 / b),
        b = b,
    )
end

function _phi_h1(λ, b)
    den = b - cos(λ)
    abs(den) < 1e-14 && return π / 2
    return atan(sin(λ) / den)
end

function _phi_h3(λ, b)
    t = sinh(λ) / (b - cosh(λ))
    abs(t) >= 1 && throw(ArgumentError("h3: |tanh φ|≥1 at λ=$λ; need 0<λ<ln(1/b)"))
    return atanh(t)
end

function _phi_h5(λ, b)
    t = (b - cosh(λ)) / sinh(λ)
    abs(t) >= 1 && throw(ArgumentError("h5: |tanh φ|≥1 at λ=$λ; need ln(1/b)<λ≤acosh(1/b)"))
    return atanh(t)
end

"""λ so that `h3(L/2) = (hi+ho)/2` (Guiggiani, matches the linear wedge at mid-pad)."""
function _lambda_h3_mid(a::Real; niter::Int=60)
    lim = film_lambda_limits(a)
    lo, hi = 1e-8, (1 - 1e-8) * lim.λstar
    target = (1 + 1 / a) / 2
    for _ in 1:niter
        λ = 0.5 * (lo + hi)
        φ = _phi_h3(λ, lim.b)
        hmid = (sinh(0.5λ + φ) / sinh(φ))^(2 / 3)
        hmid > target ? (lo = λ) : (hi = λ)
    end
    return 0.5 * (lo + hi)
end

function film_h1(; a=2.0, hi=nothing, L=1.0, λ=nothing)
    a = float(a)
    L = float(L)
    lim = film_lambda_limits(a)
    λv = λ === nothing ? lim.λ̄ : float(λ)
    0 < λv <= lim.λ̄ + 1e-12 || throw(ArgumentError("h1: need 0 < λ ≤ arccos(b)"))
    hi_ = hi === nothing ? a : float(hi)
    ho = hi_ / a
    φ = _phi_h1(λv, lim.b)
    sφ = sin(φ)
    h = function (x)
        ξ = _xi(x, L)
        return hi_ * (sin(λv * ξ + φ) / sφ)^(2 / 3)
    end
    hx = function (x)
        ξ = _xi(x, L)
        s = sin(λv * ξ + φ)
        return hi_ * (2 / 3) * (s / sφ)^(-1 / 3) * (cos(λv * ξ + φ) * λv / sφ) / L
    end
    k = (λv / L)^2
    return FilmProfile("h1", h, hx, _ -> k, a, hi_, ho, L, k, λv, φ)
end

function film_h2(; a=2.0, hi=nothing, L=1.0)
    a = float(a)
    L = float(L)
    b = film_b(a)
    hi_ = hi === nothing ? a : float(hi)
    ho = hi_ / a
    h = function (x)
        ξ = _xi(x, L)
        return hi_ * ((b - 1) * ξ + 1)^(2 / 3)
    end
    hx = function (x)
        ξ = _xi(x, L)
        s = (b - 1) * ξ + 1
        return hi_ * (2 / 3) * s^(-1 / 3) * (b - 1) / L
    end
    return FilmProfile("h2", h, hx, _ -> 0.0, a, hi_, ho, L, 0.0, 0.0, 0.0)
end

function film_h3(; a=2.0, hi=nothing, L=1.0, λ=nothing)
    a = float(a)
    L = float(L)
    lim = film_lambda_limits(a)
    λv = λ === nothing ? _lambda_h3_mid(a) : float(λ)
    0 < λv < lim.λstar || throw(ArgumentError("h3: need 0 < λ < ln(1/b)"))
    hi_ = hi === nothing ? a : float(hi)
    ho = hi_ / a
    φ = _phi_h3(λv, lim.b)
    shφ = sinh(φ)
    h = function (x)
        ξ = _xi(x, L)
        return hi_ * (sinh(λv * ξ + φ) / shφ)^(2 / 3)
    end
    hx = function (x)
        ξ = _xi(x, L)
        sh = sinh(λv * ξ + φ)
        return hi_ * (2 / 3) * (sh / shφ)^(-1 / 3) * (cosh(λv * ξ + φ) * λv / shφ) / L
    end
    k = -(λv / L)^2
    return FilmProfile("h3", h, hx, _ -> k, a, hi_, ho, L, k, λv, φ)
end

function film_h4(; a=2.0, hi=nothing, L=1.0)
    a = float(a)
    L = float(L)
    lim = film_lambda_limits(a)
    λv = lim.λstar
    hi_ = hi === nothing ? a : float(hi)
    ho = hi_ / a
    lna = log(a)
    h = function (x)
        ξ = _xi(x, L)
        return hi_ * a^(-ξ)
    end
    hx = function (x)
        ξ = _xi(x, L)
        return hi_ * a^(-ξ) * (-lna) / L
    end
    k = -(λv / L)^2
    return FilmProfile("h4", h, hx, _ -> k, a, hi_, ho, L, k, λv, 0.0)
end

function film_h5(; a=2.0, hi=nothing, L=1.0, λ=nothing)
    a = float(a)
    L = float(L)
    lim = film_lambda_limits(a)
    λv = λ === nothing ? lim.λhat : float(λ)
    lim.λstar < λv <= lim.λhat + 1e-12 ||
        throw(ArgumentError("h5: need ln(1/b) < λ ≤ arccosh(1/b)"))
    hi_ = hi === nothing ? a : float(hi)
    ho = hi_ / a
    φ = _phi_h5(λv, lim.b)
    chφ = cosh(φ)
    h = function (x)
        ξ = _xi(x, L)
        return hi_ * (cosh(λv * ξ + φ) / chφ)^(2 / 3)
    end
    hx = function (x)
        ξ = _xi(x, L)
        ch = cosh(λv * ξ + φ)
        return hi_ * (2 / 3) * (ch / chφ)^(-1 / 3) * (sinh(λv * ξ + φ) * λv / chφ) / L
    end
    k = -(λv / L)^2
    return FilmProfile("h5", h, hx, _ -> k, a, hi_, ho, L, k, λv, φ)
end

"""Linear wedge `h = hi − (hi−ho) x/L` (not a constant-`k` Guiggiani film)."""
function film_linear(; a=2.0, hi=nothing, L=1.0)
    a = float(a)
    L = float(L)
    hi_ = hi === nothing ? a : float(hi)
    ho = hi_ / a
    slope = (ho - hi_) / L
    h = function (x)
        return hi_ + slope * x
    end
    hx = function (_x)
        return slope
    end
    f = function (x)
        hv = h(x)
        return -(3 / 4) * slope^2 / hv^2
    end
    return FilmProfile("linear", h, hx, f, a, hi_, ho, L, NaN, 0.0, 0.0)
end

"""Default set for Guiggiani Figs. 2–3 (`hi/ho = a`, `λ` at the plotted extremes)."""
function guiggiani_films(; a=2.0, hi=nothing, L=1.0)
    hi_ = hi === nothing ? a : float(hi)
    return (
        h1 = film_h1(; a=a, hi=hi_, L=L),
        h2 = film_h2(; a=a, hi=hi_, L=L),
        h3 = film_h3(; a=a, hi=hi_, L=L),
        h4 = film_h4(; a=a, hi=hi_, L=L),
        h5 = film_h5(; a=a, hi=hi_, L=L),
        linear = film_linear(; a=a, hi=hi_, L=L),
    )
end

# ---------------------------------------------------------------------------
# Infinite bearing (1-D Reynolds)
# ---------------------------------------------------------------------------

function _gl01(n::Integer)
    ξ, w = gausslegendre(n)
    x = 0.5 .* (ξ .+ 1)
    return x, 0.5 .* w
end

"""
    infinite_bearing_pressure(film, x; μ=1, U=1, nq=96) -> p

Exact 1-D Reynolds (`∂/∂y = 0`) with `p(0)=p(L)=0`.
Returned `p` is dimensional; non-dimensional `p ho²/(μ U L)` follows by scaling.
"""
function infinite_bearing_pressure(film::FilmProfile, x::Real;
        μ=1.0, U=1.0, nq::Int=96)
    L = film.L
    xq, wq = _gl01(nq)
    I2 = 0.0
    I3 = 0.0
    @inbounds for i in eachindex(xq)
        hv = film.h(L * xq[i])
        I2 += wq[i] / hv^2
        I3 += wq[i] / hv^3
    end
    C = I2 / I3
    ξ = x / L
    ξ <= 0 && return 0.0
    ξ >= 1 && return 0.0
    acc = 0.0
    @inbounds for i in eachindex(xq)
        s = ξ * xq[i]
        hv = film.h(L * s)
        acc += wq[i] * (hv - C) / hv^3
    end
    return 6 * μ * U * L * ξ * acc
end

function infinite_bearing_pressure(film::FilmProfile, xs::AbstractVector;
        μ=1.0, U=1.0, nq::Int=96)
    return [infinite_bearing_pressure(film, x; μ=μ, U=U, nq=nq) for x in xs]
end

"""Closed form for the linear wedge, `H = a − (a−1)ξ`, `ho = 1` units."""
function linear_wedge_pressure(ξ, a; μ=1.0, U=1.0, L=1.0, ho=1.0)
    a = float(a)
    H = a - (a - 1) * ξ
    C = 2a / (a + 1)
    pstar = 6 / (a - 1) * (1 / H - 1 / a - (C / 2) * (1 / H^2 - 1 / a^2))
    return pstar * μ * U * L / ho^2
end

# ---------------------------------------------------------------------------
# Particular integral for h2 (Guiggiani eq. 38)
# ---------------------------------------------------------------------------

"""
    particular_h2(film, x; μ=1, U=1) -> 𝒫p

`𝒫p = −(18 μ U)/(A β) h2` with `h2 = [A(β x + 1)]^{2/3}` on `x ∈ [0, L]`.
"""
function particular_h2(film::FilmProfile, x; μ=1.0, U=1.0)
    film.name == "h2" || throw(ArgumentError("particular_h2 expects film h2"))
    A = film.hi^(3 / 2)
    β = (film_b(film) - 1) / film.L
    return -(18 * μ * U / (A * β)) * film.h(x)
end

# ---------------------------------------------------------------------------
# Pad mesh (Guiggiani Fig. 4)
# ---------------------------------------------------------------------------

"""
    mesh_guiggiani_pad(; L=1, B=0.75L, r=0.1L, ordem=2, n_long=4, n_short=2,
                       n_arc=1, nome="guiggiani_pad") -> path

Rounded rectangle, leading edge at `x=0`, trailing at `x=L`, `y ∈ [−B/2, B/2]`.
Default `4+4+2+2+1+1+1+1 = 16` quadratic elements (Guiggiani Fig. 4).
Ambient pressure: physical name `0;0` on the whole contour.
"""
function mesh_guiggiani_pad(;
        L=1.0,
        B=0.75,
        r=0.1,
        ordem=2,
        n_long=4,
        n_short=2,
        n_arc=1,
        nome="guiggiani_pad",
        show=false,
    )
    L = float(L)
    B = float(B)
    r = float(r)
    r < min(L, B) / 2 || throw(ArgumentError("fillet r=$r too large for L=$L, B=$B"))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    ymin, ymax = -B / 2, B / 2
    lc = 0.2
    p1 = gmsh.model.geo.addPoint(r, ymin, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(L - r, ymin, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(L, ymin + r, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(L, ymax - r, 0.0, lc)
    p5 = gmsh.model.geo.addPoint(L - r, ymax, 0.0, lc)
    p6 = gmsh.model.geo.addPoint(r, ymax, 0.0, lc)
    p7 = gmsh.model.geo.addPoint(0.0, ymax - r, 0.0, lc)
    p8 = gmsh.model.geo.addPoint(0.0, ymin + r, 0.0, lc)
    cbr = gmsh.model.geo.addPoint(L - r, ymin + r, 0.0, lc)
    ctr = gmsh.model.geo.addPoint(L - r, ymax - r, 0.0, lc)
    ctl = gmsh.model.geo.addPoint(r, ymax - r, 0.0, lc)
    cbl = gmsh.model.geo.addPoint(r, ymin + r, 0.0, lc)
    bot = gmsh.model.geo.addLine(p1, p2)
    abr = gmsh.model.geo.addCircleArc(p2, cbr, p3)
    rgt = gmsh.model.geo.addLine(p3, p4)
    atr = gmsh.model.geo.addCircleArc(p4, ctr, p5)
    top = gmsh.model.geo.addLine(p5, p6)
    atl = gmsh.model.geo.addCircleArc(p6, ctl, p7)
    lft = gmsh.model.geo.addLine(p7, p8)
    abl = gmsh.model.geo.addCircleArc(p8, cbl, p1)
    cl = gmsh.model.geo.addCurveLoop([bot, abr, rgt, atr, top, atl, lft, abl])
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bot, n_long + 1)
    gmsh.model.mesh.setTransfiniteCurve(top, n_long + 1)
    gmsh.model.mesh.setTransfiniteCurve(rgt, n_short + 1)
    gmsh.model.mesh.setTransfiniteCurve(lft, n_short + 1)
    for arc in (abr, atr, atl, abl)
        gmsh.model.mesh.setTransfiniteCurve(arc, n_arc + 1)
    end
    gmsh.model.addPhysicalGroup(1, [bot, abr, rgt, atr, top, atl, lft, abl], -1, "0;0")
    gmsh.model.addPhysicalGroup(2, [surf], -1, "pad")
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""Dirichlet `p = 0` (hence `𝒫 = 0`) on the whole boundary."""
function apply_ambient_pressure!(dad::BEMdata)
    @inbounds for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = 0.0
    end
    return dad
end

# ---------------------------------------------------------------------------
# Laplace interior representation (homogeneous)
# ---------------------------------------------------------------------------

"""`c u = ∫ U q − ∫ T* u` at an interior point (Laplace, no domain source)."""
function interior_laplace(dad::BEMdata{<:Laplace}, pf::Point;
        T=dad.T, q=dad.q)
    n = dad.n
    h = zeros(float(eltype(T)), n)
    g = zeros(float(eltype(T)), n)
    @inbounds for el in dad.elements
        xj = dad.Nodes[el.index]
        nn = length(el.index)
        hloc = zeros(eltype(h), nn)
        gloc = zeros(eltype(g), nn)
        integrate_element(dad, el, xj, pf, hloc, gloc)
        for (a, j) in enumerate(el.index)
            h[j] += hloc[a]
            g[j] += gloc[a]
        end
    end
    c = -sum(h)
    return (dot(g, view(q, 1:n)) - dot(h, view(T, 1:n))) / c
end

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

function _film_fields(film::FilmProfile, pts; μ=1.0, U=1.0)
    nt = length(pts)
    fv = zeros(nt)
    gv = zeros(nt)
    hv = zeros(nt)
    @inbounds for i in 1:nt
        x = pts[i][1]
        hv[i] = film.h(x)
        fv[i] = film.f(x)
        gv[i] = 6 * μ * U * film.hx(x) / hv[i]^(3 / 2)
    end
    return fv, gv, hv
end

function _add_reaction_source!(dad, fvec, gvec)
    M = dad.M
    A = dad.A
    b = dad.b
    n = dad.n
    nt = dad.nt
    @inbounds for j in 1:nt
        fj = fvec[j]
        iszero(fj) && continue
        Mj = view(M, :, j)
        if j <= n && dad.BC[j] == 0
            b .-= (fj * dad.BV[j]) .* Mj
        else
            A[:, j] .+= fj .* Mj
        end
    end
    b .+= M * gvec
    return nothing
end

"""
    solve_reynolds_dibem!(dad, film; μ=1, U=1, rbf=PHS(3; poly_deg=1), npg=16)

Laplace FS + DIBEM for Guiggiani's transformed Reynolds equation.
`dad.T` is the lubricant pressure `p`; `dad.cache.P` is `𝒫 = p h^{3/2}`.
Boundary must be Dirichlet `𝒫` (ambient `p=0` ⇒ `𝒫=0`).
"""
function solve_reynolds_dibem!(dad::BEMdata{<:Laplace}, film::FilmProfile;
        μ=1.0, U=1.0, rbf=PHS(3; poly_deg=1), npg::Int=16, kwargs...)
    has_cache(dad, :H) || assemble!(dad; npg=npg, kwargs...)
    has_cache(dad, :M) || DIBEM(dad; rbf=rbf, npg=npg)
    apply_ambient_pressure!(dad)
    pts = all_points(dad)
    fv, gv, hv = _film_fields(film, pts; μ=μ, U=U)
    applyBC(dad)
    _add_reaction_source!(dad, fv, gv)
    x = bem_linsolve(dad.A, dad.b)
    Tfull = zeros(eltype(x), dad.nt)
    qfull = zeros(eltype(x), dad.n)
    Tfull[1:length(x)] .= x
    split_sol!(dad, Tfull, qfull)
    P = Tfull[1:dad.nt]
    p = P ./ (hv .^ (3 / 2))
    set_cache!(dad; T=p, q=qfull, P=P, film=film, reynolds=:dibem)
    return dad.T
end

"""
    solve_reynolds_het!(dad, film; μ=1, U=1)

Full-film Reynolds as heterogeneous DIBEM: `∇ · (h³ ∇p) = 6μU ∂h/∂x`.
Same Poisson `u*` as `solve_heterogeneous!`. No Guiggiani `𝒫` map, no
cavitation. `dad.T` is pressure `p`.
"""
function solve_reynolds_het!(dad::BEMdata{<:Laplace}, film::FilmProfile;
        μ=1.0, U=1.0, rbf=PHS(1; poly_deg=-1),
        source_rbf=PHS(3; poly_deg=1), npg::Int=16, ambient::Bool=true, kwargs...)
    has_cache(dad, :H) || assemble!(dad; npg=npg, kwargs...)
    ambient && apply_ambient_pressure!(dad)
    K = p -> film_h(film, p)^3
    f = p -> 6 * μ * U * film_hx(film, p)
    solve_heterogeneous!(dad, K; rbf=rbf, source=f, source_rbf=source_rbf)
    set_cache!(dad; film=film, reynolds=:heterogeneous)
    return dad.T
end

# ---------------------------------------------------------------------------
# Schultz et al. 2025 Interpretation I: characteristic fixed point (CFP)
# ---------------------------------------------------------------------------
# P(θ):  ∇ · (h³ ∇p) = 6μU ∂(θ h)/∂x     (heterogeneous DIBEM)
# C(p):  θ(x) = h(x − t* e) / h(x),   t* = inf{t≥0 : p(x − t e) > p_c}
# T(θ) = min{C(P(θ)), 1},   θ^{k+1} = (1−λ_k) θ^k + λ_k T(θ^k)

"""Convergent–divergent parabola: `h(0)=h(L)=hmax`, `h(L/2)=hmin`."""
function film_parabolic(; L=76.2e-3, hmax=8e-6, hmin=4e-6)
    L = float(L); hmax = float(hmax); hmin = float(hmin)
    h = function (x)
        return hmin + (hmax - hmin) * (2 * x / L - 1)^2
    end
    hx = function (x)
        return (hmax - hmin) * 2 * (2 * x / L - 1) * (2 / L)
    end
    a = hmax / hmin
    return FilmProfile("parabolic", h, hx, _ -> NaN, a, hmax, hmin, L, NaN, 0.0, 0.0)
end

"""Unfolded journal `h = c (1 + ε cos(2π x / L))` (Schultz et al. 2025)."""
function film_journal_unfolded(; L=0.15, c=30e-6, ε=0.6)
    L = float(L); c = float(c); ε = float(ε)
    h = function (x)
        return c * (1 + ε * cos(2π * x / L))
    end
    hx = function (x)
        return c * ε * (-2π / L) * sin(2π * x / L)
    end
    hi = c * (1 + ε); ho = c * (1 - ε)
    return FilmProfile("journal", h, hx, _ -> NaN, hi / ho, hi, ho, L, NaN, 0.0, 0.0)
end

"""Misaligned journal (Schultz 2025 §6.2).

`h = c (1 + ε (1 − 2 y / W) cos(2π x / L))`. Width `W` is stored in `λ`.
"""
function film_journal_misaligned(; L=0.08, W=0.02, c=25e-6, ε=0.8)
    L = float(L); W = float(W); c = float(c); ε = float(ε)
    h = function (x, y)
        return c * (1 + ε * (1 - 2 * y / W) * cos(2π * x / L))
    end
    hx = function (x, y)
        return c * ε * (1 - 2 * y / W) * (-2π / L) * sin(2π * x / L)
    end
    hi = c * (1 + ε); ho = c * (1 - ε)
    return FilmProfile("journal_misaligned", h, hx, _ -> NaN, hi / ho, hi, ho, L, NaN, W, 0.0)
end

@inline _film_xy(film::FilmProfile) = film.name == "journal_misaligned"

"""Film height at `x` or at a point `(x, y)`. 1-D films ignore `y`."""
@inline film_h(film::FilmProfile, x::Real) = film.h(x)
@inline film_h(film::FilmProfile, x::Real, y::Real) =
    _film_xy(film) ? film.h(x, y) : film.h(x)
@inline film_h(film::FilmProfile, pt) = film_h(film, pt[1], pt[2])

@inline film_hx(film::FilmProfile, x::Real) = film.hx(x)
@inline film_hx(film::FilmProfile, x::Real, y::Real) =
    _film_xy(film) ? film.hx(x, y) : film.hx(x)
@inline film_hx(film::FilmProfile, pt) = film_hx(film, pt[1], pt[2])

"""Pairs of collocation indices on `x = x0` and `x = x1` with matching `y`."""
function periodic_x_pairs(dad::BEMdata; x0::Real=0.0, x1::Real=1.0, tol::Real=1e-6)
    left = Int[]; right = Int[]
    @inbounds for i in 1:dad.n
        x = dad.Nodes[i][1]
        abs(x - x0) <= tol && push!(left, i)
        abs(x - x1) <= tol && push!(right, i)
    end
    pairs = Tuple{Int,Int}[]
    used = falses(length(right))
    @inbounds for i in left
        yi = dad.Nodes[i][2]
        best_k, best_d = 0, Inf
        for (k, j) in enumerate(right)
            used[k] && continue
            d = abs(dad.Nodes[j][2] - yi)
            if d < best_d
                best_d, best_k = d, k
            end
        end
        if best_k > 0 && best_d <= 10 * tol
            push!(pairs, (i, right[best_k]))
            used[best_k] = true
        end
    end
    isempty(pairs) && @warn "periodic_x_pairs: no matches (x0=$x0, x1=$x1, nL=$(length(left)), nR=$(length(right)))"
    return pairs
end

"""Mark left/right edges as periodic (`BC=4`) and cache `dad.periodic_pairs`.

On each pair `(i,j)` the BIE uses `p_i = p_j` and `qn_i + qn_j = 0`
(outward normals on opposite sides of the rectangle).
"""
function mark_periodic_x!(dad::BEMdata; x0::Real=0.0, x1::Real=1.0, tol::Real=1e-6)
    pairs = periodic_x_pairs(dad; x0=x0, x1=x1, tol=tol)
    @inbounds for (i, j) in pairs
        dad.BC[i] = 4
        dad.BC[j] = 4
    end
    set_cache!(dad; periodic_pairs=pairs)
    return pairs
end

"""Mix heterogeneous `L` into a BC system. `source` is added later as `M f`."""
function _het_mixed_from_L(dad::BEMdata{<:Laplace}, L, Kv)
    G = Matrix{Float64}(dad.G)
    n = dad.n
    nt = dad.nt
    Kb = Kv[1:n]
    Asys = copy(L)
    b = zeros(nt)
    @inbounds for j in 1:n
        dad.BC[j] == 4 && continue
        if dad.BC[j] == 0
            b .-= L[:, j] .* dad.BV[j]
            Asys[:, j] .= G[:, j] .* Kb[j]
        else
            qn = -dad.BV[j] / max(Kb[j], 1e-30)
            b .-= G[:, j] .* (Kb[j] * qn)
        end
    end
    if has_cache(dad, :periodic_pairs)
        @inbounds for (i, j) in dad.periodic_pairs
            # unknown p at i (also p at j); unknown qn_i with qn_j = −qn_i
            Asys[:, i] .= view(L, :, i) .+ view(L, :, j)
            Asys[:, j] .= view(G, :, i) .* Kb[i] .- view(G, :, j) .* Kb[j]
        end
    end
    return Asys, b
end

function _split_het_sol!(dad, x, Kv)
    n = dad.n
    T = zeros(dad.nt)
    qn = zeros(n)
    is_right = falses(n)
    if has_cache(dad, :periodic_pairs)
        @inbounds for (_, j) in dad.periodic_pairs
            is_right[j] = true
        end
    end
    @inbounds for j in 1:n
        if dad.BC[j] == 4
            continue
        elseif dad.BC[j] == 0
            T[j] = dad.BV[j]
            qn[j] = x[j]
        else
            T[j] = x[j]
            qn[j] = -dad.BV[j] / max(Kv[j], 1e-30)
        end
    end
    if has_cache(dad, :periodic_pairs)
        @inbounds for (i, j) in dad.periodic_pairs
            T[i] = x[i]
            T[j] = x[i]
            qn[i] = x[j]
            qn[j] = -x[j]
        end
    end
    if dad.nt > n
        T[(n + 1):end] .= x[(n + 1):end]
    end
    q = -Kv[1:n] .* qn
    set_cache!(dad; T=T, q=q, het_K=Kv, het_qn=qn)
    return T
end

"""`∂(θ h)/∂x` by central differences on constant-`y` bins (flow in `+x`)."""
function _d_theta_h_dx(pts, θ, film; periodic_x::Bool=false)
    n = length(pts)
    dth = zeros(n)
    bins = Dict{Float64,Vector{Int}}()
    @inbounds for i in 1:n
        key = round(pts[i][2]; digits=8)
        push!(get!(bins, key, Int[]), i)
    end
    Lx = film.L
    for idxs in values(bins)
        perm = sort(idxs; by=i -> pts[i][1])
        m = length(perm)
        m == 0 && continue
        th = [θ[i] * film_h(film, pts[i]) for i in perm]
        xs = [pts[i][1] for i in perm]
        if m == 1
            dth[perm[1]] = film_hx(film, pts[perm[1]]) * θ[perm[1]]
            continue
        end
        if periodic_x
            dth[perm[1]] = (th[2] - th[m]) / (xs[2] - (xs[m] - Lx) + 1e-30)
            dth[perm[m]] = (th[1] - th[m - 1]) / ((xs[1] + Lx) - xs[m - 1] + 1e-30)
        else
            dth[perm[1]] = (th[2] - th[1]) / (xs[2] - xs[1] + 1e-30)
            dth[perm[m]] = (th[m] - th[m - 1]) / (xs[m] - xs[m - 1] + 1e-30)
        end
        @inbounds for k in 2:(m - 1)
            dth[perm[k]] = (th[k + 1] - th[k - 1]) / (xs[k + 1] - xs[k - 1] + 1e-30)
        end
    end
    return dth
end

"""
    characteristic_theta(pts, p, film; pc=0)

Schultz eq. (8): walk upstream in `x` until `p > pc` (film rupture), then
`θ = h(x_r) / h(x)`, clipped to 1. Full-film nodes get `θ = 1`.
"""
function characteristic_theta(pts, p, film; pc::Real=0.0, periodic_x::Bool=false)
    n = length(pts)
    θ = ones(n)
    bins = Dict{Float64,Vector{Int}}()
    @inbounds for i in 1:n
        key = round(pts[i][2]; digits=8)
        push!(get!(bins, key, Int[]), i)
    end
    for idxs in values(bins)
        perm = sort(idxs; by=i -> pts[i][1])
        m = length(perm)
        m == 0 && continue
        if periodic_x
            @inbounds for k in 1:m
                i = perm[k]
                if p[i] > pc
                    θ[i] = 1.0
                    continue
                end
                found = 0
                for s in 1:(m - 1)
                    kk = k - s
                    kk < 1 && (kk += m)
                    if p[perm[kk]] > pc
                        found = kk
                        break
                    end
                end
                hup = found == 0 ? film_h(film, pts[perm[1]]) : film_h(film, pts[perm[found]])
                θ[i] = min(hup / max(film_h(film, pts[i]), 1e-30), 1.0)
            end
        else
            last_full = 0
            hin = film_h(film, pts[perm[1]])
            @inbounds for k in 1:m
                i = perm[k]
                if p[i] > pc
                    last_full = k
                    θ[i] = 1.0
                else
                    hup = last_full == 0 ? hin : film_h(film, pts[perm[last_full]])
                    θ[i] = min(hup / max(film_h(film, pts[i]), 1e-30), 1.0)
                end
            end
        end
    end
    return θ
end

"""
    solve_reynolds_cfp!(dad, film; μ=1, U=1, pc=0)

Schultz–Orgassa–Rom–Müller (Tribol. Int. 2025) Interpretation I:
characteristic fixed point for JFO. `P(θ)` is heterogeneous DIBEM;
`C(p)` is the analytic characteristic (8). Krasnoselskii–Mann with
`λ_k = 1/(α k + 1)`.
"""
function solve_reynolds_cfp!(dad::BEMdata{<:Laplace}, film::FilmProfile;
        μ=1.0, U=1.0, pc::Real=0.0, α::Real=0.66, ε::Real=5e-4,
        maxiter::Int=80, rbf=PHS(1; poly_deg=-1),
        source_rbf=PHS(3; poly_deg=1), npg::Int=16, verbose::Bool=false,
        ambient::Bool=false, periodic_x::Bool=false, prescribe=nothing, kwargs...)
    has_cache(dad, :H) || assemble!(dad; npg=npg, kwargs...)
    ambient && apply_ambient_pressure!(dad)
    L, Kv, _ = heterogeneous_L(dad, p -> film_h(film, p)^3; rbf=rbf)
    has_cache(dad, :M) || DIBEM(dad; rbf=source_rbf)
    Asys, b0 = _het_mixed_from_L(dad, L, Kv)
    pts = all_points(dad)
    nt = dad.nt
    θ = ones(nt)
    hist = Float64[]
    p = zeros(nt)
    @inbounds for k in 0:maxiter
        dth = _d_theta_h_dx(pts, θ, film; periodic_x=periodic_x)
        f = (6 * μ * U) .* dth
        b = b0 + dad.M * f
        x = bem_linsolve(Asys, b)
        p = _split_het_sol!(dad, x, Kv)
        if prescribe !== nothing
            @inbounds for i in 1:nt
                v = prescribe(pts[i])
                v === nothing || (p[i] = float(v))
            end
        end
        p̃ = p .- pc
        pmax = maximum(p̃)
        pmin = minimum(p̃)
        res = abs(min(pmin, 0.0)) / max(pmax, 1e-30)
        push!(hist, res)
        verbose && println("  CFP k=", k, "  res=", res, "  pmax=", pmax, "  pmin=", pmin)
        θ̃ = characteristic_theta(pts, p, film; pc=pc, periodic_x=periodic_x)
        dθ = 0.0
        @inbounds for i in 1:nt
            dθ = max(dθ, abs(θ̃[i] - θ[i]))
        end
        k == maxiter && break
        k >= 3 && res < ε && dθ < 1e-3 && break
        λ = 1 / (α * k + 1)
        @inbounds for i in 1:nt
            θ[i] = (1 - λ) * θ[i] + λ * θ̃[i]
        end
    end
    set_cache!(dad; T=p, film=film, reynolds=:cfp, theta=θ, cfp_hist=hist)
    return dad.T, θ
end

"""
    solve_reynolds_particular!(dad, film; μ=1, U=1)

Guiggiani's particular-integral Laplace BEM, **h2 only**:
`𝒫 = 𝒫o + 𝒫p` with `∇²𝒫o = 0` and `𝒫o = −𝒫p` on `Γ`.
"""
function solve_reynolds_particular!(dad::BEMdata{<:Laplace}, film::FilmProfile;
        μ=1.0, U=1.0, npg::Int=16, kwargs...)
    film.name == "h2" || throw(ArgumentError(
        "particular integral (38) is closed form for h2; got $(film.name)"))
    has_cache(dad, :H) || assemble!(dad; npg=npg, kwargs...)
    @inbounds for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = -particular_h2(film, dad.Nodes[i][1]; μ=μ, U=U)
    end
    solve(dad)
    P_o = copy(dad.T)
    q_o = copy(dad.q)
    pts = all_points(dad)
    P = copy(P_o)
    p = zeros(length(P))
    @inbounds for i in eachindex(pts)
        Pp = particular_h2(film, pts[i][1]; μ=μ, U=U)
        P[i] += Pp
        p[i] = P[i] / film.h(pts[i][1])^(3 / 2)
    end
    set_cache!(dad; T=p, q=q_o, P=P, P_o=P_o, q_o=q_o, film=film, reynolds=:particular)
    return dad.T
end

"""Pressure at an interior point from the h2 particular-integral solve."""
function eval_reynolds_particular(dad::BEMdata{<:Laplace}, film::FilmProfile, pf::Point;
        μ=1.0, U=1.0)
    has_cache(dad, :P_o) || throw(ArgumentError("call solve_reynolds_particular! first"))
    Po = interior_laplace(dad, pf; T=dad.P_o, q=dad.q_o)
    Pp = particular_h2(film, pf[1]; μ=μ, U=U)
    return (Po + Pp) / film.h(pf[1])^(3 / 2)
end

"""Non-dimensional `p ho² / (μ U L)` from a dimensional pressure."""
function reynolds_pressure(p, film::FilmProfile; μ=1.0, U=1.0)
    return p * film.ho^2 / (μ * U * film.L)
end
