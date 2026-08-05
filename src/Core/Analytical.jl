export AnalyticalSolution, analytical, attach_analytical!, apply_analytical_bc!, rel_error
export ana_laplace_linear, ana_laplace_quadratic
export ana_heat_1d, ana_heat_insulated_sides, ana_heat_dirichlet_square
export ana_elasticity_patch
export ana_potencial1d, ana_quarto_circ, ana_moulton, ana_laquini1, ana_laquini2, ana_laquini3
# Wave-propagation analytics/meshes: data/Laplace/wave_propagation.jl (include from scripts)

"""
    AnalyticalSolution

Callable analytical reference solution attached to a BEM problem.

# Fields
- `name::String`: short identifier
- `u`: primary field (potential / temperature / displacement) as
  `u(x::Point; t=0.0) -> Number or SVector`
- `q`: boundary flux / traction as
  `q(x::Point, n::Point; t=0.0) -> Number or SVector` (optional)
- `description::String`
"""
struct AnalyticalSolution{U,Q}
    name::String
    u::U
    q::Q
    description::String
end

AnalyticalSolution(name, u; q=nothing, description="") =
    AnalyticalSolution(name, u, q, description)

(a::AnalyticalSolution)(x; t=0.0) = a.u(x; t=t)

function Base.show(io::IO, a::AnalyticalSolution)
    print(io, "AnalyticalSolution(\"$(a.name)\")")
    isempty(a.description) || print(io, " — ", a.description)
end

"""
    analytical(dad::BEMdata; t=0.0) -> Vector

Evaluate the analytical solution stored in `dad.cache.analytical` at all
boundary and internal nodes. Returns a vector compatible with `dad.T`.
"""
function analytical(dad::BEMdata; t=0.0)
    has_cache(dad, :analytical) ||
        error("No analytical solution attached. Use attach_analytical!(dad, ana).")
    ana = dad.analytical
    pts = all_points(dad)
    return _eval_ana_points(ana, pts, dad; t=t)
end

function _eval_ana_points(ana::AnalyticalSolution, pts, dad::BEMdata{<:Laplace}; t=0.0)
    return [float(ana.u(p; t=t)) for p in pts]
end

function _eval_ana_points(ana::AnalyticalSolution, pts, dad::BEMdata{<:Elasticity}; t=0.0)
    dim = dad.dimension
    out = zeros(dim * length(pts))
    for (i, p) in enumerate(pts)
        ui = ana.u(p; t=t)
        out[dim*(i-1)+1:dim*i] .= ui
    end
    return out
end

"""
    attach_analytical!(dad, ana::AnalyticalSolution)

Store an analytical solution in `dad.cache` for later comparison / BC setup.
"""
function attach_analytical!(dad::BEMdata, ana::AnalyticalSolution)
    set_cache!(dad; analytical=ana)
    return dad
end

"""
    apply_analytical_bc!(dad, ana::AnalyticalSolution)

Overwrite `dad.BC` / `dad.BV` with Dirichlet data from the analytical primary
field. Useful for patch tests.
"""
function apply_analytical_bc!(dad::BEMdata{<:Laplace}, ana::AnalyticalSolution)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = float(ana.u(dad.Nodes[i]))
    end
    attach_analytical!(dad, ana)
    return dad
end

function apply_analytical_bc!(dad::BEMdata{<:Elasticity}, ana::AnalyticalSolution)
    dim = dad.dimension
    for i in 1:dad.n
        ui = ana.u(dad.Nodes[i])
        for d in 1:dim
            dad.BC[dim*(i-1)+d] = 0
            dad.BV[dim*(i-1)+d] = float(ui[d])
        end
    end
    attach_analytical!(dad, ana)
    return dad
end

"""
    apply_analytical_bc!(dad, ana, neumann_nodes)

Dirichlet everywhere except on `neumann_nodes`, where flux/traction from
`ana.q` is imposed.
"""
function apply_analytical_bc!(
    dad::BEMdata{<:Laplace},
    ana::AnalyticalSolution,
    neumann_nodes::AbstractVector{Int},
)
    neuset = Set(neumann_nodes)
    for i in 1:dad.n
        if i in neuset && ana.q !== nothing
            dad.BC[i] = 1
            dad.BV[i] = float(ana.q(dad.Nodes[i], dad.Normal[i]))
        else
            dad.BC[i] = 0
            dad.BV[i] = float(ana.u(dad.Nodes[i]))
        end
    end
    attach_analytical!(dad, ana)
    return dad
end

# ===========================================================================
# Laplace / potential
# ===========================================================================

"""
    ana_laplace_linear(; direction = SA[1.0, 0.0], k=1.0)

Linear field ``T = d·x`` (exact for Laplace).

Flux convention in this package: ``q = -k\\,∂T/∂n`` (outward normal).
Compatible with the default `quadrado` mesh (`direction = (1,0)`, `k=1`):
left Dirichlet ``T=0``, right Neumann ``q=-1``.
"""
function ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    d = direction isa SVector ? direction : SVector{length(direction)}(direction)
    u = (x; t=0.0) -> dot(d, _as_sv(x, length(d)))
    # q = -k ∂T/∂n = -k d·n
    q = (x, n; t=0.0) -> -k * dot(d, _as_sv(n, length(d)))
    return AnalyticalSolution(
        "laplace_linear",
        u;
        q=q,
        description="T = d·x (harmonic), q=-k ∂T/∂n",
    )
end

"""
    ana_laplace_quadratic()

Quadratic harmonic field ``T = x^2 - y^2`` in 2D.
"""
function ana_laplace_quadratic(; k=1.0)
    u = (x; t=0.0) -> begin
        p = _as_sv(x, 2)
        p[1]^2 - p[2]^2
    end
    # ∂T/∂n = 2x n_x - 2y n_y ; q = -k ∂T/∂n
    q = (x, n; t=0.0) -> begin
        p = _as_sv(x, 2)
        m = _as_sv(n, 2)
        -k * (2p[1] * m[1] - 2p[2] * m[2])
    end
    return AnalyticalSolution(
        "laplace_quadratic",
        u;
        q=q,
        description="T = x² - y² (harmonic), q=-k ∂T/∂n",
    )
end

"""
    ana_heat_1d(; α=1.0, L=1.0, N=200)

1D transient heat conduction on a rod (series solution used in `intro.jl`).
Evaluates ``T(x,t)`` with the point's first coordinate as ``x``.
"""
function ana_heat_1d(; α=1.0, L=1.0, N=200)
    u = (pt; t=0.0) -> begin
        x = _as_sv(pt, 2)[1]
        s = x
        for n in 1:2:(N-1)
            s -= (8 / (n^2 * π^2)) *
                 sin(n * π / 2) *
                 sin(n * π * x / 2) *
                 exp(-(α^2 * n^2 * π^2 * t) / 4)
        end
        return s
    end
    return AnalyticalSolution(
        "heat_1d",
        u;
        description="1D transient heat series (α=$α)",
    )
end

"""
    ana_heat_insulated_sides(; α=1.0)

Transient series for a strip with insulated sides (see `ana_3adi` in `intro.jl`).
Uses the point's ``y`` coordinate.
"""
function ana_heat_insulated_sides(; α=1.0, K=200)
    u = (pt; t=0.0) -> begin
        ys = _as_sv(pt, 2)[2]
        serie = 0.0
        for k in 0:K
            serie +=
                ((-1)^k) / (2 * k + 1) *
                exp(-((2 * k + 1)^2 * π^2 * α * t) / 4) *
                cos((2 * k + 1) * π * ys / 2)
        end
        return 1 - 4 / π * serie
    end
    return AnalyticalSolution(
        "heat_insulated_sides",
        u;
        description="Transient insulated-sides series",
    )
end

"""
    ana_heat_dirichlet_square(; α=1.0, M=30, N=30)

2D transient heat on the unit square with Dirichlet data (series from `intro.jl`).
"""
function ana_heat_dirichlet_square(; α=1.0, M=30, N=30)
    u = (pt; t=0.0) -> begin
        x, y = _as_sv(pt, 2)
        s = 100.0
        for m in 1:M, n in 1:N
            c1 = 800 * n * cos(m * π) / (π * (4n^2 - 1))
            c2 = 1600 * cos(m * π) / (π^2 * (4m^2 - 1))
            s -= (c1 * (sinh(n * π * x) * sin(n * π * y) + sinh(n * π * y) * sin(n * π * x)) -
                  c2 * (sin(m * π * x) * sin(n * π * y))) *
                 exp(-α * (m^2 + n^2) * π^2 * t)
        end
        return s
    end
    return AnalyticalSolution(
        "heat_dirichlet_square",
        u;
        description="2D Dirichlet square heat series",
    )
end

# ===========================================================================
# Classic potential benchmarks (potencial_direto / dadpotencial)
# ===========================================================================

"""Alias of [`ana_laplace_linear`](@ref) with ``T = x`` (potencial1d)."""
ana_potencial1d(; k=1.0) = ana_laplace_linear(; direction=SA[1.0, 0.0], k=k)

"""
    ana_quarto_circ(; ri=1, re=2, ti=100, qe=-200, k=1)

Radial conduction on a quarter annulus:
``T = ti - qe·re·log(r/ri)`` (exact for constant outer flux `qe`).
"""
function ana_quarto_circ(; ri=1.0, re=2.0, ti=100.0, qe=-200.0, k=1.0)
    B = -qe * re   # T = ti + B log(r/ri); q_r = -k dT/dr = -k B/r
    u = (x; t=0.0) -> begin
        p = _as_sv(x, 2)
        r = hypot(p[1], p[2])
        return ti + B * log(max(r, ri * 1e-15) / ri)
    end
    q = (x, n; t=0.0) -> begin
        p = _as_sv(x, 2)
        m = _as_sv(n, 2)
        r2 = p[1]^2 + p[2]^2
        r2 < 1e-30 && return 0.0
        # ∇T = (B/r²) (x,y) ; q = -k ∇T·n
        return -k * B / r2 * (p[1] * m[1] + p[2] * m[2])
    end
    return AnalyticalSolution(
        "quarto_circ",
        u;
        q=q,
        description="T = ti - qe·re·log(r/ri) (quarter annulus)",
    )
end

"""
    ana_moulton(; k=1)

Moulton crack-tip field ``T = √r cos(θ/2)``, ``θ = atan2(y,x)``.
Harmonic; singular flux at the origin.
"""
function ana_moulton(; k=1.0)
    u = (x; t=0.0) -> begin
        p = _as_sv(x, 2)
        r = hypot(p[1], p[2])
        θ = atan(p[2], p[1])
        return sqrt(r) * cos(θ / 2)
    end
    # ∇T from polar: ∂T/∂r = cos(θ/2)/(2√r), (1/r)∂T/∂θ = -sin(θ/2)/(2√r)
    # ∇T = (∂T/∂r) ê_r + (1/r ∂T/∂θ) ê_θ
    q = (x, n; t=0.0) -> begin
        p = _as_sv(x, 2)
        m = _as_sv(n, 2)
        r = hypot(p[1], p[2])
        r < 1e-14 && return 0.0
        θ = atan(p[2], p[1])
        c, s = cos(θ / 2), sin(θ / 2)
        dTdr = c / (2 * sqrt(r))
        dTdθ_over_r = -s / (2 * sqrt(r))
        # ê_r = (cos θ, sin θ), ê_θ = (-sin θ, cos θ)
        ct, st = cos(θ), sin(θ)
        dTdx = dTdr * ct + dTdθ_over_r * (-st)
        dTdy = dTdr * st + dTdθ_over_r * ct
        return -k * (dTdx * m[1] + dTdy * m[2])
    end
    return AnalyticalSolution(
        "moulton",
        u;
        q=q,
        description="T = √r cos(θ/2) (Moulton)",
    )
end

"""
    ana_laquini1(; ns=500)

Laquini thesis problem 1 (unit square): Fourier series for T=0 on x=0,x=1,y=0
and unit Neumann data on y=1 (package flux convention matched to dadpotencial).
"""
function ana_laquini1(; ns=500)
    u = (pt; t=0.0) -> begin
        x, y = _as_sv(pt, 2)
        T = 0.0
        @inbounds for n in 1:ns
            # numerically stable sinh form
            inc =
                ((((-1)^(n + 1)) + 1) / n) *
                sin(n * π * x) *
                ((1 - exp(-2n * π * y)) / (n * π * (1 + exp(-2n * π)))) *
                exp(n * π * (y - 1))
            isnan(inc) && break
            T += inc
        end
        return (2 / π) * T
    end
    return AnalyticalSolution(
        "laquini1",
        u;
        description="Laquini-1 series (Dirichlet 0 + top Neumann)",
    )
end

"""
    ana_laquini2(; ns=500)

Laquini problem 2: ``T = x +`` Fourier correction (mixed Neumann on top/right).
"""
function ana_laquini2(; ns=500)
    u = (pt; t=0.0) -> begin
        x, y = _as_sv(pt, 2)
        T = 0.0
        @inbounds for n in 1:ns
            np = n * π
            inc =
                sin(np * x / 2) * (
                    (
                        (2 / np) * (((-1)^(n + 1) + 1) / ((np / 2) * cosh(np / 2))) +
                        (8 / (n^2 * π^2)) * sin(np / 2) * tanh(np / 2)
                    ) * sinh(np * y / 2) -
                    ((8 / (n^2 * π^2)) * sin(np / 2) * cosh(np * y / 2))
                )
            isnan(inc) && break
            T += inc
        end
        return T + x
    end
    return AnalyticalSolution(
        "laquini2",
        u;
        description="Laquini-2 series (T=x + correction)",
    )
end

"""
    ana_laquini3(; ns=1000)

Laquini problem 3: T=0 on three sides, T=1 on top (Dirichlet square).
"""
function ana_laquini3(; ns=1000)
    u = (pt; t=0.0) -> begin
        x, y = _as_sv(pt, 2)
        T = 0.0
        @inbounds for n in 1:ns
            inc =
                ((((-1)^(n + 1)) + 1) / n) *
                sin(n * π * x) *
                ((1 - exp(-2n * π * y)) / (1 - exp(-2n * π))) *
                exp(n * π * (y - 1))
            isnan(inc) && break
            T += inc
        end
        return (2 / π) * T
    end
    return AnalyticalSolution(
        "laquini3",
        u;
        description="Laquini-3 series (top Dirichlet 1)",
    )
end

# ===========================================================================
# Elasticity
# ===========================================================================

"""
    ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01, εyy=0.0, εxy=0.0)

Uniform strain patch test: ``u = ε · x``. Exact for linear elasticity.
"""
function ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01, εyy=0.0, εxy=0.0)
    u = (x; t=0.0) -> begin
        p = _as_sv(x, 2)
        SA[εxx * p[1] + εxy * p[2], εxy * p[1] + εyy * p[2]]
    end
    # Hooke plane strain traction from constant stress
    μ = E / (2(1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    σxx = λ * (εxx + εyy) + 2μ * εxx
    σyy = λ * (εxx + εyy) + 2μ * εyy
    σxy = 2μ * εxy
    q = (x, n; t=0.0) -> begin
        m = _as_sv(n, 2)
        SA[σxx * m[1] + σxy * m[2], σxy * m[1] + σyy * m[2]]
    end
    return AnalyticalSolution(
        "elasticity_patch",
        u;
        q=q,
        description="Uniform strain patch (plane strain)",
    )
end

"""
    rel_error(dad; t=0.0) -> Float64

Relative L2 error ``\\|T - T_{ana}\\| / \\|T_{ana}\\|`` using the attached
analytical solution and `dad.T` (or `dad.u` for elasticity).
"""
function rel_error(dad::BEMdata; t=0.0)
    Tana = analytical(dad; t=t)
    Tnum = if has_cache(dad, :T)
        dad.T isa AbstractMatrix ? dad.T[:, end] : dad.T
    elseif has_cache(dad, :u)
        dad.u
    else
        error("No numerical solution in cache (:T or :u). Call solve first.")
    end
    n = min(length(Tnum), length(Tana))
    num = norm(@view(Tnum[1:n]) .- @view(Tana[1:n]))
    den = norm(@view(Tana[1:n]))
    return den > 0 ? num / den : num
end

# helpers
_as_sv(x::SVector, ::Int) = x
_as_sv(x::AbstractVector, n::Int) = SVector{n,Float64}(x[1:n]...)
_as_sv(x::NTuple, n::Int) = SVector{n,Float64}(x[1:n]...)
