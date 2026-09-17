# (BEM) structures
export Point2D, Point3D, Point, BEMdata, Element, Problem, Scalar, Vectorial
export Laplace, Helmholtz, Elasticity, AnisotropicElasticity, LekhnitskiiParams
export AnisotropicElasticity3D
export OrthotropicLaplace, AnisotropicLaplace, AxisymmetricElasticity
export AbstractThinPlate, ThinPlate
export AbstractFSDT, FSDT, n_dof
export BEMCache, has_cache, set_cache!
export shear_modulus, lame_λ, lame_mu, lame_constants, plane_strain_κ, effective_nu,
       plane_stress, refresh_lame!, thermal_modulus
export singularity_orders, singularity_order_G, singularity_order_H, kernel_eltype
export point, all_points, all_points!, set_internal_nodes!

"""
    Point2D
Represents a 2D point using StaticArray.
"""
const Point2D = SVector{2,Float64}

"""
    Point3D
Represents a 3D point using StaticArray.
"""
const Point3D = SVector{3,Float64}

const Point = Union{Point2D,Point3D}

abstract type Problem end
abstract type Scalar <: Problem end
abstract type Vectorial <: Problem end

"""Kirchhoff (thin) plate — 2 DOFs/node ``(w, ∂w/∂n)``. See [`ThinPlate`](@ref)."""
abstract type AbstractThinPlate <: Vectorial end

"""
FSDT / Reissner (thick) plate — 3 DOFs/node ``(ψx, ψy, w)``.
Geometry is still 2-D (`dad.dimension == 2`); see [`n_dof`](@ref).
"""
abstract type AbstractFSDT <: Vectorial end

"""
Isotropic Kirchhoff plate. ``D = E h³ / [12(1-ν²)]``.
Distributed load ``q = q_a x + q_b y + q_c``.

`BEMdata{<:ThinPlate}` uses the same collocation assembly as 2-D elasticity
([`H_G_full_direct`](@ref)): Gauss nodes, far lumping, sinh near-field,
on-element Guiggiani.
"""
@kwdef mutable struct ThinPlate <: AbstractThinPlate
    E::Float64 = 1.0
    ν::Float64 = 0.3
    h::Float64 = 0.01
    q_a::Float64 = 0.0
    q_b::Float64 = 0.0
    q_c::Float64 = 0.0
    ρ::Float64 = 1.0
end

"""
Isotropic FSDT / Reissner plate. ``λ = √10 / h`` (κ=5/6). Load ``q = q_c``.

`BEMdata{<:FSDT}` uses the same collocation assembly as 2-D elasticity
([`H_G_full_direct`](@ref)): Gauss nodes, far lumping, sinh near-field,
on-element Guiggiani. Prefer [`FSDTProps`](@ref) when `using BEM.Plate`.
"""
@kwdef mutable struct FSDT <: AbstractFSDT
    E::Float64 = 1.0
    ν::Float64 = 0.3
    h::Float64 = 0.01
    ρ::Float64 = 1.0
    q_c::Float64 = 0.0
end

"""Field DOFs per collocation node. Distinct from geometry `dad.dimension`."""
n_dof(::Scalar, ::Integer) = 1
n_dof(::Vectorial, d::Integer) = Int(d)
n_dof(::AbstractThinPlate, ::Integer) = 2
n_dof(::AbstractFSDT, ::Integer) = 3
n_dof(p::Problem) = n_dof(p, 2)

# On-element Guiggiani Laurent orders for (single-layer G, double-layer H).
# Convention matches `guiggiani_integral`: 0 = log, -1 = CPV 1/ρ, -2 = HFP 1/ρ².
"""Laurent orders `(order_G, order_H)` for on-element BIE kernels."""
singularity_orders(::Problem) = (0, -1)
# Kirchhoff P has 1/r² (HBIE-like); G is log-singular.
singularity_orders(::AbstractThinPlate) = (0, -2)
# Vander Weeën P is CPV 1/r (elasticity-like); G is log.
singularity_orders(::AbstractFSDT) = (0, -1)

singularity_order_G(p::Problem) = singularity_orders(p)[1]
singularity_order_H(p::Problem) = singularity_orders(p)[2]

"""Element type of single-layer / double-layer kernels for `p`."""
kernel_eltype(::Problem) = Float64

# ---------------------------------------------------------------------------
# Scalar problems
# ---------------------------------------------------------------------------

"""Laplace / steady heat conduction. Conductivity `k`. Flux ``q = -k ∂T/∂n``."""
@kwdef mutable struct Laplace{T} <: Scalar
    k::T = 1.0
end

"""
    Helmholtz(; ω=1.0, c=1.0)

Helmholtz / time-harmonic acoustics with wavenumber ``κ = ω/c``.
2-D kernels use Hankel ``H_0^{(1)}, H_1^{(1)}``; 3-D kernels use
``e^{iκR}/(4πR)``. Hypersingular kernels: [`fundamental_hyper`](@ref).
"""
@kwdef mutable struct Helmholtz{T} <: Scalar
    ω::T = 1.0          # angular frequency
    c::T = 1.0          # wave speed
end

wavenumber(h::Helmholtz) = h.ω / h.c
kernel_eltype(::Helmholtz) = ComplexF64

# ---------------------------------------------------------------------------
# Vectorial problems
# ---------------------------------------------------------------------------

"""
Isotropic linear elasticity (Kelvin fundamental solution).

# Fields
- `E`, `nu`, `rho` — Young modulus, Poisson ratio, density
- `plane_strain` — `true` → plane strain; `false` → plane stress (2D Kelvin)
- `α` — linear thermal expansion
- `lambda`, `mu` — Lamé parameters **cached at construction** (and whenever
  `E` / `nu` / `plane_strain` are set via `setproperty!`)

# 2D options
```julia
Elasticity(E, ν, ρ; plane_strain=true)   # default
Elasticity(E, ν, ρ; plane_stress=true)   # sets plane_strain=false
```

Lamé constants use the **effective** Poisson ratio of the 2D model:
- plane strain: ``ν̃ = ν``
- plane stress: ``ν̃ = ν/(1+ν)`` (same map as Kelvin kernels)

```text
μ = E / (2(1+ν))                 # material shear modulus
λ = E ν̃ / ((1+ν̃)(1-2ν̃))       # first Lamé (effective 2D/3D form)
```
"""
mutable struct Elasticity{T} <: Vectorial
    E::T
    nu::T
    rho::T
    plane_strain::Bool
    α::T
    lambda::T   # λ (Lamé first parameter, effective model)
    mu::T       # μ (shear modulus)
end

"""Compute ``(λ, μ)`` from ``E, ν`` and 2D plane mode."""
function lame_constants(E::Real, nu::Real, plane_strain::Bool)
    T = float(promote_type(typeof(E), typeof(nu)))
    E, nu = T(E), T(nu)
    μ = E / (2 * (1 + nu))
    νe = plane_strain ? nu : nu / (1 + nu)   # plane stress → effective ν
    den = (1 + νe) * (1 - 2 * νe)
    λ = abs(den) < eps(T) ? T(Inf) : E * νe / den
    return λ, μ
end

function _elasticity_new(
        ::Type{T}, E, nu, rho, plane_strain::Bool, α,
    ) where {T}
    λ, μ = lame_constants(E, nu, plane_strain)
    return Elasticity{T}(T(E), T(nu), T(rho), plane_strain, T(α), T(λ), T(μ))
end

"""
    Elasticity(E, nu, rho; plane_strain=true, plane_stress=false, α=0)

Positional constructor. Pass `plane_stress=true` to select plane stress
(overrides `plane_strain`).
"""
function Elasticity(
        E::Real, nu::Real, rho::Real;
        plane_strain::Bool = true,
        plane_stress::Bool = false,
        α = 0.0,
    )
    ps = plane_stress ? false : plane_strain
    T = float(promote_type(typeof(E), typeof(nu), typeof(rho), typeof(α)))
    return _elasticity_new(T, E, nu, rho, ps, α)
end

"""Keyword constructor (defaults match historical `@kwdef`)."""
function Elasticity(; E=1.0, nu=0.3, rho=1.0, plane_strain=true, plane_stress=false, α=0.0)
    return Elasticity(E, nu, rho; plane_strain, plane_stress, α)
end

function Elasticity{T}(; E=one(T), nu=T(0.3), rho=one(T),
        plane_strain=true, plane_stress=false, α=zero(T)) where {T}
    ps = plane_stress ? false : plane_strain
    return _elasticity_new(T, E, nu, rho, ps, α)
end

"""Recompute cached Lamé fields from current `E`, `nu`, `plane_strain`."""
function refresh_lame!(e::Elasticity{T}) where {T}
    λ, μ = lame_constants(e.E, e.nu, e.plane_strain)
    setfield!(e, :lambda, T(λ))
    setfield!(e, :mu, T(μ))
    return e
end

function Base.setproperty!(e::Elasticity{T}, name::Symbol, v) where {T}
    if name === :lambda || name === :mu
        throw(ArgumentError(
            "do not set `$name` directly; set E, nu, or plane_strain (or plane_stress)"))
    end
    if name === :plane_stress
        setfield!(e, :plane_strain, !Bool(v))
        return refresh_lame!(e)
    end
    if name === :E || name === :nu || name === :α
        setfield!(e, name, convert(T, v))
        name === :α || refresh_lame!(e)
        return v
    elseif name === :plane_strain
        setfield!(e, :plane_strain, Bool(v))
        return refresh_lame!(e)
    elseif name === :rho
        return setfield!(e, :rho, convert(T, v))
    else
        return setfield!(e, name, v)
    end
end

# Accessors (prefer structure fields; keep function API stable)
shear_modulus(e::Elasticity) = e.mu
lame_mu(e::Elasticity) = e.mu
lame_λ(e::Elasticity) = e.lambda
plane_stress(e::Elasticity) = !e.plane_strain

"""Kolosov constant ``κ``: plane strain ``3-4ν``, plane stress ``(3-ν)/(1+ν)``."""
function plane_strain_κ(e::Elasticity)
    ν = e.nu
    return e.plane_strain ? (3 - 4ν) : (3 - ν) / (1 + ν)
end
plane_strain_κ(E, ν, plane_strain::Bool) =
    plane_strain ? (3 - 4ν) : (3 - ν) / (1 + ν)

"""
Thermal stress modulus ``k̂ = E α / (1-2ν̃)`` with effective Poisson ``ν̃``
(same map as Kelvin).
"""
function thermal_modulus(e::Elasticity)
    ν = effective_nu(e)
    return e.E * e.α / (1 - 2ν)
end

"""Effective Poisson ratio used inside the 2D Kelvin kernels."""
function effective_nu(e::Elasticity)
    e.plane_strain && return e.nu
    return e.nu / (1 + e.nu)   # plane stress → equivalent plane strain ν
end

"""
    LekhnitskiiParams

Complex Lekhnitskii / Stroh-like parameters for 2D anisotropic elasticity
(from the characteristic equation of the compliance matrix).

Built by [`lekhnitskii_params`](@ref) from orthotropic constants or a full
``3×3`` reduced stiffness.
"""
struct LekhnitskiiParams{T}
    mi::SVector{2,Complex{T}}   # complex roots μ₁, μ₂ (Im > 0)
    A::SMatrix{2,2,Complex{T},4}
    q::SMatrix{2,2,Complex{T},4}
    g::SMatrix{2,2,Complex{T},4}
    C::SMatrix{3,3,T,9}         # reduced stiffness (Voigt 11,22,12)
end

"""
Anisotropic 2D linear elasticity using Lekhnitskii fundamental solutions.
"""
@kwdef mutable struct AnisotropicElasticity{T} <: Vectorial
    params::LekhnitskiiParams{T}
    rho::T = one(T)
end

AnisotropicElasticity(params::LekhnitskiiParams{T}; rho=one(T)) where {T} =
    AnisotropicElasticity{T}(params, rho)

"""
3D anisotropic linear elasticity (Ting–Lee / Barnett–Lothe Green’s function).

`C` is the 6×6 Voigt stiffness. `C4` is the equivalent ``C_{ijkl}`` tensor.
"""
mutable struct AnisotropicElasticity3D{T} <: Vectorial
    C::SMatrix{6,6,T,36}
    C4::Array{T,4}
    rho::T
    nψ::Int
end
thermal_modulus(::AnisotropicElasticity3D) = 0.0

"""
    Element

Boundary element connectivity and metrics.

# Fields
- `index` — global indices into `dad.Nodes`
- `Jacobian` — ``|dx/dξ|`` samples (collocation or quadrature)
- `Length` — element arc length
- `Region` — Gmsh entity / physical region tag
"""
@kwdef mutable struct Element
    index::Vector{Int64}
    Jacobian::Vector{Float64}
    Length::Float64
    Region::Int64
    """CAD / geometric nodes (ξ = −1…1 Lagrange). Empty → geometry from `index`."""
    geo::Vector{Point2D} = Point2D[]
end

# Positional 4-arg ctor (Gmsh tags may be Int32)
function Element(
        index::AbstractVector{<:Integer},
        Jacobian::AbstractVector{<:Real},
        Length::Real,
        Region::Integer,
    )
    return Element(;
        index = collect(Int64, index),
        Jacobian = collect(Float64, Jacobian),
        Length = Float64(Length),
        Region = Int64(Region),
    )
end

Base.size(elem::Element) = (length(elem.index),)
Base.length(elem::Element) = length(elem.index)

# =============================================================================
# Working cache (replaces NamedTuple)
# =============================================================================

"""
    BEMCache

Mutable store for assembled operators, solutions, and auxiliaries attached to a
[`BEMdata`](@ref).

# Why not `NamedTuple`?
Rebuilding `dad.cache = (; dad.cache..., H=H)` is:
- **type-unstable** (a new type every time a field is added),
- easy to **desync** (forgotten fields, accidental copies),
- allocates a new tuple on every update.

`BEMCache` keeps a **fixed set of fields**, mutated in place via
[`set_cache!`](@ref). Matrix fields are `Any` because they may be dense
`Matrix` or hierarchical `HMatrix` — that residual instability is local to
field access and does not rebuild the whole object.

# Fields
| Group | Fields |
|-------|--------|
| Operators | `H`, `G`, `A`, `B`, `b`, `M` |
| Quadrature | `qsi`, `w` |
| Solutions | `T`, `q`, `u`, `traction`, `time` |
| Recovery | `strain`, `stress` (Voigt, from `∇u`) |
| Extras | `analytical`, `ode_sol`, `gmres_stats`, `extras::Dict` |

Unset fields are `nothing`. Use [`has_cache`](@ref)`(dad, :H)` or
`haskey(dad.cache, :H)`.

Unknown `set_cache!` names go in `extras`. Common keys:

| Key | Set by |
|-----|--------|
| `:cells` | `format2d` (Gmsh 2-D elements) |
| `:dibem_F`, `:dibem_c`, `:dibem_ID`, `:dibem_D`, `:dibem_rbf`, `:dibem_method` | `DIBEM` |
| `:dibem_IF`, `:dibem_IP`, `:dibem_U`, `:dibem_Q`, `:dibem_centers` | DIBEM variants |
| `:surf_dibem` | 3-D face DIBEM (`nearfield=:dibem`): `(nedge, ξ, η, c)` |
| `:bc_idx`, `:hg_blocks`, `:block_lu` | mixed-BC block path |
| `:sbm` | `assemble_sbm!` |
| `:lbem_*` | local BEM |
| `:twin`, `:eq_type` | dual BEM |
| `:topology` | `bemdata_from_loops` |
| `:modal_basis`, `:M_DA`, `:H_hyper`, `:galerkin_*` | Laplace specialty methods |
"""
mutable struct BEMCache
    H::Any
    G::Any
    A::Any
    B::Any
    b::Any
    M::Any
    qsi::Any
    w::Any
    T::Any
    q::Any
    u::Any
    traction::Any   # elasticity traction
    time::Any       # time grid for transient
    strain::Any     # Voigt ε at collocation (n × 3 or n × 6)
    stress::Any     # Voigt σ at collocation
    analytical::Any
    ode_sol::Any
    gmres_stats::Any
    extras::Dict{Symbol,Any}
end

function BEMCache()
    return BEMCache(
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing,
        nothing, nothing, nothing, nothing, nothing,
        nothing, nothing,
        nothing, nothing, nothing,
        Dict{Symbol,Any}(),
    )
end

function Base.show(io::IO, c::BEMCache)
    set = Symbol[]
    for s in fieldnames(BEMCache)
        s === :extras && continue
        getfield(c, s) !== nothing && push!(set, s)
    end
    extras = collect(keys(c.extras))
    print(io, "BEMCache(set=$(set)")
    isempty(extras) || print(io, ", extras=$(extras)")
    print(io, ")")
end

"""True if cache field `sym` is set (not `nothing`)."""
function Base.haskey(c::BEMCache, sym::Symbol)
    if hasfield(BEMCache, sym) && sym !== :extras
        return getfield(c, sym) !== nothing
    end
    return haskey(c.extras, sym) && c.extras[sym] !== nothing
end

function Base.getindex(c::BEMCache, sym::Symbol)
    if hasfield(BEMCache, sym) && sym !== :extras
        return getfield(c, sym)
    end
    return c.extras[sym]
end

function Base.setindex!(c::BEMCache, val, sym::Symbol)
    if hasfield(BEMCache, sym) && sym !== :extras
        setfield!(c, sym, val)
    else
        c.extras[sym] = val
    end
    return val
end

"""
    has_cache(dad, sym) -> Bool

Whether `dad.cache` holds a non-`nothing` value for `sym`.
"""
has_cache(dad, sym::Symbol) = haskey(dad.cache, sym)

"""
    set_cache!(dad; kwargs...)

Set one or more cache fields in place.

```julia
set_cache!(dad; H=H, G=G, T=T)
```

Unknown names go into `cache.extras`.
"""
function set_cache!(dad; kwargs...)
    c = dad.cache
    for (k, v) in pairs(kwargs)
        c[k] = v
    end
    return dad
end

# =============================================================================
# Main problem container
# =============================================================================

"""
    BEMdata

Collocation geometry lives in one vector `collocation`:

| indices | view | meaning |
|---------|------|---------|
| `1:n` | `dad.Nodes` | boundary |
| `n+1:nt` | `dad.internalNodes` | internal poles |

`all_points(dad)` is `collocation` (no copy). Prefer [`point`](@ref)`(dad,i)`
in hot loops. Replace internals with [`set_internal_nodes!`](@ref).
"""
@kwdef mutable struct BEMdata{P<:Problem}
    name::AbstractString
    dimension::Int
    elements::Vector{Element}
    element_type::AbstractPolynomial
    elem_weight::SVector
    collocation::Vector{<:Point}   # boundary then internal
    Normal::Vector{<:Point}        # boundary only, length n
    properties::P
    BC::Vector{Int}
    BV::Vector{Float64}
    n::Int64
    ni::Int64
    nt::Int64
    cache::BEMCache = BEMCache()
end

# Forward cache fields as `dad.H`, `dad.T`, … and views for Nodes / internalNodes
function Base.getproperty(dad::BEMdata, sym::Symbol)
    if sym === :Nodes
        c = getfield(dad, :collocation)
        n = Int(getfield(dad, :n))
        return view(c, 1:n)
    elseif sym === :internalNodes
        c = getfield(dad, :collocation)
        n = Int(getfield(dad, :n))
        nt = Int(getfield(dad, :nt))
        return view(c, (n + 1):nt)
    elseif sym === :points
        return getfield(dad, :collocation)
    elseif sym in fieldnames(typeof(dad))
        return getfield(dad, sym)
    end
    c = getfield(dad, :cache)
    if sym === :t
        if c.time !== nothing
            return c.time
        elseif c.traction !== nothing
            return c.traction
        end
        error("cache.t is not set (neither time nor traction)")
    end
    if hasfield(BEMCache, sym) && sym !== :extras
        v = getfield(c, sym)
        v === nothing && error("cache.$sym is not set — assemble/solve first?")
        return v
    end
    haskey(c.extras, sym) && return c.extras[sym]
    error("BEMdata has no property or cache field `$sym`")
end

function Base.setproperty!(dad::BEMdata, sym::Symbol, val)
    if sym === :internalNodes
        return set_internal_nodes!(dad, val)
    elseif sym === :Nodes
        throw(ArgumentError(
            "dad.Nodes is a view into collocation[1:n]; assign elements or rebuild BEMdata"))
    elseif sym in fieldnames(typeof(dad))
        return setfield!(dad, sym, val)
    end
    if sym === :t
        c = getfield(dad, :cache)
        c.time = val
        return val
    end
    c = getfield(dad, :cache)
    c[sym] = val
    return val
end

function Base.propertynames(dad::BEMdata, private::Bool=false)
    c = getfield(dad, :cache)
    cached = Symbol[s for s in fieldnames(BEMCache) if s !== :extras && getfield(c, s) !== nothing]
    return (:Nodes, :internalNodes, :points, fieldnames(typeof(dad))..., cached..., keys(c.extras)...)
end

# ---------------------------------------------------------------------------
# Collocation point access (boundary + internal)
# ---------------------------------------------------------------------------

"""
    point(dad, i) -> Point

Collocation point `collocation[i]` (`i ∈ 1:nt`). Zero allocations.
"""
@inline function point(dad::BEMdata, i::Integer)
    @boundscheck (1 <= i <= dad.nt) || throw(BoundsError(dad, i))
    return @inbounds getfield(dad, :collocation)[i]
end

"""
    all_points(dad) -> Vector

Return the live `collocation` storage (boundary then internal). **No copy.**
`ClusterTree` defaults to `copy_elements=true`, so it will not permute in place.
"""
all_points(dad::BEMdata) = getfield(dad, :collocation)

n_dof(dad::BEMdata) = n_dof(dad.properties, dad.dimension)

"""
    all_points!(pts, dad) -> pts

Copy collocation into `pts` (resize if needed). Use when the consumer mutates.
"""
function all_points!(pts::AbstractVector, dad::BEMdata)
    c = getfield(dad, :collocation)
    nt = length(c)
    length(pts) == nt || resize!(pts, nt)
    copyto!(pts, c)
    return pts
end

"""
    set_internal_nodes!(dad, pts) -> dad

Replace internal collocation poles. Boundary `1:n` is unchanged.
Updates `ni`, `nt`, and resizes `collocation`.
"""
function set_internal_nodes!(dad::BEMdata, pts)
    n = Int(getfield(dad, :n))
    coll = getfield(dad, :collocation)
    ni = length(pts)
    resize!(coll, n + ni)
    @inbounds for k in 1:ni
        coll[n + k] = pts[k]
    end
    setfield!(dad, :ni, Int64(ni))
    setfield!(dad, :nt, Int64(n + ni))
    return dad
end

function Base.show(io::IO, d::BEMdata{P}) where {P<:Problem}
    n = length(d.Nodes)
    ni = length(d.internalNodes)
    ne = length(d.elements)
    println(io, "BEMdata \"$(d.name)\"")
    println(io, "  dimension: $(d.dimension), problem: $(typeof(d.properties))")
    println(io, "  nodes: $n, internal nodes: $ni, total nodes: $(n + ni)")
    println(io, "  elements: $ne")
    if d.properties isa Laplace
        println(io, "  properties: k=$(d.properties.k)")
    elseif d.properties isa OrthotropicLaplace
        println(io, "  properties: k1=$(d.properties.k1), k2=$(d.properties.k2)")
    elseif d.properties isa AnisotropicLaplace
        println(io, "  properties: anisotropic Laplace K=$(d.properties.K)")
    elseif d.properties isa Helmholtz
        println(io, "  properties: ω=$(d.properties.ω), c=$(d.properties.c), κ=$(wavenumber(d.properties))")
    elseif d.properties isa Elasticity
        println(io, "  properties: E=$(d.properties.E), ν=$(d.properties.nu), ρ=$(d.properties.rho), plane_strain=$(d.properties.plane_strain)")
    elseif d.properties isa AnisotropicElasticity
        println(io, "  properties: anisotropic (Lekhnitskii), ρ=$(d.properties.rho)")
    elseif d.properties isa AnisotropicElasticity3D
        println(io, "  properties: anisotropic 3D, ρ=$(d.properties.rho)")
    elseif d.properties isa ThinPlate
        println(io, "  properties: Kirchhoff E=$(d.properties.E), ν=$(d.properties.ν), h=$(d.properties.h)")
    elseif d.properties isa AbstractThinPlate
        println(io, "  properties: Kirchhoff anisotropic")
    else
        println(io, "  properties: $(d.properties)")
    end
    println(io, "  $(d.cache)")
end
