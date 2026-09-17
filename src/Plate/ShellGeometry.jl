# Mid-surface geometry → Donnell curvatures κ_αβ ≈ z,αβ (shallow).
# Used by [`LaminatedShell`](@ref). Constant-κ tests wrap as `ConstantCurvature`.

export ShellGeometry, FlatShell, SphericalShell, CylindricalShell
export ConstantCurvature, HeightGraph, curvature_at, curvature_fields

"""Undeformed mid-surface. Donnell `κ_αβ` from [`curvature_at`](@ref)."""
abstract type ShellGeometry end

"""Planar mid-surface (`κ = 0`)."""
struct FlatShell <: ShellGeometry end

"""Spherical cap, Donnell `κ₁ = κ₂ = 1/R`."""
struct SphericalShell <: ShellGeometry
    R::Float64
end
SphericalShell(R::Real) = SphericalShell(Float64(R))

"""Circular cylinder of radius `R`.

`hoop=:x` → curvature in `x` (`κ₁=1/R`, generators along `y`).
`hoop=:y` → `κ₂=1/R`.
"""
struct CylindricalShell <: ShellGeometry
    R::Float64
    hoop::Symbol
end
CylindricalShell(R::Real; hoop::Symbol=:x) = CylindricalShell(Float64(R), hoop)

"""Uniform Donnell `κ₁, κ₂, κ₁₂` (legacy numeric constructor)."""
struct ConstantCurvature <: ShellGeometry
    κ1::Float64
    κ2::Float64
    κ12::Float64
end
ConstantCurvature(κ1::Real, κ2::Real, κ12::Real=0) =
    ConstantCurvature(Float64(κ1), Float64(κ2), Float64(κ12))

"""Graph `z(x,y)`. Donnell `κ = Hess z` (shallow second fundamental form)."""
struct HeightGraph{F} <: ShellGeometry
    z::F
end

"""Donnell `(κ₁, κ₂, κ₁₂)` at point `p` on the mid-surface."""
function curvature_at end
curvature_at(::FlatShell, p) = (0.0, 0.0, 0.0)
curvature_at(g::SphericalShell, p) = (1 / g.R, 1 / g.R, 0.0)
function curvature_at(g::CylindricalShell, p)
    invR = 1 / g.R
    g.hoop === :y && return (0.0, invR, 0.0)
    g.hoop === :x && return (invR, 0.0, 0.0)
    error("CylindricalShell hoop must be :x or :y, got $(g.hoop)")
end
curvature_at(g::ConstantCurvature, p) = (g.κ1, g.κ2, g.κ12)

function curvature_at(g::HeightGraph, p)
    x = SVector{2,Float64}(Float64(p[1]), Float64(p[2]))
    H = ForwardDiff.hessian(q -> g.z(q[1], q[2]), x)
    return (H[1, 1], H[2, 2], H[1, 2])
end

"""Nodal Donnell `(κ₁, κ₂, κ₁₂)` at collocation `pts`."""
function curvature_fields(g::ShellGeometry, pts)
    n = length(pts)
    κ1 = Vector{Float64}(undef, n)
    κ2 = Vector{Float64}(undef, n)
    κ12 = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        κ1[i], κ2[i], κ12[i] = curvature_at(g, pts[i])
    end
    return κ1, κ2, κ12
end
