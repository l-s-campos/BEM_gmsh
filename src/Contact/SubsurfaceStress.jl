"""
    SubsurfaceStress

Johnson / Boussinesq–Cerruti interior stress kernels and Love-style rectangle
integrals (Juliá Lerma Paper 1 Apps. A, C).

Stresses are those in **one** body `α` (use that body's `ν`). Tractions
`(px, py, pn)` are the surface contact tractions of the half-space problem.
"""
module SubsurfaceStress

using LinearAlgebra
using ..ContactHalfSpace

export von_mises, hertz_axis_stress
export point_stress_normal, point_stress_shear_x, point_stress_shear_y
export subsurface_stress, subsurface_plane

"""Von Mises equivalent from Cauchy components."""
function von_mises(σxx, σyy, σzz, σxy, σxz, σyz)
    J2 = 0.5 * ((σxx - σyy)^2 + (σyy - σzz)^2 + (σzz - σxx)^2) +
         3 * (σxy^2 + σxz^2 + σyz^2)
    return sqrt(max(J2, 0.0))
end

"""
Hertz sphere, on-axis stresses in the elastic body (Johnson).
`ζ = z/a`; returns `(σr, σz, σVM)` (σθ = σr).
"""
function hertz_axis_stress(z, a, p0, ν)
    ζ = z / a
    den = 1 + ζ^2
    σz = -p0 / den
    σr = -p0 * ((1 + ν) * (1 - ζ * atan(1 / ζ)) - 0.5 / den)
    σVM = abs(σr - σz)
    return σr, σz, σVM
end

# -----------------------------------------------------------------------------
# Point kernels Tij (unit force at the origin)
# -----------------------------------------------------------------------------

@inline _ρ(x, y, z) = sqrt(x * x + y * y + z * z)
@inline _r2(x, y) = x * x + y * y

function point_stress_normal(x, y, z, ν)
    ρ = _ρ(x, y, z)
    ρ == 0 && return _nan6()
    r2 = max(_r2(x, y), 1e-30 * (ρ * ρ))
    c = 1 / (2π)
    ρ3 = ρ^3
    ρ5 = ρ^5
    Tzz = -3 * z^3 * c / ρ5
    Txz = -3 * x * z^2 * c / ρ5
    Tyz = -3 * y * z^2 * c / ρ5
    zmρ = 1 - z / ρ
    Txx = c * ((1 - 2ν) / r2 * (zmρ * (x * x - y * y) / r2 + z * y * y / ρ3) - 3 * z * x * x / ρ5)
    Tyy = c * ((1 - 2ν) / r2 * (zmρ * (y * y - x * x) / r2 + z * x * x / ρ3) - 3 * z * y * y / ρ5)
    Txy = c * ((1 - 2ν) / r2 * (zmρ * (x * y) / r2 - x * y * z / ρ3) - 3 * x * y * z / ρ5)
    return (xx=Txx, yy=Tyy, zz=Tzz, xy=Txy, xz=Txz, yz=Tyz)
end

function point_stress_shear_x(x, y, z, ν)
    ρ = _ρ(x, y, z)
    ρ == 0 && return _nan6()
    c = 1 / (2π)
    ρ3 = ρ^3
    ρ5 = ρ^5
    ρz = ρ + z
    ρz2 = ρz * ρz
    ρz3 = ρz2 * ρz
    νf = 1 - 2ν
    Txx = c * (-3 * x^3 / ρ5 + νf * (x / ρ3 - 3x / (ρ * ρz2) + x^3 / (ρ3 * ρz2) + 2 * x^3 / (ρ^2 * ρz3)))
    Tyy = c * (-3 * x * y^2 / ρ5 + νf * (x / ρ3 - x / (ρ * ρz2) + x * y^2 / (ρ3 * ρz2) + 2 * x * y^2 / (ρ^2 * ρz3)))
    Tzz = -3 * x * z^2 * c / ρ5
    Txy = c * (-3 * x^2 * y / ρ5 + νf * (-y / (ρ * ρz2) + x^2 * y / (ρ3 * ρz2) + 2 * x^2 * y / (ρ^2 * ρz3)))
    Txz = -3 * x^2 * z * c / ρ5
    Tyz = -3 * x * y * z * c / ρ5
    return (xx=Txx, yy=Tyy, zz=Tzz, xy=Txy, xz=Txz, yz=Tyz)
end

function point_stress_shear_y(x, y, z, ν)
    # Tij^Sy(x,y,z) = Tij^Sx(y,x,z) with xx↔yy, xz↔yz
    S = point_stress_shear_x(y, x, z, ν)
    return (xx=S.yy, yy=S.xx, zz=S.zz, xy=S.xy, xz=S.yz, yz=S.xz)
end

_nan6() = (xx=NaN, yy=NaN, zz=NaN, xy=NaN, xz=NaN, yz=NaN)

# -----------------------------------------------------------------------------
# Antiderivatives T̄ (Paper 1 App. C) — four-corner Love stencil
# -----------------------------------------------------------------------------

@inline function _atan_xyz(x, y, z, ρ)
    # atan(xy / (ρ z)); z→0 → (π/2) sign(xy)
    if abs(z) < 1e-30 * max(abs(x), abs(y), 1.0)
        return (π / 2) * copysign(1.0, x * y)
    end
    return atan(x * y, ρ * z)
end

function tbar_normal(x, y, z, ν)
    ρ = _ρ(x, y, z)
    ρ == 0 && return _nan6()
    ρy = ρ + y
    ρx = ρ + x
    at = _atan_xyz(x, y, z, ρ)
    Txx = -2ν * at + 2(1 - 2ν) * (atan(x, ρ + y + z) - x * z / (ρ * ρy + eps(ρ)))
    Tyy = -2ν * _atan_xyz(y, x, z, ρ) + 2(1 - 2ν) * (atan(y, ρ + x + z) - y * z / (ρ * ρx + eps(ρ)))
    Tzz = -at + x * z / (ρ * ρy + eps(ρ)) + y * z / (ρ * ρx + eps(ρ))
    Txy = (2ν - 1) * log(max(ρ + z, floatmin(ρ))) - z / ρ
    Txz = -z^2 / (ρ * ρy + eps(ρ))
    Tyz = -z^2 / (ρ * ρx + eps(ρ))
    return (xx=Txx, yy=Tyy, zz=Tzz, xy=Txy, xz=Txz, yz=Tyz)
end

function tbar_shear_x(x, y, z, ν)
    ρ = _ρ(x, y, z)
    ρ == 0 && return _nan6()
    ρy = ρ + y
    ρz = ρ + z
    νf = 1 - 2ν
    Txx = 2 * log(max(ρy, floatmin(ρ))) + z * νf * (y / (ρ * ρz + eps(ρ)) + z / (ρ * ρy + eps(ρ))) -
          2ν * x^2 / (ρ * ρy + eps(ρ))
    Tyy = 2ν * log(max(ρy, floatmin(ρ))) - z * νf * y / (ρ * ρz + eps(ρ)) - 2ν * y / ρ
    Tzz = -z^2 / (ρ * ρy + eps(ρ))
    Txy = log(max(x + ρ, floatmin(ρ))) - z * νf * x / (ρ * ρz + eps(ρ)) - 2ν * x / ρ
    Txz = -x * z / (ρ * ρy + eps(ρ)) - _atan_xyz(x, y, z, ρ)
    Tyz = -z / ρ
    return (xx=Txx, yy=Tyy, zz=Tzz, xy=Txy, xz=Txz, yz=Tyz)
end

function tbar_shear_y(x, y, z, ν)
    S = tbar_shear_x(y, x, z, ν)
    return (xx=S.yy, yy=S.xx, zz=S.zz, xy=S.xy, xz=S.yz, yz=S.xz)
end

@inline function _corners(tbar, x, y, z, hx, hy, ν)
    hx2, hy2 = hx / 2, hy / 2
    tpp = tbar(x + hx2, y + hy2, z, ν)
    tmm = tbar(x - hx2, y - hy2, z, ν)
    tmp_ = tbar(x - hx2, y + hy2, z, ν)
    tpm = tbar(x + hx2, y - hy2, z, ν)
    return tpp, tmm, tmp_, tpm
end

"""Rectangle influence ``B_ij`` for a unit traction on a cell of size `(hx,hy)`
centred at the origin, observer at `(x,y,z)` (relative)."""
function B_normal(x, y, z, hx, hy, ν)
    tpp, tmm, tm_p, tp_m = _corners(tbar_normal, x, y, z, hx, hy, ν)
    return _stencil(tpp, tmm, tm_p, tp_m)
end

function B_shear_x(x, y, z, hx, hy, ν)
    tpp, tmm, tm_p, tp_m = _corners(tbar_shear_x, x, y, z, hx, hy, ν)
    return _stencil(tpp, tmm, tm_p, tp_m)
end

function B_shear_y(x, y, z, hx, hy, ν)
    tpp, tmm, tm_p, tp_m = _corners(tbar_shear_y, x, y, z, hx, hy, ν)
    return _stencil(tpp, tmm, tm_p, tp_m)
end

function _stencil(tpp, tmm, tm_p, tp_m)
    s = 1 / (2π)
    return (
        xx = s * (tpp.xx + tmm.xx - tm_p.xx - tp_m.xx),
        yy = s * (tpp.yy + tmm.yy - tm_p.yy - tp_m.yy),
        zz = s * (tpp.zz + tmm.zz - tm_p.zz - tp_m.zz),
        xy = s * (tpp.xy + tmm.xy - tm_p.xy - tp_m.xy),
        xz = s * (tpp.xz + tmm.xz - tm_p.xz - tp_m.xz),
        yz = s * (tpp.yz + tmm.yz - tm_p.yz - tp_m.yz),
    )
end

# -----------------------------------------------------------------------------
# Superposition on a query point / plane
# -----------------------------------------------------------------------------

"""
    subsurface_stress(xq, yq, zq, px, py, pn, x, y, hs, ν_body) -> NamedTuple

Stress at a single point `(xq,yq,zq)` due to cell-wise constant surface tractions
on the grid with centres `x, y`.
"""
function subsurface_stress(
    xq::Real, yq::Real, zq::Real,
    px::AbstractMatrix, py::AbstractMatrix, pn::AbstractMatrix,
    x::AbstractVector, y::AbstractVector,
    hs::ElasticHalfSpace, ν_body::Real,
)
    hx, hy = hs.hx, hs.hy
    nx, ny = length(x), length(y)
    acc = zeros(6)
    @inbounds for j in 1:ny, i in 1:nx
        xr = xq - x[i]
        yr = yq - y[j]
        if pn[i, j] != 0
            B = B_normal(xr, yr, zq, hx, hy, ν_body)
            acc[1] += B.xx * pn[i, j]
            acc[2] += B.yy * pn[i, j]
            acc[3] += B.zz * pn[i, j]
            acc[4] += B.xy * pn[i, j]
            acc[5] += B.xz * pn[i, j]
            acc[6] += B.yz * pn[i, j]
        end
        if px[i, j] != 0
            B = B_shear_x(xr, yr, zq, hx, hy, ν_body)
            acc[1] += B.xx * px[i, j]
            acc[2] += B.yy * px[i, j]
            acc[3] += B.zz * px[i, j]
            acc[4] += B.xy * px[i, j]
            acc[5] += B.xz * px[i, j]
            acc[6] += B.yz * px[i, j]
        end
        if py[i, j] != 0
            B = B_shear_y(xr, yr, zq, hx, hy, ν_body)
            acc[1] += B.xx * py[i, j]
            acc[2] += B.yy * py[i, j]
            acc[3] += B.zz * py[i, j]
            acc[4] += B.xy * py[i, j]
            acc[5] += B.xz * py[i, j]
            acc[6] += B.yz * py[i, j]
        end
    end
    return (xx=acc[1], yy=acc[2], zz=acc[3], xy=acc[4], xz=acc[5], yz=acc[6],
            VM=von_mises(acc[1], acc[2], acc[3], acc[4], acc[5], acc[6]))
end

"""
    subsurface_plane(xs, zs, y0, px, py, pn, x, y, hs, ν) -> Matrix

Evaluate `σ_VM` on the plane `y = y0` at pairs `(xs[i], zs[k])`.
Returns a `length(xs) × length(zs)` matrix.
"""
function subsurface_plane(
    xs::AbstractVector, zs::AbstractVector, y0::Real,
    px::AbstractMatrix, py::AbstractMatrix, pn::AbstractMatrix,
    x::AbstractVector, y::AbstractVector,
    hs::ElasticHalfSpace, ν_body::Real,
)
    σVM = zeros(length(xs), length(zs))
    @inbounds for k in eachindex(zs), i in eachindex(xs)
        σ = subsurface_stress(xs[i], y0, zs[k], px, py, pn, x, y, hs, ν_body)
        σVM[i, k] = σ.VM
    end
    return σVM
end

end # module
