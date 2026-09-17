# Geometric properties via boundary integrals (Green identity)
#
# 2D: A = ½ ∫_Γ x·n dΓ,  ∫_Ω x_i dΩ = ∫_Γ (x_i²/2) n_i dΓ
# 3D: V = ⅓ ∫_Γ x·n dΓ,  same per-component centroid formula

export GeometricProps2D, GeometricProps3D
export geometric_props, geometric_props_2d, geometric_props_3d
export geometric_props_2d_polygon

"""2D geometric properties of a planar region."""
struct GeometricProps2D
    perimeter::Float64
    area::Float64
    centroid::SVector{2,Float64}
end

"""3D geometric properties of a closed solid."""
struct GeometricProps3D
    surface_area::Float64
    volume::Float64
    centroid::SVector{3,Float64}
end

"""
    geometric_props(dad::BEMdata; npg_radial=12, npg_boundary=nothing) -> GeometricProps2D/3D

Perimeter/area/centroid (2D) or surface/volume/centroid (3D) of the domain
bounded by `dad`. `npg_radial` is ignored (closed-form Green identity).
`npg_boundary=nothing` reuses mesh quadrature; an integer re-integrates
each element with that many Gauss points.
"""
function geometric_props(dad::BEMdata; npg_radial=12, npg_boundary=nothing)
    if dad.dimension == 2
        return geometric_props_2d(dad; npg_boundary=npg_boundary)
    elseif dad.dimension == 3
        return geometric_props_3d(dad; npg_boundary=npg_boundary)
    else
        error("geometric_props: unsupported dimension $(dad.dimension)")
    end
end

function _geom_accum_2d(x, n, wJ, P, A, Sx, Sy)
    P += wJ
    A += 0.5 * dot(x, n) * wJ
    Sx += 0.5 * x[1]^2 * n[1] * wJ
    Sy += 0.5 * x[2]^2 * n[2] * wJ
    return P, A, Sx, Sy
end

function geometric_props_2d(dad::BEMdata; npg_radial=12, npg_boundary=nothing)
    P = A = Sx = Sy = 0.0
    if npg_boundary === nothing
        w_el = dad.elem_weight
        for elem in dad.elements
            for k in eachindex(elem.index)
                i = elem.index[k]
                wJ = elem.Jacobian[k] * w_el[k]
                P, A, Sx, Sy = _geom_accum_2d(dad.Nodes[i], dad.Normal[i], wJ, P, A, Sx, Sy)
            end
        end
    else
        poly = dad.element_type
        qsi, w = gausslegendre(Int(npg_boundary))
        for elem in dad.elements
            X = dad.Nodes[elem.index]
            nn = length(X)
            for (ig, ξ) in enumerate(qsi)
                L, dLdξ = shapefun(poly, ξ)
                x = zero(X[1])
                dx = zero(X[1])
                @inbounds for k in 1:nn
                    x += L[k] * X[k]
                    dx += dLdξ[k] * X[k]
                end
                J = norm(dx)
                J < 1e-16 && continue
                n = tan2normal(dx / J)
                P, A, Sx, Sy = _geom_accum_2d(x, n, J * w[ig], P, A, Sx, Sy)
            end
        end
    end
    c = A == 0 ? zero(SVector{2,Float64}) : SVector(Sx / A, Sy / A)
    return GeometricProps2D(P, A, c)
end

"""
    geometric_props_2d_polygon(verts; npg=12)

Closed polygon (ordered vertices, CCW). Uses shoelace area and the exact
polygon-centroid formula. `npg` is ignored.
"""
function geometric_props_2d_polygon(verts::AbstractVector{<:SVector{2}}; npg=12)
    n = length(verts)
    n >= 3 || error("need ≥ 3 vertices")
    A = polygon_area(verts)
    P = 0.0
    Cx = Cy = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        a, b = verts[i], verts[j]
        P += norm(b - a)
        cross = a[1] * b[2] - b[1] * a[2]
        Cx += (a[1] + b[1]) * cross
        Cy += (a[2] + b[2]) * cross
    end
    c = A == 0 ? zero(SVector{2,Float64}) : SVector(Cx / (6A), Cy / (6A))
    return GeometricProps2D(P, A, c)
end

function geometric_props_2d_polygon(xy::AbstractMatrix; npg=12)
    verts = [SVector{2,Float64}(xy[i, 1], xy[i, 2]) for i in 1:size(xy, 1)]
    return geometric_props_2d_polygon(verts; npg=npg)
end

function _geom_accum_3d(x, n, wJ, Area, Vol, Sx, Sy, Sz)
    Area += wJ
    Vol += (1 / 3) * dot(x, n) * wJ
    Sx += 0.5 * x[1]^2 * n[1] * wJ
    Sy += 0.5 * x[2]^2 * n[2] * wJ
    Sz += 0.5 * x[3]^2 * n[3] * wJ
    return Area, Vol, Sx, Sy, Sz
end

function geometric_props_3d(dad::BEMdata; npg_radial=12, npg_boundary=nothing)
    Area = Vol = Sx = Sy = Sz = 0.0
    poly = dad.element_type
    if npg_boundary === nothing
        w_el = dad.elem_weight
        for elem in dad.elements
            for k in eachindex(elem.index)
                i = elem.index[k]
                wJ = elem.Jacobian[k] * w_el[k]
                Area, Vol, Sx, Sy, Sz = _geom_accum_3d(
                    dad.Nodes[i], dad.Normal[i], wJ, Area, Vol, Sx, Sy, Sz)
            end
        end
    else
        npg = Int(npg_boundary)
        qsi, w = gausslegendre(npg)
        for elem in dad.elements
            X = dad.Nodes[elem.index]
            nn = length(X)
            if nn == 3
                for l in 1:npg, m in 1:npg
                    η = (qsi[l] + 1) / 2
                    ξ = (1 - η) * (qsi[m] + 1) / 2
                    N = SVector(ξ, η, 1 - ξ - η)
                    dNξ = SVector(1.0, 0.0, -1.0)
                    dNη = SVector(0.0, 1.0, -1.0)
                    x = N[1] * X[1] + N[2] * X[2] + N[3] * X[3]
                    dxξ = dNξ[1] * X[1] + dNξ[2] * X[2] + dNξ[3] * X[3]
                    dxη = dNη[1] * X[1] + dNη[2] * X[2] + dNη[3] * X[3]
                    nvec = cross(dxξ, dxη)
                    J = norm(nvec)
                    J < 1e-16 && continue
                    jac_map = (1 - η) / 4
                    ww = w[l] * w[m] * jac_map
                    Area, Vol, Sx, Sy, Sz = _geom_accum_3d(
                        x, nvec / J, ww * J, Area, Vol, Sx, Sy, Sz)
                end
            else
                for l in 1:npg, m in 1:npg
                    L, Lx, Ly = shapefun2D(poly, poly, qsi[m], qsi[l])
                    x = zero(X[1])
                    dxξ = zero(X[1])
                    dxη = zero(X[1])
                    @inbounds for k in 1:nn
                        x += L[k] * X[k]
                        dxξ += Lx[k] * X[k]
                        dxη += Ly[k] * X[k]
                    end
                    nvec = cross(dxξ, dxη)
                    J = norm(nvec)
                    J < 1e-16 && continue
                    ww = w[l] * w[m]
                    Area, Vol, Sx, Sy, Sz = _geom_accum_3d(
                        x, nvec / J, ww * J, Area, Vol, Sx, Sy, Sz)
                end
            end
        end
    end
    c = Vol == 0 ? zero(SVector{3,Float64}) : SVector(Sx / Vol, Sy / Vol, Sz / Vol)
    return GeometricProps3D(Area, Vol, c)
end
