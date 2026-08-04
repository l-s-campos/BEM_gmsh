# Geometric properties via boundary integrals (propgeo / propgeo3d)
#
# Primary API uses a prepared [`BEMdata`](@ref) (`dad`) and the mesh quadrature /
# shape functions already stored on it (`element_type`, `elem_weight`,
# `Element.Jacobian`, `Nodes`, `Normal`).

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

# =============================================================================
# Dispatch on BEMdata
# =============================================================================

"""
    geometric_props(dad::BEMdata; npg_radial=12, npg_boundary=nothing) -> GeometricProps2D/3D

Compute geometric properties of the domain bounded by `dad`'s mesh.

- **2D** (`dad.dimension == 2`): perimeter, area, centroid  
- **3D** (`dad.dimension == 3`): surface area, volume, centroid  

Integration uses the default element shape functions (`dad.element_type` /
[`shapefun`](@ref) / [`shapefun2D`](@ref)) and the mesh normals/Jacobians.

# Keywords
- `npg_radial`: Gauss points for the radial particular integral ``F = ∫_0^R f ρ\\,dρ``
- `npg_boundary`: if `nothing` (default), reuse the collocation quadrature already
  on the mesh (`elem.Jacobian` × `dad.elem_weight`). If an integer, re-integrate
  each element with that many Gauss points via `shapefun`.
"""
function geometric_props(dad::BEMdata; npg_radial=12, npg_boundary=nothing)
    if dad.dimension == 2
        return geometric_props_2d(dad; npg_radial=npg_radial, npg_boundary=npg_boundary)
    elseif dad.dimension == 3
        return geometric_props_3d(dad; npg_radial=npg_radial, npg_boundary=npg_boundary)
    else
        error("geometric_props: unsupported dimension $(dad.dimension)")
    end
end

# =============================================================================
# 2D
# =============================================================================

function geometric_props_2d(dad::BEMdata; npg_radial=12, npg_boundary=nothing)
    qsi_r, w_r = gausslegendre(npg_radial)
    P = A = xdA = ydA = 0.0

    if npg_boundary === nothing
        # ---- default: collocation quadrature already on dad ----
        w_el = dad.elem_weight
        for elem in dad.elements
            for k in eachindex(elem.index)
                i = elem.index[k]
                x = dad.Nodes[i]
                n = dad.Normal[i]
                wJ = elem.Jacobian[k] * w_el[k]
                P += wJ
                r = norm(x)
                r < 1e-14 && continue
                rx, ry = x[1] / r, x[2] / r
                nr = n[1] * rx + n[2] * ry
                Fa, Fx, Fy = _calc_F_2d(r, atan(ry, rx), qsi_r, w_r)
                A += Fa * nr / r * wJ
                xdA += Fx * nr / r * wJ
                ydA += Fy * nr / r * wJ
            end
        end
    else
        # ---- re-integrate with shapefun(dad.element_type, ξ) ----
        poly = dad.element_type
        qsi, w = gausslegendre(Int(npg_boundary))
        for elem in dad.elements
            X = dad.Nodes[elem.index]
            nn = length(X)
            for (ig, ξ) in enumerate(qsi)
                L, dLdξ = shapefun(poly, ξ)          # 1×nn at ξ
                x = zero(X[1])
                dx = zero(X[1])
                @inbounds for k in 1:nn
                    x += L[k] * X[k]
                    dx += dLdξ[k] * X[k]
                end
                J = norm(dx)
                J < 1e-16 && continue
                n = Point2D(dx[2] / J, -dx[1] / J)   # left normal
                wJ = J * w[ig]
                P += wJ
                r = norm(x)
                r < 1e-14 && continue
                rx, ry = x[1] / r, x[2] / r
                nr = n[1] * rx + n[2] * ry
                Fa, Fx, Fy = _calc_F_2d(r, atan(ry, rx), qsi_r, w_r)
                A += Fa * nr / r * wJ
                xdA += Fx * nr / r * wJ
                ydA += Fy * nr / r * wJ
            end
        end
    end

    c = A == 0 ? zero(SVector{2,Float64}) : SVector(xdA / A, ydA / A)
    return GeometricProps2D(P, A, c)
end

function _calc_F_2d(r, theta, qsi, w)
    dro = r / 2
    Fa = Fx = Fy = 0.0
    @inbounds for i in eachindex(qsi)
        ρ = r / 2 * (qsi[i] + 1)
        x = ρ * cos(theta)
        y = ρ * sin(theta)
        Fa += ρ * dro * w[i]
        Fx += x * ρ * dro * w[i]
        Fy += y * ρ * dro * w[i]
    end
    return Fa, Fx, Fy
end

"""
    geometric_props_2d_polygon(verts; npg=12)

Closed polygon (ordered vertices, CCW). Builds a temporary linear-element
`BEMdata` and calls [`geometric_props_2d`](@ref).
"""
function geometric_props_2d_polygon(verts::AbstractVector{<:SVector{2}}; npg=12)
    n = length(verts)
    n >= 3 || error("need ≥ 3 vertices")
    # collocation at segment midpoints + endpoints via 2-node Gauss
    poly = Equispaced(1)   # linear
    qsi, wi = gausslegendre(2)
    Nodes = Point2D[]
    Normal = Point2D[]
    elements = Element[]
    for e in 1:n
        p0 = verts[e]
        p1 = verts[mod1(e + 1, n)]
        X = (p0, p1)
        idx0 = length(Nodes) + 1
        Lmat, dL = shapefun(poly, qsi)   # 2×2
        for k in 1:2
            x = Lmat[k, 1] * X[1] + Lmat[k, 2] * X[2]
            dx = dL[k, 1] * X[1] + dL[k, 2] * X[2]
            J = norm(dx)
            push!(Nodes, Point2D(x))
            push!(Normal, Point2D(dx[2] / J, -dx[1] / J))
        end
        idx = collect(idx0:(idx0 + 1))
        # Jacobian at the two Gauss points
        Jvec = Float64[]
        for k in 1:2
            dx = dL[k, 1] * X[1] + dL[k, 2] * X[2]
            push!(Jvec, norm(dx))
        end
        push!(elements, Element(idx, Jvec, sum(Jvec .* wi), 1))
    end
    # minimal BEMdata
    dad = BEMdata(;
        name="polygon",
        dimension=2,
        elements=elements,
        element_type=poly,
        elem_weight=SVector{2,Float64}(wi),
        Nodes=Nodes,
        Normal=Normal,
        internalNodes=Point2D[],
        properties=Laplace(1.0),
        BC=ones(Int, length(Nodes)),
        BV=zeros(length(Nodes)),
        n=length(Nodes),
        ni=0,
        nt=length(Nodes),
    )
    return geometric_props_2d(dad; npg_radial=npg)
end

function geometric_props_2d_polygon(xy::AbstractMatrix; npg=12)
    verts = [SVector{2,Float64}(xy[i, 1], xy[i, 2]) for i in 1:size(xy, 1)]
    return geometric_props_2d_polygon(verts; npg=npg)
end

# =============================================================================
# 3D
# =============================================================================

function geometric_props_3d(dad::BEMdata; npg_radial=12, npg_boundary=nothing)
    qsi_r, w_r = gausslegendre(npg_radial)
    Area = Vol = xdV = ydV = zdV = 0.0
    poly = dad.element_type

    if npg_boundary === nothing
        w_el = dad.elem_weight
        for elem in dad.elements
            for k in eachindex(elem.index)
                i = elem.index[k]
                x = dad.Nodes[i]
                n = dad.Normal[i]
                wJ = elem.Jacobian[k] * w_el[k]
                Area += wJ
                R = norm(x)
                R < 1e-14 && continue
                jfac = dot(n, x) / R^3
                Fv, Fx, Fy, Fz = _calc_F_3d(x, qsi_r, w_r)
                Vol += Fv * jfac * wJ
                xdV += Fx * jfac * wJ
                ydV += Fy * jfac * wJ
                zdV += Fz * jfac * wJ
            end
        end
    else
        npg = Int(npg_boundary)
        qsi, w = gausslegendre(npg)
        for elem in dad.elements
            X = dad.Nodes[elem.index]
            nn = length(X)
            # triangle (3) or quad (4 / 9…) via shapefun2D tensor product
            if nn == 3
                # Duffy / collapsed quad for triangle using shapefun on Equispaced(1)
                # map (ξ,η)∈[-1,1]² → barycentric
                for l in 1:npg, m in 1:npg
                    η = (qsi[l] + 1) / 2
                    ξ = (1 - η) * (qsi[m] + 1) / 2
                    # linear triangle shape
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
                    Area += ww * J
                    R = norm(x)
                    R < 1e-14 && continue
                    n̂ = nvec / J
                    jfac = dot(n̂, x) / R^3
                    Fv, Fx, Fy, Fz = _calc_F_3d(x, qsi_r, w_r)
                    Vol += Fv * jfac * ww * J
                    xdV += Fx * jfac * ww * J
                    ydV += Fy * jfac * ww * J
                    zdV += Fz * jfac * ww * J
                end
            else
                # tensor-product quad with dad.element_type
                for l in 1:npg, m in 1:npg
                    L, Lx, Ly = shapefun2D(poly, poly, qsi[m], qsi[l])
                    # L is 1×nn (row), physical gradients need Jacobian of mapping
                    x = zero(X[1])
                    dxξ = zero(X[1])
                    dxη = zero(X[1])
                    @inbounds for k in 1:nn
                        x += L[k] * X[k]
                        # Lx, Ly are ∂N/∂ξ, ∂N/∂η at nodes of reference element
                        dxξ += Lx[k] * X[k]
                        dxη += Ly[k] * X[k]
                    end
                    nvec = cross(dxξ, dxη)
                    J = norm(nvec)
                    J < 1e-16 && continue
                    ww = w[l] * w[m]
                    Area += ww * J
                    R = norm(x)
                    R < 1e-14 && continue
                    n̂ = nvec / J
                    jfac = dot(n̂, x) / R^3
                    Fv, Fx, Fy, Fz = _calc_F_3d(x, qsi_r, w_r)
                    Vol += Fv * jfac * ww * J
                    xdV += Fx * jfac * ww * J
                    ydV += Fy * jfac * ww * J
                    zdV += Fz * jfac * ww * J
                end
            end
        end
    end

    c = Vol == 0 ? zero(SVector{3,Float64}) : SVector(xdV / Vol, ydV / Vol, zdV / Vol)
    return GeometricProps3D(Area, Vol, c)
end

function _calc_F_3d(rvec, qsi_raw, w)
    R = norm(rvec)
    qsi = (qsi_raw .+ 1) ./ 2
    dro = R
    Fv = Fx = Fy = Fz = 0.0
    @inbounds for i in eachindex(qsi)
        ρ = rvec * qsi[i]
        ro = norm(ρ)
        fac = ro^2 * dro * w[i] / 2
        Fv += fac
        Fx += ρ[1] * fac
        Fy += ρ[2] * fac
        Fz += ρ[3] * fac
    end
    return Fv, Fx, Fy, Fz
end
