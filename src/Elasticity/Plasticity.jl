# Constant-cell initial-stress BEM (Telles 1983 / Gao & Davies 2002).
# 2-D von Mises, linear isotropic hardening. Cells from format2d Gmsh surfaces.

export VonMises, inplane_stiffness
export cell_integral_Estrain, cell_integral_Estress, cell_integral_Estress_gao
export assemble_plastic_ops!, solve_elastoplastic!
export ana_thick_cylinder_plastic, ana_thick_cylinder_tresca, apply_radius_pressure!

"""von Mises with initial yield `σY` and linear isotropic hardening `H′`."""
struct VonMises{T}
    σY::T
    H′::T
end
VonMises(; σY, H′=0.0) = VonMises(float(σY), float(H′))

"""In-plane Voigt ``C`` ``(ε_{11},ε_{22},γ_{12})→(σ_{11},σ_{22},σ_{12})``."""
function inplane_stiffness(props::Elasticity)
    if props.plane_strain
        λ, μ = props.lambda, props.mu
        return @SMatrix [λ+2μ  λ  0; λ  λ+2μ  0; 0  0  μ]
    end
    E, ν = props.E, props.nu
    c = E / (1 - ν^2)
    μ = E / (2 * (1 + ν))
    return @SMatrix [c  c*ν  0; c*ν  c  0; 0  0  μ]
end

# ---------------------------------------------------------------------------
# Cell-edge integrals
# ---------------------------------------------------------------------------

function _ccw_verts(cell::DomainCell)
    verts = cell.verts
    return polygon_area(verts) >= 0 ? verts : reverse(verts)
end

"""
    cell_integral_Estrain(props, x0, cell; npg=12) -> SMatrix{2,3}

``∫_{Ω_c} E_{ijk}\\,dΩ`` as a cell-edge integral of Kelvin ``U``:
``∫_{∂Ω_c} U_{ij}\\,n_k\\,dΓ`` (symmetric ``σ``).
"""
function cell_integral_Estrain(props::Elasticity, x0::SVector{2},
        cell::DomainCell; npg::Int=12)
    verts = _ccw_verts(cell)
    nv = length(verts)
    nv < 3 && return zero(SMatrix{2,3,Float64,6})
    ηs, ws = gausslegendre(npg)
    n0 = SVector(1.0, 0.0)
    acc = zero(MMatrix{2,3,Float64})
    @inbounds for k in 1:nv
        a = verts[k]
        b = verts[k == nv ? 1 : k + 1]
        tvec = b - a
        J = norm(tvec)
        J < 1e-16 && continue
        n̂ = tan2normal(tvec / J)
        Nmat = @SMatrix [n̂[1]  0.0  n̂[2]; 0.0  n̂[2]  n̂[1]]
        for q in eachindex(ηs)
            y = ((1 - ηs[q]) * a + (1 + ηs[q]) * b) / 2
            r = y - x0
            R = norm(r)
            R < 1e-14 && continue
            Umat = _to_smat(fundamental(props, r, n0).U)
            acc .+= (Umat * Nmat) * (ws[q] * (J / 2))
        end
    end
    return SMatrix(acc)
end

"""
    cell_integral_Estress(props, x0, cell; npg=16, free_term=false) -> SMatrix{3,3}

Interior-stress cell operator from the **source derivative of the displacement
BIE**, not from a strongly singular ``E_{ijkl}`` volume integral.

On a constant cell ``∫_{Ω_c} E_{ijk}\\,dΩ = ∫_{∂Ω_c} U_{ij} n_k\\,dΓ``. The
centroid is off the edges, so ``∂/∂p`` of that edge integral is a regular
``1/r`` quadrature of [`fundamental_grad`](@ref). Hooke maps the strain to
``σ^e``; `free_term=true` subtracts ``I`` because ``σ = σ^e - σ^p`` when
``p`` lies in the cell.
"""
function cell_integral_Estress(props::Elasticity, x0::SVector{2},
        cell::DomainCell; npg::Int=16, free_term::Bool=false)
    verts = _ccw_verts(cell)
    nv = length(verts)
    nv < 3 && return zero(SMatrix{3,3,Float64,9})
    ηs, ws = gausslegendre(npg)
    n0 = SVector(1.0, 0.0)
    dQdx = zero(MMatrix{2,3,Float64})
    dQdy = zero(MMatrix{2,3,Float64})
    @inbounds for k in 1:nv
        a = verts[k]
        b = verts[k == nv ? 1 : k + 1]
        tvec = b - a
        J = norm(tvec)
        J < 1e-16 && continue
        n̂ = tan2normal(tvec / J)
        Nmat = @SMatrix [n̂[1]  0.0  n̂[2]; 0.0  n̂[2]  n̂[1]]
        for q in eachindex(ηs)
            y = ((1 - ηs[q]) * a + (1 + ηs[q]) * b) / 2
            r = y - x0
            R = norm(r)
            R < 1e-14 && continue
            Ux, _, Uy, _ = fundamental_grad(props, r, n0)
            wJ = ws[q] * (J / 2)
            # ∂U/∂p = −∂U/∂y  (r = y − p)
            dQdx .+= (-_to_smat(Ux) * Nmat) * wJ
            dQdy .+= (-_to_smat(Uy) * Nmat) * wJ
        end
    end
    # ε = M σ^p  with γ12 = ∂u1/∂p2 + ∂u2/∂p1
    M = @SMatrix [
        dQdx[1, 1]  dQdx[1, 2]  dQdx[1, 3]
        dQdy[2, 1]  dQdy[2, 2]  dQdy[2, 3]
        (dQdy[1, 1] + dQdx[2, 1])  (dQdy[1, 2] + dQdx[2, 2])  (dQdy[1, 3] + dQdx[2, 3])
    ]
    E = inplane_stiffness(props) * M
    return free_term ? E - I : E
end

"""Gao–Davies ``∫_{∂Ω_c} E_{ijkl}(r·n)\\ln r\\,dΓ`` (+ ``F``). Check only."""
function cell_integral_Estress_gao(props::Elasticity, x0::SVector{2},
        cell::DomainCell; npg::Int=16, free_term::Bool=false)
    verts = _ccw_verts(cell)
    nv = length(verts)
    nv < 3 && return zero(SMatrix{3,3,Float64,9})
    ηs, ws = gausslegendre(npg)
    acc = zero(MMatrix{3,3,Float64})
    @inbounds for k in 1:nv
        a = verts[k]
        b = verts[k == nv ? 1 : k + 1]
        tvec = b - a
        J = norm(tvec)
        J < 1e-16 && continue
        n̂ = tan2normal(tvec / J)
        for q in eachindex(ηs)
            y = ((1 - ηs[q]) * a + (1 + ηs[q]) * b) / 2
            r = y - x0
            R = norm(r)
            R < 1e-14 && continue
            w = ws[q] * (J / 2) * dot(r, n̂) * log(R)
            acc .+= initial_stress_kernel(props, r) * w
        end
    end
    E = SMatrix(acc)
    return free_term ? E + initial_stress_free_term(props) : E
end

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

"""
    assemble_plastic_ops!(dad; domain=:cells, npg=12, npg_stress=16, threaded=true)

`domain=:cells` — piecewise-constant cell integrals.
`domain=:dibem` — DIBEM of ``∫ E_{ijk} σ^p`` with RBF centres at **cell centroids**.

- `plastic_Q`  — ``2 n_t × 3 n_c``,  ``H u = G t + Q σ^p``
- `plastic_Su`, `plastic_St` — interior Voigt stress from boundary ``u,t``
- `plastic_Sσ` — stress at the same `n_c` points from ``σ^p``
"""
function assemble_plastic_ops!(dad::BEMdata{<:Elasticity};
        domain::Symbol=:cells, npg::Int=12, npg_stress::Int=16,
        threaded::Bool=true, rbf=PHS(3; poly_deg=1), remainder::Symbol=:shepard)
    domain in (:cells, :dibem) || throw(ArgumentError(
        "domain must be :cells or :dibem (got $domain)"))
    dad.dimension == 2 || error("plasticity operators are 2-D only")
    has_cache(dad, :H) || error("assemble_plastic_ops!: call assemble! first")
    if domain === :dibem
        return assemble_plastic_dibem!(dad; npg=npg, npg_stress=npg_stress,
            threaded=threaded, rbf=rbf, remainder=remainder)
    end
    cells = extract_domain_cells(dad)
    isempty(cells) && error("assemble_plastic_ops!: no Gmsh 2-D cells")
    props = dad.properties
    nc = length(cells)
    nt = dad.nt
    Q = zeros(2nt, 3nc)
    Sσ = zeros(3nc, 3nc)
    _collocation_loop!(threaded, nt) do i
        pf = point(dad, i)
        rows = 2i-1:2i
        @inbounds for k in 1:nc
            Q[rows, 3k-2:3k] .= cell_integral_Estrain(props, pf, cells[k]; npg=npg)
        end
    end
    _collocation_loop!(threaded, nc) do k
        pf = cells[k].centroid
        rows = 3k-2:3k
        @inbounds for c in 1:nc
            Sσ[rows, 3c-2:3c] .= cell_integral_Estress(props, pf, cells[c];
                npg=npg_stress, free_term=(c == k))
        end
    end
    pts = [c.centroid for c in cells]
    Su, St = _plastic_boundary_stress_ops(dad, pts)
    set_cache!(dad; plastic_Q=Q, plastic_Sσ=Sσ, plastic_Su=Su, plastic_St=St,
        plastic_nc=nc, plastic_pts=pts, plastic_domain=:cells)
    return (; Q, Sσ, Su, St, cells)
end

function _plastic_boundary_stress_ops(dad::BEMdata{<:Elasticity},
        pts::AbstractVector)
    nc = length(pts)
    n = dad.n
    Su = zeros(3nc, 2n)
    St = zeros(3nc, 2n)
    has_cache(dad, :qsi) || _init_quadrature!(dad, 16)
    @inbounds for k in 1:nc
        pf = pts[k]
        rows = 3k-2:3k
        for el in dad.elements
            xj = dad.Nodes[el.index]
            N, r, nrm, wwJ = _quad_geom(dad, el, xj, pf)
            nn = size(N, 2)
            for iq in eachindex(wwJ)
                sk = fundamental_stress(dad, r[iq], nrm[iq])
                DD, SS = sk.D, sk.S
                wi = wwJ[iq]
                Dblk = @SMatrix [
                    DD[1, 1, 1] DD[2, 1, 1]
                    DD[1, 2, 2] DD[2, 2, 2]
                    DD[1, 1, 2] DD[2, 1, 2]
                ]
                Sblk = @SMatrix [
                    SS[1, 1, 1] SS[2, 1, 1]
                    SS[1, 2, 2] SS[2, 2, 2]
                    SS[1, 1, 2] SS[2, 1, 2]
                ]
                for a in 1:nn
                    ja = el.index[a]
                    cols = 2ja-1:2ja
                    Na = N[iq, a] * wi
                    St[rows, cols] .+= Dblk .* Na
                    Su[rows, cols] .-= Sblk .* Na
                end
            end
        end
    end
    return Su, St
end

_plastic_boundary_stress_ops(dad::BEMdata{<:Elasticity}, cells::Vector{DomainCell}) =
    _plastic_boundary_stress_ops(dad, [c.centroid for c in cells])

# ---------------------------------------------------------------------------
# DIBEM domain integral (centres = cell centroids)
# ---------------------------------------------------------------------------

function _nearest_center(p, ξ)
    imin = 1
    dmin = Inf
    @inbounds for k in eachindex(ξ)
        d = sum(abs2, ξ[k] - p)
        if d < dmin
            dmin = d
            imin = k
        end
    end
    return imin, dmin
end

"""``∫_Γ U n\\,dΓ`` (2×3): displacement of a uniform initial stress."""
function _boundary_integral_Estrain(dad::BEMdata{<:Elasticity}, x0; npg::Int=12)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    acc = zero(MMatrix{2,3,Float64})
    n0 = SVector(1.0, 0.0)
    @inbounds for el in dad.elements
        xj = dad.Nodes[el.index]
        _, r, nrm, wwJ = _quad_geom(dad, el, xj, x0)
        for iq in eachindex(wwJ)
            R = norm(r[iq])
            R < 1e-14 && continue
            Umat = _to_smat(fundamental(dad.properties, r[iq], n0).U)
            n̂ = nrm[iq]
            Nmat = @SMatrix [n̂[1]  0.0  n̂[2]; 0.0  n̂[2]  n̂[1]]
            acc .+= (Umat * Nmat) * wwJ[iq]
        end
    end
    return SMatrix(acc)
end

"""``C(∂/∂p ∫_Γ U n) − I``: stress of a uniform initial stress at an interior point."""
function _boundary_integral_Estress(dad::BEMdata{<:Elasticity}, x0; npg::Int=16,
        free_term::Bool=true)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    dQdx = zero(MMatrix{2,3,Float64})
    dQdy = zero(MMatrix{2,3,Float64})
    n0 = SVector(1.0, 0.0)
    @inbounds for el in dad.elements
        xj = dad.Nodes[el.index]
        _, r, nrm, wwJ = _quad_geom(dad, el, xj, x0)
        for iq in eachindex(wwJ)
            R = norm(r[iq])
            R < 1e-14 && continue
            Ux, _, Uy, _ = fundamental_grad(dad.properties, r[iq], n0)
            n̂ = nrm[iq]
            Nmat = @SMatrix [n̂[1]  0.0  n̂[2]; 0.0  n̂[2]  n̂[1]]
            w = wwJ[iq]
            dQdx .+= (-_to_smat(Ux) * Nmat) * w
            dQdy .+= (-_to_smat(Uy) * Nmat) * w
        end
    end
    M = @SMatrix [
        dQdx[1, 1]  dQdx[1, 2]  dQdx[1, 3]
        dQdy[2, 1]  dQdy[2, 2]  dQdy[2, 3]
        (dQdy[1, 1] + dQdx[2, 1])  (dQdy[1, 2] + dQdx[2, 2])  (dQdy[1, 3] + dQdx[2, 3])
    ]
    E = inplane_stiffness(dad.properties) * M
    return free_term ? E - I : E
end

"""
    assemble_plastic_dibem!(dad; rbf=PHS(3; poly_deg=1), npg=12)

DIBEM for ``∫_Ω E_{ijk} σ^p\\,dΩ``. Centres are the **internal collocation
points from [`format2d`](@ref)** (`pontointerno=true`, cell centroids).
Internal rows and near-boundary cell pairs use the regular cell-edge
``∫ U n``; far pairs use ``c_k E(p_i,ξ_k)``. Remainder (optional) via
`_dibem_center_Q` so a uniform field on the boundary reproduces
``∫_Γ U n``. This is a far-field quadrature of the same constant-cell
operator, not a higher-order interpolant of ``σ^p``.
"""
function assemble_plastic_dibem!(dad::BEMdata{<:Elasticity};
        rbf=PHS(3; poly_deg=1), npg::Int=12, npg_stress::Int=16,
        threaded::Bool=true, remainder::Symbol=:shepard)
    dad.ni > 0 || error("plastic DIBEM needs format2d internals (pontointerno=true)")
    n = dad.n
    ξ = [point(dad, i) for i in (n + 1):dad.nt]
    nc = length(ξ)
    nt = dad.nt
    pts = all_points(dad)
    F = zeros(nc, nc)
    @inbounds for k in 1:nc, j in 1:nc
        j == k && continue
        F[j, k] = rbf(norm(ξ[j] - ξ[k]))
    end
    _dibem_ridge_F!(F)
    IF = _dibem_rbf_IF(dad, rbf, ξ; npg=npg)
    IP = _dibem_monomial_IP(dad, rbf)
    c = _dibem_poly_c(F, IF, ξ, rbf; IP=IP)
    cells = extract_domain_cells(dad)
    length(cells) == nc || error("plastic DIBEM: ni=$(nc) ≠ ncells=$(length(cells))")
    Q = zeros(2nt, 3nc)
    Sσ = zeros(3nc, 3nc)
    _collocation_loop!(threaded, nt) do i
        pf = pts[i]
        rows = 2i-1:2i
        if i > n
            # Internal collocation sits on a format2d centre: regular cell-edge
            # ∫U n (same as domain=:cells).
            @inbounds for k in 1:nc
                Q[rows, 3k-2:3k] .= cell_integral_Estrain(dad.properties, pf,
                    cells[k]; npg=npg)
            end
        else
            IE = _boundary_integral_Estrain(dad, pf; npg=npg)
            acc = zero(MMatrix{2,3,Float64})
            @inbounds for k in 1:nc
                r = ξ[k] - pf
                R2 = sum(abs2, r)
                hk = sqrt(cells[k].area)
                if R2 < (2.5 * hk)^2
                    blk = cell_integral_Estrain(dad.properties, pf, cells[k]; npg=npg)
                else
                    blk = c[k] * initial_strain_kernel(dad.properties, r)
                end
                Q[rows, 3k-2:3k] .= blk
                acc .+= blk
            end
            rem = IE - SMatrix(acc)
            _plastic_dibem_remainder!(Q, rows, rem, pf, ξ, remainder)
        end
    end
    # Stress at the same internals. Cell-edge ∂U/∂p (regular at the centroid).
    _collocation_loop!(threaded, nc) do m
        pf = ξ[m]
        rows = 3m-2:3m
        @inbounds for k in 1:nc
            Sσ[rows, 3k-2:3k] .= cell_integral_Estress(dad.properties, pf, cells[k];
                npg=npg_stress, free_term=(k == m))
        end
    end
    Su, St = _plastic_boundary_stress_ops(dad, ξ)
    set_cache!(dad; plastic_Q=Q, plastic_Sσ=Sσ, plastic_Su=Su, plastic_St=St,
        plastic_nc=nc, plastic_pts=ξ, plastic_domain=:dibem, plastic_dibem_c=c,
        plastic_dibem_remainder=remainder)
    return (; Q, Sσ, Su, St, c, pts=ξ)
end

"""Put `IE − ∑ c E` on centre columns. `:shepard` weights centres (sum 1);
coincident format2d internals take the whole remainder. `:nearest` lumps
on the closest centre. Both keep `Q 1 = IE` on boundary rows."""
function _plastic_dibem_remainder!(Q, rows, rem, pf, ξ, remainder::Symbol)
    nc = length(ξ)
    if remainder === :none
        return
    elseif remainder === :nearest
        kn, _ = _nearest_center(pf, ξ)
        Q[rows, 3kn-2:3kn] .+= rem
        return
    elseif remainder !== :shepard
        throw(ArgumentError("DIBEM remainder must be :shepard, :nearest, or :none"))
    end
    @inbounds for k in 1:nc
        if sum(abs2, ξ[k] - pf) < 1e-24
            Q[rows, 3k-2:3k] .+= rem
            return
        end
    end
    wsum = 0.0
    ws = Vector{Float64}(undef, nc)
    @inbounds for k in 1:nc
        d2 = sum(abs2, ξ[k] - pf)
        ws[k] = 1 / max(d2, 1e-16)
        wsum += ws[k]
    end
    @inbounds for k in 1:nc
        Q[rows, 3k-2:3k] .+= (ws[k] / wsum) * rem
    end
    return
end

# ---------------------------------------------------------------------------
# Constitutive update
# ---------------------------------------------------------------------------

@inline function _vmises_plane(σx, σy, τ)
    return sqrt(max(σx * σx - σx * σy + σy * σy + 3 * τ * τ, 0.0))
end

function _dvmises_plane(σx, σy, τ)
    q = _vmises_plane(σx, σy, τ)
    q < 1e-16 && return zero(SVector{3,Float64})
    return SVector((2σx - σy) / (2q), (2σy - σx) / (2q), 3τ / q)
end

"""Plane-stress J2 return from start-of-increment `(σ, κ)` and `Δε` Voigt."""
function _return_plane_stress(σ, κ, Δε, C, mat::VonMises)
    σt = σ + C * Δε
    q = _vmises_plane(σt[1], σt[2], σt[3])
    σY = mat.σY + mat.H′ * κ
    q <= σY + 1e-14 && return σt, κ
    Δγ = 0.0
    σc = σt
    @inbounds for _ in 1:25
        q = _vmises_plane(σc[1], σc[2], σc[3])
        f = q - (mat.σY + mat.H′ * (κ + Δγ))
        abs(f) < 1e-12 * (1 + mat.σY) && break
        n = _dvmises_plane(σc[1], σc[2], σc[3])
        d = dot(n, C * n) + mat.H′
        abs(d) < 1e-30 && break
        Δγ = max(Δγ + f / d, 0.0)
        σc = σt - Δγ * (C * n)
    end
    return σc, κ + Δγ
end

"""Plane-strain J2 return. `σ4 = (σx,σy,σz,τ)`, `Δε = (Δεx,Δεy,Δγ)` with `Δεz=0`."""
function _return_plane_strain(σ4, κ, Δε, λ, μ, mat::VonMises)
    σt = SVector(
        σ4[1] + (λ + 2μ) * Δε[1] + λ * Δε[2],
        σ4[2] + λ * Δε[1] + (λ + 2μ) * Δε[2],
        σ4[3] + λ * (Δε[1] + Δε[2]),
        σ4[4] + μ * Δε[3],
    )
    p = (σt[1] + σt[2] + σt[3]) / 3
    s = SVector(σt[1] - p, σt[2] - p, σt[3] - p, σt[4])
    J2 = 0.5 * (s[1]^2 + s[2]^2 + s[3]^2) + s[4]^2
    q = sqrt(max(3 * J2, 0.0))
    σY = mat.σY + mat.H′ * κ
    q <= σY + 1e-14 && return σt, κ
    Δγ = (q - σY) / (3μ + mat.H′)
    fac = 1 - 3μ * Δγ / q
    σn = SVector(fac * s[1] + p, fac * s[2] + p, fac * s[3] + p, fac * s[4])
    return σn, κ + Δγ
end

"""Pack `nc×3` Voigt rows into a length-`3nc` vector (copy, no alias)."""
function _pack_voigt(σ::AbstractMatrix)
    nc = size(σ, 1)
    v = Vector{Float64}(undef, 3 * nc)
    @inbounds for k in 1:nc
        v[3k-2] = σ[k, 1]
        v[3k-1] = σ[k, 2]
        v[3k] = σ[k, 3]
    end
    return v
end

function _unpack_voigt!(σ::AbstractMatrix, v::AbstractVector)
    nc = size(σ, 1)
    @inbounds for k in 1:nc
        σ[k, 1] = v[3k-2]
        σ[k, 2] = v[3k-1]
        σ[k, 3] = v[3k]
    end
    return σ
end

"""Add the particular-stress contribution to `σB` (already `Su u + St t`)."""
function _add_plastic_stress!(σB::AbstractVector, Sσ::AbstractMatrix,
        σp_vec::AbstractVector, props, stress_coupling::Symbol)
    nc = length(σp_vec) ÷ 3
    if stress_coupling === :full
        σB .+= Sσ * σp_vec
    elseif stress_coupling === :local
        @inbounds for k in 1:nc
            i0 = 3k - 2
            s1, s2, s3 = σp_vec[i0], σp_vec[i0 + 1], σp_vec[i0 + 2]
            σB[i0]     += Sσ[i0, i0] * s1     + Sσ[i0, i0 + 1] * s2     + Sσ[i0, i0 + 2] * s3
            σB[i0 + 1] += Sσ[i0 + 1, i0] * s1 + Sσ[i0 + 1, i0 + 1] * s2 + Sσ[i0 + 1, i0 + 2] * s3
            σB[i0 + 2] += Sσ[i0 + 2, i0] * s1 + Sσ[i0 + 2, i0 + 1] * s2 + Sσ[i0 + 2, i0 + 2] * s3
        end
    elseif stress_coupling === :jump
        Fblk = initial_stress_free_term(props)
        @inbounds for k in 1:nc
            i0 = 3k - 2
            v = Fblk * SVector(σp_vec[i0], σp_vec[i0 + 1], σp_vec[i0 + 2])
            σB[i0] += v[1]
            σB[i0 + 1] += v[2]
            σB[i0 + 2] += v[3]
        end
    elseif stress_coupling !== :none
        throw(ArgumentError("stress_coupling must be :full, :local, :jump, or :none"))
    end
    return σB
end

# ---------------------------------------------------------------------------
# Incremental solver (Telles initial-stress iteration)
# ---------------------------------------------------------------------------

"""
    solve_elastoplastic!(dad, mat::VonMises; nsteps=10, maxiter=40, tol=1e-4)

Monotonic load-control. Domain integral `:cells` (default) or `:dibem`
(RBF centres = cell centroids). Reuses `dad.A` from [`applyBC`](@ref).
Stores `dad.u`, `dad.traction`, `dad.stress` (Voigt at plastic points),
`plastic_strain`, and `plastic_initial_stress`.

`stress_coupling`: `:jump` (free term `F` only, default), `:local` (owning-cell
`Sσ` block), `:full` (all of `Sσ`), or `:none`. Inner iteration damped
`:picard` (default) or `:broyden` (fewer iters, same cylinder path).
"""
function solve_elastoplastic!(dad::BEMdata{<:Elasticity}, mat::VonMises;
        nsteps::Int=10, maxiter::Int=80, tol::Float64=1e-4,
        npg::Int=12, npg_stress::Int=16, threaded::Bool=true,
        save_history::Bool=false, domain::Symbol=:cells, rbf=PHS(3; poly_deg=1),
        relax::Float64=0.4, stress_coupling::Symbol=:jump, inner::Symbol=:picard,
        remainder::Symbol=:shepard)
    stress_coupling in (:full, :local, :jump, :none) ||
        throw(ArgumentError("stress_coupling must be :full, :local, :jump, or :none"))
    inner in (:picard, :broyden) ||
        throw(ArgumentError("inner must be :picard or :broyden"))
    has_cache(dad, :H) || assemble!(dad)
    has_cache(dad, :plastic_Q) ||
        assemble_plastic_ops!(dad; domain=domain, npg=npg, npg_stress=npg_stress,
            threaded=threaded, rbf=rbf, remainder=remainder)
    applyBC(dad)
    A = dad.A
    bfull = copy(dad.b)
    Q = dad.plastic_Q
    Su = dad.plastic_Su
    St = dad.plastic_St
    Sσ = dad.plastic_Sσ
    nc = Int(dad.plastic_nc)
    n = dad.n
    dim = 2
    C = inplane_stiffness(dad.properties)
    Cinv = inv(C)
    λ, μ = dad.properties.lambda, dad.properties.mu
    plane = dad.properties.plane_strain

    σ = zeros(nc, 3)
    σz = zeros(nc)
    σp = zeros(nc, 3)
    κ = zeros(nc)
    u = zeros(dim * n)
    traction = zeros(dim * n)

    AF = factorize(A)
    x = zeros(length(bfull))
    nplast = 0
    history = save_history ? NamedTuple[] : nothing
    @inbounds for step in 1:nsteps
        load = step / nsteps
        σn = copy(σ)
        σzn = copy(σz)
        κn = copy(κ)
        εn = zeros(nc, 3)
        for k in 1:nc
            εn[k, :] = Cinv * SVector(σn[k, 1] + σp[k, 1],
                σn[k, 2] + σp[k, 2], σn[k, 3] + σp[k, 3])
        end
        σp_vec = _pack_voigt(σp)
        update! = function (sp)
            return _plastic_return!(σ, σz, κ, σp, u, traction, x, dad, AF, load,
                bfull, Q, Su, St, Sσ, sp, σn, σzn, κn, εn, C, Cinv, λ, μ, mat,
                plane, stress_coupling)
        end
        if inner === :broyden
            resid, nit, nplast = _broyden_sigma!(σp_vec, update!; maxiter=maxiter, tol=tol)
            if resid >= 10 * tol
                trial = copy(σp_vec)
                r2, n2, np2 = _picard_sigma!(trial, update!; maxiter=maxiter,
                    tol=tol, relax=relax)
                if isfinite(r2) && r2 < resid
                    copyto!(σp_vec, trial)
                    resid, nit, nplast = r2, nit + n2, np2
                end
                # Picard trial mutates σ,u; refresh from the accepted σp.
                _, nplast = update!(σp_vec)
            end
        else
            resid, nit, nplast = _picard_sigma!(σp_vec, update!; maxiter=maxiter,
                tol=tol, relax=relax)
        end
        _unpack_voigt!(σp, σp_vec)
        resid < 10 * tol || @warn "elastoplastic step $step residual $resid"
        if save_history
            push!(history, (; step, load, u=copy(u), traction=copy(traction),
                stress=copy(σ), plastic_strain=copy(κ), resid, niters=nit,
                nplast))
        end
    end
    uint = length(x) > dim * n ? x[dim*n+1:end] : Float64[]
    set_cache!(dad; u=u, traction=traction, T=u, uint=uint, stress=σ,
        plastic_strain=κ, plastic_initial_stress=σp, plastic_sigmaz=σz,
        plastic_ncells=nplast, plastic_history=history)
    return u
end

function _picard_sigma!(σp_vec, update!; maxiter::Int, tol::Float64, relax::Float64)
    ω = clamp(relax, 0.05, 1.0)
    resid = Inf
    nit = 0
    nplast = 0
    @inbounds for iter in 1:maxiter
        nit = iter
        σp_new, nplast = update!(σp_vec)
        any(!isfinite, σp_new) && return (Inf, nit, nplast)
        resid = norm(σp_new - σp_vec) / (norm(σp_new) + 1.0)
        @. σp_vec = (1 - ω) * σp_vec + ω * σp_new
        resid < tol && break
    end
    return resid, nit, nplast
end

function _broyden_sigma!(σp_vec, update!; maxiter::Int, tol::Float64)
    x = copy(σp_vec)
    fx, nplast = update!(x)
    any(!isfinite, fx) && return (Inf, 1, nplast)
    r = fx - x
    resid = norm(r) / (norm(fx) + 1.0)
    resid < tol && (copyto!(σp_vec, fx); return resid, 1, nplast)
    n = length(x)
    # r(x) = f(x) − x; if f′ ≈ 0 then J ≈ −I, so H = J^{-1} starts at −I (Picard).
    H = Matrix{Float64}(-I, n, n)
    nit = 1
    @inbounds for iter in 2:maxiter
        nit = iter
        dx = -(H * r)
        nrm = norm(dx)
        cap = 10 * (norm(x) + 1)
        nrm > cap && (dx .*= cap / nrm)
        α = 1.0
        xnew = x + dx
        fx, nplast = update!(xnew)
        rnew = fx - xnew
        resid_try = any(!isfinite, fx) ? Inf : norm(rnew) / (norm(fx) + 1.0)
        while resid_try > resid && α > 0.06
            α *= 0.5
            xnew = x + α * dx
            fx, nplast = update!(xnew)
            rnew = fx - xnew
            resid_try = any(!isfinite, fx) ? Inf : norm(rnew) / (norm(fx) + 1.0)
        end
        any(!isfinite, fx) && return (Inf, nit, nplast)
        resid = resid_try
        copyto!(σp_vec, xnew)
        resid < tol && return resid, nit, nplast
        s = xnew - x
        y = rnew - r
        Hy = H * y
        den = dot(s, Hy)
        if abs(den) > 1e-18 * (norm(s) * norm(Hy) + 1)
            Hs = H' * s
            H .+= ((s - Hy) * Hs') / den
        end
        x, r = xnew, rnew
    end
    return resid, nit, nplast
end

function _plastic_return!(σ, σz, κ, σp, u, traction, x, dad, AF, load, bfull,
        Q, Su, St, Sσ, σp_vec, σn, σzn, κn, εn, C, Cinv, λ, μ, mat, plane,
        stress_coupling)
    n = dad.n
    nc = size(σ, 1)
    dim = 2
    rhs = load * bfull + Q * σp_vec
    sol = AF \ rhs
    copyto!(x, sol)
    split_sol!(dad, view(x, 1:dim * n), u, traction)
    σB = Su * u + St * traction
    _add_plastic_stress!(σB, Sσ, σp_vec, dad.properties, stress_coupling)
    nplast = 0
    @inbounds for k in 1:nc
        σel = SVector(σB[3k - 2] + σp_vec[3k - 2],
            σB[3k - 1] + σp_vec[3k - 1],
            σB[3k] + σp_vec[3k])
        Δε = Cinv * σel - SVector(εn[k, 1], εn[k, 2], εn[k, 3])
        if plane
            σ4 = SVector(σn[k, 1], σn[k, 2], σzn[k], σn[k, 3])
            σ4n, κk = _return_plane_strain(σ4, κn[k], Δε, λ, μ, mat)
            σ[k, 1] = σ4n[1]
            σ[k, 2] = σ4n[2]
            σ[k, 3] = σ4n[4]
            σz[k] = σ4n[3]
            κ[k] = κk
            σt = σ4 + SVector(
                (λ + 2μ) * Δε[1] + λ * Δε[2],
                λ * Δε[1] + (λ + 2μ) * Δε[2],
                λ * (Δε[1] + Δε[2]),
                μ * Δε[3],
            )
            σp[k, 1] = σt[1] - σ4n[1]
            σp[k, 2] = σt[2] - σ4n[2]
            σp[k, 3] = σt[4] - σ4n[4]
        else
            σk = SVector(σn[k, 1], σn[k, 2], σn[k, 3])
            σnew, κk = _return_plane_stress(σk, κn[k], Δε, C, mat)
            σ[k, 1] = σnew[1]
            σ[k, 2] = σnew[2]
            σ[k, 3] = σnew[3]
            κ[k] = κk
            σt = σk + C * Δε
            σp[k, 1] = σt[1] - σnew[1]
            σp[k, 2] = σt[2] - σnew[2]
            σp[k, 3] = σt[3] - σnew[3]
        end
        nplast += κ[k] > κn[k] + 1e-16
    end
    return _pack_voigt(σp), nplast
end

# ---------------------------------------------------------------------------
# Thick-cylinder closed form (von Mises, perfect plasticity, plane strain)
# ---------------------------------------------------------------------------

"""
    ana_thick_cylinder_plastic(r; a, b, p, σY, E, ν)

Hill / Hodge fields for an internally pressurised tube, von Mises
``k=σ_Y/√3``, associated perfect plasticity. Returns
`(σr, σθ, u, c, elastic)` at radius `r`. Plastic front `c` from

```math
p = k\\bigl(1 - c^2/b^2 + 2\\ln(c/a)\\bigr).
```
"""
function ana_thick_cylinder_plastic(r::Real; a, b, p, σY, E, ν)
    k = σY / sqrt(3)
    pel = k * (1 - (a / b)^2)
    plim = 2k * log(b / a)
    p >= plim - 1e-14 && error("pressure $p ≥ limit $plim")
    if p <= pel + 1e-14
        A = p * a^2 / (b^2 - a^2)
        σr = -A * (b^2 / r^2 - 1)
        σθ = A * (b^2 / r^2 + 1)
        u = r * (1 + ν) / E * ((1 - ν) * σθ - ν * σr)
        return (; σr, σθ, u, c=a, elastic=true, pel, plim)
    end
    c = _plastic_front(a, b, p, k)
    if r >= c
        σr = k * c^2 * (1 / b^2 - 1 / r^2)
        σθ = k * c^2 * (1 / b^2 + 1 / r^2)
    else
        σr = k * (c^2 / b^2 - 1) - 2k * log(c / r)
        σθ = σr + 2k
    end
    u = r >= c ?
        r * (1 + ν) / E * ((1 - ν) * σθ - ν * σr) :
        begin
            # incompressible plastic core matched at r=c
            σrc = k * (c^2 / b^2 - 1)
            σθc = σrc + 2k
            uc = c * (1 + ν) / E * ((1 - ν) * σθc - ν * σrc)
            uc * c / r   # u ∝ 1/r for ν_pl = 1/2
        end
    return (; σr, σθ, u, c, elastic=false, pel, plim)
end

"""Set Neumann traction ``t = -p n`` on nodes with radius ``≈ R`` (inner pressure)."""
function apply_radius_pressure!(dad::BEMdata, p::Real; R::Real, tol::Real=0.0)
    n = dad.n
    δ = tol > 0 ? tol : 0.05 * (R + 1e-12)
    @inbounds for i in 1:n
        pt = dad.Nodes[i]
        abs(hypot(pt[1], pt[2]) - R) > δ && continue
        n̂ = dad.Normal[i]
        dad.BC[2i-1] = 1
        dad.BC[2i] = 1
        dad.BV[2i-1] = -p * n̂[1]
        dad.BV[2i] = -p * n̂[2]
    end
    return dad
end

"""
    ana_thick_cylinder_tresca(r; a, b, p, σY, E, ν)

Escudero appendix O / Tresca closed form. Plastic front `c` from

```math
p = σ_Y\\bigl(\\ln(c/a) + (b^2-c^2)/(2b^2)\\bigr).
```
"""
function ana_thick_cylinder_tresca(r::Real; a, b, p, σY, E, ν)
    pel = σY * (b^2 - a^2) / (2 * b^2)
    plim = σY * log(b / a)
    p >= plim - 1e-14 && error("pressure $p ≥ Tresca limit $plim")
    if p <= pel + 1e-14
        A = p * a^2 / (b^2 - a^2)
        σr = -A * (b^2 / r^2 - 1)
        σθ = A * (b^2 / r^2 + 1)
        u = r * (1 + ν) / E * ((1 - ν) * σθ - ν * σr)
        return (; σr, σθ, u, c=a, elastic=true, pel, plim)
    end
    c = _plastic_front_tresca(a, b, p, σY)
    if r >= c
        q = σY * (b^2 - c^2) / (2 * b^2)
        σr = q * c^2 * (r^2 - b^2) / (r^2 * (b^2 - c^2))
        σθ = q * c^2 * (r^2 + b^2) / (r^2 * (b^2 - c^2))
    else
        σr = σY * (log(r / c) + (c^2 - b^2) / (2 * b^2))
        σθ = σY * (log(r / c) + (c^2 + b^2) / (2 * b^2))
    end
    u = (1 + ν) * σY * c^2 / (2 * E * b^2) * ((1 - 2ν) * r + b^2 / r)
    return (; σr, σθ, u, c, elastic=false, pel, plim)
end

function _plastic_front_tresca(a, b, p, σY)
    f(c) = log(c / a) + (b^2 - c^2) / (2 * b^2) - p / σY
    lo, hi = a * (1 + 1e-9), b * (1 - 1e-9)
    c = 0.5 * (a + b)
    @inbounds for _ in 1:50
        fc = f(c)
        abs(fc) < 1e-12 && return c
        df = 1 / c - c / b^2
        c = clamp(c - fc / df, lo, hi)
    end
    return c
end

function _plastic_front(a, b, p, k)
    # p/k = 1 - (c/b)^2 + 2 ln(c/a)
    f(c) = 1 - (c / b)^2 + 2 * log(c / a) - p / k
    lo, hi = a * (1 + 1e-9), b * (1 - 1e-9)
    f(lo) > 0 && return a
    f(hi) < 0 && return b
    c = 0.5 * (a + b)
    @inbounds for _ in 1:50
        fc = f(c)
        abs(fc) < 1e-12 && return c
        # d/dc [-(c/b)^2 + 2 ln(c/a)] = -2c/b^2 + 2/c
        df = -2c / b^2 + 2 / c
        c = clamp(c - fc / df, lo, hi)
    end
    return c
end
