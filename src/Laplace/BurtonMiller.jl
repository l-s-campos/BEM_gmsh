# =============================================================================
# Time-domain Burton–Miller correlato (Laplace FS + domain mass)
# =============================================================================
# Combined field on boundary rows only:
#   (H + α H′) u − (G + α G′) q = (M + α M′) ü
# Interior collocation stays CBIE (no unique nξ).
# Coupling α is a real length (default 1). α = 0 reproduces CBIE.

export H_G_hyper, combine_burton_miller, assemble_wave_burton_miller!
export drm_hyper_mass, dibem_hyper_mass, cell_hyper_mass, pad_hyper

const _HBIE_ORDERS = (-1, -2)   # CPV G′, HFP H′

# ---------------------------------------------------------------------------
# Hypersingular collocation (boundary sources)
# ---------------------------------------------------------------------------

"""
    H_G_hyper(dad; npg=20, threaded=true) -> (H′, G′)

Collocation HBIE matrices for 2-D Laplace and 2-D/3-D Helmholtz. Rows are
boundary sources (`1:n`) with normal `dad.Normal[i]`.

- `H′` is `n × nt` (kernel ∂T/∂nξ)
- `G′` is `n × n`  (kernel ∂U/∂nξ)

On-element: Guiggiani orders `(-1, -2)`. No rigid-body row-sum on `H′`
(HFP from Guiggiani). The CBIE jump differentiated along `nξ` adds
`G′_ii -= 1/(2k)` (Laplace) or `G′_ii -= 1/2` (Helmholtz, `q=∂u/∂n`).
Stores `H_hyper`, `G_hyper` on `dad`.
`near_factor=:auto` is [`auto_near_factor`](@ref).
`laurent=:interp` (default) samples `ninterp` Gauss nodes on the singular
element for coefficients and the regularized integral; `:auto` uses
closed-form tensors then Richardson; `:richardson` always extrapolates.
"""
function H_G_hyper(dad::BEMdata{<:Union{Laplace,Helmholtz}}; npg::Int=20,
        threaded::Bool=true, laurent::Symbol=:interp, ninterp::Int=20,
        near_factor::Union{Real,Symbol,Nothing}=:auto)
    if dad.properties isa Laplace
        dad.dimension == 2 || error("H_G_hyper Laplace: 2D only")
    end
    set_cache!(dad; laurent=laurent, ninterp=ninterp)
    _init_quadrature!(dad, npg)
    n, nt = dad.n, dad.nt
    T = kernel_eltype(dad.properties)
    Hp = zeros(T, n, nt)
    Gp = zeros(T, n, n)
    elems = dad.elements
    nE = length(elems)
    nE == 0 && (set_cache!(dad; H_hyper=Hp, G_hyper=Gp); return Hp, Gp)
    nodes_el, hbufs, gbufs = _scalar_elem_bufs(dad, elems, T)
    nfct = _resolve_near_factor(dad, near_factor)
    set_cache!(dad; near_factor=nfct)
    _collocation_loop!(threaded, n) do i
        pf = point(dad, i)
        nf = dad.Normal[i]
        f = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
        tid = _assembly_tid(threaded)
        hbuf = hbufs[tid]
        gbuf = gbufs[tid]
        @inbounds for eidx in 1:nE
            el = elems[eidx]
            xj = nodes_el[eidx]
            if _near_element(pf, xj, el; factor=nfct)
                nn = length(el)
                hloc = view(hbuf, 1:nn)
                gloc = view(gbuf, 1:nn)
                fill!(hloc, zero(T))
                fill!(gloc, zero(T))
                integrate_element(dad, el, xj, pf, hloc, gloc, f; orders=_HBIE_ORDERS)
                for (k, j) in enumerate(el.index)
                    Hp[i, j] += hloc[k]
                    Gp[i, j] += gloc[k]
                end
            else
                _far_nodal_hyper!(Hp, Gp, dad, el, pf, nf, i)
            end
        end
    end
    # Laplace: (1/2) ∂u/∂nξ = −q/(2k)  →  G′_ii -= 1/(2k)
    # Helmholtz (q = ∂u/∂n): (1/2) q jump  →  G′_ii -= 1/2
    jump = dad.properties isa Laplace ? 0.5 / float(dad.properties.k) : T(0.5)
    @inbounds for i in 1:n
        Gp[i, i] -= jump
    end
    set_cache!(dad; H_hyper=Hp, G_hyper=Gp)
    return Hp, Gp
end

function _far_nodal_hyper!(Hp, Gp, dad, el, pf, nf, i)
    @inbounds for k in eachindex(el.index)
        node = el.index[k]
        Uh, Th = fundamental_hyper(dad, dad.Nodes[node] - pf, dad.Normal[node], nf)
        wjk = el.Jacobian[k] * dad.elem_weight[k]
        Hp[i, node] += Th * wjk
        Gp[i, node] += Uh * wjk
    end
    return nothing
end

"""Pad boundary-only hyper operators to `(nt × ·)` with zero interior rows."""
function pad_hyper(Hp::AbstractMatrix, Gp::AbstractMatrix, nt::Integer)
    n = size(Hp, 1)
    H = zeros(eltype(Hp), nt, size(Hp, 2))
    G = zeros(eltype(Gp), nt, size(Gp, 2))
    H[1:n, :] .= Hp
    G[1:n, :] .= Gp
    return H, G
end

# ---------------------------------------------------------------------------
# Combine CBIE + α HBIE (boundary rows)
# ---------------------------------------------------------------------------

"""
    combine_burton_miller(H, G, M, H′, G′, M′, α; n) -> (Hc, Gc, Mc)

`Xc[1:n, :] = X[1:n, :] + α X′`. Interior rows of `H, G, M` unchanged.
`α = 0` copies CBIE. Does not mutate inputs.
"""
function combine_burton_miller(H, G, M, Hp, Gp, Mp, α::Real; n::Integer)
    Hc = copy(H)
    Gc = copy(G)
    Mc = copy(M)
    iszero(α) && return Hc, Gc, Mc
    a = float(α)
    Hc[1:n, :] .+= a .* Hp
    Gc[1:n, :] .+= a .* Gp
    Mc[1:n, :] .+= a .* view(Mp, 1:n, :)
    return Hc, Gc, Mc
end

# ---------------------------------------------------------------------------
# Domain M′
# ---------------------------------------------------------------------------

"""DRM mass from given `H, G` and factors returned by [`build_drm_matrices`](@ref)."""
function drm_mass_from_HG(H, G, drm)
    C = H * drm.Ψ - G * drm.η
    if drm.npoly > 0 && drm.Ψp !== nothing
        Cfull = [C  H * drm.Ψp - G * drm.ηp]
        return Cfull * drm.W
    end
    return C / drm.F
end

"""
    drm_hyper_mass(drm, H′, G′) -> M′

`M′ = (H′ Ψ − G′ η) F⁻¹` with interior rows zero (`H′, G′` padded).
"""
function drm_hyper_mass(drm, Hp, Gp)
    nt = size(drm.F, 1)
    Hpp, Gpp = pad_hyper(Hp, Gp, nt)
    return drm_mass_from_HG(Hpp, Gpp, drm)
end

"""RIM of `U*` at an arbitrary source (same far/near split as DIBEM ID)."""
function _dibem_ID_at(dad::BEMdata{<:Laplace}, x; geos=nothing, near_factor::Float64=1.5)
    gcache = geos === nothing ? _rim_build_elements(dad) : geos
    props = dad.properties
    dimv = Val(Int(dad.dimension))
    ID = 0.0
    @inbounds for g in gcache
        ID += _dibem_elem_ID(x, g, props, dimv, near_factor)
    end
    return ID
end

# ∂/∂nξ of Φ(R)·(n·r)/R^d. Φ' = U* R^{d-1}, ∇_x R = −e, r = y − x.
@inline function _radial_dID_dnξ(props::Laplace, r, R, nrm, nf, ::Val{2})
    k = float(props.k)
    invR = 1 / R
    rn = dot(nrm, r) * invR
    rnf = dot(nf, r) * invR
    U = -log(R) / (2π * k)
    Φ = radial_integral(props, R, Val(2))
    return -U * rn * rnf + Φ * (2 * rn * rnf - dot(nrm, nf)) * (invR * invR)
end

@inline function _radial_dID_dnξ(props::Laplace, r, R, nrm, nf, ::Val{3})
    k = float(props.k)
    invR = 1 / R
    rn = dot(nrm, r) * invR
    rnf = dot(nf, r) * invR
    U = invR / (4π * k)
    Φ = radial_integral(props, R, Val(3))
    return -U * rn * rnf + Φ * (3 * rn * rnf - dot(nrm, nf)) * (invR * invR * invR)
end

function _dibem_elem_IDp(x, nf, g, props, dimv, near_factor::Float64)
    acc = 0.0
    if _near_element(x, g.nodes, g.el; factor=near_factor)
        @inbounds for q in eachindex(g.wJ)
            wJ = g.wJ[q]
            wJ == 0 && continue
            r = g.y[q] - x
            R = norm(r)
            R < 1e-14 && continue
            acc += _radial_dID_dnξ(props, r, R, g.n[q], nf, dimv) * wJ
        end
    else
        @inbounds for j in eachindex(g.xj)
            r = g.xj[j] - x
            R = norm(r)
            R < 1e-10 && continue
            acc += _radial_dID_dnξ(props, r, R, g.nj[j], nf, dimv) * g.wj[j]
        end
    end
    return acc
end

"""RIM of `∂U*/∂nξ` at source `x` (analytic ∂/∂nξ of [`_dibem_ID_at`](@ref))."""
function _dibem_IDp_at(dad::BEMdata{<:Laplace}, x, nf; geos=nothing,
        near_factor::Float64=1.5)
    gcache = geos === nothing ? _rim_build_elements(dad) : geos
    props = dad.properties
    dimv = Val(Int(dad.dimension))
    acc = 0.0
    @inbounds for g in gcache
        acc += _dibem_elem_IDp(x, nf, g, props, dimv, near_factor)
    end
    return acc
end

"""
    dibem_hyper_mass(dad; rbf=PHS(3; poly_deg=1)) -> M′

DIBEM M′ on all collocation points (`nt × nt`). Requires internal nodes
(`dad.ni > 0`): the RBF weights `c` and interpolation centers are the
full cloud (boundary + internals), same as CBIE DIBEM.

- Boundary rows `1:n`: kernel `∂U*/∂nξ`, remainder `ID′` (analytic
  `∂/∂nξ` of the RIM of `U*`).
- Interior rows `n+1:nt`: CBIE DIBEM (`dad.M`) — no unique `nξ`.

`ε` is accepted and ignored (old finite-difference step).
"""
function dibem_hyper_mass(dad::BEMdata{<:Laplace}; rbf=PHS(3; poly_deg=1),
        ε::Float64=1e-6, threaded::Bool=true, near_factor::Real=1.5)
    dad.ni > 0 || throw(ArgumentError(
        "dibem_hyper_mass: DIBEM needs internal points (pontointerno=true or set_internal_nodes!)"))
    if !has_cache(dad, :dibem_c)
        DIBEM(dad; method=:dense, rbf=rbf)
    end
    _ = ε
    c = dad.dibem_c
    n, nt = dad.n, dad.nt
    length(c) == nt || throw(DimensionMismatch(
        "dibem_c length $(length(c)) ≠ nt=$nt"))
    pts = all_points(dad)
    props = dad.properties
    n0 = dad.Normal[1]
    Dp = zeros(n, nt)
    @inbounds for i in 1:n, j in 1:nt
        r = pts[j] - pts[i]
        R = norm(r)
        R < 1e-14 && continue
        Dp[i, j] = fundamental_hyper(props, r, n0, dad.Normal[i]).U
    end
    geos = _rim_build_elements(dad)
    dimv = Val(Int(dad.dimension))
    nfct = float(near_factor)
    IDp = zeros(n)
    _collocation_loop!(threaded, n) do i
        x = pts[i]
        nfi = dad.Normal[i]
        acc = 0.0
        @inbounds for g in geos
            acc += _dibem_elem_IDp(x, nfi, g, props, dimv, nfct)
        end
        IDp[i] = acc
    end
    Mp = zeros(nt, nt)
    @inbounds for i in 1:n
        for j in 1:nt
            Mp[i, j] = Dp[i, j] * c[j]
        end
        Mp[i, i] = 0
        Mp[i, i] = -sum(view(Mp, i, :)) + IDp[i]
    end
    # Interior collocation stays CBIE DIBEM (same M rows).
    M = Matrix(dad.M)
    Mp[n + 1:nt, :] .= @view M[n + 1:nt, :]
    set_cache!(dad; M_hyper=Mp)
    return Mp
end

"""
    cell_hyper_mass(dad; npg=12, P=nothing, cells=nothing) -> M′

Piecewise-constant ∫_cell ∂U/∂nξ, Shepard-mapped to collocation. Interior
rows zero.
"""
function cell_hyper_mass(dad::BEMdata{<:Laplace}; npg::Int=12, P=nothing, cells=nothing)
    cells0 = cells === nothing ? extract_domain_cells(dad) : cells
    isempty(cells0) && error("cell_hyper_mass: no Gmsh 2-D cells")
    pts = all_points(dad)
    n, nt = dad.n, dad.nt
    nc = length(cells0)
    props = dad.properties
    Mc = zeros(n, nc)
    @inbounds for k in 1:nc, i in 1:n
        Mc[i, k] = cell_integral_dUdnξ(props, pts[i], dad.Normal[i], cells0[k]; npg=npg)
    end
    P0 = P === nothing ? _cell_shepard_P(pts, [c.centroid for c in cells0]) : P
    Mp = zeros(nt, nt)
    Mp[1:n, :] .= Mc * P0
    set_cache!(dad; M_hyper=Mp, cell_M_hyper=Mc)
    return Mp
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    assemble_wave_burton_miller!(dad; mass=:dibem, α=1.0, npg=12, rbf=PHS(3; poly_deg=1))

Assemble CBIE `(H,G,M)`, HBIE `(H′,G′,M′)`, and combined operators with
coupling `α`. Mass `:drm`, `:dibem`, or `:cells`.

Caches `H,G,M` as the **combined** operators (ready for `solve_Houbolt`).
CBIE copies live in `H_cbie, G_cbie, M_cbie`.
"""
function assemble_wave_burton_miller!(dad::BEMdata{<:Laplace};
        mass::Symbol=:dibem,
        α::Real=1.0,
        npg::Int=12,
        rbf=PHS(3; poly_deg=1),
        basis=nothing,
        threaded::Bool=true)
    mass in (:drm, :dibem, :cells) || throw(ArgumentError(
        "mass must be :drm, :dibem, or :cells (got $mass)"))
    has_cache(dad, :H) || H_G_full_direct(dad; npg=npg, threaded=threaded)
    Hp, Gp = H_G_hyper(dad; npg=npg, threaded=threaded)
    H = Matrix(dad.H)
    G = Matrix(dad.G)

    M, Mp = if mass === :drm
        b = basis === nothing ? rbf : basis
        drm = build_drm_matrices(dad, b; npg=npg)
        drm.M, drm_hyper_mass(drm, Hp, Gp)
    elseif mass === :dibem
        has_cache(dad, :M) || DIBEM(dad; method=:dense, rbf=rbf)
        Matrix(dad.M), dibem_hyper_mass(dad; rbf=rbf)
    else
        cm = build_cell_mass(dad; npg=npg)
        cm.M, cell_hyper_mass(dad; npg=npg, P=cm.P, cells=cm.cells)
    end

    Hc, Gc, Mc = combine_burton_miller(H, G, M, Hp, Gp, Mp, α; n=dad.n)
    set_cache!(dad; H=Hc, G=Gc, M=Mc, H_hyper=Hp, G_hyper=Gp, M_hyper=Mp,
        H_cbie=H, G_cbie=G, M_cbie=M, burton_alpha=float(α), burton_mass=mass)
    return dad
end
