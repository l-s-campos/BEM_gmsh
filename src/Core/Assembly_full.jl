export H_G_full_direct, H_G_hyper, assemble!, integrate_element, auto_near_factor

# =============================================================================
# Dense H, G assembly
# =============================================================================

"""Largest `κ L` at which Helmholtz still uses far lumping (`≳10` pts / λ)."""
const _HELMHOLTZ_LUMP_KL = 0.6

"""
    auto_near_factor(dad) -> Float64

Far-lumping cutoff for dense H/G.

- Not Helmholtz: `1.5` (same as vectorial).
- Helmholtz: `1.5` if `κ L_max ≤ 0.6` (≳10 points per wavelength),
  otherwise `Inf` (no nodal lumping). `L_max` is the longest element.

Pass an explicit `near_factor` to [`assemble!`](@ref) to override.
"""
function auto_near_factor(dad::BEMdata)
    dad.properties isa Helmholtz || return 1.5
    isempty(dad.elements) && return 1.5
    κ = abs(float(wavenumber(dad.properties)))
    Lmax = 0.0
    @inbounds for el in dad.elements
        el.Length > Lmax && (Lmax = el.Length)
    end
    return κ * Lmax ≤ _HELMHOLTZ_LUMP_KL ? 1.5 : Inf
end

function _resolve_near_factor(dad, near_factor)
    if near_factor === nothing || near_factor === :auto
        return auto_near_factor(dad)
    end
    near_factor isa Real || throw(ArgumentError(
        "near_factor must be a Real, :auto, or nothing; got $(repr(near_factor))"))
    return float(near_factor)
end

"""
    H_G_full_direct(dad; npg=20, threaded=true, near_factor=:auto)

Assemble dense influence matrices `H` and `G`.

**2D:** on-element Guiggiani (topological: source node on the element);
nearly singular map from [`default_nearfield`](@ref) (override with
`dad.nearfield`); far nodal lumping when the source is farther
than `near_factor * L` (`Inf` disables lumping).
**3D:** on-element polar Guiggiani (radial Laurent subtraction);
nearly singular polar/tensor sinh; far nodal lumping.

Scalar default `near_factor=:auto` is [`auto_near_factor`](@ref)
(Helmholtz: lump only if `κ L_max ≤ 0.6`). Pass a `Real` to override.
Vectorial default is `1.5`.

Rows = collocation sources (`Threads.@threads` when `threaded`).
A geometrically close but distinct face (finite-width crack) is sinh, not
Guiggiani. Dual-BEM coincident twins pass `twins` into `integrate_element`.
"""
function H_G_full_direct(dad::BEMdata{<:Scalar}; npg=20, threaded=true,
        near_factor::Union{Real,Symbol,Nothing}=:auto)
    _init_quadrature!(dad, npg)
    T = kernel_eltype(dad.properties)
    H = zeros(T, dad.nt, dad.nt)
    G = zeros(T, dad.nt, dad.n)
    nf = _resolve_near_factor(dad, near_factor)
    set_cache!(dad; H, G, near_factor=nf)
    elems = dad.elements
    nE = length(elems)
    nE == 0 && return H, G
    nodes_el, hbufs, gbufs = _scalar_elem_bufs(dad, elems, T)
    _collocation_loop!(threaded, dad.nt) do i
        pf = point(dad, i)
        tid = _assembly_tid(threaded)
        hbuf = hbufs[tid]
        gbuf = gbufs[tid]
        @inbounds for eidx in 1:nE
            _accumulate_element_scalar!(H, G, dad, elems[eidx], pf, i;
                xj=nodes_el[eidx], hbuf=hbuf, gbuf=gbuf, near_factor=nf)
        end
    end
    _rowsum_diag!(H)
    return H, G
end

H_G_full_direct(dad::BEMdata{<:Scalar}, npg::Integer; threaded=true,
        near_factor::Union{Real,Symbol,Nothing}=:auto) =
    H_G_full_direct(dad; npg=npg, threaded=threaded, near_factor=near_factor)

"""
    assemble!(dad; method=:dense, npg=20, kwargs...)

Student-facing collocation assembly. Mutates `dad.H` and `dad.G`.

| `method` | Backend |
|----------|---------|
| `:dense` | [`H_G_full_direct`](@ref) (every physics with a dense kernel) |
| `:hmatrix` | [`H_G_Hmat`](@ref) (Laplace hierarchical) |
| `:gpu` | [`H_G_gpu`](@ref) (2-D Laplace or isotropic elasticity; KernelAbstractions) |

Positional `assemble!(dad, npg)` is the dense path (same as the old
`H_G_full_direct(dad, npg)` call). Dense kwargs include `threaded`,
`near_factor` (`:auto` → [`auto_near_factor`](@ref); `Inf` disables far
lumping), and for vectorial `singular`.
"""
function assemble!(dad::BEMdata; method::Symbol=:dense, npg=20, kwargs...)
    if method === :dense
        return H_G_full_direct(dad; npg=npg, kwargs...)
    elseif method === :hmatrix
        return H_G_Hmat(dad; kwargs...)
    elseif method === :gpu
        return H_G_gpu(dad; npg=npg, kwargs...)
    else
        throw(ArgumentError("assemble! method must be :dense, :hmatrix, or :gpu; got $method"))
    end
end
assemble!(dad::BEMdata, npg::Integer; kwargs...) = assemble!(dad; npg=npg, kwargs...)

function H_G_full_direct(dad::BEMdata{<:Vectorial}; npg=20, threaded=true,
        near_factor::Real=1.5, singular::Symbol=:guiggiani)
    (singular === :guiggiani || singular === :telles) ||
        throw(ArgumentError("singular must be :guiggiani or :telles; got $singular"))
    _init_quadrature!(dad, npg)
    dim = n_dof(dad)
    H = zeros(dim * dad.nt, dim * dad.nt)
    G = zeros(dim * dad.nt, dim * dad.n)
    elems = dad.elements
    nf = float(near_factor)
    set_cache!(dad; H, G, singular=singular, near_factor=nf)
    nE = length(elems)
    nE == 0 && return H, G
    nodes_el = Vector{typeof(dad.Nodes[elems[1].index])}(undef, nE)
    jj_el = Vector{UnitRange{Int}}(undef, nE)
    maxcols = 1
    @inbounds for eidx in 1:nE
        el = elems[eidx]
        nodes_el[eidx] = dad.Nodes[el.index]
        jj = expand(el.index, dim)
        jj_el[eidx] = jj
        maxcols = max(maxcols, length(jj))
    end
    nb = Threads.maxthreadid()
    T = eltype(H)
    hbufs = [zeros(T, dim, maxcols) for _ in 1:nb]
    gbufs = [zeros(T, dim, maxcols) for _ in 1:nb]
    _collocation_loop!(threaded, dad.nt) do i
        pf = point(dad, i)
        ii = expand(i, dim)
        f = _source_kernel(dad, i)
        tid = _assembly_tid(threaded)
        hbuf = hbufs[tid]
        gbuf = gbufs[tid]
        @inbounds for eidx in 1:nE
            el = elems[eidx]
            xj = nodes_el[eidx]
            if _near_element(pf, xj, el; factor=nf)
                jj = jj_el[eidx]
                nc = length(jj)
                hloc = view(hbuf, 1:dim, 1:nc)
                gloc = view(gbuf, 1:dim, 1:nc)
                fill!(hloc, 0)
                fill!(gloc, 0)
                integrate_element(dad, el, xj, pf, hloc, gloc, f; source=i)
                H[ii, jj] .+= hloc
                G[ii, jj] .+= gloc
            else
                _far_nodal_vec!(H, G, dad, el, pf, ii, dim, f)
            end
        end
    end
    _vectorial_free_term!(H, dad)
    _after_vectorial_assemble!(dad, H, G)
    return H, G
end

H_G_full_direct(dad::BEMdata{<:Vectorial}, npg::Integer; threaded=true,
        near_factor::Real=1.5, singular::Symbol=:guiggiani) =
    H_G_full_direct(dad; npg=npg, threaded=threaded, near_factor=near_factor,
        singular=singular)

"""
    H_G_hyper(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity}}; npg=50, threaded=true)

2-D traction BIE (`fundamental_hyper`). On-element orders `(-1, -2)`:
Guiggiani, with Cordeiro SST leading tensors when they exist.
Jump ``\\tfrac12 t`` is moved onto `G` (`G_{ii} \\mathrel{-}= \\tfrac12 I`).
Rigid-body row-sum on `H` enforces ``c_{ij}=0`` (paper Table 7).
Stores `H`, `G` (and `H_hyper`, `G_hyper`) so [`solve`](@ref) can run the
hypersingular system. Default `npg=50` matches the paper's HBIE applications.
`laurent=:interp` (default) samples 20 Gauss nodes on the singular
element for coefficients and the regularized integral; `:auto` is
closed-form / SST then Richardson; `:richardson` always extrapolates.
"""
function H_G_hyper(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity}};
        npg::Int=50, threaded::Bool=true, laurent::Symbol=:interp, ninterp::Int=20)
    dad.dimension == 2 || error("H_G_hyper elasticity: 2D only")
    set_cache!(dad; laurent=laurent, ninterp=ninterp)
    _init_quadrature!(dad, npg)
    dim = dad.dimension
    n = dad.n
    H = zeros(dim * n, dim * n)
    G = zeros(dim * n, dim * n)
    elems = dad.elements
    orders = (-1, -2)
    _collocation_loop!(threaded, n) do i
        pf = point(dad, i)
        nf = dad.Normal[i]
        ii = expand(i, dim)
        f = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
        @inbounds for el in elems
            xj = dad.Nodes[el.index]
            jj = expand(el.index, dim)
            # Never one-point-lump 1/r²: always SST/Guiggiani (on-element) or sinh.
            hloc = zeros(eltype(H), dim, length(jj))
            gloc = zeros(eltype(G), dim, length(jj))
            integrate_element(dad, el, xj, pf, hloc, gloc, f;
                orders=orders, source=i)
            H[ii, jj] .+= hloc
            G[ii, jj] .+= gloc
        end
    end
    @inbounds for i in 1:n
        ii = expand(i, dim)
        for d in 1:dim
            G[ii[d], ii[d]] -= 0.5
        end
    end
    # Traction BIE free term cij = 0 (Cordeiro Table 7).
    @views for i in 1:n
        ii = expand(i, dim)
        H[ii, ii] .= 0.0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end
    set_cache!(dad; H, G, H_hyper=H, G_hyper=G)
    return H, G
end

# =============================================================================
# Element integration
# =============================================================================

"""
    integrate_element(dad, elem, x, pf, h, g, f=fundamental; orders, source, twins)

Double/single-layer contribution of one element for source `pf` into local
blocks `h`, `g`.

2D: on-element → Guiggiani; nearly singular → sinh GL.
3D: on-element → polar Guiggiani (radial Laurent subtraction);
nearly singular → polar/tensor sinh via [`transform_surface`](@ref),
or surface DIBEM when `dad.nearfield === :dibem` and `d/L < 0.05`.

On-element detection is **topological** when `source` (collocation index) is
given: Guiggiani only if `source ∈ elem.index` or the dual-BEM twin of
`source` sits on `elem`. A nearby parallel face (finite-width crack) stays
on the sinh path even if the gap is tiny. Without `source`, a geometric
`dist ≤ 1e-12` still selects Guiggiani (legacy / tests).
"""
function integrate_element(dad::BEMdata, elem, x::AbstractVector{<:Point}, pf::Point,
                           h, g, f=fundamental; orders=nothing,
                           source::Union{Nothing,Integer}=nothing,
                           twins=nothing)
    poly = dad.element_type
    if pf isa Point2D && dad.dimension == 2
        a, _, dist = closest_point_1d(poly, x, pf; ξ0=_seed_1d(poly, x, pf))
        singular = if source !== nothing
            _source_on_element(elem, source) ||
                _source_on_twin_element(elem, source, twins)
        else
            dist <= 1e-12
        end
        if singular
            # Field normal follows the integration element (`_quad_geom`).
            # Source `n_ξ` is `dad.Normal[source]` and is passed separately so
            # closed-form Laurent tensors can flip on dual-BEM twins
            # (`n_ξ·n_el<0`). Do not orient `n` to `n_ξ` here: that made the
            # samples look like a self-element and double-counted the flip.
            nref = dad.Normal[elem.index[1]]
            if has_cache(dad, :singular) && dad.singular === :telles
                _integrate_singular_telles!(h, g, dad, x, pf, a, f; geo=elem.geo, nref=nref)
            else
                _integrate_singular_guiggiani!(h, g, dad, x, pf, a, f;
                    orders=orders, source=source, geo=elem.geo, nref=nref)
            end
            return nothing
        end
    elseif pf isa Point3D && dad.dimension == 3
        aξ0, aη0 = _seed_2d(poly, x, pf)
        aξ, aη, _, dist = closest_point_2d(poly, x, pf; ξ0=(aξ0, aη0))
        singular = if source !== nothing
            _source_on_element(elem, source) ||
                _source_on_twin_element(elem, source, twins)
        else
            dist <= 1e-12
        end
        if singular
            _integrate_singular_guiggiani_surface!(h, g, dad, x, pf, aξ, aη, f;
                orders=orders)
            return nothing
        end
        if _surface_mode(dad) === :dibem &&
                dist / max(elem.Length, eps()) < 0.05
            integrate_element_dibem!(h, g, dad, elem, x, pf, f)
            return nothing
        end
    end
    N, r, nrm, wwJ = _quad_geom(dad, elem, x, pf)
    _accumulate_kernels!(h, g, dad, N, r, nrm, wwJ, f)
    return nothing
end

@inline function _source_on_element(elem, source::Integer)
    @inbounds for k in eachindex(elem.index)
        elem.index[k] == source && return true
    end
    return false
end

@inline function _source_on_twin_element(elem, source::Integer, twins)
    twins === nothing && return false
    t = twins[source]
    t == 0 && return false
    return _source_on_element(elem, t)
end

"""Thread id for assembly buffers (`1` when serial)."""
@inline _assembly_tid(threaded::Bool) =
    (threaded && Threads.nthreads() > 1) ? Threads.threadid() : 1

"""Cached element nodes + thread-local `hloc`/`gloc` for scalar H/G."""
function _scalar_elem_bufs(dad, elems, T)
    nE = length(elems)
    nodes_el = Vector{typeof(dad.Nodes[elems[1].index])}(undef, nE)
    maxnn = 1
    @inbounds for eidx in 1:nE
        el = elems[eidx]
        nodes_el[eidx] = dad.Nodes[el.index]
        maxnn = max(maxnn, length(el))
    end
    nb = Threads.maxthreadid()
    hbufs = [zeros(T, maxnn) for _ in 1:nb]
    gbufs = [zeros(T, maxnn) for _ in 1:nb]
    return nodes_el, hbufs, gbufs
end

"""One-element scalar contribution: Guiggiani/sinh if near, else `far!`.

Pass `hbuf`/`gbuf` (length ≥ `nn`) to reuse storage; otherwise allocates.
`xj` is the element nodes (computed if omitted).
"""
function _accumulate_element_scalar!(H, G, dad, el, pf, i;
        f=fundamental, orders=nothing, twins=nothing, far=_far_nodal_scalar!,
        xj=nothing, hbuf=nothing, gbuf=nothing, near_factor::Real=1.5)
    xj0 = xj === nothing ? dad.Nodes[el.index] : xj
    near = _near_element(pf, xj0, el; factor=float(near_factor)) ||
           _source_on_element(el, i) ||
           _source_on_twin_element(el, i, twins)
    if near
        nn = length(el)
        hloc, gloc = if hbuf === nothing
            zeros(eltype(H), nn), zeros(eltype(G), nn)
        else
            hv = view(hbuf, 1:nn)
            gv = view(gbuf, 1:nn)
            fill!(hv, zero(eltype(hv)))
            fill!(gv, zero(eltype(gv)))
            hv, gv
        end
        integrate_element(dad, el, xj0, pf, hloc, gloc, f;
            orders=orders, source=i, twins=twins)
        @inbounds for (k, j) in enumerate(el.index)
            H[i, j] += hloc[k]
            G[i, j] += gloc[k]
        end
    else
        far(H, G, dad, el, pf, i)
    end
    return nothing
end

function _rowsum_diag!(H::AbstractMatrix)
    @inbounds for i in 1:size(H, 1)
        H[i, i] = 0
        H[i, i] = -sum(view(H, i, :))
    end
    return H
end

# =============================================================================
# On-element Guiggiani (fused G + H)
# =============================================================================

"""Fused Guiggiani; Laurent orders from `singularity_orders(dad.properties)`."""
function _integrate_singular_guiggiani!(h::AbstractVector, g::AbstractVector,
        dad, nodes, pf::Point2D, a::Real, f; orders=nothing, source=nothing,
        geo=nothing, nref=nothing)
    fill!(h, 0); fill!(g, 0)
    poly = dad.element_type
    nN = length(h)
    a = clamp(float(a), nextfloat(-1.0), prevfloat(1.0))
    og, oh = orders === nothing ? singularity_orders(dad.properties) : orders
    z = zeros(eltype(h), nN)
    method, ninterp = _dad_laurent_opts(dad)
    nf = source !== nothing ? dad.Normal[source] : nothing
    Ig, Ih = guiggiani_GH(a; order_G=og, order_H=oh, qsi=dad.qsi, w=dad.w,
        props=dad.properties, poly=poly, nodes=nodes, nf=nf,
        laurent=method, ninterp=ninterp) do ξ
        samp = _sample_kernel(dad, poly, nodes, pf, ξ, f; geo=geo, nref=nref)
        samp === nothing && return z, z
        U, T, Nrow, J = samp
        Fg = similar(z); Fh = similar(z)
        @inbounds for j in 1:nN
            NjJ = Nrow[j] * J
            Fg[j] = U * NjJ
            Fh[j] = T * NjJ
        end
        return Fg, Fh
    end
    g .= Ig; h .= Ih
    return nothing
end

function _integrate_singular_guiggiani!(h::AbstractMatrix, g::AbstractMatrix,
        dad, nodes, pf::Point2D, a::Real, f; orders=nothing, source=nothing,
        geo=nothing, nref=nothing)
    fill!(h, 0); fill!(g, 0)
    poly = dad.element_type
    dim = size(h, 1)
    nN = size(h, 2) ÷ dim
    a = clamp(float(a), nextfloat(-1.0), prevfloat(1.0))
    og, oh = orders === nothing ? singularity_orders(dad.properties) : orders
    ncols = dim * nN
    z = zeros(eltype(h), dim, ncols)
    fpair = ξ -> begin
        samp = _sample_kernel(dad, poly, nodes, pf, ξ, f; geo=geo, nref=nref)
        samp === nothing && return z, z
        U, T, Nrow, J = samp
        Fg = zeros(eltype(h), dim, ncols); Fh = zeros(eltype(h), dim, ncols)
        @inbounds for j in 1:nN
            cols = expand(j, dim)
            NjJ = Nrow[j] * J
            Fg[:, cols] .= U .* NjJ
            Fh[:, cols] .= T .* NjJ
        end
        return Fg, Fh
    end
    nf = source !== nothing ? dad.Normal[source] : nothing
    method, ninterp = _dad_laurent_opts(dad)
    Ig, Ih = guiggiani_GH(fpair, a; order_G=og, order_H=oh, qsi=dad.qsi, w=dad.w,
        props=dad.properties, poly=poly, nodes=nodes, nf=nf,
        laurent=method, ninterp=ninterp)
    g .= Ig; h .= Ih
    return nothing
end

"""Laurent coefficient method stored on `dad` (`:auto`, `:interp`, `:richardson`)."""
function _dad_laurent_opts(dad)
    method = has_cache(dad, :laurent) ? dad.laurent : :interp
    ninterp = has_cache(dad, :ninterp) ? Int(dad.ninterp) : 20
    return method, ninterp
end

"""Telles (1987) cubic: `η(γ)` with `η'=η''=0` at the image of `eet` (MATLAB `telles`)."""
function _telles_map(γ::Real, eet::Real)
    eet = float(eet)
    eest = eet^2 - 1
    term1 = cbrt(eet * eest + abs(eest))
    term2 = cbrt(eet * eest - abs(eest))
    Γ = term1 + term2 + eet
    Q = 1 + 3 * Γ^2
    A = 1 / Q
    B = -3 * Γ / Q
    C = 3 * Γ^2 / Q
    D = -B
    η = A * γ^3 + B * γ^2 + C * γ + D
    Jt = 3 * A * γ^2 + 2 * B * γ + C
    return η, Jt
end

"""On-element Telles quadrature (Contato `calc_gh`). `a` is the parent collocation."""
function _integrate_singular_telles!(h::AbstractVector, g::AbstractVector,
        dad, nodes, pf::Point2D, a::Real, f; geo=nothing, nref=nothing)
    fill!(h, 0); fill!(g, 0)
    poly = dad.element_type
    nN = length(h)
    a = clamp(float(a), nextfloat(-1.0), prevfloat(1.0))
    qsi, w = dad.qsi, dad.w
    @inbounds for i in eachindex(qsi)
        η, Jt = _telles_map(qsi[i], a)
        samp = _sample_kernel(dad, poly, nodes, pf, η, f; geo=geo, nref=nref)
        samp === nothing && continue
        U, T, Nrow, J = samp
        wJ = Jt * w[i] * J
        for j in 1:nN
            Nj = Nrow[j] * wJ
            g[j] += U * Nj
            h[j] += T * Nj
        end
    end
    return nothing
end

function _integrate_singular_telles!(h::AbstractMatrix, g::AbstractMatrix,
        dad, nodes, pf::Point2D, a::Real, f; geo=nothing, nref=nothing)
    fill!(h, 0); fill!(g, 0)
    poly = dad.element_type
    dim = size(h, 1)
    nN = size(h, 2) ÷ dim
    a = clamp(float(a), nextfloat(-1.0), prevfloat(1.0))
    qsi, w = dad.qsi, dad.w
    @inbounds for i in eachindex(qsi)
        η, Jt = _telles_map(qsi[i], a)
        samp = _sample_kernel(dad, poly, nodes, pf, η, f; geo=geo, nref=nref)
        samp === nothing && continue
        U, T, Nrow, J = samp
        wJ = Jt * w[i] * J
        for j in 1:nN
            cols = expand(j, dim)
            Nj = Nrow[j] * wJ
            g[:, cols] .+= U .* Nj
            h[:, cols] .+= T .* Nj
        end
    end
    return nothing
end

"""Fused polar Guiggiani on a parent square; Laurent orders from the BIE pair."""
function _integrate_singular_guiggiani_surface!(h::AbstractVector, g::AbstractVector,
        dad, nodes, pf::Point3D, aξ::Real, aη::Real, f; orders=nothing)
    fill!(h, 0); fill!(g, 0)
    poly = dad.element_type
    nN = length(h)
    og, oh = orders === nothing ? singularity_orders(dad.properties) : orders
    z = zeros(eltype(h), nN)
    method, ninterp = _dad_laurent_opts(dad)
    Ig, Ih = guiggiani_GH_surface(aξ, aη; order_G=og, order_H=oh,
        qsi=dad.qsi, w=dad.w, props=dad.properties, poly=poly, nodes=nodes,
        laurent=method, ninterp=ninterp) do ξ, η
        samp = _sample_kernel_surface(dad, poly, nodes, pf, ξ, η, f)
        samp === nothing && return z, z
        U, T, Nrow, J = samp
        Fg = similar(z); Fh = similar(z)
        @inbounds for j in 1:nN
            NjJ = Nrow[j] * J
            Fg[j] = U * NjJ
            Fh[j] = T * NjJ
        end
        return Fg, Fh
    end
    g .= Ig; h .= Ih
    return nothing
end

function _integrate_singular_guiggiani_surface!(h::AbstractMatrix, g::AbstractMatrix,
        dad, nodes, pf::Point3D, aξ::Real, aη::Real, f; orders=nothing)
    fill!(h, 0); fill!(g, 0)
    poly = dad.element_type
    dim = size(h, 1)
    nN = size(h, 2) ÷ dim
    og, oh = orders === nothing ? singularity_orders(dad.properties) : orders
    ncols = dim * nN
    z = zeros(eltype(h), dim, ncols)
    method, ninterp = _dad_laurent_opts(dad)
    Ig, Ih = guiggiani_GH_surface(aξ, aη; order_G=og, order_H=oh,
        qsi=dad.qsi, w=dad.w, props=dad.properties, poly=poly, nodes=nodes,
        laurent=method, ninterp=ninterp) do ξ, η
        samp = _sample_kernel_surface(dad, poly, nodes, pf, ξ, η, f)
        samp === nothing && return z, z
        U, T, Nrow, J = samp
        Fg = zeros(eltype(h), dim, ncols); Fh = zeros(eltype(h), dim, ncols)
        @inbounds for j in 1:nN
            cols = expand(j, dim)
            NjJ = Nrow[j] * J
            Fg[:, cols] .= U .* NjJ
            Fh[:, cols] .= T .* NjJ
        end
        return Fg, Fh
    end
    g .= Ig; h .= Ih
    return nothing
end

@inline function _sample_kernel_surface(dad, poly, nodes, pf::Point3D, ξ, η, f)
    L, Lξ, Lη = shapefun2D(poly, poly, ξ, η)
    nN = size(L, 2)
    pg = zero(eltype(nodes))
    xξ = zero(eltype(nodes))
    xη = zero(eltype(nodes))
    @inbounds for k in 1:nN
        pg += L[1, k] * nodes[k]
        xξ += Lξ[1, k] * nodes[k]
        xη += Lη[1, k] * nodes[k]
    end
    Jv = cross(xξ, xη)
    J = norm(Jv)
    J < 1e-30 && return nothing
    r = pg - pf
    norm(r) < 1e-30 && return nothing
    nrm = Jv / J
    U, T = f(dad, r, nrm)
    return U, T, view(L, 1, :), J
end

"""Field `N` at `ξ`; geometry `pg, dx` from CAD `geo` when present (Contato)."""
@inline function _shape_geom(poly, nodes, ξ, geo)
    N, dN = shapefun(poly, ξ)
    if !isempty(geo)
        Ng, dNg = shapefun(Equispaced(length(geo) - 1), ξ)
        pg = zero(eltype(geo))
        dx = zero(eltype(geo))
        @inbounds for k in eachindex(geo)
            pg += Ng[1, k] * geo[k]
            dx += dNg[1, k] * geo[k]
        end
        return N, pg, dx
    end
    pg = zero(eltype(nodes))
    dx = zero(eltype(nodes))
    @inbounds for k in eachindex(nodes)
        pg += N[1, k] * nodes[k]
        dx += dN[1, k] * nodes[k]
    end
    return N, pg, dx
end

@inline function _sample_kernel(dad, poly, nodes, pf, ξ, f; geo=nothing,
        nref=nothing)
    g = geo === nothing ? Point2D[] : geo
    N, pg, dx = _shape_geom(poly, nodes, ξ, g)
    J = norm(dx)
    J < 1e-30 && return nothing
    r = pg - pf
    norm(r) < 1e-30 && return nothing
    nrm = tan2normal(dx / J)
    if nref !== nothing && (nrm[1] * nref[1] + nrm[2] * nref[2]) < 0
        nrm = Point2D(-nrm[1], -nrm[2])
    end
    U, T = f(dad, r, nrm)
    return U, T, view(N, 1, :), J
end

# =============================================================================
# Near-field quadrature + kernel accumulation
# =============================================================================

function _quad_geom(dad, elem, x::AbstractVector{<:Point2D}, pf::Point2D)
    eta, ww = transform(dad, elem, x, pf)
    poly = dad.element_type
    N, dN = shapefun(poly, eta)
    if !isempty(elem.geo)
        Ng, dNg = shapefun(Equispaced(length(elem.geo) - 1), eta)
        pg = Ng * elem.geo
        dx = dNg * elem.geo
    else
        pg = N * x
        dx = dN * x
    end
    r = pg .- Ref(pf)
    J = norm.(dx)
    nrm = tan2normal.(dx ./ J)
    nref = dad.Normal[elem.index[1]]
    if !isempty(nrm) && (nrm[1][1] * nref[1] + nrm[1][2] * nref[2]) < 0
        nrm = [Point2D(-n[1], -n[2]) for n in nrm]
    end
    return N, r, nrm, J .* ww
end

function _quad_geom(dad, elem, x::AbstractVector{<:Point3D}, pf::Point3D)
    η1, η2, ww = transform_surface(dad, elem, pf; qsi2=dad.qsi)
    N, dN1, dN2 = shapefun2D_points(dad.element_type, η1, η2)
    pg = N * x
    dx = cross.(dN1 * x, dN2 * x)
    r = pg .- Ref(pf)
    J = norm.(dx)
    nrm = dx ./ J
    return N, r, nrm, J .* ww
end


function _accumulate_kernels!(h::AbstractVector, g::AbstractVector, dad, N, r, nrm, wwJ, f)
    nn = size(N, 2)
    @inbounds for i in eachindex(wwJ)
        U, T = f(dad, r[i], nrm[i])
        wi = wwJ[i]
        for j in 1:nn
            Nj = N[i, j] * wi
            h[j] += T * Nj
            g[j] += U * Nj
        end
    end
    return nothing
end

function _accumulate_kernels!(h::AbstractMatrix, g::AbstractMatrix, dad, N, r, nrm, wwJ, f)
    dim = size(h, 1)
    nn = size(N, 2)
    @inbounds for i in eachindex(wwJ)
        U, T = f(dad, r[i], nrm[i])
        wi = wwJ[i]
        for j in 1:nn
            cols = expand(j, dim)
            Nj = N[i, j] * wi
            for d in 1:dim
                h[d, cols] .+= T[d, :] * Nj
                g[d, cols] .+= U[d, :] * Nj
            end
        end
    end
    return nothing
end

# =============================================================================
# Assembly helpers
# =============================================================================

function _init_quadrature!(dad, npg)
    qq, ww = gausslegendre(npg)
    set_cache!(dad; qsi=qq, w=ww)
    return nothing
end

"""Near if the source is within `factor * L` of the **element**, not just its nodes.

`d_node` is an upper bound on true distance `d*`, AABB a lower bound:
- `d_node < factor L` → near
- `d_AABB ≥ factor L` → far
- otherwise piecewise-linear closest point (`_dist_element`; exact for linear
  segments / planar triangles and quads). Newton `closest_point_*` is not used
  here (thousands of times slower than a node loop).

`factor = Inf` disables far lumping.
"""
function _near_element(pf, xg, el; factor::Float64=1.5)
    (factor == Inf || factor > 1e20) && return true
    n = length(xg)
    n == 0 && return true
    lim = factor * el.Length
    lim2 = lim * lim
    dn2 = Inf
    lo = hi = xg[1]
    @inbounds for k in eachindex(xg)
        q = xg[k]
        lo = min.(lo, q)
        hi = max.(hi, q)
        d2 = sum(abs2, pf - q)
        d2 < dn2 && (dn2 = d2)
    end
    dn2 < lim2 && return true
    _dist_aabb(pf, lo, hi) >= lim && return false
    return _dist_element(pf, xg) < lim
end

function _collocation_loop!(body, threaded::Bool, n::Int)
    if threaded && Threads.nthreads() > 1
        Threads.@threads :static for i in 1:n
            body(i)
        end
    else
        for i in 1:n
            body(i)
        end
    end
    return nothing
end

"""Kernel `(dad, r, n) → (U, T)` at collocation `i`. Plates close over source `n_ξ`."""
_source_kernel(dad::BEMdata, i::Integer) = fundamental
function _source_kernel(dad::BEMdata{<:AbstractThinPlate}, i::Integer)
    nf = i <= dad.n ? dad.Normal[i] : zero(eltype(dad.Normal))
    return (d, r, nrm) -> fundamental(d, r, nrm, nf)
end

"""Elasticity: rigid-body row-sum. Kirchhoff: ``\\tfrac12 I`` jump on the boundary."""
function _vectorial_free_term!(H, dad::BEMdata)
    dim = n_dof(dad)
    @views for i in 1:dad.nt
        ii = expand(i, dim)
        H[ii, ii] .= 0.0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end
    return H
end
function _vectorial_free_term!(H, dad::BEMdata{<:AbstractThinPlate})
    dim = n_dof(dad)
    @inbounds for i in 1:dad.n
        ii = expand(i, dim)
        for d in 1:dim
            H[ii[d], ii[d]] += 0.5
        end
    end
    return H
end

_after_vectorial_assemble!(dad::BEMdata, H, G) = nothing

function _far_nodal_scalar!(H, G, dad, el, pf, i)
    @inbounds for k in eachindex(el.index)
        node = el.index[k]
        Tast, Qast = fundamental(dad, dad.Nodes[node] - pf, dad.Normal[node])
        wjk = el.Jacobian[k] * dad.elem_weight[k]
        H[i, node] += Qast * wjk
        G[i, node] += Tast * wjk
    end
    return nothing
end

function _far_nodal_vec!(H, G, dad, el, pf, ii, dim, f=fundamental)
    @inbounds for k in eachindex(el.index)
        node = el.index[k]
        U, T = f(dad, dad.Nodes[node] - pf, dad.Normal[node])
        wjk = el.Jacobian[k] * dad.elem_weight[k]
        jji = expand(node, dim)
        H[ii, jji] .+= T * wjk
        G[ii, jji] .+= U * wjk
    end
    return nothing
end

"""Far HBIE lumping: `fundamental_hyper` with collocation normal `nf`."""
function _far_nodal_hyper_vec!(H, G, dad, el, pf, nf, ii, dim)
    @inbounds for k in eachindex(el.index)
        node = el.index[k]
        U, T = fundamental_hyper(dad, dad.Nodes[node] - pf, dad.Normal[node], nf)
        wjk = el.Jacobian[k] * dad.elem_weight[k]
        jji = expand(node, dim)
        H[ii, jji] .+= T * wjk
        G[ii, jji] .+= U * wjk
    end
    return nothing
end

expand(i::Integer, dim::Integer) = (dim * (i - 1) + 1):(dim * i)
expand(idx::AbstractVector{<:Integer}, dim::Integer) =
    (dim * (first(idx) - 1) + 1):(dim * last(idx))

