# Shared DIBEM infrastructure (scalar factored form + F-solve + format helpers)
#
# Physics-specific pieces stay in:
#   Laplace/Domain.jl, Domain_fast.jl
#   Elasticity/Domain.jl, Domain_fast.jl

export DibemFactoredOperator, DibemFKernel
export _zero_rowsum_diag!
export build_cell_mass, cell_integral_U

# ---------------------------------------------------------------------------
# RBF Gram kernel
# ---------------------------------------------------------------------------

"""RBF Gram ``F[i,j] = φ(‖x_i−x_j‖)`` (diagonal left 0; apply [`_zero_rowsum_diag!`](@ref))."""
struct DibemFKernel{R,P} <: AbstractMatrix{Float64}
    rbf::R
    points::Vector{P}
end
Base.size(K::DibemFKernel) = (length(K.points), length(K.points))
function Base.getindex(K::DibemFKernel, i::Int, j::Int)
    i == j && return 0.0
    return float(K.rbf(norm(K.points[i] - K.points[j])))
end

"""
    _zero_rowsum_diag!(F) -> F

Set ``F_{ii} = 1 − ∑_{j≠i} F_{ij}`` so each row sums to one (``F\\mathbf{1}=\\mathbf{1}``).
With a zero diagonal on entry, ``s = F\\mathbf{1}`` is the off-diagonal row sum and
``F_{ii} = 1 − s_i``.
"""
function _zero_rowsum_diag!(F::AbstractMatrix)
    n = size(F, 1)
    size(F, 2) == n || throw(DimensionMismatch("F must be square"))
    s = F * ones(eltype(F), n)
    @inbounds for i in 1:n
        F[i, i] = 1 - s[i]
    end
    return F
end

# ---------------------------------------------------------------------------
# Factored operator  M x = D (c ∘ x) + diag ∘ x
# ---------------------------------------------------------------------------

"""Scalar DIBEM `M` is a [`ColWeightedOp`](@ref): `M x = D (c ∘ x) + d ∘ x`."""
const DibemFactoredOperator = ColWeightedOp

"""Build factored M from compressed D, weights c, Galerkin ID."""
function _dibem_factored_M(D, c::Vector{Float64}, ID::Vector{Float64}, method::Symbol)
    n = length(c)
    rowsum_off = D * c
    diagv = .-rowsum_off .+ ID
    return ColWeightedOp(D, c, diagv, n, n)
end

# ---------------------------------------------------------------------------
# Format map + F solve
# ---------------------------------------------------------------------------

function _dibem_struct_format(method::Symbol)
    m = method
    m in (:hmatrix, :Hmat, :hmat, :H, :HMatrix) && return :H
    m in (:h2, :H2, :H2Matrix, :nnca, :NNCA) && return :H2
    m in (:blr, :BLR) && return :BLR
    m in (:dense,) && return :dense
    m in (:fmm, :FMM) && return :fmm
    return nothing
end

"""RIM surface factor: ``(n·r)/R^{d}`` so ``∫_Ω φ = ∫_Γ [∫_0^R φ ρ^{d-1}dρ](n·e_r)/R^{d-1}``."""
@inline _rim_factor(n, r, R, ::Val{2}) = dot(n, r) / (R * R)
@inline _rim_factor(n, r, R, ::Val{3}) = dot(n, r) / (R * R * R)
@inline _rim_factor(n, r, R, dim::Integer) =
    dim == 2 ? _rim_factor(n, r, R, Val(2)) : _rim_factor(n, r, R, Val(3))

function _dibem_src_loop!(body, n::Int, threaded::Bool)
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            body(i)
        end
    else
        for i in 1:n
            body(i)
        end
    end
    return nothing
end

"""Per-element Gauss + nodal geometry for RIM (built once per `DIBEM` call)."""
struct RimElemGeom{P}
    el::Element
    nodes::Vector{P}
    y::Vector{P}
    n::Vector{P}
    wJ::Vector{Float64}
    xj::Vector{P}
    nj::Vector{P}
    wj::Vector{Float64}
end

function _rim_shape_matrices(poly, ηs, ws, dim::Integer)
    if dim == 2
        N, dN = shapefun(poly, ηs)
        return N, dN, nothing, ws
    end
    N, dNξ, dNη = shapefun2D(poly, ηs)
    return N, dNξ, dNη, kron(ws, ws)
end

function _rim_build_elements(dad, ηs=nothing, ws=nothing)
    if ηs === nothing
        has_cache(dad, :qsi) || _init_quadrature!(dad, 16)
        ηs, ws = dad.qsi, dad.w
    elseif ws === nothing
        has_cache(dad, :w) || _init_quadrature!(dad, length(ηs))
        ws = dad.w
    end
    dim = dad.dimension
    N, dN, dNη, wq = _rim_shape_matrices(dad.element_type, ηs, ws, dim)
    P = typeof(point(dad, 1))
    nq = size(N, 1)
    geos = Vector{RimElemGeom{P}}(undef, length(dad.elements))
    @inbounds for (eidx, el) in enumerate(dad.elements)
        nn = length(el.index)
        nodes = Vector{P}(undef, nn)
        xj = Vector{P}(undef, nn)
        nj = Vector{P}(undef, nn)
        wj = Vector{Float64}(undef, nn)
        nref = dad.Normal[el.index[1]]
        for k in 1:nn
            ind = el.index[k]
            pk = dad.Nodes[ind]
            nodes[k] = pk
            xj[k] = pk
            nj[k] = dad.Normal[ind]
            wj[k] = dad.elem_weight[k] * el.Jacobian[k]
        end
        pg = N * nodes
        y = Vector{P}(undef, nq)
        nrm = Vector{P}(undef, nq)
        wJ = Vector{Float64}(undef, nq)
        if dim == 2
            dx = dN * nodes
            for q in 1:nq
                Jv = dx[q]
                J = norm(Jv)
                nnrm = J < 1e-16 ? nref : tan2normal(Jv / J)
                nnrm ⋅ nref < 0 && (nnrm = -nnrm)
                y[q] = pg[q]
                nrm[q] = nnrm
                wJ[q] = J < 1e-16 ? 0.0 : wq[q] * J
            end
        else
            tξ = dN * nodes
            tη = dNη * nodes
            for q in 1:nq
                Jv = cross(tξ[q], tη[q])
                J = norm(Jv)
                nnrm = J < 1e-16 ? nref : Jv / J
                nnrm ⋅ nref < 0 && (nnrm = -nnrm)
                y[q] = pg[q]
                nrm[q] = nnrm
                wJ[q] = J < 1e-16 ? 0.0 : wq[q] * J
            end
        end
        geos[eidx] = RimElemGeom{P}(el, nodes, y, nrm, wJ, xj, nj, wj)
    end
    return geos
end

@inline function _rim_from_geom!(f, x0, g::RimElemGeom, dim)
    @inbounds for q in eachindex(g.wJ)
        wJ = g.wJ[q]
        wJ == 0 && continue
        y = g.y[q]
        r = y - x0
        R = norm(r)
        R < 1e-14 && continue
        f(wJ * _rim_factor(g.n[q], r, R, dim), R, r / R, y)
    end
    return nothing
end

"""Gauss RIM of one boundary element as seen from `x0`. Calls `f(wJn, R, e, y)`."""
function _rim_element!(f, dad, el, nodes, x0, ηs, ws)
    dim = dad.dimension
    nref = dad.Normal[el.index[1]]
    if dim == 2
        N, dN = shapefun(dad.element_type, ηs)
        pg = N * nodes
        dx = dN * nodes
        @inbounds for q in eachindex(ηs)
            Jv = dx[q]
            J = norm(Jv)
            J < 1e-16 && continue
            n = tan2normal(Jv / J)
            n ⋅ nref < 0 && (n = -n)
            y = pg[q]
            r = y - x0
            R = norm(r)
            R < 1e-14 && continue
            f(ws[q] * J * _rim_factor(n, r, R, Val(2)), R, r / R, y)
        end
    else
        N, dNξ, dNη = shapefun2D(dad.element_type, ηs)
        w2 = kron(ws, ws)
        pg = N * nodes
        tξ = dNξ * nodes
        tη = dNη * nodes
        @inbounds for q in eachindex(w2)
            Jv = cross(tξ[q], tη[q])
            J = norm(Jv)
            J < 1e-16 && continue
            n = Jv / J
            n ⋅ nref < 0 && (n = -n)
            y = pg[q]
            r = y - x0
            R = norm(r)
            R < 1e-14 && continue
            f(w2[q] * J * _rim_factor(n, r, R, Val(3)), R, r / R, y)
        end
    end
    return nothing
end

"""Call `f(wJn, R, e, y)` at every boundary Gauss point from source `x0`."""
function _rim_foreach(f, dad, x0; ηs=nothing, ws=nothing, geos=nothing)
    if geos !== nothing
        dim = dad.dimension
        @inbounds for g in geos
            _rim_from_geom!(f, x0, g, dim)
        end
        return nothing
    end
    geos = _rim_build_elements(dad, ηs, ws)
    dim = dad.dimension
    @inbounds for g in geos
        _rim_from_geom!(f, x0, g, dim)
    end
    return nothing
end

"""RIM of `φ` and of `U*` at every collocation source (Gauss on every element)."""
function _dibem_IF_ID_gauss(dad, rbf, pts, npg::Int; threaded::Bool=true)
    nt = length(pts)
    props = dad.properties
    dim = dad.dimension
    IF = zeros(nt)
    ID = zeros(nt)
    ηs, ws = gausslegendre(npg)
    geos = _rim_build_elements(dad, ηs, ws)
    _dibem_src_loop!(nt, threaded) do i
        x0 = pts[i]
        accF = 0.0
        accD = 0.0
        @inbounds for g in geos
            for q in eachindex(g.wJ)
                wJ = g.wJ[q]
                wJ == 0 && continue
                y = g.y[q]
                r = y - x0
                R = norm(r)
                R < 1e-14 && continue
                wJn = wJ * _rim_factor(g.n[q], r, R, dim)
                accF += int(rbf, x0, y) * wJn
                accD += radial_integral(props, R, dim) * wJn
            end
        end
        IF[i] += accF
        ID[i] += accD
    end
    return IF, ID
end

"""Interior source for RIM of monomials (avoid half-solid-angle at a boundary node)."""
function _dibem_interior_source(dad)
    if dad.ni > 0
        return dad.internalNodes[1]
    end
    try
        return geometric_props(dad).centroid
    catch
        return mean(dad.Nodes)
    end
end

"""Nodal monomial matrix ``P_{ik}=p_k(x_i)`` (`nt × npoly`)."""
function _dibem_poly_P(pts, rbf)
    pdeg = max(poly_deg(rbf), -1)
    pdeg < 0 && return zeros(length(pts), 0)
    dim = length(pts[1])
    npoly = rbf_npoly(dim, pdeg)
    npoly == 0 && return zeros(length(pts), 0)
    mon = MonomialBasis(dim, pdeg)
    P = zeros(length(pts), npoly)
    @inbounds for i in eachindex(pts)
        P[i, :] = mon(pts[i])
    end
    return P
end

"""``∫_Ω p_k dΩ`` by RIM from an interior point."""
function _dibem_monomial_IP(dad, rbf; npg::Int=16)
    pdeg = max(poly_deg(rbf), -1)
    pdeg < 0 && return zeros(0)
    dim = dad.dimension
    npoly = rbf_npoly(dim, pdeg)
    npoly == 0 && return zeros(0)
    mon = MonomialBasis(dim, pdeg)
    IP = zeros(npoly)
    x0 = _dibem_interior_source(dad)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    _rim_foreach(dad, x0; ηs=dad.qsi, ws=dad.w) do wJn, R, e, y
        IP .+= int(mon, x0, y) * wJn
    end
    return IP
end

"""
CPD-augmented weights: ``K[c;λ]=[IF;IP]`` with ``K=[F P; P' 0]``.
`poly_deg < 0` → `F \\ IF`.
"""
function _dibem_poly_c(F::AbstractMatrix, IF::AbstractVector, pts, rbf;
        IP::Union{Nothing,AbstractVector}=nothing)
    nt = length(IF)
    size(F, 1) == nt || throw(DimensionMismatch("F vs IF"))
    P = _dibem_poly_P(pts, rbf)
    npoly = size(P, 2)
    if npoly == 0
        return F \ IF
    end
    ip = IP === nothing ? zeros(npoly) : collect(float.(IP))
    length(ip) == npoly || throw(DimensionMismatch("IP length $(length(ip)) ≠ npoly=$npoly"))
    K = [F P; P' zeros(npoly, npoly)]
    rhs = [collect(float.(IF)); ip]
    coef = K \ rhs
    return coef[1:nt]
end

function _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=:dense, atol=1e-6,
    rtol=1e-6, nmax=32, eta=3.0, threads=true, rank=typemax(Int),
    alpha=1.0, IP=nothing)
    nt = length(IF)
    use_zrs = rbf isa FundamentalRBF
    pdeg = max(poly_deg(rbf), -1)
    want_poly = pdeg >= 0 && rbf_npoly(length(pts[1]), pdeg) > 0 && !use_zrs
    # CPD KKT is dense; do not GMRES a hierarchical F when polynomials are on.
    dense_F = f_method === :dense || (f_method === :auto && nt <= 256) || want_poly
    if dense_F
        F = zeros(nt, nt)
        @inbounds for j in 1:nt, i in 1:nt
            i == j && continue
            F[i, j] = rbf(norm(pts[i] - pts[j]))
        end
        if use_zrs
            _zero_rowsum_diag!(F)   # F 1 = 1
            ε = 1e-14 * (norm(F, 1) / nt + 1)
        else
            # classical ridge on empty diagonal (PHS / MQ / …)
            ε = 1e-12 * (sum(abs, F) / max(nt * (nt - 1), 1) + 1)
        end
        @inbounds for i in 1:nt
            F[i, i] += ε
        end
        set_cache!(dad; dibem_F=F)
        ip = IP === nothing && want_poly ? _dibem_monomial_IP(dad, rbf) : IP
        return _dibem_poly_c(F, IF, pts, rbf; IP=ip)
    end

    splitter = hmatrix_splitter(; nmax=nmax)
    tree = ClusterTree(pts, splitter)
    KF = DibemFKernel(rbf, pts)
    fmt = _dibem_struct_format(f_method)
    fmt === nothing && (fmt = :H)
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)
    adm = StrongAdmissibilityStd(; eta=eta)
    Fst = assemble_structured(KF, tree; format=fmt, adm=adm, comp=comp,
        threads=threads, rtol=rtol, rank=rank, alpha=alpha,
        global_index=true)
    if Fst isa HMatrices.HMatrix
        use_zrs && _hmat_zero_rowsum_diag!(Fst)
        _hmat_add_diag_ridge!(Fst, use_zrs ? 1e-14 : 1e-12)
    end
    c, stats = Krylov.gmres(Fst, IF; atol=rtol, rtol=rtol, itmax=max(4nt, 200))
    if !stats.solved
        @warn "DIBEM: GMRES(F) incomplete" f_method stats.status
    end
    set_cache!(dad; dibem_F_h=Fst)
    return c
end

function _hmat_add_diag_ridge!(Hmat::HMatrices.HMatrix, ε::Float64)
    piv = HMatrices.pivot(Hmat)
    for block in HMatrices.nodes(Hmat)
        HMatrices.hasdata(block) || continue
        HMatrices.isadmissible(block) && continue
        data = HMatrices.data(block)
        data isa Matrix || continue
        irange = HMatrices.rowrange(block) .- piv[1] .+ 1
        jrange = HMatrices.colrange(block) .- piv[2] .+ 1
        irangeg = HMatrices.rowperm(Hmat)[irange]
        jrangeg = HMatrices.colperm(Hmat)[jrange]
        for (iloc, ig) in enumerate(irangeg), (jloc, jg) in enumerate(jrangeg)
            if ig == jg && iloc <= size(data, 1) && jloc <= size(data, 2)
                data[iloc, jloc] += ε * (tr(data) / max(size(data, 1), 1) + 1)
            end
        end
    end
    return nothing
end

"""Near-field diagonal leaves so global rows of `F` sum to one: ``F_{ii}=1−∑_{j≠i}F_{ij}``."""
function _hmat_zero_rowsum_diag!(Hmat::HMatrices.HMatrix)
    n = size(Hmat, 1)
    s = Hmat * ones(n)   # off-diagonal row sums when diag ≈ 0
    piv = HMatrices.pivot(Hmat)
    for block in HMatrices.nodes(Hmat)
        HMatrices.hasdata(block) || continue
        HMatrices.isadmissible(block) && continue
        data = HMatrices.data(block)
        data isa Matrix || continue
        irange = HMatrices.rowrange(block) .- piv[1] .+ 1
        jrange = HMatrices.colrange(block) .- piv[2] .+ 1
        irangeg = HMatrices.rowperm(Hmat)[irange]
        jrangeg = HMatrices.colperm(Hmat)[jrange]
        for (iloc, ig) in enumerate(irangeg), (jloc, jg) in enumerate(jrangeg)
            if ig == jg && iloc <= size(data, 1) && jloc <= size(data, 2)
                data[iloc, jloc] = 1 - s[ig]
            end
        end
    end
    return nothing
end

"""Scale FMM kernel: `(αA)x = α(Ax)`."""
struct _ScaledFMM{TA} <: AbstractMatrix{Float64}
    A::TA
    α::Float64
end
Base.size(S::_ScaledFMM) = size(S.A)
Base.size(S::_ScaledFMM, d) = size(S.A, d)
Base.IndexStyle(::Type{<:_ScaledFMM}) = IndexCartesian()
Base.getindex(S::_ScaledFMM, i::Int, j::Int) = S.α * S.A[i, j]
function LinearAlgebra.mul!(y::AbstractVector, S::_ScaledFMM, x::AbstractVector)
    mul!(y, S.A, x)
    y .*= S.α
    return y
end
function LinearAlgebra.mul!(y::AbstractVector, S::_ScaledFMM, x::AbstractVector,
    α::Number, β::Number)
    if iszero(β)
        mul!(y, S, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, S, x)
        y .= β .* y .+ α .* t
    end
    return y
end
Base.:*(S::_ScaledFMM, x::AbstractVector) = mul!(similar(x, Float64), S, x)
function LinearAlgebra.mul!(Y::AbstractMatrix, S::_ScaledFMM, X::AbstractMatrix)
    mul!(Y, S.A, X)
    Y .*= S.α
    return Y
end
function LinearAlgebra.mul!(y::AbstractVector, St::Adjoint{<:Any,<:_ScaledFMM}, x::AbstractVector)
    S = parent(St)
    mul!(y, adjoint(S.A), x)
    y .*= S.α
    return y
end
function LinearAlgebra.mul!(Y::AbstractMatrix, St::Adjoint{<:Any,<:_ScaledFMM}, X::AbstractMatrix)
    S = parent(St)
    mul!(Y, adjoint(S.A), X)
    Y .*= S.α
    return Y
end

# ---------------------------------------------------------------------------
# Constant cells (RIM of U on each Gmsh 2-D element)
# ---------------------------------------------------------------------------

_cell_U_primitive(props::Laplace, R, e) = radial_integral(props, R, 2)

"""RIM of the fundamental `U*` over a polygon cell. Skip `R→0` (integrand → 0)."""
function cell_integral_U(props, x0::SVector{2}, cell::DomainCell; npg::Int=12)
    verts = cell.verts
    nv = length(verts)
    nv < 3 && return _cell_U_zero(props)
    ηs, ws = gausslegendre(npg)
    acc = _cell_U_zero(props)
    @inbounds for k in 1:nv
        a = verts[k]
        b = verts[k == nv ? 1 : k + 1]
        tvec = b - a
        J = norm(tvec)
        J < 1e-16 && continue
        n = tan2normal(tvec / J)
        for q in eachindex(ηs)
            ξ = ηs[q]
            y = ((1 - ξ) * a + (1 + ξ) * b) / 2
            r = y - x0
            R = norm(r)
            R < 1e-14 && continue
            e = r / R
            wJn = ws[q] * (J / 2) * (n ⋅ r) / R^2
            acc += _cell_U_primitive(props, R, e) * wJn
        end
    end
    return acc
end

_cell_U_zero(::Laplace) = 0.0
_cell_U_zero(::Elasticity) = zero(SMatrix{2,2,Float64,4})

"""
    cell_integral_dUdnξ(props, x0, nf, cell; npg=12)

∫_cell ∂U/∂nξ dΩ by Duffy–Gauss on triangles from the centroid.
Integrand ``(r·nξ)/(2π k R²)`` is weakly singular in 2D (skip R→0).
"""
function cell_integral_dUdnξ(props::Laplace, x0::SVector{2}, nf::SVector{2},
        cell::DomainCell; npg::Int=12)
    verts = cell.verts
    nv = length(verts)
    nv < 3 && return 0.0
    c = cell.centroid
    kcond = float(props.k)
    ηs, ws = gausslegendre(npg)
    acc = 0.0
    @inbounds for i in 1:nv
        a = verts[i]
        b = verts[i == nv ? 1 : i + 1]
        Jtri = abs((a[1] - c[1]) * (b[2] - c[2]) - (a[2] - c[2]) * (b[1] - c[1]))
        Jtri < 1e-30 && continue
        for p in eachindex(ηs), q in eachindex(ηs)
            ξ = (ηs[p] + 1) / 2
            t = (ηs[q] + 1) / 2
            η = t * (1 - ξ)
            x = (1 - ξ - η) * c + ξ * a + η * b
            r = x - x0
            R = norm(r)
            R < 1e-14 && continue
            g = dot(r, nf) / (2π * kcond * R^2)
            acc += g * Jtri * (1 - ξ) * ws[p] * ws[q] / 4
        end
    end
    return acc
end

function _cell_shepard_P(pts, centroids; power::Float64=2.0)
    nt = length(pts)
    nc = length(centroids)
    P = zeros(nc, nt)
    @inbounds for k in 1:nc
        s = 0.0
        for j in 1:nt
            d2 = sum(abs2, centroids[k] - pts[j])
            w = 1 / (d2^(power / 2) + 1e-30)
            P[k, j] = w
            s += w
        end
        invs = 1 / s
        for j in 1:nt
            P[k, j] *= invs
        end
    end
    return P
end

"""
    build_cell_mass(dad; npg=12) -> (; M, M_cell, cells, P)

Piecewise-constant domain operator: `M_cell[i,k] = ∫_{ω_k} U*(x_i,x) dx`.
Square collocation `M = M_cell * P` with Shepard `P` from nodes to centroids.
Stores `M` on `dad` (same slot as DIBEM/DRM).
"""
function build_cell_mass(dad::BEMdata; npg::Int=12)
    cells = extract_domain_cells(dad)
    isempty(cells) && error("build_cell_mass: no Gmsh 2-D cells (mesh a surface, pontointerno)")
    props = dad.properties
    pts = all_points(dad)
    nt = dad.nt
    nc = length(cells)
    vecial = props isa Vectorial
    nd = vecial ? 2 : 1
    Mc = zeros(nd * nt, nd * nc)
    @inbounds for k in 1:nc, i in 1:nt
        Uω = cell_integral_U(props, pts[i], cells[k]; npg=npg)
        if vecial
            Mc[2i-1:2i, 2k-1:2k] .= Uω
        else
            Mc[i, k] = Uω
        end
    end
    P0 = _cell_shepard_P(pts, [c.centroid for c in cells])
    if vecial
        P = zeros(2nc, 2nt)
        @inbounds for k in 1:nc, j in 1:nt
            a = P0[k, j]
            P[2k-1, 2j-1] = a
            P[2k, 2j] = a
        end
    else
        P = P0
    end
    M = Mc * P
    props isa Elasticity && (M .*= props.rho)
    set_cache!(dad; M=M, cell_M=Mc, cells=cells, cell_P=P, dibem_method=:cells)
    return (; M, M_cell=Mc, cells, P)
end

# ---------------------------------------------------------------------------
# Volume-center DIBEM (PHS centers = cell centroids, not boundary collocation)
# ---------------------------------------------------------------------------

"""Cell centroids as RBF centers. Requires Gmsh 2-D cells on `dad`."""
function _dibem_volume_centers(dad::BEMdata)
    cells = extract_domain_cells(dad)
    isempty(cells) && error("DIBEM centers=:cells needs Gmsh 2-D cells (mesh a surface)")
    return [c.centroid for c in cells]
end

"""Map collocation values → center values. Exact match if a node coincides."""
function _dibem_center_Q(pts, centers)
    Q = _cell_shepard_P(pts, centers)
    @inbounds for k in eachindex(centers)
        for j in eachindex(pts)
            if sum(abs2, pts[j] - centers[k]) < 1e-24
                Q[k, :] .= 0
                Q[k, j] = 1
                break
            end
        end
    end
    return Q
end

"""RIM of `φ` at arbitrary interior sources (Gauss on every boundary element)."""
function _dibem_rbf_IF(dad, rbf, sources; npg::Int=16, threaded::Bool=true)
    nc = length(sources)
    IF = zeros(nc)
    ηs, ws = gausslegendre(npg)
    geos = _rim_build_elements(dad, ηs, ws)
    dim = dad.dimension
    _dibem_src_loop!(nc, threaded) do i
        x = sources[i]
        acc = 0.0
        @inbounds for g in geos
            for q in eachindex(g.wJ)
                wJ = g.wJ[q]
                wJ == 0 && continue
                y = g.y[q]
                r = y - x
                R = norm(r)
                R < 1e-14 && continue
                acc += int(rbf, x, y) * (wJ * _rim_factor(g.n[q], r, R, dim))
            end
        end
        IF[i] += acc
    end
    return IF
end

function _dibem_ridge_F!(F)
    n = size(F, 1)
    ε = 1e-12 * (sum(abs, F) / max(n * (n - 1), 1) + 1)
    @inbounds for i in 1:n
        F[i, i] += ε
    end
    return F
end

"""Scalar `M = D diag(c) Q + diag(ID − D c)` (volume-center remainder form)."""
function _dibem_M_volume(D, c, Q, ID::AbstractVector)
    dc = D * c
    M = (D .* c') * Q
    @inbounds for i in eachindex(ID)
        M[i, i] += ID[i] - dc[i]
    end
    return M
end

"""Vectorial 2-D `M` with nodal `c` and scalar pullback `Q` (`nc × nt`)."""
function _dibem_M_volume_elast(D, c, Q, ID::AbstractMatrix)
    nc = length(c)
    nt = size(Q, 2)
    size(D, 2) == 2nc && size(D, 1) == 2nt || throw(DimensionMismatch("D vs c,Q"))
    size(ID, 1) == 2nt && size(ID, 2) == 2 || throw(DimensionMismatch("ID"))
    Q2 = zeros(2nc, 2nt)
    @inbounds for k in 1:nc, j in 1:nt
        a = Q[k, j]
        Q2[2k-1, 2j-1] = a
        Q2[2k, 2j] = a
    end
    Dc = copy(D)
    @inbounds for k in 1:nc
        Dc[:, 2k-1] .*= c[k]
        Dc[:, 2k] .*= c[k]
    end
    M = Dc * Q2
    e1 = zeros(2nc)
    e2 = zeros(2nc)
    @inbounds for k in 1:nc
        e1[2k-1] = c[k]
        e2[2k] = c[k]
    end
    s1 = D * e1
    s2 = D * e2
    @inbounds for i in 1:nt
        r1, r2 = 2i - 1, 2i
        M[r1, r1] += ID[r1, 1] - s1[r1]
        M[r2, r1] += ID[r2, 1] - s1[r2]
        M[r1, r2] += ID[r1, 2] - s2[r1]
        M[r2, r2] += ID[r2, 2] - s2[r2]
    end
    return M
end


