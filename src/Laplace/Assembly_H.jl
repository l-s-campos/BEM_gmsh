# Hierarchical H,G assembly for Laplace (factored form)
#
# Generic operators: Core/Assembly_factored.jl (`ColWeightedOp`, `node_weights`, …)
# Dense assembly:    Core/Assembly_full.jl

export H_G_Hmat, assemble_hmatrix!, corrige_diagonais!, correct_nearfield!, mixed_bc_rhs

# ---------------------------------------------------------------------------
# Bare kernels (no weights) — Fundamental.jl
# ---------------------------------------------------------------------------

"""
Double-layer bare kernel: ``(Dq)_{ij} = ∂u*/∂n_j(x_i,x_j)`` (no `w_j`).
Internal columns and the diagonal are 0 (free term later).
"""
struct LaplaceDqKernel{P,Prop} <: AbstractMatrix{Float64}
    points::Vector{P}
    normals::Vector{P}
    props::Prop
    n_boundary::Int
end
Base.size(K::LaplaceDqKernel) = (length(K.points), length(K.points))
function Base.getindex(K::LaplaceDqKernel, i::Int, j::Int)
    (i == j || j > K.n_boundary) && return 0.0
    r = K.points[j] - K.points[i]
    _R2(r) < 1e-30 && return 0.0
    return fundamental_T(K.props, r, K.normals[j])
end

"""
Single-layer bare kernel on boundary columns: ``(Du)_{ij} = u*(x_i,x_j)`` (no `w_j`).
Size `(n_total × n_boundary)`.
"""
struct LaplaceDuKernel{P,Prop} <: AbstractMatrix{Float64}
    points::Vector{P}
    props::Prop
    n_boundary::Int
    n_dummy::P
end
Base.size(K::LaplaceDuKernel) = (length(K.points), K.n_boundary)
function Base.getindex(K::LaplaceDuKernel, i::Int, j::Int)
    i == j && return 0.0
    r = K.points[j] - K.points[i]
    _R2(r) < 1e-30 && return 0.0
    return fundamental_U(K.props, r)
end

function HMatrices.getblock!(out, K::LaplaceDqKernel, irange_, jrange_)
    irange = irange_ isa Colon ? axes(K, 1) : irange_
    jrange = jrange_ isa Colon ? axes(K, 2) : jrange_
    pts, nrm, props, nb = K.points, K.normals, K.props, K.n_boundary
    @inbounds for (jloc, j) in enumerate(jrange)
        if j > nb
            for iloc in 1:length(irange)
                out[iloc, jloc] = 0.0
            end
            continue
        end
        pj = pts[j]
        nj = nrm[j]
        for (iloc, i) in enumerate(irange)
            if i == j
                out[iloc, jloc] = 0.0
            else
                r = pj - pts[i]
                out[iloc, jloc] = _R2(r) < 1e-30 ? 0.0 : fundamental_T(props, r, nj)
            end
        end
    end
    return out
end

function HMatrices.getblock!(out, K::LaplaceDuKernel, irange_, jrange_)
    irange = irange_ isa Colon ? axes(K, 1) : irange_
    jrange = jrange_ isa Colon ? axes(K, 2) : jrange_
    pts, props = K.points, K.props
    @inbounds for (jloc, j) in enumerate(jrange)
        pj = pts[j]
        for (iloc, i) in enumerate(irange)
            if i == j
                out[iloc, jloc] = 0.0
            else
                r = pj - pts[i]
                out[iloc, jloc] = _R2(r) < 1e-30 ? 0.0 : fundamental_U(props, r)
            end
        end
    end
    return out
end

function HMatrices._kernel_row!(row::AbstractVector, K::LaplaceDqKernel, i::Int,
        J::Vector{Int}, ws=nothing)
    xi = K.points[i]
    nrm, props, nb = K.normals, K.props, K.n_boundary
    @inbounds for t in eachindex(J)
        j = J[t]
        if i == j || j > nb
            row[t] = 0.0
        else
            r = K.points[j] - xi
            row[t] = _R2(r) < 1e-30 ? 0.0 : fundamental_T(props, r, nrm[j])
        end
    end
    return row
end

function HMatrices._kernel_col!(col::AbstractVector, K::LaplaceDqKernel, j::Int,
        I::Vector{Int}, ws=nothing)
    if j > K.n_boundary
        fill!(view(col, 1:length(I)), 0.0)
        return col
    end
    pj = K.points[j]
    nj = K.normals[j]
    props = K.props
    @inbounds for t in eachindex(I)
        i = I[t]
        if i == j
            col[t] = 0.0
        else
            r = pj - K.points[i]
            col[t] = _R2(r) < 1e-30 ? 0.0 : fundamental_T(props, r, nj)
        end
    end
    return col
end

function HMatrices._kernel_row!(row::AbstractVector, K::LaplaceDuKernel, i::Int,
        J::Vector{Int}, ws=nothing)
    xi = K.points[i]
    props = K.props
    @inbounds for t in eachindex(J)
        j = J[t]
        if i == j
            row[t] = 0.0
        else
            r = K.points[j] - xi
            row[t] = _R2(r) < 1e-30 ? 0.0 : fundamental_U(props, r)
        end
    end
    return row
end

function HMatrices._kernel_col!(col::AbstractVector, K::LaplaceDuKernel, j::Int,
        I::Vector{Int}, ws=nothing)
    pj = K.points[j]
    props = K.props
    @inbounds for t in eachindex(I)
        i = I[t]
        if i == j
            col[t] = 0.0
        else
            r = pj - K.points[i]
            col[t] = _R2(r) < 1e-30 ? 0.0 : fundamental_U(props, r)
        end
    end
    return col
end

function HMatrices._kernel_block!(out::AbstractMatrix, K::LaplaceDqKernel,
        I::Vector{Int}, J::Vector{Int})
    return HMatrices.getblock!(out, K, I, J)
end
function HMatrices._kernel_block!(out::AbstractMatrix, K::LaplaceDuKernel,
        I::Vector{Int}, J::Vector{Int})
    return HMatrices.getblock!(out, K, I, J)
end

# ---------------------------------------------------------------------------
# H-matrix assembly (factored)
# ---------------------------------------------------------------------------

"""
    H_G_Hmat(dad; atol=1e-6, nmax=32, eta=3.0, threads=true, format=:H)

Assemble hierarchical Laplace ``H`` and ``G`` in **factored** form
([`ColWeightedOp`](@ref) over bare kernels).

`format` is `:H` (default) or `:H2` (NNCA for square `Dq` and rectangular
`Du` on a dual tree: all collocation × boundary).
"""
function H_G_Hmat(
    dad::BEMdata{<:Laplace};
    atol = 1e-6,
    rtol = 0.0,
    nmax = 32,
    eta = 3.0,
    threads = true,
    format::Symbol = :H,
    rank = typemax(Int),
    alpha = 1.0,
    splitter = nothing,
    nearfield::Bool = true,
    near_factor::Real = 1.5,
    eps = 1e-6,
    device = :host,
)
    has_cache(dad, :qsi) || _init_quadrature!(dad, 16)
    points = collect(all_points(dad))
    w = node_weights(dad)
    n = dad.n
    nt = length(points)
    props = dad.properties
    n_dummy = points[1] isa SVector{2} ? SVector(0.0, 1.0) : SVector(0.0, 0.0, 1.0)

    wH = zeros(nt)
    wH[1:n] .= w

    splitter = splitter === nothing ? hmatrix_splitter(; nmax=nmax) : splitter
    Xclt = ClusterTree(points, splitter)
    Yclt_H = ClusterTree(copy(points), splitter)
    Yclt_G = ClusterTree(collect(dad.Nodes), splitter)

    adm = StrongAdmissibilityStd(; eta=eta)
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)

    KDq = LaplaceDqKernel(points, dad.Normal, props, n)
    KDu = LaplaceDuKernel(points, props, n, n_dummy)

    fmt = format
    @info "H_G_Hmat factored" format=fmt nt n

    Dq, Du, extras = _assemble_HG_bare(KDq, KDu, Xclt, Yclt_H, Yclt_G, fmt;
        adm=adm, comp=comp, threads=threads, rtol=rtol, rank=rank,
        alpha=alpha, points=points, props=props,
        normals=dad.Normal, n_boundary=n, eps=eps, nmax=nmax, eta=eta,
        device=device)

    Dq = gpu_wrap(Dq, device)
    Du = gpu_wrap(Du, device)
    H = ColWeightedOp(Dq, wH, zeros(nt), nt, nt)
    G = ColWeightedOp(Du, copy(w), zeros(nt), nt, n)

    nearfield && correct_nearfield!(dad, H, G; factor=near_factor)
    corrige_diagonais!(dad, H, G)
    set_cache!(dad; H=H, G=G, H_bare=Dq, G_bare=Du, bem_weights=w,
        G_bare_square=get(extras, :Du_sq, nothing))
    return H, G
end

"""Teaching alias for [`H_G_Hmat`](@ref)."""
const assemble_hmatrix! = H_G_Hmat

"""
    correct_nearfield!(dad, H, G; factor=1.5)

Near-field correction for Laplace factored operators (see dense
[`integrate_element`](@ref)).
"""
function correct_nearfield!(
    dad::BEMdata{<:Laplace},
    H::ColWeightedOp,
    G::ColWeightedOp;
    factor::Real = 2.0,
)
    nt = H.nrows
    n = G.ncols
    n == dad.n || throw(DimensionMismatch("G columns"))
    Ih = Int[]; Jh = Int[]; Vh = Float64[]
    Ig = Int[]; Jg = Int[]; Vg = Float64[]

    elems = dad.elements
    Xel = [[dad.Nodes[j] for j in elem.index] for elem in elems]

    nn_max = maximum(length(el) for el in elems; init=3)
    hloc = zeros(nn_max)
    gloc = zeros(nn_max)

    @inbounds for i in 1:nt
        pf = point(dad, i)
        for (ej, elem) in enumerate(elems)
            xj = Xel[ej]
            r0 = norm(pf - xj[1])
            r0 < factor * elem.Length || continue

            nn = length(elem)
            fill!(hloc, 0.0)
            fill!(gloc, 0.0)
            hv = @view hloc[1:nn]
            gv = @view gloc[1:nn]
            integrate_element(dad, elem, xj, pf, hv, gv)

            for k in 1:nn
                j = elem.index[k]
                j > n && continue
                hp = 0.0
                gp = 0.0
                if i != j
                    r_node = dad.Nodes[j] - pf
                    n_node = dad.Normal[j]
                    U, Tker = fundamental(dad, r_node, n_node)
                    hp = Tker * H.w[j]
                    gp = U * G.w[j]
                end
                if i == j
                    H.d[i] += hv[k]
                    dg = gv[k] - gp
                    if abs(dg) > 0
                        push!(Ig, i); push!(Jg, j); push!(Vg, dg)
                    end
                    continue
                end
                dh = hv[k] - hp
                dg = gv[k] - gp
                if abs(dh) > 0
                    push!(Ih, i); push!(Jh, j); push!(Vh, dh)
                end
                if abs(dg) > 0
                    push!(Ig, i); push!(Jg, j); push!(Vg, dg)
                end
            end
        end
    end

    H.corr = isempty(Vh) ? spzeros(nt, nt) : sparse(Ih, Jh, Vh, nt, nt)
    G.corr = isempty(Vg) ? spzeros(nt, n) : sparse(Ig, Jg, Vg, nt, n)
    return nothing
end

"""Return `(Dq, Du, extras::Dict)`."""
function _assemble_HG_bare(KDq, KDu, Xclt, Yclt_H, Yclt_G, fmt::Symbol;
        adm, comp, threads, rtol, rank, alpha,
        points=nothing, props=nothing, normals=nothing, n_boundary=nothing,
        eps=1e-6, nmax=32, eta=3.0, device=:host)
    extras = Dict{Symbol,Any}()
    if fmt in (:H, :HMatrix, :hmatrix)
        Dq = assemble_hmatrix(KDq, Xclt, Yclt_H; adm=adm, comp=comp, threads=threads,
            device=device)
        Du = assemble_hmatrix(KDu, Xclt, Yclt_G; adm=adm, comp=comp, threads=threads,
            device=device)
        return Dq, Du, extras
    elseif fmt in (:H2, :h2)
        rtol_h2 = float(rtol) > 0 ? float(rtol) : (float(comp.atol) > 0 ? float(comp.atol) : 1e-6)
        Dq = assemble_h2(KDq, Xclt; rtol=rtol_h2, rank=rank, threads=threads,
            device=device)
        Ypts = points[1:n_boundary]
        Yclt = ClusterTree(Ypts, DyadicSplitter(; nmax=nmax, tight=false);
            cube=true, container=HMatrices.container(Xclt))
        Du = assemble_h2(KDu, Xclt, Yclt; rtol=rtol_h2, rank=rank, threads=threads,
            device=device)
        extras[:Yclt_G] = Yclt
        return Dq, Du, extras
    else
        throw(ArgumentError("H_G_Hmat format must be :H or :H2; got $fmt"))
    end
end

"""
    corrige_diagonais!(dad, H, G; free_term=:explicit)

Diagonal coefficients for Laplace [`ColWeightedOp`](@ref) / `HMatrix`.
"""
function corrige_diagonais!(dad::BEMdata{<:Laplace}, H::ColWeightedOp, G::ColWeightedOp;
        free_term::Symbol = :explicit)
    n = dad.n
    nt = dad.nt
    k = float(dad.properties.k)

    if free_term === :explicit
        @inbounds for i in 1:nt
            H.d[i] += BEM.free_term(dad, i)
        end
    elseif free_term === :rowsum
        fill!(H.d, 0.0)
        hsum = H * ones(nt)
        @inbounds for i in 1:nt
            H.d[i] = -hsum[i]
        end
    else
        throw(ArgumentError("free_term must be :explicit or :rowsum, got $free_term"))
    end

    pts = all_points(dad)
    if dad.dimension == 2
        xlin = [p[1] + p[2] for p in pts]
        qlin = [-k * (dad.Normal[j][1] + dad.Normal[j][2]) for j in 1:n]
    else
        xlin = [p[1] + p[2] + p[3] for p in pts]
        qlin = [-k * sum(dad.Normal[j]) for j in 1:n]
    end
    fill!(G.d, 0.0)
    resid = H * xlin - G * qlin
    @inbounds for i in 1:n
        denom = qlin[i]
        G.d[i] = abs(denom) < 1e-14 ? 0.0 : resid[i] / denom
    end
    return nothing
end

function corrige_diagonais!(dad::BEMdata{<:Laplace}, Hmat::HMatrix, Gmat::HMatrix;
        free_term::Symbol = :explicit)
    n = dad.n
    nt = dad.nt
    k = float(dad.properties.k)

    if free_term === :explicit
        _set_diagonal!(Hmat) do i
            BEM.free_term(dad, i)
        end
    elseif free_term === :rowsum
        _set_diagonal!(Hmat) do _
            0.0
        end
        hsum = Hmat * ones(nt)
        _set_diagonal!(Hmat) do i
            -hsum[i]
        end
    else
        throw(ArgumentError("free_term must be :explicit or :rowsum, got $free_term"))
    end

    pts = all_points(dad)
    if dad.dimension == 2
        xlin = [p[1] + p[2] for p in pts]
        qlin = [-k * (dad.Normal[j][1] + dad.Normal[j][2]) for j in 1:n]
    else
        xlin = [p[1] + p[2] + p[3] for p in pts]
        qlin = [-k * sum(dad.Normal[j]) for j in 1:n]
    end
    resid = Hmat * xlin - Gmat * qlin
    _set_diagonal!(Gmat) do i
        i > n && return 0.0
        denom = qlin[i]
        abs(denom) < 1e-14 && return 0.0
        return resid[i] / denom
    end
    return nothing
end

"""
Build RHS `b` consistent with mixed BCs:
`b = -H * T_known + G * q_known`.
"""
function mixed_bc_rhs(H, G, dad::BEMdata{<:Laplace})
    Tknown = zeros(dad.nt)
    qknown = zeros(dad.n)
    @inbounds for j in 1:dad.n
        if dad.BC[j] == 0
            Tknown[j] = dad.BV[j]
        else
            qknown[j] = dad.BV[j]
        end
    end
    return -(H * Tknown) + (G * qknown)
end
