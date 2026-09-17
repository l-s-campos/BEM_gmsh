export DIBEM_dense, dibem!, solve_poisson_dibem!

"""Far: nodal lumping. Near: regular Gauss. Radial primitive is regular — no sinh."""
function _dibem_accumulate_IF_ID!(::Nothing, ID, dad, rbf; threaded::Bool=true,
        geos=nothing, near_factor::Float64=1.5)
    gcache = geos === nothing ? _rim_build_elements(dad) : geos
    props = dad.properties
    dimv = Val(Int(dad.dimension))
    nf = near_factor
    _dibem_src_loop!(dad.nt, threaded) do i
        _dibem_source_rim!(nothing, ID, i, point(dad, i), gcache, rbf, props, dimv, nf)
    end
    return nothing
end

function _dibem_accumulate_IF_ID!(IF::AbstractVector, ID, dad, rbf; threaded::Bool=true,
        geos=nothing, near_factor::Float64=1.5)
    gcache = geos === nothing ? _rim_build_elements(dad) : geos
    props = dad.properties
    dimv = Val(Int(dad.dimension))
    nf = near_factor
    _dibem_src_loop!(dad.nt, threaded) do i
        _dibem_source_rim!(IF, ID, i, point(dad, i), gcache, rbf, props, dimv, nf)
    end
    return nothing
end

function _dibem_source_rim!(::Nothing, ID, i, x, geos, rbf, props, dimv, near_factor::Float64)
    accD = 0.0
    @inbounds for g in geos
        accD += _dibem_elem_ID(x, g, props, dimv, near_factor)
    end
    ID[i] += accD
    return nothing
end

function _dibem_source_rim!(IF::AbstractVector, ID, i, x, geos, rbf, props, dimv, near_factor::Float64)
    accF = 0.0
    accD = 0.0
    @inbounds for g in geos
        dF, dD = _dibem_elem_IF_ID(x, g, rbf, props, dimv, near_factor)
        accF += dF
        accD += dD
    end
    IF[i] += accF
    ID[i] += accD
    return nothing
end

function _dibem_elem_ID(x, g, props, dimv, near_factor::Float64)
    accD = 0.0
    if _near_element(x, g.nodes, g.el; factor=near_factor)
        @inbounds for q in eachindex(g.wJ)
            wJ = g.wJ[q]
            wJ == 0 && continue
            y = g.y[q]
            r = y - x
            R = norm(r)
            R < 1e-14 && continue
            accD += radial_integral(props, R, dimv) * (wJ * _rim_factor(g.n[q], r, R, dimv))
        end
    else
        @inbounds for j in eachindex(g.xj)
            xj = g.xj[j]
            r = xj - x
            R = norm(r)
            R < 1e-10 && continue
            accD += radial_integral(props, R, dimv) * (g.wj[j] * _rim_factor(g.nj[j], r, R, dimv))
        end
    end
    return accD
end

function _dibem_elem_IF_ID(x, g, rbf, props, dimv, near_factor::Float64)
    accF = 0.0
    accD = 0.0
    if _near_element(x, g.nodes, g.el; factor=near_factor)
        @inbounds for q in eachindex(g.wJ)
            wJ = g.wJ[q]
            wJ == 0 && continue
            y = g.y[q]
            r = y - x
            R = norm(r)
            R < 1e-14 && continue
            wJn = wJ * _rim_factor(g.n[q], r, R, dimv)
            accF += int(rbf, x, y) * wJn
            accD += radial_integral(props, R, dimv) * wJn
        end
    else
        @inbounds for j in eachindex(g.xj)
            xj = g.xj[j]
            r = xj - x
            R = norm(r)
            R < 1e-10 && continue
            wJn = g.wj[j] * _rim_factor(g.nj[j], r, R, dimv)
            accF += int(rbf, x, xj) * wJn
            accD += radial_integral(props, R, dimv) * wJn
        end
    end
    return accF, accD
end

function _dibem_far_lump!(IF, ID, dad, el, x, i, rbf, props, dim; mon=nothing, IP=nothing, i0=false)
    @inbounds for j in eachindex(el.index)
        ind = el.index[j]
        xj = dad.Nodes[ind]
        r = xj - x
        R = norm(r)
        R < 1e-10 && continue
        wJn = dad.elem_weight[j] * el.Jacobian[j] * _rim_factor(dad.Normal[ind], r, R, dim)
        IF !== nothing && (IF[i] += int(rbf, x, xj) * wJn)
        ID[i] += radial_integral(props, R, dim) * wJn
    end
    return nothing
end

function _dibem_near_gauss!(IF, ID, dad, el, nodes, x, i, rbf, ηs, ws, props, dim;
        mon=nothing, IP=nothing, i0=false)
    _rim_element!(dad, el, nodes, x, ηs, ws) do wJn, R, e, y
        IF !== nothing && (IF[i] += int(rbf, x, y) * wJn)
        ID[i] += radial_integral(props, R, dim) * wJn
    end
    return nothing
end


"""
    DIBEM_dense(dad; rbf=PHS())

Dense **Direct Interpolation BEM** operator `M`:

```
∫_Ω β(X) u*(ξ,X) dΩ  ≈  (M β)(ξ)
```

Stores `M` in `dad.cache.M` and returns it.

For large problems see [`DIBEM`](@ref) with `method=:hmatrix` or `:fmm`
([`DIBEM_Hmat`](@ref), [`DIBEM_FMM`](@ref) in `Domain_fast.jl`).
"""
function DIBEM_dense(dad::BEMdata{<:Laplace}; rbf=PHS(), centers::Symbol=:collocation,
        npg::Int=16, rim::Symbol=:lumped, threaded::Bool=true, kwargs...)
    # Fundamental-as-RBF: F≡D (off-diag), IF≡ID — specialized path
    rbf isa FundamentalRBF && return _DIBEM_dense_fundamental(dad, rbf; threaded=threaded)
    centers in (:collocation, :cells) || throw(ArgumentError(
        "DIBEM centers must be :collocation or :cells (got $centers)"))
    rim in (:lumped, :full_gauss) || throw(ArgumentError(
        "DIBEM rim must be :lumped or :full_gauss (got $rim)"))

    if centers === :cells
        return _DIBEM_dense_volume_centers(dad, rbf; npg=npg, threaded=threaded)
    end

    F = zeros(dad.nt, dad.nt)
    D = zeros(dad.nt, dad.nt)
    IF = zeros(dad.nt)
    ID = zeros(dad.nt)

    pts = all_points(dad)
    _dibem_fill_FD!(F, D, dad, rbf; threaded=threaded)
    _dibem_ridge_F!(F)

    if rim === :full_gauss
        IF, ID = _dibem_IF_ID_gauss(dad, rbf, pts, npg; threaded=threaded)
    else
        _dibem_accumulate_IF_ID!(IF, ID, dad, rbf; threaded=threaded)
    end
    IP = _dibem_monomial_IP(dad, rbf)
    c = _dibem_poly_c(F, IF, pts, rbf; IP=IP)

    M = D .* c'
    @inbounds for i in 1:dad.nt
        M[i, i] = 0
        M[i, i] = -sum(view(M, i, :)) + ID[i]
    end
    set_cache!(dad; M, dibem_F=F, dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_IF=IF,
        dibem_IP=IP, dibem_rbf=rbf, dibem_method=:dense, dibem_centers=:collocation)
    return M
end

function _dibem_fill_FD!(F, D, dad, rbf; threaded::Bool=true)
    pts = all_points(dad)
    nt = dad.nt
    k = float(dad.properties.k)
    if dad.dimension == 2
        cU = -1 / (2π * k)
        _dibem_src_loop!(nt, threaded) do j
            xj = pts[j]
            @inbounds for i in 1:nt
                i == j && continue
                R = norm(xj - pts[i])
                R > 0 || continue
                F[i, j] = rbf(R)
                D[i, j] = cU * log(R)
            end
        end
    else
        cU = 1 / (4π * k)
        _dibem_src_loop!(nt, threaded) do j
            xj = pts[j]
            @inbounds for i in 1:nt
                i == j && continue
                R = norm(xj - pts[i])
                R > 0 || continue
                F[i, j] = rbf(R)
                D[i, j] = cU / R
            end
        end
    end
    return nothing
end

function _dibem_fill_U!(F, dad; threaded::Bool=true)
    pts = all_points(dad)
    nt = dad.nt
    k = float(dad.properties.k)
    if dad.dimension == 2
        cU = -1 / (2π * k)
        _dibem_src_loop!(nt, threaded) do j
            xj = pts[j]
            @inbounds for i in 1:nt
                i == j && continue
                R = norm(xj - pts[i])
                R > 0 || continue
                F[i, j] = cU * log(R)
            end
        end
    else
        cU = 1 / (4π * k)
        _dibem_src_loop!(nt, threaded) do j
            xj = pts[j]
            @inbounds for i in 1:nt
                i == j && continue
                R = norm(xj - pts[i])
                R > 0 || continue
                F[i, j] = cU / R
            end
        end
    end
    return nothing
end

"""PHS centers at cell centroids; collocation only evaluates `I`."""
function _DIBEM_dense_volume_centers(dad::BEMdata{<:Laplace}, rbf; npg::Int=16,
        threaded::Bool=true)
    ξ = _dibem_volume_centers(dad)
    pts = all_points(dad)
    nc = length(ξ)
    nt = dad.nt
    k = float(dad.properties.k)
    F = zeros(nc, nc)
    D = zeros(nt, nc)
    dim = dad.dimension
    _dibem_src_loop!(nc, threaded) do kk
        ξk = ξ[kk]
        @inbounds for j in 1:nc
            r = norm(ξ[j] - ξk)
            r > 0 && (F[j, kk] = rbf(r))
        end
        if dim == 2
            cU = -1 / (2π * k)
            @inbounds for i in 1:nt
                R = norm(ξk - pts[i])
                R > 1e-15 && (D[i, kk] = cU * log(R))
            end
        else
            cU = 1 / (4π * k)
            @inbounds for i in 1:nt
                R = norm(ξk - pts[i])
                R > 1e-15 && (D[i, kk] = cU / R)
            end
        end
    end
    _dibem_ridge_F!(F)
    IF = _dibem_rbf_IF(dad, rbf, ξ; npg=npg, threaded=threaded)
    ID = zeros(nt)
    _dibem_accumulate_IF_ID!(nothing, ID, dad, rbf; threaded=threaded)
    IP = _dibem_monomial_IP(dad, rbf)
    c = _dibem_poly_c(F, IF, ξ, rbf; IP=IP)
    Q = _dibem_center_Q(pts, ξ)
    M = _dibem_M_volume(D, c, Q, ID)
    set_cache!(dad; M, dibem_F=F, dibem_c=c, dibem_ID=ID, dibem_rbf=rbf,
        dibem_Q=Q, dibem_method=:dense, dibem_centers=:cells)
    return M
end

"""
DIBEM when φ = u* (fundamental as RBF).

- Off-diagonal: ``F_{ij} = D_{ij} = u*(x_i,x_j)``
- Diagonal: F_ii from SBM OIF when ni==0 (same idea as SBM for the U kernel diagonal u_ii), instead of row-sum or ridge heuristic. This gives more accurate self-interaction for the singular fundamental basis.
- ``IF = ID`` (same radial integral of u*)
- ``c`` from ridge-regularized off-diagonal solve ``A c = ID``
- ``M_{ij} = F_{ij} c_j`` (off-diag), then ``M_{ii}`` fixed so ``M\\mathbf{1}=ID``
"""
function _DIBEM_dense_fundamental(dad::BEMdata{<:Laplace}, rbf::FundamentalRBF;
        threaded::Bool=true)
    nt = dad.nt
    F = zeros(nt, nt)   # also D off-diagonal
    ID = zeros(nt)

    # SBM OIF for F diagonal when there are no internals (2-D only).
    if dad.dimension == 2 && dad.ni == 0
        sd = assemble_sbm!(dad)
        F = sd.G  # U-eval matrix with SBM OIF diagonal
    else
        _dibem_fill_U!(F, dad; threaded=threaded)
        ε = 1e-10
        @inbounds for i in 1:nt
            F[i, i] += ε
        end
    end

    _dibem_accumulate_IF_ID!(nothing, ID, dad, rbf; threaded=threaded)

    # c-solve on off-diagonal + ridge (A has the small diag from SBM or ridge)
    A = copy(F)
    ε = 1e-10
    @inbounds for i in 1:nt
        A[i, i] = ε
    end
    c = A \ ID

    # M_ij = F_ij c_j (off-diag); then diagonal fix using ID so M 1 = ID
    M = copy(F)
    @inbounds for i in 1:nt
        M[i, i] = 0.0
    end
    M .*= c'
    @inbounds for i in 1:nt
        M[i, i] = -sum(view(M, i, :)) + ID[i]
    end
    set_cache!(dad; M, dibem_F=F, dibem_c=c, dibem_ID=ID, dibem_rbf=rbf,
        dibem_method=:dense_fs)
    return M
end

"""
    dibem_matrix(dad; rbf=PHS(), rebuild=false, method=:dense) -> M

Return the DIBEM operator `M`. Rebuilds via [`DIBEM`](@ref) if missing or
`rebuild=true`.
"""
function dibem_matrix(dad::BEMdata{<:Laplace}; rbf=PHS(), rebuild::Bool=false,
    method::Symbol=:dense, kwargs...)
    if rebuild || !has_cache(dad, :M)
        return DIBEM(dad; method=method, rbf=rbf, kwargs...)
    end
    return dad.M
end

export dibem_matrix

"""
Radial integral of the fundamental solution
∫ u* ρdρ in 2d
∫ u* ρ^2dρ in 3d
"""
@inline function radial_integral(props::Laplace, R::Real, ::Val{2})
    k = float(props.k)
    return -(2 * R^2 * log(R) - R^2) / (8 * π * k)
end
@inline function radial_integral(props::Laplace, R::Real, ::Val{3})
    return R^2 / (8 * π * float(props.k))
end
radial_integral(props::Laplace, R::Real, dim::Int) =
    dim == 2 ? radial_integral(props, R, Val(2)) : radial_integral(props, R, Val(3))

"""
    solve_poisson_dibem!(dad, f; rbf=PHS(3; poly_deg=1), npg=16) -> T

Poisson ``∇²u = f`` with DIBEM mass: ``H u − G q = M f``.
`f` is a number, `f(x)` , or a nodal vector on `all_points(dad)`.

When `f` is a scalar and `method=:dense`, only the RIM vector `ID` is built
(``M f = f·ID``). Call [`DIBEM`](@ref) first if you need `dad.M`.
"""
function solve_poisson_dibem!(dad::BEMdata{<:Laplace}, f;
        rbf=PHS(3; poly_deg=1), npg::Int=16, method::Symbol=:dense, kwargs...)
    has_cache(dad, :H) || assemble!(dad; npg=npg, kwargs...)
    const_f = f isa Number
    if const_f && method === :dense && !has_cache(dad, :M)
        if !has_cache(dad, :dibem_ID)
            ID = zeros(dad.nt)
            _dibem_accumulate_IF_ID!(nothing, ID, dad, rbf)
            set_cache!(dad; dibem_ID=ID, dibem_rbf=rbf)
        end
    else
        has_cache(dad, :M) || DIBEM(dad; rbf=rbf, method=method)
    end
    applyBC(dad)
    if const_f
        ID = has_cache(dad, :dibem_ID) ? dad.dibem_ID : dad.M * ones(dad.nt)
        dad.b .+= float(f) .* ID
    else
        fv = _eval_field(f, all_points(dad))
        dad.b .+= dad.M * fv
    end
    x = bem_linsolve(dad.A, dad.b)
    Tfull = zeros(eltype(x), dad.nt)
    qfull = zeros(eltype(x), dad.n)
    Tfull[1:length(x)] .= x
    split_sol!(dad, Tfull, qfull)
    set_cache!(dad; T=Tfull[1:dad.nt], q=qfull)
    return dad.T
end
