# =============================================================================
# Fast DIBEM — elasticity (factored form, compressed backends)
# =============================================================================
#
# Discrete dense (Domain.jl):
#   F c = IF ,   M-block_ij = c_j U*(x_i,x_j)  (i≠j) ,
#   M-block_ii = ID_i − ∑_{j≠i} M-block_ij
#
# Factored matvec:
#   M x = D (c ∘_nodes x) + blockdiag · x
#
# U* backends (node-major (d n)×(d n), blocksize=d):
#   :dense     — dense Matrix
#   :fmm       — KelvinFMMMatrix (2-D) or KelvinFMMMatrix3D (3-D)
#   :hmatrix   — H-matrix of SMatrix{d,d} on the point tree
#   :h2        — block NNCA (point skeletons, (d n)×(d n) apply)
# =============================================================================

export DibemElastFactoredOperator, DibemKelvinBlockKernel
# DIBEM entry points exported by Laplace/Domain_fast.jl

# ---------------------------------------------------------------------------
# IF (scalar) and ID (2×2 blocks stacked as 2nt×2)
# ---------------------------------------------------------------------------

"""RIM of the RBF (`IF`) and of Kelvin `U*` (`ID`).

`rim=:lumped` (default): near-element Gauss, far nodal lumping — same split
as Laplace [`_dibem_accumulate_IF_ID!`](@ref). `rim=:full_gauss`: Gauss on
every element (`calc_md` / `Monta_M_RIMd`). Integrand is regular (`ρ log ρ`,
`ρ³`); no sinh.
"""
function _dibem_elast_IF_ID(dad::BEMdata{<:Elasticity}, rbf; npg::Int=12,
        threaded::Bool=true, rim::Symbol=:lumped, near_factor::Float64=1.5)
    rim in (:lumped, :full_gauss) || throw(ArgumentError(
        "DIBEM rim must be :lumped or :full_gauss (got $rim)"))
    nt = dad.nt
    dim = dad.dimension
    IF = zeros(nt)
    ID = zeros(dim * nt, dim)
    ηs, ws = gausslegendre(npg)
    geos = _rim_build_elements(dad, ηs, ws)
    dim == 2 ? _dibem_elast_rim!(IF, ID, dad, rbf, geos, Val(2), rim, near_factor, threaded) :
               _dibem_elast_rim!(IF, ID, dad, rbf, geos, Val(3), rim, near_factor, threaded)
    return IF, ID, all_points(dad)
end

function _dibem_elast_rim!(IF, ID, dad, rbf, geos, dimv::Val{D}, rim::Symbol,
        near_factor::Float64, threaded::Bool) where {D}
    props = dad.properties
    pts = all_points(dad)
    lump = rim === :lumped
    _dibem_src_loop!(dad.nt, threaded) do i
        x = pts[i]
        accF = 0.0
        accD = zero(SMatrix{D,D,Float64,D * D})
        @inbounds for g in geos
            dF, dD = if lump && !_near_element(x, g.nodes, g.el; factor=near_factor)
                _dibem_elast_far_ID(x, g, rbf, props, dimv)
            else
                _dibem_elast_near_ID(x, g, rbf, props, dimv)
            end
            accF += dF
            accD += dD
        end
        IF[i] += accF
        i0 = D * (i - 1)
        @inbounds for β in 1:D, α in 1:D
            ID[i0 + α, β] += accD[α, β]
        end
    end
    return nothing
end

function _dibem_elast_near_ID(x, g, rbf, props, ::Val{D}) where {D}
    accF = 0.0
    accD = zero(SMatrix{D,D,Float64,D * D})
    @inbounds for q in eachindex(g.wJ)
        wJ = g.wJ[q]
        wJ == 0 && continue
        y = g.y[q]
        r = y - x
        R = norm(r)
        R < 1e-14 && continue
        e = r / R
        wJn = wJ * _rim_factor(g.n[q], r, R, Val(D))
        accF += int(rbf, x, y) * wJn
        accD += _galerkin_Ustar(props, R, e) * wJn
    end
    return accF, accD
end

function _dibem_elast_far_ID(x, g, rbf, props, ::Val{D}) where {D}
    accF = 0.0
    accD = zero(SMatrix{D,D,Float64,D * D})
    @inbounds for j in eachindex(g.xj)
        xj = g.xj[j]
        r = xj - x
        R = norm(r)
        R < 1e-10 && continue
        e = r / R
        wJn = g.wj[j] * _rim_factor(g.nj[j], r, R, Val(D))
        accF += int(rbf, x, xj) * wJn
        accD += _galerkin_Ustar(props, R, e) * wJn
    end
    return accF, accD
end

# ---------------------------------------------------------------------------
# Block Kelvin kernel  (eltype SMatrix{D,D})
# ---------------------------------------------------------------------------

"""
Bare Kelvin single-layer as a point-indexed matrix of `SMatrix{D,D}`:

```
K[i,j] = U*(x_i, x_j) ∈ ℝ^{D×D}   (i≠j),   K[i,i] = 0
```

H-matrix and block NNCA assemble this on the **point** tree (tensor entries).
"""
struct DibemKelvinBlockKernel{P, Prop, D, L} <: AbstractMatrix{SMatrix{D, D, Float64, L}}
    points::Vector{P}
    props::Prop
    n_dummy::P
end

function DibemKelvinBlockKernel(points::Vector{P}, props, n_dummy) where {P}
    D = length(eltype(points))
    L = D * D
    return DibemKelvinBlockKernel{P, typeof(props), D, L}(points, props, n_dummy)
end

Base.size(K::DibemKelvinBlockKernel) = (length(K.points), length(K.points))
Base.IndexStyle(::Type{<:DibemKelvinBlockKernel}) = IndexCartesian()

function Base.getindex(K::DibemKelvinBlockKernel{P, Prop, D, L}, i::Int, j::Int) where {P, Prop, D, L}
    i == j && return zero(SMatrix{D, D, Float64, L})
    r = K.points[j] - K.points[i]
    norm(r) < 1e-15 && return zero(SMatrix{D, D, Float64, L})
    return _to_smat(fundamental(K.props, r, K.n_dummy).U)
end

"""Dense `(d n)×(d n)` node-major Kelvin matrix (diagonal blocks zero)."""
function _dibem_kelvin_dense(pts, props, n_dummy)
    nt = length(pts)
    dim = length(eltype(pts))
    D = zeros(dim * nt, dim * nt)
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        r = pts[j] - pts[i]
        norm(r) < 1e-15 && continue
        D[expand(i, dim), expand(j, dim)] .= _to_smat(fundamental(props, r, n_dummy).U)
    end
    return D
end

"""
Assemble compressed Kelvin `D` on collocation poles as a single `(d n)×(d n)`
operator (never four/nine separate scalar matrices).
"""
function _dibem_elast_compress_D(dad, pts, method::Symbol; atol=1e-6, rtol=1e-6,
        nmax=32, eta=3.0, threads=true, rank=typemax(Int), alpha=1.0,
        eps=1e-6, device=:host)
    props = dad.properties
    nt = length(pts)
    dim = length(eltype(pts))
    n_dummy = pts[1] isa SVector{2} ? SVector(0.0, 1.0) : SVector(0.0, 0.0, 1.0)
    fmt = _dibem_struct_format(method)

    if fmt === :fmm
        Pmat = Matrix{Float64}(undef, dim, nt)
        @inbounds for j in 1:nt, a in 1:dim
            Pmat[a, j] = pts[j][a]
        end
        if dim == 2
            return FMM.fmm_kelvin2d_matrix(Pmat;
                μ=props.mu, ν=effective_nu(props),
                eps=Float64(eps), nmax=nmax, η=Float64(eta))
        else
            return FMM.fmm_kelvin3d_matrix(Pmat;
                μ=props.mu, ν=props.nu,
                eps=Float64(eps), nmax=nmax, η=Float64(eta))
        end
    end

    if fmt === :dense || fmt === nothing
        return _dibem_kelvin_dense(pts, props, n_dummy)
    end

    KB = DibemKelvinBlockKernel(pts, props, n_dummy)
    splitter = hmatrix_splitter(; nmax=nmax)
    tree = ClusterTree(collect(pts), splitter)
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)
    adm = StrongAdmissibilityStd(; eta=eta)

    if fmt === :H
        return assemble_hmatrix(KB, tree, tree; adm=adm, comp=comp, threads=threads,
            device=device)
    elseif fmt === :H2
        return assemble_h2(KB, tree; rtol=rtol, rank=rank, threads=threads,
            device=device)
    end
    throw(ArgumentError(
        "elasticity DIBEM compression must be :dense, :hmatrix, :h2, or :fmm; got $method"))
end

# ---------------------------------------------------------------------------
# Factored operator  M x = D (c ∘_nodes x) + block-diag · x
# ---------------------------------------------------------------------------

"""
    DibemElastFactoredOperator

Matrix-free elasticity DIBEM `M` (`(d n)×(d n)`):

```
(M x)_i = U* (c ∘ x)_i   +   DiagBlock_i · x_i
```

`D` is any `(d n)×(d n)` matvec (dense, tensor H-matrix, block NNCA, Kelvin FMM, …).
"""
struct DibemElastFactoredOperator{TD} <: AbstractMatrix{Float64}
    n::Int
    dim::Int
    c::Vector{Float64}
    diagB::Matrix{Float64}       # (d n) × d
    D::TD
    method::Symbol
end

Base.size(A::DibemElastFactoredOperator) = (A.dim * A.n, A.dim * A.n)
Base.IndexStyle(::Type{<:DibemElastFactoredOperator}) = IndexCartesian()

function Base.getindex(A::DibemElastFactoredOperator, α::Int, β::Int)
    d = A.dim
    i = (α - 1) ÷ d + 1
    j = (β - 1) ÷ d + 1
    b = β - d * (j - 1)
    if i == j
        return A.diagB[α, b]
    end
    return A.c[j] * A.D[α, β]
end

function _scale_nodes_c(c::Vector{Float64}, x::AbstractVector, dim::Int)
    n = length(c)
    z = similar(x)
    @inbounds for j in 1:n
        a = c[j]
        base = dim * (j - 1)
        for p in 1:dim
            z[base + p] = a * x[base + p]
        end
    end
    return z
end
_scale_nodes_c(c::Vector{Float64}, x::AbstractVector) = _scale_nodes_c(c, x, 2)

function LinearAlgebra.mul!(y::AbstractVector, A::DibemElastFactoredOperator, x::AbstractVector)
    n2 = A.dim * A.n
    length(x) == n2 && length(y) == n2 || throw(DimensionMismatch())
    z = _scale_nodes_c(A.c, x, A.dim)
    mul!(y, A.D, z)
    d = A.dim
    @inbounds for i in 1:A.n
        rows = expand(i, d)
        for a in 1:d
            s = zero(eltype(y))
            for b in 1:d
                s += A.diagB[rows[a], b] * x[rows[b]]
            end
            y[rows[a]] += s
        end
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::DibemElastFactoredOperator, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, A, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, A, x)
        y .= β .* y .+ α .* t
    end
    return y
end

Base.:*(A::DibemElastFactoredOperator, x::AbstractVector) = mul!(similar(x, Float64), A, x)

function _dibem_elast_factored_M(D, c::Vector{Float64}, ID::Matrix{Float64}, method::Symbol)
    n = length(c)
    dim = size(ID, 2)
    size(ID, 1) == dim * n || throw(DimensionMismatch("ID"))
    ndof = dim * n
    e = zeros(ndof)
    diagB = copy(ID)
    @inbounds for d in 1:dim
        fill!(e, 0.0)
        for j in 1:n
            e[dim * (j - 1) + d] = c[j]
        end
        s = D * e
        diagB[:, d] .-= s
    end
    return DibemElastFactoredOperator(n, dim, c, diagB, D, method)
end

# ---------------------------------------------------------------------------
# Unified compressed DIBEM (elasticity)
# ---------------------------------------------------------------------------

function DIBEM_compressed(
    dad::BEMdata{<:Elasticity};
    method::Symbol = :hmatrix,
    rbf = PHS(),
    f_method::Symbol = :auto,
    atol = 1e-6,
    rtol = 1e-6,
    nmax = 32,
    eta = 3.0,
    threads = true,
    rank = typemax(Int),
    alpha = 1.0,
    eps = 1e-6,
    f_nmax = nothing,
    npg::Int = 12,
    rim::Symbol = :lumped,
    device = :host,
    kwargs...,
)
    IF, ID, pts = _dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=threads, rim=rim)
    nt = dad.nt

    fm = f_method
    if fm === :auto
        fm = nt <= 256 ? :dense :
             method in (:fmm, :FMM) ? :hmatrix : method
    end

    @info "DIBEM_compressed (elasticity)" method f_method=fm nt
    c = _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=fm, atol=atol, rtol=rtol,
        nmax = something(f_nmax, nmax), eta=eta, threads=threads, rank=rank,
        alpha=alpha, IP=_dibem_monomial_IP(dad, rbf))

    D = _dibem_elast_compress_D(dad, pts, method; atol=atol, rtol=rtol, nmax=nmax,
        eta=eta, threads=threads, rank=rank, alpha=alpha,
        eps=eps, device=device)

    M = _dibem_elast_factored_M(D, c, ID, method)
    ρ = dad.properties.rho
    if ρ != 1
        if M isa Matrix
            M .*= ρ
        else
            M.c .*= ρ
            M.diagB .*= ρ
        end
    end
    set_cache!(dad; M=M, dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_rbf=rbf,
        dibem_method=method)
    return M
end

"""
    DIBEM(dad::BEMdata{<:Elasticity}; method=:dense, rbf=PHS(), kwargs...)

| `method` | `U*` compression | notes |
|----------|------------------|--------|
| `:dense` | dense `(d n)×(d n)` | `Domain.jl` |
| `:hmatrix` | H-matrix of `SMatrix{d,d}` | point tree |
| `:h2` | block NNCA | point skeletons, `(d n)×(d n)` apply |
| `:fmm` | Kelvin FMM | 2-D: 3× Laplace; 3-D: 4× Laplace-3D |
| `:gpu` | dense on GPU (2-D; KernelAbstractions) | dense `Matrix` |

```
M x = D (c ∘_nodes x) + blockdiag(ID − D c)
```
"""
function DIBEM(dad::BEMdata{<:Elasticity}; method::Symbol=:dense, rbf=PHS(),
        centers::Symbol=:collocation, kwargs...)
    if method === :dense
        return DIBEM_dense(dad; rbf=rbf, centers=centers, kwargs...)
    elseif method === :gpu
        return DIBEM_gpu(dad; rbf=rbf, centers=centers, kwargs...)
    elseif centers !== :collocation
        throw(ArgumentError("DIBEM centers=:cells is dense-only"))
    elseif _dibem_struct_format(method) !== nothing || method in (:fmm, :FMM)
        return DIBEM_compressed(dad; method=method, rbf=rbf, kwargs...)
    else
        throw(ArgumentError(
            "DIBEM method must be :dense, :gpu, :hmatrix, :h2, or :fmm; got $method"))
    end
end
