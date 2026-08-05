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
# U* backends (all node-major 2n×2n, blocksize=2):
#   :dense     — dense Matrix
#   :fmm       — KelvinFMMMatrix
#   :hmatrix / :hodlr / :hss / :hbs / :h2 — one structured matrix on
#                 expand_tree(point_tree, 2) via ScalarizedMatrix of 2×2 blocks
#   :hss + hss_method=:fmm — HMatrices.HSSMatrix from Kelvin FMM samples
# =============================================================================

export DibemElastFactoredOperator, DibemKelvinBlockKernel
# DIBEM entry points exported by Laplace/Domain_fast.jl

# ---------------------------------------------------------------------------
# IF (scalar) and ID (2×2 blocks stacked as 2nt×2)
# ---------------------------------------------------------------------------

function _dibem_elast_IF_ID(dad::BEMdata{<:Elasticity}, rbf)
    nt = dad.nt
    props = dad.properties
    pts = all_points(dad)
    IF = zeros(nt)
    ID = zeros(2nt, 2)
    @inbounds for i in 1:nt
        x = point(dad, i)
        for elem in dad.elements
            for j in eachindex(elem.index)
                ind = elem.index[j]
                xj = dad.Nodes[ind]
                r = xj - x
                R = norm(r)
                R < 1e-10 && continue
                wJn = dad.elem_weight[j] * elem.Jacobian[j] * dot(dad.Normal[ind], r) / R^2
                IF[i] += int(rbf, x, xj) * wJn
                e = r / R
                ID[2i-1:2i, :] .+= _galerkin_Ustar(props, R, e) * wJn
            end
        end
    end
    return IF, ID, pts
end

# ---------------------------------------------------------------------------
# Block Kelvin kernel  (eltype SMatrix{2,2}) → ScalarizedMatrix + expand_tree
# ---------------------------------------------------------------------------

"""
Bare 2D Kelvin single-layer as a point-indexed matrix of `SMatrix{2,2}`:

```
K[i,j] = U*(x_i, x_j) ∈ ℝ^{2×2}   (i≠j),   K[i,i] = 0
```

Compress with [`ScalarizedMatrix`](@ref) + [`expand_tree`](@ref)`(tree, 2)` so
H / HODLR / HSS see one `(2n)×(2n)` scalar matrix (node-major DOFs).
"""
struct DibemKelvinBlockKernel{P,Prop} <: AbstractMatrix{SMatrix{2,2,Float64,4}}
    points::Vector{P}
    props::Prop
    n_dummy::P
end

Base.size(K::DibemKelvinBlockKernel) = (length(K.points), length(K.points))
Base.IndexStyle(::Type{<:DibemKelvinBlockKernel}) = IndexCartesian()

const _ZERO22 = zero(SMatrix{2,2,Float64})

function Base.getindex(K::DibemKelvinBlockKernel, i::Int, j::Int)
    i == j && return _ZERO22
    r = K.points[j] - K.points[i]
    norm(r) < 1e-15 && return _ZERO22
    return _to_smat(fundamental(K.props, r, K.n_dummy).U)
end

"""
H² free-space sample: displacement component `a` at point `i_point` due to a
unit load in direction `b` at proxy location `yp`.
Matches `getindex` convention `r = source − field`.
"""
function HMatrices.h2_proxy_block(K::DibemKelvinBlockKernel, i_point::Int, a::Int, yp, b::Int)
    x = K.points[i_point]
    r = yp - x                    # source yp, field x  (same as col − row)
    R2 = sum(abs2, r)
    R2 < 1e-30 && return 0.0
    U = _to_smat(fundamental(K.props, r, K.n_dummy).U)
    return float(U[a, b])
end

"""Dense 2n×2n node-major Kelvin matrix (diagonal blocks zero)."""
function _dibem_kelvin_dense(pts, props, n_dummy)
    nt = length(pts)
    D = zeros(2nt, 2nt)
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        r = pts[j] - pts[i]
        norm(r) < 1e-15 && continue
        D[2i-1:2i, 2j-1:2j] .= _to_smat(fundamental(props, r, n_dummy).U)
    end
    return D
end

"""
Assemble compressed Kelvin `D` on collocation poles as a single `(2n)×(2n)`
operator (never four separate scalar matrices).
"""
function _dibem_elast_compress_D(dad, pts, method::Symbol; atol=1e-6, rtol=1e-6,
        nmax=32, eta=3.0, threads=true, rank=typemax(Int), alpha=1.0,
        hss_method=:dense, eps=1e-6)
    props = dad.properties
    nt = length(pts)
    n_dummy = pts[1] isa SVector{2} ? SVector(0.0, 1.0) : SVector(0.0, 0.0, 1.0)
    fmt = _dibem_struct_format(method)

    if fmt === :fmm
        dad.dimension == 2 || throw(ArgumentError("Kelvin FMM DIBEM is 2D only"))
        Pmat = Matrix{Float64}(undef, 2, nt)
        @inbounds for j in 1:nt
            Pmat[1, j] = pts[j][1]
            Pmat[2, j] = pts[j][2]
        end
        return FMM.fmm_kelvin2d_matrix(Pmat;
            μ=props.mu, ν=effective_nu(props),
            eps=Float64(eps), nmax=nmax, η=Float64(eta))
    end

    if fmt === :dense || fmt === nothing
        return _dibem_kelvin_dense(pts, props, n_dummy)
    end

    # HSS via Kelvin FMM matvecs + expand_tree(·, 2)
    if fmt in (:HSS, :HBS) && hss_method in (:fmm, :FMM, :matvec)
        dad.dimension == 2 || throw(ArgumentError("Kelvin HSS-FMM is 2D only"))
        Pmat = Matrix{Float64}(undef, 2, nt)
        @inbounds for j in 1:nt
            Pmat[1, j] = pts[j][1]
            Pmat[2, j] = pts[j][2]
        end
        KF = FMM.fmm_kelvin2d_matrix(Pmat;
            μ=props.mu, ν=effective_nu(props),
            eps=Float64(eps), nmax=nmax, η=Float64(eta))
        return assemble_hss_fmm_kernel(KF, pts;
            rtol=rtol, rank=rank == typemax(Int) ? 48 : rank,
            nmax=nmax, blocksize=2)
    end

    # One structured matrix on expanded DOF tree
    dad.dimension == 2 || throw(ArgumentError("structured Kelvin DIBEM is 2D only"))
    KB = DibemKelvinBlockKernel(pts, props, n_dummy)
    S = ScalarizedMatrix(KB)                 # (2n)×(2n) Float64 view
    splitter = PrincipalComponentSplitter(; nmax=nmax)
    tree = ClusterTree(collect(pts), splitter)
    etree = expand_tree(tree, 2)             # block sampler
    length(etree) == 2nt || throw(DimensionMismatch("expand_tree size"))

    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)
    adm = StrongAdmissibilityStd(; eta=eta)

    if fmt === :H2
        # FMM Kelvin matvecs → nested H² on expand_tree; else entry/proxy assemble_h2
        if hss_method in (:fmm, :FMM, :matvec)
            Pmat_h2 = Matrix{Float64}(undef, 2, nt)
            @inbounds for j in 1:nt
                Pmat_h2[1, j] = pts[j][1]
                Pmat_h2[2, j] = pts[j][2]
            end
            KF = FMM.fmm_kelvin2d_matrix(Pmat_h2;
                μ=props.mu, ν=effective_nu(props),
                eps=Float64(eps), nmax=nmax, η=Float64(eta))
            rH2 = rank == typemax(Int) ? 48 : Int(rank)
            return FMM.assemble_h2_fmm(KF, etree; rtol=rtol, rank=rH2, alpha=alpha,
                nsample=max(64, 2rH2), global_index=true)
        end
        # True H² on block-expanded tree: proxies × 2 load dirs → nested ID bases.
        # far_method=:aca → on-the-fly PartialACA for far B (near always dense).
        far_m = hss_method in (:aca, :ACA) ? :aca : :dense
        return assemble_h2(Float64, S, etree; rtol=rtol, rank=rank, alpha=alpha,
            global_index=true, symmetric=true, far_method=far_m,
            comp=far_m === :aca ? PartialACA(; rtol=rtol, rank=rank) : nothing)
    end

    return assemble_structured(S, etree; format=fmt, adm=adm, comp=comp,
        threads=threads, rtol=rtol, rank=rank, alpha=alpha,
        method=hss_method, global_index=true)
end

# ---------------------------------------------------------------------------
# Factored operator  M x = D (c ∘_nodes x) + block-diag · x
# ---------------------------------------------------------------------------

"""
    DibemElastFactoredOperator

Matrix-free elasticity DIBEM `M` (`2n × 2n`):

```
(M x)_i = U* (c ∘ x)_i   +   DiagBlock_i · x_i
```

`D` is any `(2n)×(2n)` matvec (dense, H-matrix on `expand_tree`, Kelvin FMM, …).
"""
struct DibemElastFactoredOperator{TD} <: AbstractMatrix{Float64}
    n::Int
    c::Vector{Float64}
    diagB::Matrix{Float64}       # 2n × 2
    D::TD
    method::Symbol
end

Base.size(A::DibemElastFactoredOperator) = (2A.n, 2A.n)
Base.IndexStyle(::Type{<:DibemElastFactoredOperator}) = IndexCartesian()

function Base.getindex(A::DibemElastFactoredOperator, α::Int, β::Int)
    i = (α + 1) ÷ 2
    j = (β + 1) ÷ 2
    a = α - 2(i - 1)
    b = β - 2(j - 1)
    if i == j
        return A.diagB[α, b]
    end
    return A.c[j] * A.D[α, β]
end

function _scale_nodes_c(c::Vector{Float64}, x::AbstractVector)
    n = length(c)
    z = similar(x)
    @inbounds for j in 1:n
        z[2j-1] = c[j] * x[2j-1]
        z[2j]   = c[j] * x[2j]
    end
    return z
end

function LinearAlgebra.mul!(y::AbstractVector, A::DibemElastFactoredOperator, x::AbstractVector)
    n2 = 2A.n
    length(x) == n2 && length(y) == n2 || throw(DimensionMismatch())
    z = _scale_nodes_c(A.c, x)
    mul!(y, A.D, z)
    @inbounds for i in 1:A.n
        r1, r2 = 2i - 1, 2i
        y[r1] += A.diagB[r1, 1] * x[r1] + A.diagB[r1, 2] * x[r2]
        y[r2] += A.diagB[r2, 1] * x[r1] + A.diagB[r2, 2] * x[r2]
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
    e1 = zeros(2n)
    e2 = zeros(2n)
    @inbounds for j in 1:n
        e1[2j-1] = c[j]
        e2[2j]   = c[j]
    end
    s1 = D * e1
    s2 = D * e2
    diagB = copy(ID)
    @inbounds for i in 1:n
        r1, r2 = 2i - 1, 2i
        diagB[r1, 1] -= s1[r1]
        diagB[r2, 1] -= s1[r2]
        diagB[r1, 2] -= s2[r1]
        diagB[r2, 2] -= s2[r2]
    end
    return DibemElastFactoredOperator(n, c, diagB, D, method)
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
    hss_method = :dense,
    eps = 1e-6,
    f_nmax = nothing,
)
    dad.dimension == 2 || error("DIBEM_compressed(Elasticity) is 2D only")
    IF, ID, pts = _dibem_elast_IF_ID(dad, rbf)
    nt = dad.nt

    fm = f_method
    if fm === :auto
        fm = nt <= 256 ? :dense :
             method in (:fmm, :FMM, :h2, :H2) ? :hmatrix : method
    end

    @info "DIBEM_compressed (elasticity)" method f_method=fm nt
    c = _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=fm, atol=atol, rtol=rtol,
        nmax = something(f_nmax, nmax), eta=eta, threads=threads, rank=rank,
        hss_method=hss_method, alpha=alpha)

    D = _dibem_elast_compress_D(dad, pts, method; atol=atol, rtol=rtol, nmax=nmax,
        eta=eta, threads=threads, rank=rank, alpha=alpha, hss_method=hss_method,
        eps=eps)

    M = _dibem_elast_factored_M(D, c, ID, method)
    set_cache!(dad; M=M, dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_rbf=rbf,
        dibem_method=method)
    return M
end

"""
    DIBEM(dad::BEMdata{<:Elasticity}; method=:dense, rbf=PHS(), kwargs...)

| `method` | `U*` compression | notes |
|----------|------------------|--------|
| `:dense` | dense `(2n)×(2n)` | `Domain.jl` |
| `:hmatrix` / `:hodlr` / `:hss` / `:hbs` | one structured matrix | `expand_tree(·,2)` |
| `:h2` | H² nested bases | proxies × 2 load dirs on expanded tree |
| `:fmm` | Kelvin FMM | 3× Laplace |
| `:hss` + `hss_method=:fmm` | HSS from Kelvin FMM samples | block tree |

```
M x = D (c ∘_nodes x) + blockdiag(ID − D c)
```
"""
function DIBEM(dad::BEMdata{<:Elasticity}; method::Symbol=:dense, rbf=PHS(), kwargs...)
    if method === :dense
        return DIBEM_dense(dad; rbf=rbf)
    elseif _dibem_struct_format(method) !== nothing || method in (:fmm, :FMM)
        return DIBEM_compressed(dad; method=method, rbf=rbf, kwargs...)
    else
        throw(ArgumentError(
            "DIBEM method must be :dense, :hmatrix, :hodlr, :hss, :hbs, :h2, or :fmm; got $method"))
    end
end
