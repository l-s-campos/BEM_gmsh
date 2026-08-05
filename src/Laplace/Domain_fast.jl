# =============================================================================
# Fast DIBEM — Laplace (factored form, compressed backends)
# =============================================================================
#
# Shared infrastructure: Core/DIBEM_common.jl
#   DibemFactoredOperator, DibemFKernel, _dibem_solve_Fc!, _dibem_struct_format, …
#
# Discrete dense (Domain.jl):
#   F c = IF ,   M_ij = c_j u*(x_i,x_j)  (i≠j) ,   M_ii = ID_i − ∑_{j≠i} M_ij
#
# Factored matvec:
#   M x = D (c ∘ x) + diag ∘ x ,   diag = ID − D c
# =============================================================================

export DIBEM, DIBEM_Hmat, DIBEM_HODLR, DIBEM_HSS, DIBEM_HBS, DIBEM_H2
export DIBEM_FMM, dibem!

# ---------------------------------------------------------------------------
# Geometry / IF, ID
# ---------------------------------------------------------------------------

function _dibem_IF_ID(dad::BEMdata{<:Laplace}, rbf)
    nt = dad.nt
    props = dad.properties
    pts = all_points(dad)
    IF = zeros(nt)
    ID = zeros(nt)
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
                ID[i] += _galerkin_n_dot_gradG(props, R) * wJn
            end
        end
    end
    return IF, ID, pts
end

# ---------------------------------------------------------------------------
# Bare single-layer kernel D = u*
# ---------------------------------------------------------------------------

"""
Square single-layer kernel on all collocation points (DIBEM factor `D`):

```
D_ij = u*(x_i, x_j) = fundamental(props, x_j−x_i, ·).U   (i≠j),   D_ii = 0
```
"""
struct DibemUStarKernel{P,Prop} <: AbstractMatrix{Float64}
    points::Vector{P}
    props::Prop
    n_dummy::P
end
Base.size(K::DibemUStarKernel) = (length(K.points), length(K.points))
function Base.getindex(K::DibemUStarKernel, i::Int, j::Int)
    i == j && return 0.0
    r = K.points[j] - K.points[i]
    norm(r) < 1e-15 && return 0.0
    return float(fundamental(K.props, r, K.n_dummy).U)
end

function _dibem_ustar_kernelmatrix(pts, props, n_dummy)
    function ustar(x, y)::Float64
        r = y - x
        R = norm(r)
        R < 1e-15 && return 0.0
        return float(fundamental(props, r, n_dummy).U)
    end
    return KernelMatrix{typeof(ustar), typeof(pts), typeof(pts), Float64}(ustar, pts, pts)
end

"""
Assemble compressed single-layer `D` on all collocation points (same `u*` as G).
"""
function _dibem_compress_D(dad, pts, method::Symbol; atol=1e-6, rtol=1e-6,
        nmax=32, eta=3.0, threads=true, rank=typemax(Int), alpha=1.0,
        hss_method=:dense, eps=1e-6)
    props = dad.properties
    k = float(props.k)
    dim = dad.dimension
    n_dummy = pts[1] isa SVector{2} ? SVector(0.0, 1.0) : SVector(0.0, 0.0, 1.0)
    nt = length(pts)
    fmt = _dibem_struct_format(method)

    if fmt === :fmm
        d = length(pts[1])
        Pmat = Matrix{Float64}(undef, d, nt)
        @inbounds for j in 1:nt, a in 1:d
            Pmat[a, j] = pts[j][a]
        end
        if dim == 2
            raw = FMM.fmm_laplace2d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
            return _ScaledFMM(raw, -1 / (2π * k))
        else
            raw = FMM.fmm_laplace3d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
            return _ScaledFMM(raw, 1 / k)
        end
    end

    if fmt === :dense || fmt === nothing
        D = zeros(nt, nt)
        KD = DibemUStarKernel(pts, props, n_dummy)
        @inbounds for j in 1:nt, i in 1:nt
            D[i, j] = KD[i, j]
        end
        return D
    end

    splitter = PrincipalComponentSplitter(; nmax=nmax)
    tree = ClusterTree(pts, splitter)
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)
    adm = StrongAdmissibilityStd(; eta=eta)

    if fmt === :H2
        # FMM matvecs → nested H² (HARA); else entry/proxy assemble_h2
        if hss_method in (:fmm, :FMM, :matvec)
            d = length(pts[1])
            Pmat = Matrix{Float64}(undef, d, nt)
            @inbounds for j in 1:nt, a in 1:d
                Pmat[a, j] = pts[j][a]
            end
            raw = dim == 2 ? FMM.fmm_laplace2d_matrix(Pmat; eps=Float64(eps), nmax=nmax) :
                  FMM.fmm_laplace3d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
            α = dim == 2 ? -1 / (2π * k) : 1 / k
            KF = _ScaledFMM(raw, α)
            rH2 = rank == typemax(Int) ? 48 : Int(rank)
            return FMM.assemble_h2_fmm(KF, tree; rtol=rtol, rank=rH2, alpha=alpha,
                nsample=max(64, 2rH2), global_index=true)
        end
        KD = _dibem_ustar_kernelmatrix(pts, props, n_dummy)
        far_m = hss_method in (:aca, :ACA) ? :aca : :dense
        return assemble_h2(Float64, KD, tree; rtol=rtol, rank=rank, alpha=alpha,
            global_index=true, symmetric=true, far_method=far_m,
            comp=far_m === :aca ? PartialACA(; rtol=rtol, rank=rank) : nothing)
    end

    # HSS via FMM matvecs → same HMatrices.HSSMatrix as method=:dense/:randomized
    if fmt in (:HSS, :HBS) && hss_method in (:fmm, :FMM, :matvec)
        k = float(props.k)
        dim = dad.dimension
        Pmat = Matrix{Float64}(undef, dim, nt)
        @inbounds for j in 1:nt, a in 1:dim
            Pmat[a, j] = pts[j][a]
        end
        raw = dim == 2 ? FMM.fmm_laplace2d_matrix(Pmat; eps=Float64(eps), nmax=nmax) :
              FMM.fmm_laplace3d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
        α = dim == 2 ? -1 / (2π * k) : 1 / k
        KF = _ScaledFMM(raw, α)
        return assemble_hss_fmm_kernel(KF, pts; rtol=rtol, rank=rank == typemax(Int) ? 48 : rank,
            nmax=nmax, blocksize=1)
    end

    KD = DibemUStarKernel(pts, props, n_dummy)
    return assemble_structured(KD, tree; format=fmt, adm=adm, comp=comp,
        threads=threads, rtol=rtol, rank=rank, alpha=alpha,
        method=hss_method, global_index=true)
end

# ---------------------------------------------------------------------------
# Unified compressed DIBEM
# ---------------------------------------------------------------------------

"""
    DIBEM_compressed(dad; method=:hmatrix, rbf=PHS(), f_method=:auto, kwargs...)

Factored Laplace DIBEM for any compression of `D = u*`.
"""
function DIBEM_compressed(
    dad::BEMdata{<:Laplace};
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
    IF, ID, pts = _dibem_IF_ID(dad, rbf)
    nt = dad.nt

    fm = f_method
    if fm === :auto
        fm = nt <= 256 ? :dense :
             method in (:fmm, :FMM, :h2, :H2) ? :hmatrix : method
    end

    @info "DIBEM_compressed" method f_method=fm nt
    c = _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=fm, atol=atol, rtol=rtol,
        nmax = something(f_nmax, nmax), eta=eta, threads=threads, rank=rank,
        hss_method=hss_method, alpha=alpha)

    D = _dibem_compress_D(dad, pts, method; atol=atol, rtol=rtol, nmax=nmax,
        eta=eta, threads=threads, rank=rank, alpha=alpha, hss_method=hss_method,
        eps=eps)

    M = _dibem_factored_M(D, c, ID, method)
    set_cache!(dad; M=M, dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_rbf=rbf,
        dibem_method=method)
    return M
end

# Named entry points (all factored)
DIBEM_Hmat(dad; kwargs...)  = DIBEM_compressed(dad; method=:hmatrix, kwargs...)
DIBEM_HODLR(dad; kwargs...) = DIBEM_compressed(dad; method=:hodlr, kwargs...)
DIBEM_HSS(dad; kwargs...)   = DIBEM_compressed(dad; method=:hss, kwargs...)
DIBEM_HBS(dad; kwargs...)   = DIBEM_compressed(dad; method=:hbs, kwargs...)
DIBEM_H2(dad; kwargs...)    = DIBEM_compressed(dad; method=:h2, f_method=:dense, kwargs...)
DIBEM_FMM(dad; kwargs...)   = DIBEM_compressed(dad; method=:fmm, kwargs...)

# ---------------------------------------------------------------------------
# Unified DIBEM entry
# ---------------------------------------------------------------------------

"""
    DIBEM(dad; method=:dense, rbf=PHS(), kwargs...)

| `method` | `D = u*` compression | `M` type |
|----------|----------------------|----------|
| `:dense` | dense | dense `Matrix` (`Domain.jl`) |
| `:hmatrix` | H-matrix | [`DibemFactoredOperator`](@ref) |
| `:hodlr` | HODLR | factored |
| `:hss` / `:hbs` | HSS | factored |
| `:h2` | H² | factored |
| `:fmm` | FMM | factored |
"""
function DIBEM(dad::BEMdata{<:Laplace}; method::Symbol=:dense, rbf=PHS(), kwargs...)
    if method === :dense
        return DIBEM_dense(dad; rbf=rbf)
    elseif _dibem_struct_format(method) !== nothing || method in (:fmm, :FMM)
        return DIBEM_compressed(dad; method=method, rbf=rbf, kwargs...)
    else
        throw(ArgumentError(
            "DIBEM method must be :dense, :hmatrix, :hodlr, :hss, :hbs, :h2, or :fmm; got $method"))
    end
end

const dibem! = DIBEM
