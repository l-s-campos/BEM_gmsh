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

export DIBEM, DIBEM_Hmat, DIBEM_H2
export DIBEM_FMM, dibem!

# ---------------------------------------------------------------------------
# Geometry / IF, ID
# ---------------------------------------------------------------------------

function _dibem_IF_ID(dad::BEMdata{<:Laplace}, rbf; threaded::Bool=true)
    nt = dad.nt
    IF = zeros(nt)
    ID = zeros(nt)
    _dibem_accumulate_IF_ID!(IF, ID, dad, rbf; threaded=threaded)
    return IF, ID, all_points(dad)
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
    _R2(r) < 1e-30 && return 0.0
    return fundamental_U(K.props, r)
end

function _dibem_ustar_kernelmatrix(pts, props, n_dummy)
    function ustar(x, y)::Float64
        r = y - x
        _R2(r) < 1e-30 && return 0.0
        return fundamental_U(props, r)
    end
    return KernelMatrix{typeof(ustar), typeof(pts), typeof(pts), Float64}(ustar, pts, pts)
end

"""
Assemble compressed single-layer `D` on all collocation points (same `u*` as G).
"""
function _dibem_compress_D(dad, pts, method::Symbol; atol=1e-6, rtol=1e-6,
        nmax=32, eta=3.0, threads=true, rank=typemax(Int), alpha=1.0,
        eps=1e-6, device=:host)
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

    splitter = hmatrix_splitter(; nmax=nmax)
    tree = ClusterTree(pts, splitter)
    comp = PartialACA(; atol=atol, rtol=rtol, rank=rank)
    adm = StrongAdmissibilityStd(; eta=eta)

    KD = DibemUStarKernel(pts, props, n_dummy)
    return assemble_structured(KD, tree; format=fmt, adm=adm, comp=comp,
        threads=threads, rtol=rtol, rank=rank, alpha=alpha,
        global_index=true, device=device)
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
    eps = 1e-6,
    f_nmax = nothing,
    device = :host,
)
    # φ = u*: F≡D off-diag, IF≡ID — build D once, derive F/c from it
    if rbf isa FundamentalRBF
        return _DIBEM_compressed_fundamental(dad; method=method, rbf=rbf,
            f_method=f_method, atol=atol, rtol=rtol, nmax=nmax, eta=eta,
            threads=threads, rank=rank, alpha=alpha,
            eps=eps, f_nmax=f_nmax, device=device)
    end

    IF, ID, pts = _dibem_IF_ID(dad, rbf; threaded=threads)
    nt = dad.nt

    fm = f_method
    if fm === :auto
        fm = nt <= 256 ? :dense :
             method in (:fmm, :FMM) ? :hmatrix : method
    end

    @info "DIBEM_compressed" method f_method=fm nt
    c = _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=fm, atol=atol, rtol=rtol,
        nmax = something(f_nmax, nmax), eta=eta, threads=threads, rank=rank,
        alpha=alpha)

    D = _dibem_compress_D(dad, pts, method; atol=atol, rtol=rtol, nmax=nmax,
        eta=eta, threads=threads, rank=rank, alpha=alpha,
        eps=eps, device=device)

    M = _dibem_factored_M(D, c, ID, method)
    set_cache!(dad; M=M, dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_rbf=rbf,
        dibem_method=method)
    return M
end

"""Compressed DIBEM with φ=u*: assemble D once; F is D + zero-row-sum diag; IF=ID."""
function _DIBEM_compressed_fundamental(
    dad::BEMdata{<:Laplace};
    method::Symbol = :hmatrix,
    rbf::FundamentalRBF,
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
    device = :host,
)
    # ID only (IF ≡ ID); D = u*
    _, ID, pts = _dibem_IF_ID(dad, rbf; threaded=threads)   # int(FS)=radial_integral(Laplace) ⇒ IF=ID
    nt = dad.nt
    D = _dibem_compress_D(dad, pts, method; atol=atol, rtol=rtol, nmax=nmax,
        eta=eta, threads=threads, rank=rank, alpha=alpha,
        eps=eps, device=device)

    # F off-diag ≡ D (φ=u*); densify hierarchical D by matvecs
    F = zeros(nt, nt)
    if D isa Matrix
        @inbounds for j in 1:nt, i in 1:nt
            i == j && continue
            F[i, j] = D[i, j]
        end
    else
        x = zeros(nt)
        col = zeros(nt)
        @inbounds for j in 1:nt
            fill!(x, 0.0); x[j] = 1.0
            mul!(col, D, x)
            @inbounds for i in 1:nt
                i == j && continue
                F[i, j] = col[i]
            end
        end
    end
    _zero_rowsum_diag!(F)   # stored F: F1=0

    # c-solve on off-diagonal F + ridge diagonal (keep stored F pure zrs)
    ε = 1e-10
    A = copy(F)
    @inbounds for i in 1:nt
        A[i, i] = ε
    end
    c = A \ ID

    M = _dibem_factored_M(D, c, ID, method)
    set_cache!(dad; M=M, dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_F=F,
        dibem_rbf=rbf, dibem_method=Symbol(String(method) * "_fs"))
    return M
end
# Named entry points (all factored)
DIBEM_Hmat(dad; kwargs...)  = DIBEM_compressed(dad; method=:hmatrix, kwargs...)
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
| `:h2` | NNCA H² | factored |
| `:fmm` | FMM | factored |
| `:gpu` | dense on GPU (2-D; KernelAbstractions) | dense `Matrix` |
"""
function DIBEM(dad::BEMdata{<:Laplace}; method::Symbol=:dense, rbf=PHS(),
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

"""Teaching alias for [`DIBEM`](@ref)."""
const dibem! = DIBEM
