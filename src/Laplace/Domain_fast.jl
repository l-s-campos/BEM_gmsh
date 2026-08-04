# =============================================================================
# Fast DIBEM: hierarchical (H-matrix) and FMM-backed M
# =============================================================================
# Classic dense path: Domain.jl → DIBEM
# Here:
#   DIBEM_Hmat  — ACA H-matrices for F (RBF) and M (column-scaled FS)
#   DIBEM_FMM   — H-matrix/dense F + FMM matvec for the Laplace factor of M
#   DIBEM(...; method=:dense|:hmatrix|:fmm)
# =============================================================================

export DIBEM, DIBEM_Hmat, DIBEM_FMM, DibemFMMOperator, dibem!

# ---------------------------------------------------------------------------
# Shared geometry / boundary integrals (IF, ID)
# ---------------------------------------------------------------------------

function _dibem_collocation_points(dad::BEMdata)
    return isempty(dad.internalNodes) ? collect(dad.Nodes) :
           vcat(collect(dad.Nodes), collect(dad.internalNodes))
end

"""
Boundary integrals for DIBEM:
- `IF[i] = ∑_Γ int(rbf, x_i, X) (n·r/R²) dΓ`  (primitive of RBF)
- `ID[i] = ∑_Γ G*_k n_k dΓ`  (Galerkin remainder for u*)
"""
function _dibem_IF_ID(dad::BEMdata{<:Laplace}, rbf)
    nt = dad.nt
    k = float(dad.properties.k)
    pts = _dibem_collocation_points(dad)
    IF = zeros(nt)
    ID = zeros(nt)
    @inbounds for i in 1:nt
        x = pts[i]
        for elem in dad.elements
            for j in eachindex(elem.index)
                ind = elem.index[j]
                xj = dad.Nodes[ind]
                r = xj - x
                R = norm(r)
                R < 1e-10 && continue
                wJ = dad.elem_weight[j] * elem.Jacobian[j]
                n_dot = dot(dad.Normal[ind], r) / R^2
                IF[i] += int(rbf, x, xj) * wJ * n_dot
                ID[i] += -(2 * R^2 * log(R) - R^2) / (8 * π * k) * wJ * n_dot
            end
        end
    end
    return IF, ID, pts
end

# ---------------------------------------------------------------------------
# Kernel matrices for H-assembly
# ---------------------------------------------------------------------------

"""RBF Gram matrix ``F[i,j] = φ(‖x_i - x_j‖²)``."""
struct DibemFKernel{R,P} <: AbstractMatrix{Float64}
    rbf::R
    points::Vector{P}
end
Base.size(K::DibemFKernel) = (length(K.points), length(K.points))
function Base.getindex(K::DibemFKernel, i::Int, j::Int)
    return float(K.rbf(sqeuclidean(K.points[i], K.points[j])))
end

"""
Column-scaled fundamental kernel for DIBEM off-diagonal:
``M[i,j] = c[j] · u*(x_i, x_j)`` (diagonal left 0 until correction).
"""
struct DibemMKernel{P} <: AbstractMatrix{Float64}
    points::Vector{P}
    c::Vector{Float64}
    k::Float64
    dim::Int
end
Base.size(K::DibemMKernel) = (length(K.points), length(K.points))
function Base.getindex(K::DibemMKernel, i::Int, j::Int)
    i == j && return 0.0
    r = K.points[j] - K.points[i]
    R = norm(r)
    R < 1e-15 && return 0.0
    # Match Domain.jl: -log(R²)/(4πk) = -log(R)/(2πk)
    u = if K.dim == 2
        -log(R) / (2π * K.k)
    else
        1 / (4π * K.k * R)
    end
    return K.c[j] * u
end

# ---------------------------------------------------------------------------
# H-matrix DIBEM
# ---------------------------------------------------------------------------

"""
    DIBEM_Hmat(dad; rbf=PHS(), atol=1e-6, nmax=32, eta=3.0, threads=true)

Build the DIBEM operator `M` with **hierarchical matrices**:

1. `F` (RBF) as H-matrix → solve `F c = IF` (GMRES)
2. Assemble `M[i,j] = c[j] u*(x_i,x_j)` as H-matrix
3. Diagonal regularization: `M_ii = -∑_{j≠i} M_ij + ID_i`

Stores `dad.cache.M` (HMatrix) and returns it.
"""
function DIBEM_Hmat(
    dad::BEMdata{<:Laplace};
    rbf = PHS(),
    atol = 1e-6,
    rtol = 1e-6,
    nmax = 32,
    eta = 3.0,
    threads = true,
    gmres_itmax = 0,
)
    nt = dad.nt
    k = float(dad.properties.k)
    dim = dad.dimension
    IF, ID, pts = _dibem_IF_ID(dad, rbf)

    splitter = PrincipalComponentSplitter(; nmax=nmax)
    Xclt = ClusterTree(pts, splitter)
    Yclt = ClusterTree(copy(pts), splitter)
    adm = StrongAdmissibilityStd(; eta=eta)
    comp = PartialACA(; atol=atol)

    # --- F (RBF) ---
    KF = DibemFKernel(rbf, pts)
    @info "DIBEM_Hmat: assembling F (RBF)" nt
    Fh = assemble_hmatrix(KF, Xclt, Yclt; adm=adm, comp=comp, threads=threads)

    # ridge on dense diagonal blocks for stability
    _hmat_add_diag_ridge!(Fh, 1e-12)

    itmax = gmres_itmax > 0 ? gmres_itmax : max(4 * nt, 200)
    c, stats = Krylov.gmres(Fh, IF; atol=rtol, rtol=rtol, itmax=itmax)
    if !stats.solved
        @warn "DIBEM_Hmat: GMRES on F did not fully converge" stats.niter stats.status
    end

    # --- M off-diagonal as H-matrix ---
    KM = DibemMKernel(pts, c, k, dim)
    @info "DIBEM_Hmat: assembling M" nt
    # fresh trees (same geometry)
    Xclt2 = ClusterTree(pts, splitter)
    Yclt2 = ClusterTree(copy(pts), splitter)
    Mh = assemble_hmatrix(KM, Xclt2, Yclt2; adm=adm, comp=comp, threads=threads)

    # Diagonal: M_ii = -∑_j M_ij + ID_i  (row sum of current M has M_ii=0)
    rowsum = Mh * ones(nt)
    _set_diagonal!(Mh) do i
        -rowsum[i] + ID[i]
    end

    set_cache!(dad; M=Mh, dibem_c=c, dibem_ID=ID, dibem_rbf=rbf, dibem_method=:hmatrix)
    return Mh
end

function _hmat_add_diag_ridge!(Hmat::HMatrices.HMatrix, ε::Float64)
    piv = HMatrices.pivot(Hmat)
    n = size(Hmat, 1)
    # accumulate current diag via matvec is hard; add ε on dense diagonal blocks only
    for block in HMatrices.nodes(Hmat)
        HMatrices.hasdata(block) || continue
        HMatrices.isadmissible(block) && continue
        data = HMatrices.data(block)
        data isa Matrix || continue
        irange = HMatrices.rowrange(block) .- piv[1] .+ 1
        jrange = HMatrices.colrange(block) .- piv[2] .+ 1
        irangeg = HMatrices.rowperm(Hmat)[irange]
        jrangeg = HMatrices.colperm(Hmat)[jrange]
        for (iloc, ig) in enumerate(irangeg)
            for (jloc, jg) in enumerate(jrangeg)
                if ig == jg && iloc <= size(data, 1) && jloc <= size(data, 2)
                    data[iloc, jloc] += ε * (tr(data) / max(size(data, 1), 1) + 1)
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# FMM-backed matrix-free M
# ---------------------------------------------------------------------------

"""
    DibemFMMOperator

Matrix-free DIBEM `M` with FMM Laplace matvec:

```
(M x)_i = [D (c ∘ x)]_i + diag_i x_i
```

where `D` is the fundamental-solution matrix (FMM), `c = F \\ IF`,
and `diag` enforces the Galerkin/regularization diagonal.
"""
struct DibemFMMOperator{TD} <: AbstractMatrix{Float64}
    n::Int
    c::Vector{Float64}
    diag::Vector{Float64}
    D::TD                 # FMMKernelMatrix or similar with mul!
end

Base.size(A::DibemFMMOperator) = (A.n, A.n)
Base.IndexStyle(::Type{<:DibemFMMOperator}) = IndexCartesian()

function Base.getindex(A::DibemFMMOperator, i::Int, j::Int)
    i == j && return A.diag[i]
    # entry of D scaled by c[j]
    return A.c[j] * A.D[i, j]
end

function LinearAlgebra.mul!(y::AbstractVector, A::DibemFMMOperator, x::AbstractVector)
    length(x) == A.n && length(y) == A.n || throw(DimensionMismatch())
    # y = D * (c .* x)
    tmp = A.c .* x
    mul!(y, A.D, tmp)
    @inbounds for i in 1:A.n
        y[i] += A.diag[i] * x[i]
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector, A::DibemFMMOperator, x::AbstractVector,
    α::Number, β::Number) = begin
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

"""
    DIBEM_FMM(dad; rbf=PHS(), eps=1e-6, nmax=50, f_method=:hmatrix)

Build DIBEM `M` as a **matrix-free** [`DibemFMMOperator`](@ref):

| Factor | Method |
|--------|--------|
| `F` (RBF) | H-matrix (`:hmatrix`) or dense (`:dense`) → `c = F \\ IF` |
| `D` (Laplace FS) | FMM matvec (`FMM.fmm_laplace2d/3d_matrix`) |
| diagonal | regularization with `ID` |

Stores `dad.cache.M` and returns the operator.
"""
function DIBEM_FMM(
    dad::BEMdata{<:Laplace};
    rbf = PHS(),
    eps = 1e-6,
    rtol = 1e-6,
    nmax = 50,
    eta = 3.0,
    f_method::Symbol = :hmatrix,
    f_nmax = 32,
    threads = true,
)
    nt = dad.nt
    k = float(dad.properties.k)
    dim = dad.dimension
    IF, ID, pts = _dibem_IF_ID(dad, rbf)

    # --- solve F c = IF ---
    c = _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=f_method, atol=eps,
        rtol=rtol, nmax=f_nmax, eta=eta, threads=threads)

    # --- FMM operator for D ~ u* ---
    # Domain: u* = -log(R)/(2π k) in 2D,  1/(4π k R) in 3D
    P = reduce(hcat, pts)  # d × nt if pts are SVectors - need d×N
    d = length(pts[1])
    Pmat = Matrix{Float64}(undef, d, nt)
    @inbounds for j in 1:nt, a in 1:d
        Pmat[a, j] = pts[j][a]
    end

    D_fmm = if dim == 2
        # fmm returns log(R); we need -log(R)/(2πk)
        raw = FMM.fmm_laplace2d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
        _ScaledFMM(raw, -1 / (2π * k))
    else
        # fmm returns 1/(4π R); Domain uses 1/(4π k R)
        raw = FMM.fmm_laplace3d_matrix(Pmat; eps=Float64(eps), nmax=nmax)
        _ScaledFMM(raw, 1 / k)
    end

    # off-diagonal row sums: (D c)_i = ∑_j D_ij c_j
    rowsum_off = D_fmm * c
    diag = -rowsum_off .+ ID

    Mop = DibemFMMOperator(nt, c, diag, D_fmm)
    set_cache!(dad; M=Mop, dibem_c=c, dibem_ID=ID, dibem_rbf=rbf, dibem_method=:fmm)
    return Mop
end

"""Scale an FMM kernel matrix: `(αA)*x = α (A*x)`."""
struct _ScaledFMM{TA}
    A::TA
    α::Float64
end
Base.size(S::_ScaledFMM, d) = size(S.A, d)
Base.size(S::_ScaledFMM) = size(S.A)
Base.getindex(S::_ScaledFMM, i::Int, j::Int) = S.α * S.A[i, j]
function LinearAlgebra.mul!(y::AbstractVector, S::_ScaledFMM, x::AbstractVector)
    mul!(y, S.A, x)
    y .*= S.α
    return y
end
Base.:*(S::_ScaledFMM, x::AbstractVector) = mul!(similar(x, Float64), S, x)
Base.:*(A::DibemFMMOperator, x::AbstractVector) = mul!(similar(x, Float64), A, x)

function _dibem_solve_Fc!(dad, pts, IF, rbf; f_method=:hmatrix, atol=1e-6,
        rtol=1e-6, nmax=32, eta=3.0, threads=true)
    nt = length(IF)
    if f_method === :dense || nt <= 256
        F = zeros(nt, nt)
        @inbounds for j in 1:nt, i in 1:nt
            F[i, j] = rbf(sqeuclidean(pts[i], pts[j]))
        end
        ε = 1e-12 * (tr(F) / nt + 1)
        @inbounds for i in 1:nt
            F[i, i] += ε
        end
        set_cache!(dad; dibem_F=F)
        return F \ IF
    elseif f_method === :hmatrix
        splitter = PrincipalComponentSplitter(; nmax=nmax)
        Xclt = ClusterTree(pts, splitter)
        Yclt = ClusterTree(copy(pts), splitter)
        adm = StrongAdmissibilityStd(; eta=eta)
        comp = PartialACA(; atol=atol)
        Fh = assemble_hmatrix(DibemFKernel(rbf, pts), Xclt, Yclt;
            adm=adm, comp=comp, threads=threads)
        _hmat_add_diag_ridge!(Fh, 1e-12)
        c, stats = Krylov.gmres(Fh, IF; atol=rtol, rtol=rtol, itmax=max(4nt, 200))
        if !stats.solved
            @warn "DIBEM_FMM: GMRES(F) incomplete" stats.status
        end
        set_cache!(dad; dibem_F_h=Fh)
        return c
    else
        throw(ArgumentError("f_method must be :dense or :hmatrix, got $f_method"))
    end
end

# ---------------------------------------------------------------------------
# Unified entry
# ---------------------------------------------------------------------------

"""
    DIBEM(dad; method=:dense, rbf=PHS(), kwargs...)

| `method` | Backend |
|----------|---------|
| `:dense` | Original dense `Domain.jl` assembly |
| `:hmatrix` | [`DIBEM_Hmat`](@ref) |
| `:fmm` | [`DIBEM_FMM`](@ref) |
"""
function DIBEM(dad::BEMdata{<:Laplace}; method::Symbol=:dense, rbf=PHS(), kwargs...)
    if method === :dense
        return DIBEM_dense(dad; rbf=rbf)
    elseif method === :hmatrix || method === :Hmat || method === :hmat
        return DIBEM_Hmat(dad; rbf=rbf, kwargs...)
    elseif method === :fmm || method === :FMM
        return DIBEM_FMM(dad; rbf=rbf, kwargs...)
    else
        throw(ArgumentError("DIBEM method must be :dense, :hmatrix, or :fmm; got $method"))
    end
end

const dibem! = DIBEM
