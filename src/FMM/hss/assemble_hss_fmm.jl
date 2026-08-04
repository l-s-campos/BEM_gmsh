# =============================================================================
# Martinsson (2011) randomized HSS compression via FMM matvecs
#
# SIAM J. Matrix Anal. Appl. 32(4):1251–1274
#
# Cost: T_total ∼ T_mult · 2(k+p) + T_entry · 2 N k + O(N k²)
# When T_mult = O(N) (FMM), overall O(N k²).
# =============================================================================

"""
    assemble_hss_fmm(A::FMMKernelMatrix, tree; rtol=1e-6, rank=48, oversampling=10)

Build an [`HSSMatrix`](@ref) for the FMM-backed kernel matrix `A` on the binary
`ClusterTree` `tree` using Martinsson's randomized algorithm:

1. Draw Gaussian test matrices `Ωr, Ωc` of width `k+p`
2. Form samples `Sr = A*Ωr`, `Sc = A'*Ωc` via FMM (parallelizable)
3. Bottom-up interpolative decomposition of off-diagonal ranges
4. Extract sibling coupling blocks by entry evaluation

# Arguments
- `A` — [`FMMKernelMatrix`](@ref) (fast matvec + entry access)
- `tree` — binary [`ClusterTree`](@ref) over the *same* point ordering as `A`
  (build with `ClusterTree` on `A.points` columns, `copy_elements=false` after
  permuting points into tree order, or pass `global_index=true` and a tree on
  the original points)

# Keywords
- `rtol` — ID relative tolerance
- `rank` — max HSS rank
- `oversampling` — extra Gaussian samples `p` (default 10; failure prob ≲ 10⁻⁹)
"""
function assemble_hss_fmm(
    A::FMMKernelMatrix{T},
    tree;
    rtol::Real=1e-6,
    rank::Int=48,
    oversampling::Int=10,
    global_index::Bool=true,
) where {T}
    # Ensure tree ordering matches A: if global_index, wrap entry access
    n = A.n
    length(tree) == n || throw(DimensionMismatch("tree length $(length(tree)) ≠ matrix size $n"))

    # Work in tree-local ordering
    l2g = loc2glob(tree)
    if global_index
        # A is in original order; samples need original indices
        # Build samples in original order, then permute to local
        rt = min(n, rank)
        rs = min(n, max(rt + oversampling, oversampling + 1))
        Ωr = randn(T, n, rs)
        Ωc = randn(T, n, rs)
        Sr = sample_matvec(A, Ωr)          # original order
        Sc = sample_matvec_adj(A, Ωc)
        # permute everything to local
        Sr_loc = Sr[l2g, :]
        Sc_loc = Sc[l2g, :]
        Ωr_loc = Ωr[l2g, :]
        Ωc_loc = Ωc[l2g, :]
        # local entry wrapper
        entry_loc = (i, j) -> A.entry(l2g[i], l2g[j])
    else
        rt = min(n, rank)
        rs = min(n, max(rt + oversampling, oversampling + 1))
        Ωr_loc = randn(T, n, rs)
        Ωc_loc = randn(T, n, rs)
        Sr_loc = sample_matvec(A, Ωr_loc)
        Sc_loc = sample_matvec_adj(A, Ωc_loc)
        entry_loc = A.entry
    end

    root = _build_hss_tree(HSSMatrix{typeof(tree),T}, tree)
    _hss_compress_fmm!(root, entry_loc, Sr_loc, Sc_loc, Ωr_loc, Ωc_loc, float(rtol), Int(rank))
    return root
end

"""
    assemble_hss_fmm(points; kernel=:laplace3d, kwargs...)

Convenience: build tree + FMM kernel matrix + HSS in one call.

# Keywords
- `kernel` — `:laplace3d`, `:laplace2d`, or `:yukawa3d`
- `κ` — Yukawa parameter (if needed)
- `eps`, `nmax`, `η` — FMM parameters
- `rtol`, `rank`, `oversampling` — HSS compression
- `splitter` — tree splitter (default `GeometricSplitter(nmax=64)`)
"""
function assemble_hss_fmm(
    points::AbstractMatrix{<:Real};
    kernel::Symbol=:laplace3d,
    κ::Float64=1.0,
    eps::Float64=1e-8,
    nmax::Int=40,
    η::Float64=1.0,
    rtol::Real=1e-6,
    rank::Int=48,
    oversampling::Int=10,
    splitter=GeometricSplitter(nmax=64),
)
    d = size(points, 1)
    # Build tree (permutes a copy of points into local order)
    pts_sv = [SVector{d,Float64}(ntuple(k -> Float64(points[k, i]), d)) for i in 1:size(points, 2)]
    tree = ClusterTree(pts_sv, splitter; copy_elements=true)
    l2g = loc2glob(tree)
    # points in local order
    pts_loc = Matrix{Float64}(undef, d, length(l2g))
    @inbounds for i in eachindex(l2g)
        for k in 1:d
            pts_loc[k, i] = points[k, l2g[i]]
        end
    end

    A = if kernel === :laplace3d
        d == 3 || throw(ArgumentError("laplace3d needs 3×N points"))
        fmm_laplace3d_matrix(pts_loc; eps=eps, nmax=nmax, η=η)
    elseif kernel === :laplace2d
        d == 2 || throw(ArgumentError("laplace2d needs 2×N points"))
        fmm_laplace2d_matrix(pts_loc; eps=eps, nmax=nmax, η=η)
    elseif kernel === :yukawa3d
        d == 3 || throw(ArgumentError("yukawa3d needs 3×N points"))
        fmm_yukawa3d_matrix(pts_loc, κ; eps=eps, nmax=nmax, η=η)
    else
        throw(ArgumentError("unknown kernel $kernel"))
    end

    # tree elements already match pts_loc order (ClusterTree permutes in place)
    return assemble_hss_fmm(A, tree; rtol=rtol, rank=rank, oversampling=oversampling,
                            global_index=false)
end

function _extract_entries(entry, I::Vector{Int}, J::Vector{Int}, ::Type{T}) where {T}
    B = Matrix{T}(undef, length(I), length(J))
    @inbounds for jj in eachindex(J), ii in eachindex(I)
        B[ii, jj] = entry(I[ii], J[jj])
    end
    return B
end

"""
Bottom-up HSS compression from samples (Martinsson Alg. structure).

At leaves, peel diagonal so the ID sees the off-diagonal range only.
"""
function _hss_compress_fmm!(
    H::HSSMatrix{R,T},
    entry,
    Sr::Matrix{T},
    Sc::Matrix{T},
    Ωr::Matrix{T},
    Ωc::Matrix{T},
    rtol::Float64,
    rmax::Int,
) where {R,T}
    I = collect(index_range(H))
    if isleaf(H)
        m = length(I)
        D = _extract_entries(entry, I, I, T)
        H.D = D
        # Off-diagonal samples: S[I,:] - D*Ω[I,:]
        Brow = Sr[I, :] - D * Ωr[I, :]
        Bcol = Sc[I, :] - D' * Ωc[I, :]
        U, Jr_loc = row_id(Brow, rtol, min(rmax, m, size(Brow, 2)))
        V, Jc_loc = row_id(Bcol, rtol, min(rmax, m, size(Bcol, 2)))
        H.U, H.V = U, V
        Jr = isempty(Jr_loc) ? Int[] : I[Jr_loc]
        Jc = isempty(Jc_loc) ? Int[] : I[Jc_loc]
        return Jr, Jc
    else
        c0, c1 = H.children[1], H.children[2]
        Jr0, Jc0 = _hss_compress_fmm!(c0, entry, Sr, Sc, Ωr, Ωc, rtol, rmax)
        Jr1, Jc1 = _hss_compress_fmm!(c1, entry, Sr, Sc, Ωr, Ωc, rtol, rmax)
        H.B01 = _extract_entries(entry, Jr0, Jc1, T)
        H.B10 = _extract_entries(entry, Jr1, Jc0, T)
        Jrows = vcat(Jr0, Jr1)
        Jcols = vcat(Jc0, Jc1)
        # Martinsson: peel diagonal block A(J, I_α) Ω(I_α)
        # so the ID sees only the off-diagonal range A(J, I_α^c) Ω(I_α^c).
        if isempty(Jrows)
            Brow = zeros(T, 0, size(Sr, 2))
        else
            AI = _extract_entries(entry, Jrows, I, T)   # |J| × |I_α|
            Brow = Sr[Jrows, :] - AI * Ωr[I, :]
        end
        if isempty(Jcols)
            Bcol = zeros(T, 0, size(Sc, 2))
        else
            AIt = _extract_entries(entry, I, Jcols, T)  # |I_α| × |J|
            Bcol = Sc[Jcols, :] - AIt' * Ωc[I, :]
        end
        U, Jr_loc = row_id(Brow, rtol, min(rmax, max(length(Jrows), 1), size(Brow, 2)))
        V, Jc_loc = row_id(Bcol, rtol, min(rmax, max(length(Jcols), 1), size(Bcol, 2)))
        H.U, H.V = U, V
        Jr = isempty(Jr_loc) ? Int[] : Jrows[Jr_loc]
        Jc = isempty(Jc_loc) ? Int[] : Jcols[Jc_loc]
        return Jr, Jc
    end
end
