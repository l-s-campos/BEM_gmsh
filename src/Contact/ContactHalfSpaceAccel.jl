# Hierarchical / FMM backends for Pohrt–Li kernels.
# Same assembly path as Laplace DIBEM (`assemble_hmatrix`)
# and HalfSpaceBEM (`lfmm3d` + exact near panels).

"""
    PohrtKernel(comp, hs, nx, ny)

Flattened ``N×N`` Love / Cerruti matrix, ``N = n_x n_y``, column-major in
``(i_x, i_y)`` so `vec(reshape(p, nx, ny))` matches the grid used by FFT.
Entry `(i,j)` is [`influence_coeff`](@ref)`(comp, Δi, Δj, hs)`.
"""
struct PohrtKernel <: AbstractMatrix{Float64}
    comp::InfluenceComponent
    hs::ElasticHalfSpace
    nx::Int
    ny::Int
end

Base.size(K::PohrtKernel) = (K.nx * K.ny, K.nx * K.ny)

@inline function _pohrt_ij(K::PohrtKernel, i::Int)
    ix = ((i - 1) % K.nx) + 1
    iy = ((i - 1) ÷ K.nx) + 1
    return ix, iy
end

function Base.getindex(K::PohrtKernel, i::Int, j::Int)
    ix, iy = _pohrt_ij(K, i)
    jx, jy = _pohrt_ij(K, j)
    return influence_coeff(K.comp, ix - jx, iy - jy, K.hs)
end

function Base.getindex(K::PohrtKernel, I::AbstractVector{Int}, J::AbstractVector{Int})
    M = Matrix{Float64}(undef, length(I), length(J))
    @inbounds for (jj, j) in enumerate(J), (ii, i) in enumerate(I)
        M[ii, jj] = K[i, j]
    end
    return M
end

"""Collocation centres, column-major in `(i_x, i_y)`."""
function pohrt_grid_points(nx::Int, ny::Int, hs::ElasticHalfSpace)
    pts = Vector{SVector{2,Float64}}(undef, nx * ny)
    @inbounds for iy in 1:ny, ix in 1:nx
        pts[ix + (iy - 1) * nx] = SVector((ix - 0.5) * hs.hx, (iy - 0.5) * hs.hy)
    end
    return pts
end

function _pohrt_tree(nx, ny, hs; nmax=32)
    pts = pohrt_grid_points(nx, ny, hs)
    splitter = HMatrices.hmatrix_splitter(; nmax=nmax)
    return ClusterTree(pts, splitter), pts
end

function _assemble_hmatrix(comp, nx, ny, hs; nmax=32, atol=1e-8, eta=3.0, kwargs...)
    clt, _ = _pohrt_tree(nx, ny, hs; nmax=nmax)
    adm = StrongAdmissibilityStd(; eta=eta)
    aca = PartialACA(; atol=atol)
    return assemble_hmatrix(PohrtKernel(comp, hs, nx, ny), clt, clt;
        adm=adm, comp=aca, threads=false)
end

function _assemble_h2(comp, nx, ny, hs; nmax=32, rtol=1e-8, method=:nnca, kwargs...)
    if method === :cheb || method === :chebyshev
        K, pts = _pohrt_kernelmatrix(comp, nx, ny, hs; round_off = false)
        tree = ClusterTree(pts, HMatrices.hmatrix_splitter(; nmax = nmax))
        return assemble_h2(K, tree; rtol = rtol, method = :cheb, kwargs...)
    end
    tree, _ = _pohrt_tree(nx, ny, hs; nmax=nmax)
    return assemble_h2(PohrtKernel(comp, hs, nx, ny), tree; rtol=rtol, kwargs...)
end

"""Love kernel as [`KernelMatrix`](@ref) so HSS can use the circle proxy."""
function _pohrt_kernelmatrix(comp, nx, ny, hs; round_off::Bool = true)
    pts = pohrt_grid_points(nx, ny, hs)
    hx, hy = hs.hx, hs.hy
    f = if round_off
        function (a, b)
            di = round(Int, (a[1] - b[1]) / hx)
            dj = round(Int, (a[2] - b[2]) / hy)
            return influence_coeff(comp, di, dj, hs)
        end
    else
        function (a, b)
            return influence_coeff(comp, (a[1] - b[1]) / hx, (a[2] - b[2]) / hy, hs)
        end
    end
    return KernelMatrix{typeof(f), typeof(pts), typeof(pts), Float64}(f, pts, pts), pts
end

function _assemble_hss(comp, nx, ny, hs; nmax=32, rtol=1e-8, kwargs...)
    K, pts = _pohrt_kernelmatrix(comp, nx, ny, hs)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=nmax))
    return assemble_hss(K, tree; rtol=rtol, kwargs...)
end

# -----------------------------------------------------------------------------
# FMM (Kzz): Laplace DIBEM far field + HalfSpaceBEM near correction
#
# Laplace 3-D FMM (`fmm_laplace3d_matrix`) is 1/(4π r), skip-self — same
# plan as DIBEM. Scaled by 4A/E* that is the point Boussinesq
# A/(π E* r). Pohrt FFT / H / H² use the Love *rectangle* integral, which
# differs from 1/r on neighbouring cells. Mirror HalfSpaceBEM `lfmm3d`:
# keep FMM in the far field, replace a near stencil with Love
# `influence_coeff` (including the self term).
# -----------------------------------------------------------------------------

struct PohrtFMMOp
    n::Int
    nx::Int
    ny::Int
    hs::ElasticHalfSpace
    α::Float64                          # 4A / E*
    nb::Int                             # near half-width (cells)
    F::FMM.FMMKernelMatrix{Float64}     # 1/(4π r), skip-self
end

Base.size(op::PohrtFMMOp) = (op.n, op.n)
Base.eltype(::PohrtFMMOp) = Float64
Base.:*(op::PohrtFMMOp, x::AbstractVector) = mul!(similar(x, Float64), op, x)

function _assemble_fmm_kzz(nx, ny, hs; eps=1e-8, nmax=-1, η=1.0,
        near_factor=8.0, kwargs...)
    n = nx * ny
    pts = pohrt_grid_points(nx, ny, hs)
    P = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        P[1, i] = pts[i][1]
        P[2, i] = pts[i][2]
        P[3, i] = 0.0
    end
    F = FMM.fmm_laplace3d_matrix(P; eps=float(eps), nmax=Int(nmax), η=float(η),
        full_fmm=true)
    A = hs.hx * hs.hy
    α = 4 * A / contact_modulus(hs)
    nb = max(1, Int(ceil(float(near_factor))))
    return PohrtFMMOp(n, nx, ny, hs, α, nb, F)
end

const _INV4PI = 1 / (4 * π)

function LinearAlgebra.mul!(y::AbstractVector, op::PohrtFMMOp, p::AbstractVector)
    length(y) == op.n == length(p) || throw(DimensionMismatch())
    mul!(y, op.F, p)
    α = op.α
    @inbounds @simd for i in eachindex(y)
        y[i] *= α
    end
    nx, ny, nb, hs = op.nx, op.ny, op.nb, op.hs
    hx, hy = hs.hx, hs.hy
    @inbounds for iy in 1:ny, ix in 1:nx
        i = ix + (iy - 1) * nx
        acc = y[i]
        i0 = max(1, ix - nb)
        i1 = min(nx, ix + nb)
        j0 = max(1, iy - nb)
        j1 = min(ny, iy + nb)
        for jy in j0:j1, jx in i0:i1
            j = jx + (jy - 1) * nx
            pj = p[j]
            if j != i
                r = hypot((ix - jx) * hx, (iy - jy) * hy)
                acc -= α * _INV4PI / r * pj
            end
            acc += influence_coeff(Kzz, ix - jx, iy - jy, hs) * pj
        end
        y[i] = acc
    end
    return y
end

"""
    build_pohrt_operator(hs, nx, ny, comp=Kzz; method=:fft, kwargs...)

Single-component operator. `method` is `:fft`, `:dense`, `:hmatrix`, `:h2`,
`:hss`, or `:fmm` (`Kzz` only; Laplace FMM far field + Love near stencil).
FFT returns the `precompute_kernels` NamedTuple so [`fc_forward`](@ref) can
convolve; the others return a matvec object.
"""
function build_pohrt_operator(
    hs::ElasticHalfSpace, nx::Int, ny::Int, comp::InfluenceComponent=Kzz;
    method::Symbol=:fft, kwargs...,
)
    method === :fft && return precompute_kernels(nx, ny, hs; components=(comp,), method=:fft)
    method === :dense && return Matrix(PohrtKernel(comp, hs, nx, ny))
    method === :hmatrix && return _assemble_hmatrix(comp, nx, ny, hs; kwargs...)
    method === :h2 && return _assemble_h2(comp, nx, ny, hs; kwargs...)
    method === :cheb && return _assemble_h2(comp, nx, ny, hs; method = :cheb, kwargs...)
    method === :hss && return _assemble_hss(comp, nx, ny, hs; kwargs...)
    if method === :fmm
        comp === Kzz || throw(ArgumentError(
            "FMM backend is the Boussinesq 1/r operator (Kzz); use :hmatrix, :h2, or :hss for $comp"))
        return _assemble_fmm_kzz(nx, ny, hs; kwargs...)
    end
    throw(ArgumentError("unknown method $method — use :fft, :dense, :hmatrix, :h2, :cheb, :hss, :fmm"))
end
