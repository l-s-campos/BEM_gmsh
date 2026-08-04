# =============================================================================
# Kernel interface for KIFMM
# =============================================================================

"""Abstract kernel: implement `pot_p2p!` (and optionally `grad_p2p!`)."""
abstract type KIKernel end

"""
    pot_p2p!(pot, sources, charges, targets, kernel; exclude_self=false)

Increment `pot[j]` by Σ_i q_i K(target_j, source_i).
"""
function pot_p2p! end

"""Laplace kernel K = 1/(4π r)."""
struct KILaplace3D <: KIKernel end

function pot_p2p!(
    pot::AbstractVector{<:Real},
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Real},
    targets::Vector{SVector{3,Float64}},
    ::KILaplace3D;
    exclude_self::Bool=false,
    src_range=eachindex(sources),
    trg_range=eachindex(targets),
)
    @inbounds for (jt, j) in enumerate(trg_range)
        tj = targets[j]
        tx, ty, tz = tj[1], tj[2], tj[3]
        s = 0.0
        for i in src_range
            exclude_self && i == j && continue
            si = sources[i]
            rx = tx - si[1]; ry = ty - si[2]; rz = tz - si[3]
            r2 = rx * rx + ry * ry + rz * rz
            r2 < 1e-30 && continue
            s += charges[i] * INV4PI / sqrt(r2)
        end
        pot[jt] += s
    end
    return pot
end

"""Yukawa kernel K = e^{-κr}/(4π r)."""
struct KIYukawa3D <: KIKernel
    κ::Float64
end

function pot_p2p!(
    pot::AbstractVector{<:Real},
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Real},
    targets::Vector{SVector{3,Float64}},
    ker::KIYukawa3D;
    exclude_self::Bool=false,
    src_range=eachindex(sources),
    trg_range=eachindex(targets),
)
    κ = ker.κ
    @inbounds for (jt, j) in enumerate(trg_range)
        tj = targets[j]
        s = 0.0
        for i in src_range
            exclude_self && i == j && continue
            r = norm(tj - sources[i])
            r < 1e-30 && continue
            s += charges[i] * exp(-κ * r) * INV4PI / r
        end
        pot[jt] += s
    end
    return pot
end

"""Helmholtz kernel K = e^{ikr}/(4π r)."""
struct KIHelmholtz3D <: KIKernel
    zk::ComplexF64
end

function pot_p2p!(
    pot::AbstractVector{<:Complex},
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Number},
    targets::Vector{SVector{3,Float64}},
    ker::KIHelmholtz3D;
    exclude_self::Bool=false,
    src_range=eachindex(sources),
    trg_range=eachindex(targets),
)
    zk = ker.zk
    @inbounds for (jt, j) in enumerate(trg_range)
        tj = targets[j]
        s = zero(eltype(pot))
        for i in src_range
            exclude_self && i == j && continue
            r = norm(tj - sources[i])
            r < 1e-30 && continue
            s += complex(charges[i]) * exp(im * zk * r) * INV4PI / r
        end
        pot[jt] += s
    end
    return pot
end

"""
Kernel matrix A[j,i] = K(target_j, source_i).
Size ntrg × nsrc.
"""
function kernel_matrix(
    sources::Vector{SVector{3,Float64}},
    targets::Vector{SVector{3,Float64}},
    ker::KIKernel,
)
    T = ker isa KIHelmholtz3D ? ComplexF64 : Float64
    nsrc, ntrg = length(sources), length(targets)
    A = zeros(T, ntrg, nsrc)
    q = ones(T, 1)
    col = zeros(T, ntrg)
    for i in 1:nsrc
        fill!(col, 0)
        pot_p2p!(col, [sources[i]], q, targets, ker)
        A[:, i] .= col
    end
    return A
end
