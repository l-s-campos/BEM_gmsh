# =============================================================================
# Direct 2D Laplace kernels (complex log)
#
#   pot(t) += Σ_j q_j log|t - s_j|
#          + Σ_j dipstr_j * v_j · ∇_s log|t - s_j|
#
# Self interactions with |t-s| ≤ thresh are skipped.
# Gradients are physical (∂/∂x, ∂/∂y).
# =============================================================================

"""Convert complex d/dz gradient to physical (∂x, ∂y)."""
@inline function dz_to_physical(g::Complex)
    return real(g), -imag(g)
end

"""
    direct_laplace!(pot, grad, sources, charges, dipstr, dipvec, targets;
                    thresh=0.0, skip_self=false)

O(N²) evaluation of the complex log-Laplace potential.
`pot` / `grad` are over targets. `grad` may be `nothing`.
`dipstr`/`dipvec` may be `nothing`.
"""
function direct_laplace!(
    pot::AbstractVector{ComplexF64},
    grad::Union{Nothing,AbstractMatrix{ComplexF64}},
    sources::AbstractMatrix{<:Real},
    charges::Union{Nothing,AbstractVector{<:Number}},
    dipstr::Union{Nothing,AbstractVector{<:Number}},
    dipvec::Union{Nothing,AbstractMatrix{<:Real}},
    targets::AbstractMatrix{<:Real};
    thresh::Float64=0.0,
    skip_self::Bool=false,
)
    ns = size(sources, 2)
    nt = size(targets, 2)
    thresh2 = thresh * thresh
    has_charge = charges !== nothing
    has_dipole = dipstr !== nothing && dipvec !== nothing
    want_grad = grad !== nothing

    @inbounds for j in 1:nt
        tx = targets[1, j]
        ty = targets[2, j]
        pj = zero(ComplexF64)
        gx = zero(ComplexF64)
        gy = zero(ComplexF64)
        for i in 1:ns
            dx = tx - sources[1, i]
            dy = ty - sources[2, i]
            r2 = dx * dx + dy * dy
            if r2 <= thresh2 || (skip_self && i == j && r2 == 0)
                continue
            end
            if r2 == 0
                continue
            end
            if has_charge
                q = complex(charges[i])
                pj += q * (log(r2) / 2)
                if want_grad
                    invr2 = 1 / r2
                    gx += q * dx * invr2
                    gy += q * dy * invr2
                end
            end
            if has_dipole
                # pot += dipstr * (v · ∇_src log|t-s|) = dipstr * (-v·(t-s)/r²)
                d = complex(dipstr[i])
                vx = dipvec[1, i]
                vy = dipvec[2, i]
                invr2 = 1 / r2
                pj += d * (-(vx * dx + vy * dy) * invr2)
                if want_grad
                    # ∇_t [ -(v·r)/r² ] = -v/r² + 2(v·r)r/r⁴
                    vdot = vx * dx + vy * dy
                    gx += d * (-vx * invr2 + 2 * vdot * dx * invr2 * invr2)
                    gy += d * (-vy * invr2 + 2 * vdot * dy * invr2 * invr2)
                end
            end
        end
        pot[j] += pj
        if want_grad
            grad[1, j] += gx
            grad[2, j] += gy
        end
    end
    return nothing
end

"""Direct evaluation on tree-local storage for a **real** density channel.

`pot` is `AbstractVector{<:Real}` (physical log potential). `grad` stores complex
`d/dz` of the analytic potential (convert via [`dz_to_physical`](@ref)).
"""
function direct_laplace_sv!(
    pot::AbstractVector{<:Real},
    grad::Union{Nothing,AbstractVector{ComplexF64}},
    sources::AbstractVector{SVector{2,Float64}},
    charges::Union{Nothing,AbstractVector{<:Real}},
    dipstr::Union{Nothing,AbstractVector{<:Real}},
    dipvec::Union{Nothing,AbstractVector{SVector{2,Float64}}},
    src_range::AbstractUnitRange,
    targets::AbstractVector{SVector{2,Float64}},
    targ_range::AbstractUnitRange;
    thresh::Float64=0.0,
    exclude_self::Bool=false,
)
    thresh2 = thresh * thresh
    has_charge = charges !== nothing
    has_dipole = dipstr !== nothing && dipvec !== nothing
    want_grad = grad !== nothing

    for (jt, j) in enumerate(targ_range)
        t = targets[j]
        pj = 0.0
        gj = zero(ComplexF64)
        for i in src_range
            if exclude_self && i == j
                continue
            end
            dxy = t - sources[i]
            dx, dy = dxy[1], dxy[2]
            r2 = dx * dx + dy * dy
            if r2 <= thresh2
                continue
            end
            z = complex(dx, dy)
            if has_charge
                q = Float64(charges[i])
                pj += q * (log(r2) / 2)
                if want_grad
                    gj += q / z
                end
            end
            if has_dipole
                d = Float64(dipstr[i])
                vx, vy = dipvec[i][1], dipvec[i][2]
                invr2 = 1 / r2
                pj += d * (-(vx * dx + vy * dy) * invr2)
                if want_grad
                    dconv = d * (-complex(vx, vy))
                    gj -= dconv / (z * z)
                end
            end
        end
        pot[jt] += pj
        if want_grad
            grad[jt] += gj
        end
    end
    return nothing
end
