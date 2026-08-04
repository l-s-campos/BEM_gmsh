# =============================================================================
# Cube-surface discretization (exafmm-t geometry::surface)
# nsurf = 6*(p-1)^2 + 2
# =============================================================================

"""Number of surface points for expansion order `p`."""
nsurf(p::Int) = 6 * (p - 1)^2 + 2

"""
    surface_points(p, half_width, center; α=1.05)

Coordinates of equivalent/check surface points for a cube of half-width
`half_width` centered at `center`, scaled by ratio `α`
(exafmm-t: 1.05 = inner, 2.95 = outer).
"""
function surface_points(
    p::Int,
    half_width::Float64,
    center::SVector{3,Float64};
    α::Float64=1.05,
)
    n = nsurf(p)
    # unit cube surface in [-1,1]^3
    coords = Vector{SVector{3,Float64}}(undef, n)
    coords[1] = SVector(-1.0, -1.0, -1.0)
    count = 2
    # face x = -1
    for i in 0:(p - 2)
        for j in 0:(p - 2)
            coords[count] = SVector(-1.0, (2.0 * (i + 1) - p + 1) / (p - 1), (2.0 * j - p + 1) / (p - 1))
            count += 1
        end
    end
    # face y = -1
    for i in 0:(p - 2)
        for j in 0:(p - 2)
            coords[count] = SVector((2.0 * i - p + 1) / (p - 1), -1.0, (2.0 * (j + 1) - p + 1) / (p - 1))
            count += 1
        end
    end
    # face z = -1
    for i in 0:(p - 2)
        for j in 0:(p - 2)
            coords[count] = SVector((2.0 * (i + 1) - p + 1) / (p - 1), (2.0 * j - p + 1) / (p - 1), -1.0)
            count += 1
        end
    end
    # opposite three faces by central inversion
    half = n ÷ 2
    for i in 1:half
        coords[half + i] = -coords[i]
    end
    b = α * half_width
    return [center + b * coords[i] for i in 1:n]
end
