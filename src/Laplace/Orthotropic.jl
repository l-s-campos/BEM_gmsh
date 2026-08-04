# Orthotropic 2D Laplace / anisotropic conductivity
# From potencial2d orto (calsolfund with k1, k2)

export OrthotropicLaplace

"""
    OrthotropicLaplace(; k1=1.0, k2=1.0)

2D steady heat conduction with diagonal conductivity
``q = -K ∇T``, ``K = \\mathrm{diag}(k_1, k_2)``.

Fundamental solution (Chang–Tadeu / standard anisotropic):
```math
G = -\\frac{\\log\\sqrt{k_1 k_2}\\,\\sqrt{r_1^2/k_1 + r_2^2/k_2}}{2π\\sqrt{k_1 k_2}},\\quad
H = \\frac{r·n_K}{2π\\sqrt{k_1 k_2}\\,(r_1^2/k_1 + r_2^2/k_2)}
```
with the flux normal ``n_K`` absorbed as ``r·n`` form matching the reference.
"""
@kwdef mutable struct OrthotropicLaplace{T} <: Scalar
    k1::T = 1.0
    k2::T = 1.0
end

function fundamental(props::OrthotropicLaplace, r::SVector{2}, n::SVector{2})
    k1, k2 = props.k1, props.k2
    r1, r2 = r[1], r[2]
    sk = sqrt(k1 * k2)
    den = r1^2 / k1 + r2^2 / k2
    den = max(den, 1e-30)
    # single layer (Tast in reference)
    G = -log(sqrt(den) * sk) / (2π * sk)
    # double layer / flux kernel (Qast)
    H = dot(r, n) / (2π * sk * den)
    return KernelPair(G, H)
end

function fundamental(dad::BEMdata{<:OrthotropicLaplace}, r::Point2D, n::Point2D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end

# isotropic limit check helper
is_isotropic(p::OrthotropicLaplace; tol=1e-12) = abs(p.k1 - p.k2) < tol * (1 + abs(p.k1))
