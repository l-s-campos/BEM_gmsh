# Laplace fundamental solutions

function fundamental(dad::BEMdata{<:Laplace}, r::Point2D, n::Point2D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end
function fundamental(dad::BEMdata{<:Laplace}, r::Point3D, n::Point3D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end

# 2-D: U = -log(R)/(2πk) = -log(R²)/(4πk); T needs R² only (no sqrt).
@inline function fundamental_U(props::Laplace, r::SVector{2})
    return -log(_R2(r)) / (4π * props.k)
end
@inline function fundamental_T(props::Laplace, r::SVector{2}, n::SVector{2})
    return dot(r, n) / (_R2(r) * 2π)
end
function fundamental(props::Laplace, r::SVector{2}, n::SVector{2})
    R2 = _R2(r)
    inv2π = 1 / (2π)
    G = -log(R2) * (0.5 * inv2π / props.k)
    H = dot(r, n) * (inv2π / R2)
    return KernelPair(G, H)
end

@inline function fundamental_U(props::Laplace, r::SVector{3})
    return 1 / (4π * props.k * sqrt(_R2(r)))
end
@inline function fundamental_T(props::Laplace, r::SVector{3}, n::SVector{3})
    R2 = _R2(r)
    return dot(r, n) / (4π * R2 * sqrt(R2))
end
function fundamental(props::Laplace, r::SVector{3}, n::SVector{3})
    R2 = _R2(r)
    R = sqrt(R2)
    inv4π = 1 / (4π)
    G = inv4π / (props.k * R)
    H = dot(r, n) * inv4π / (R2 * R)
    return KernelPair(G, H)
end

function fundamental_hyper(props::Laplace, r::SVector{2}, n::SVector{2}, nf::SVector{2})
    R = _R(r)
    invR = 1 / R
    rn = dot(r, n) * invR          # e · n
    rnf = dot(r, nf) * invR        # e · nξ
    # ∂U/∂nξ, U = -log(R)/(2π k);  ∂T/∂nξ with T = (r·n)/(2π R²)
    G_hyper = rnf / (R * 2π * props.k)
    H_hyper = -(dot(nf, n) - 2 * rn * rnf) / (R^2 * 2π)
    return KernelPair(G_hyper, H_hyper)
end

function fundamental_hyper(props::Laplace, r::SVector{3}, n::SVector{3}, nf::SVector{3})
    R = _R(r)
    invR = 1 / R
    rn = dot(r, n) * invR
    rnf = dot(r, nf) * invR
    G_hyper = rnf / (R^2 * 4π * props.k)
    H_hyper = -(dot(nf, n) - 3 * rn * rnf) / (R^3 * 4π)
    return KernelPair(G_hyper, H_hyper)
end

function fundamental_hyper(dad::BEMdata{<:Laplace}, r, n, nf)
    kp = fundamental_hyper(dad.properties, r, n, nf)
    return kp.U, kp.T
end

"""
    fundamental_grad(props::Laplace, r, n) -> (∇_d U, ∇_d T)

Derivatives of the Laplace kernels with respect to the **source** `d`,
where `r = x - d` and `n` is the outward normal at the field point `x`.

2-D:

```
U = -log(R)/(2πk)     ∇_d U = r / (2π k R²)
T = (r·n)/(2π R²)     ∇_d T = [-n R² + 2 (r·n) r] / (2π R⁴)
```

3-D:

```
U = 1/(4π k R)        ∇_d U = r / (4π k R³)
T = (r·n)/(4π R³)     ∇_d T = [-n R² + 3 (r·n) r] / (4π R⁵)
```
"""
function fundamental_grad(props::Laplace, r::SVector{2}, n::SVector{2})
    R2 = dot(r, r)
    R2 < 1e-30 && return zero(r), zero(r)
    inv2π = 1 / (2π)
    dU = (inv2π / props.k) * (r / R2)
    rn = dot(r, n)
    dT = inv2π * (-n * R2 + 2 * rn * r) / (R2 * R2)
    return dU, dT
end

"""
3-D Laplace kernels differentiated w.r.t. the source `d` (`r = x − d`).

```
U = 1/(4π k R)           ∇_d U = r / (4π k R³)
T = (r·n)/(4π R³)        ∇_d T = [−n R² + 3 (r·n) r] / (4π R⁵)
```
"""
function fundamental_grad(props::Laplace, r::SVector{3}, n::SVector{3})
    R2 = dot(r, r)
    R2 < 1e-30 && return zero(r), zero(r)
    R = sqrt(R2)
    inv4π = 1 / (4π)
    dU = (inv4π / props.k) * (r / (R * R2))
    rn = dot(r, n)
    dT = inv4π * (-n * R2 + 3 * rn * r) / (R2 * R2 * R)
    return dU, dT
end

function fundamental_grad(dad::BEMdata{<:Laplace}, r, n)
    return fundamental_grad(dad.properties, r, n)
end
