# Laplace fundamental solutions

function fundamental(dad::BEMdata{<:Laplace}, r::Point2D, n::Point2D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end
function fundamental(dad::BEMdata{<:Laplace}, r::Point3D, n::Point3D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end

function fundamental(props::Laplace, r::SVector{2}, n::SVector{2})
    k = props.k
    R = _R(r)
    G = -log(R) / (2π * k)
    H = dot(r, n) / (R^2 * 2π)
    return KernelPair(G, H)
end

function fundamental(props::Laplace, r::SVector{3}, n::SVector{3})
    k = props.k
    R = _R(r)
    G = 1 / (4π * k * R)
    H = dot(r, n) / (4π * R^3)
    return KernelPair(G, H)
end

function fundamental_hyper(props::Laplace, r::SVector{2}, n::SVector{2}, nf::SVector{2})
    R = _R(r)
    invR = 1 / R
    rn = dot(r, n) * invR
    rnf = dot(r, nf) * invR
    G_hyper = rnf / (R^2 * 2π * props.k)
    H_hyper = -(dot(nf, n) - 2 * rn * rnf) / (R^2 * 2π)
    return KernelPair(G_hyper, H_hyper)
end

fundamental_hyper(dad::BEMdata{<:Laplace}, r, n, nf) =
    fundamental_hyper(dad.properties, r, n, nf)
