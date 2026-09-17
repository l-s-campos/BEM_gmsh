# Helmholtz fundamental solutions
# r = x_field − x_source. Double layer H is ∂G/∂n_x (acoustic q = ∂u/∂n).
# As κ → 0 with k = 1: G → Laplace G and H → −Laplace H.
using SpecialFunctions: hankelh1

@inline function fundamental_U(props::Helmholtz, r::SVector{2})
    z = wavenumber(props) * _R(r)
    return im / 4 * hankelh1(0, z)
end
@inline function fundamental_T(props::Helmholtz, r::SVector{2}, n::SVector{2})
    R = _R(r)
    κ = wavenumber(props)
    return -κ * im / 4 * hankelh1(1, κ * R) * dot(r, n) / R
end
function fundamental(props::Helmholtz, r::SVector{2}, n::SVector{2})
    R = _R(r)
    κ = wavenumber(props)
    z = κ * R
    G = im / 4 * hankelh1(0, z)
    H = -κ * im / 4 * hankelh1(1, z) * dot(r, n) / R
    return KernelPair(G, H)
end

@inline function fundamental_U(props::Helmholtz, r::SVector{3})
    R = _R(r)
    return cis(wavenumber(props) * R) / (4π * R)
end
@inline function fundamental_T(props::Helmholtz, r::SVector{3}, n::SVector{3})
    R = _R(r)
    κ = wavenumber(props)
    e = cis(κ * R)
    inv4πR = 1 / (4π * R)
    return e * (im * κ * R - 1) * dot(r, n) * inv4πR / (R * R)
end
function fundamental(props::Helmholtz, r::SVector{3}, n::SVector{3})
    R = _R(r)
    κ = wavenumber(props)
    e = cis(κ * R)
    inv4πR = 1 / (4π * R)
    G = e * inv4πR
    H = e * (im * κ * R - 1) * dot(r, n) * inv4πR / (R * R)
    return KernelPair(G, H)
end

function fundamental(dad::BEMdata{<:Helmholtz}, r, n)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end

function fundamental_hyper(props::Helmholtz, r::SVector{2}, n::SVector{2}, nf::SVector{2})
    R = _R(r)
    κ = wavenumber(props)
    z = κ * R
    invR = 1 / R
    rn = dot(r, n) * invR
    rnf = dot(r, nf) * invR
    H1 = hankelh1(1, z)
    H2 = hankelh1(2, z)
    G_h = κ * im / 4 * H1 * rnf
    H_h = κ * im / 4 * invR * H1 * dot(nf, n) - κ^2 * im / 4 * H2 * rn * rnf
    return KernelPair(G_h, H_h)
end

function fundamental_hyper(props::Helmholtz, r::SVector{3}, n::SVector{3}, nf::SVector{3})
    R = _R(r)
    κ = wavenumber(props)
    e = cis(κ * R)
    invR = 1 / R
    inv4πR3 = e / (4π * R * R * R)
    z = κ * R
    rn = dot(r, n) * invR
    rnf = dot(r, nf) * invR
    nn = dot(nf, n)
    G_h = e * (1 - im * z) * rnf / (4π * R * R)
    H_h = inv4πR3 * ((1 - im * z) * nn + (z * z - 3 + 3 * im * z) * rn * rnf)
    return KernelPair(G_h, H_h)
end

function fundamental_hyper(dad::BEMdata{<:Helmholtz}, r, n, nf)
    kp = fundamental_hyper(dad.properties, r, n, nf)
    return kp.U, kp.T
end
