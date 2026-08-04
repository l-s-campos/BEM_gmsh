# Helmholtz fundamental solutions
using SpecialFunctions: hankelh1

function fundamental(props::Helmholtz, r::SVector{2}, n::SVector{2})
    R = _R(r)
    κ = wavenumber(props)
    z = κ * R
    G = im / 4 * hankelh1(0, z)
    H = -κ * im / 4 * hankelh1(1, z) * dot(r, n) / R
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

fundamental_hyper(dad::BEMdata{<:Helmholtz}, r, n, nf) =
    fundamental_hyper(dad.properties, r, n, nf)
