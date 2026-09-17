"""
    LayeredAniso

Fourier-domain surface compliance of a (possibly layered) anisotropic
half-space, for the Pohrt FFT contact stack.

Homogeneous isotropic kernels stay in [`ContactHalfSpace`](@ref). This module
builds the same nine `K_ab` via Stroh + a layer propagator (Bagault, Nélias,
Baietto, Ovaert, *Int. J. Solids Struct.* **50** (2013) 743–754, implemented
in `q`-space rather than the real-space *N*-series).

Isotropic materials are fed to Stroh as cubic with Coulomb modulus ±1 %
(Bagault §4.1) to split the repeated eigenvalues.
"""
module LayeredAniso

using LinearAlgebra
using FFTW
using ..ContactHalfSpace
import ..ContactHalfSpace: precompute_kernels, default_penalties, solve_normal_contact

export isotropic_C, cubic_C, cubic_almost_isotropic, orthotropic_C, rotate_C, rotate_C_about_x
export Layer, LayeredHalfSpace, combined_layered, homogeneous
export isotropic_halfspace, isotropic_coated, with_grid, bagault_orthotropic
export surface_compliance, layered_compliance, kernel_self
export solve_sphere_load, hertz_rigid_sphere

const _VOIGT = Int[1 6 5; 6 2 4; 5 4 3]
@inline _vij(i, j) = _VOIGT[i, j]
@inline _Cten(C, i, j, k, l) = C[_vij(i, j), _vij(k, l)]

# ---------------------------------------------------------------------------
# Voigt stiffness
# ---------------------------------------------------------------------------

"""Isotropic `C` from Young's modulus and Poisson ratio."""
function isotropic_C(E::Real, ν::Real)
    G = E / (2(1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    C = zeros(6, 6)
    C[1,1] = C[2,2] = C[3,3] = λ + 2G
    C[1,2] = C[1,3] = C[2,1] = C[2,3] = C[3,1] = C[3,2] = λ
    C[4,4] = C[5,5] = C[6,6] = G
    return C
end

"""Cubic `C` from `E`, `ν`, and Coulomb modulus `G` (independent of `E/2(1+ν)`)."""
function cubic_C(E::Real, ν::Real, G::Real)
    C11 = E * (1 - ν) / ((1 + ν) * (1 - 2ν))
    C12 = E * ν / ((1 + ν) * (1 - 2ν))
    C = zeros(6, 6)
    C[1,1] = C[2,2] = C[3,3] = C11
    C[1,2] = C[1,3] = C[2,1] = C[2,3] = C[3,1] = C[3,2] = C12
    C[4,4] = C[5,5] = C[6,6] = G
    return C
end

"""Bagault almost-isotropic cubic: Coulomb modulus `G_iso*(1+δG)`."""
function cubic_almost_isotropic(E::Real, ν::Real; δG=0.01)
    G = E / (2(1 + ν))
    return cubic_C(E, ν, G * (1 + δG))
end

"""Orthotropic `C` in its principal axes (engineering constants).

Poisson ratios are clipped so `|ν_ij| < √(Ei/Ej)` (thermodynamic stability).
Extreme `E1/E2` with a shared `ν` (Bagault Fig. 5) otherwise makes `S` indefinite.
"""
function orthotropic_C(E1, E2, E3, ν12, ν13, ν23, G12, G13, G23)
    ν12 = _clip_nu(ν12, E1, E2)
    ν13 = _clip_nu(ν13, E1, E3)
    ν23 = _clip_nu(ν23, E2, E3)
    ν21 = ν12 * E2 / E1
    ν31 = ν13 * E3 / E1
    ν32 = ν23 * E3 / E2
    S = zeros(6, 6)
    S[1,1] = 1 / E1
    S[2,2] = 1 / E2
    S[3,3] = 1 / E3
    S[1,2] = S[2,1] = -ν12 / E1
    S[1,3] = S[3,1] = -ν13 / E1
    S[2,3] = S[3,2] = -ν23 / E2
    S[4,4] = 1 / G23
    S[5,5] = 1 / G13
    S[6,6] = 1 / G12
    return inv(S)
end

@inline function _clip_nu(ν, Ei, Ej)
    lim = 0.49 * sqrt(Ei / Ej)
    return sign(ν) * min(abs(ν), lim)
end

"""Bond-transform `C` by a 3×3 rotation (`C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs`)."""
function rotate_C(C::AbstractMatrix, R::AbstractMatrix)
    C2 = zeros(eltype(C), 6, 6)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        acc = zero(eltype(C))
        for p in 1:3, q in 1:3, r in 1:3, s in 1:3
            acc += R[i, p] * R[j, q] * R[k, r] * R[l, s] * _Cten(C, p, q, r, s)
        end
        C2[_vij(i, j), _vij(k, l)] = acc
    end
    return 0.5 .* (C2 .+ C2')
end

"""Rotation about axis 1 (rolling) by `θ` (paper Fig. 4, `θ_m`)."""
function rotate_C_about_x(C, θ)
    c, s = cos(θ), sin(θ)
    R = [1.0 0.0 0.0; 0.0 c -s; 0.0 s c]
    return rotate_C(C, R)
end

# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

"""One coating or the infinite substrate (`thickness = Inf`)."""
struct Layer
    C::Matrix{Float64}
    thickness::Float64
end
Layer(C::AbstractMatrix; thickness=Inf) = Layer(Matrix{Float64}(C), Float64(thickness))

"""Coated half-space. `layers[end]` is the infinite substrate."""
struct LayeredHalfSpace
    layers::Vector{Layer}
    hx::Float64
    hy::Float64
end

function LayeredHalfSpace(layers::Vector{Layer}; hx=1.0, hy=hx)
    isempty(layers) && error("need at least a substrate")
    isfinite(layers[end].thickness) &&
        error("last layer must be the infinite substrate (thickness=Inf)")
    return LayeredHalfSpace(layers, Float64(hx), Float64(hy))
end

homogeneous(C::AbstractMatrix; hx=1.0, hy=hx) =
    LayeredHalfSpace([Layer(C; thickness=Inf)]; hx=hx, hy=hy)

"""Orthotropic `C` with in-plane `E` and depth `E3`, same ν and Coulomb modulus as isotropic `E,ν` (Bagault §4.2)."""
function bagault_orthotropic(E, E3, ν)
    G = E / (2(1 + ν))
    return orthotropic_C(E, E, E3, ν, ν, ν, G, G, G)
end

with_grid(hs::LayeredHalfSpace, hx, hy=hx) = LayeredHalfSpace(hs.layers, Float64(hx), Float64(hy))

function isotropic_halfspace(E, ν; hx=1.0, hy=hx, δG=0.01)
    return homogeneous(cubic_almost_isotropic(E, ν; δG=δG); hx=hx, hy=hy)
end

function isotropic_coated(E_c, ν_c, Z_c, E_s, ν_s; hx=1.0, hy=hx, δG=0.01)
    Z_c <= 0 && return isotropic_halfspace(E_s, ν_s; hx=hx, hy=hy, δG=δG)
    return LayeredHalfSpace([
        Layer(cubic_almost_isotropic(E_c, ν_c; δG=δG); thickness=Z_c),
        Layer(cubic_almost_isotropic(E_s, ν_s; δG=δG); thickness=Inf),
    ]; hx=hx, hy=hy)
end

# ---------------------------------------------------------------------------
# Stroh
# ---------------------------------------------------------------------------

"""Stroh `Q, R, T` for in-plane direction `(n1, n2)` (Bagault eqs. 1)."""
function stroh_QRT(C, n1, n2)
    n = (n1, n2, 0.0)
    Q = zeros(ComplexF64, 3, 3)
    R = zeros(ComplexF64, 3, 3)
    T = zeros(ComplexF64, 3, 3)
    @inbounds for i in 1:3, k in 1:3
        q = 0.0; r = 0.0; t = 0.0
        for α in 1:2, β in 1:2
            q += _Cten(C, i, α, k, β) * n[α] * n[β]
        end
        for α in 1:2
            r += _Cten(C, i, α, k, 3) * n[α]
        end
        t += _Cten(C, i, 3, k, 3)
        Q[i, k] = q; R[i, k] = r; T[i, k] = t
    end
    return Q, R, T
end

const _STROH = Dict{Tuple{UInt,Float64,Float64},Any}()
function _stroh_key(C, n1, n2)
    return (objectid(C), round(n1; digits=6), round(n2; digits=6))
end
clear_stroh_cache!() = empty!(_STROH)

"""Decaying (`Im p > 0`) and growing Stroh triples `A, B, p`."""
function stroh_modes(C, n1, n2)
    get!(_STROH, _stroh_key(C, n1, n2)) do
        _stroh_modes(C, n1, n2)
    end
end
function _stroh_modes(C, n1, n2)
    Q, R, Tm = stroh_QRT(C, n1, n2)
    Tinv = inv(Tm)
    N1 = -Tinv * R'
    N2 = Tinv
    N3 = R * Tinv * R' - Q
    N = Matrix{ComplexF64}(undef, 6, 6)
    N[1:3, 1:3] = N1
    N[1:3, 4:6] = N2
    N[4:6, 1:3] = N3
    N[4:6, 4:6] = N1'
    vals, vecs = eigen(N)
    # decaying: Im(p) > 0 for field ~ exp(i η p z), z into the solid
    perm = sortperm(vals; by=p -> (-imag(p), real(p)))
    vals = vals[perm]
    vecs = vecs[:, perm]
    plus = findall(p -> imag(p) > 0, vals)
    if length(plus) != 3
        # fallback: take the three largest Im(p)
        plus = collect(1:3)
    end
    minus = setdiff(1:6, plus)
    length(minus) != 3 && (minus = collect(4:6))
    function pack(idx)
        A = vecs[1:3, idx]
        B = vecs[4:6, idx]
        p = vals[idx]
        # column-normalize a
        for j in 1:3
            nrm = norm(A[:, j])
            nrm < 1e-30 && continue
            A[:, j] ./= nrm
            B[:, j] ./= nrm
        end
        return A, B, p
    end
    Ap, Bp, pp = pack(plus)
    Am, Bm, pm = pack(minus)
    return Ap, Bp, pp, Am, Bm, pm
end

"""Homogeneous half-space surface compliance `û = Û t̂` (3×3)."""
function surface_compliance(C::AbstractMatrix, qx::Real, qy::Real)
    η = hypot(qx, qy)
    η < 1e-14 && return _compliance_dc(C)
    n1, n2 = qx / η, qy / η
    A, B, _, _, _, _ = stroh_modes(C, n1, n2)
    # Stroh decaying: û = A (i η B)^{-1} t̂ with t = σ·n_outward.
    # Contact traction is into the solid (opposite outward), so Û_contact = −Û_stroh.
    Û = -(A / (im * η * B))
    return 0.5 .* (Û .+ Û')
end

function _compliance_dc(C)
    # finite stand-in used only for the q=0 FFT bin (rigid translation)
    E = _young_from_C(C)
    ν = 0.3
    G = E / (2(1 + ν))
    return Matrix{ComplexF64}(I, 3, 3) .* (2(1 - ν) / G)
end

function _young_from_C(C)
    S = inv(real.(C))
    e = 1 / max(real(S[1, 1]), 1e-30)
    return e
end

# ---------------------------------------------------------------------------
# Layer propagator (one or more finite coatings + substrate)
# ---------------------------------------------------------------------------

"""Surface compliance of a layered stack at wavevector `(qx, qy)`."""
function layered_compliance(layers::Vector{Layer}, qx::Real, qy::Real)
    length(layers) == 1 && return surface_compliance(layers[1].C, qx, qy)
    η = hypot(qx, qy)
    η < 1e-14 && return surface_compliance(layers[end].C, qx, qy)
    n1, n2 = qx / η, qy / η
    return _layered_compliance_η(layers, η, n1, n2)
end

function _layered_compliance_η(layers, η, n1, n2)
    nlay = length(layers)
    # substrate: decaying only
    As, Bs, _, _, _, _ = stroh_modes(layers[end].C, n1, n2)
    # walk coatings from the substrate interface up to the surface
    # For a single coating (the Bagault case) use the stably scaled 9×9 solve.
    nlay == 2 && return _one_coating_compliance(layers[1], As, Bs, η, n1, n2)
    # n coatings: successive interface matching (stable scaling per layer)
    return _n_coating_compliance(layers, η, n1, n2)
end

function _one_coating_compliance(coat::Layer, As, Bs, η, n1, n2)
    h = coat.thickness
    Ap, Bp, pp, Am, Bm, pm = stroh_modes(coat.C, n1, n2)
    eph = [exp(im * η * pp[j] * h) for j in 1:3]          # decaying, |e|≤1
    emh = [exp(im * η * pm[j] * (-h)) for j in 1:3]        # growing measured from interface
    # unknowns: c+(3), c-(3), cs(3)
    # interface z=h: Ap <eph> c+ + Am c- = As cs
    #                Bp <eph> c+ + Bm c- = Bs cs
    # surface  z=0:  Bp c+ + Bm <emh> c- = t0 / (iη)
    M = zeros(ComplexF64, 9, 9)
    # rows 1:3 interface u
    M[1:3, 1:3] = Ap .* reshape(eph, 1, 3)
    M[1:3, 4:6] = Am
    M[1:3, 7:9] = -As
    # rows 4:6 interface t/(iη)
    M[4:6, 1:3] = Bp .* reshape(eph, 1, 3)
    M[4:6, 4:6] = Bm
    M[4:6, 7:9] = -Bs
    # rows 7:9 surface traction
    M[7:9, 1:3] = Bp
    M[7:9, 4:6] = Bm .* reshape(emh, 1, 3)
    # rhs columns = unit tractions / (iη) on rows 7:9
    Û = zeros(ComplexF64, 3, 3)
    rhs = zeros(ComplexF64, 9)
    F = lu(M)
    for k in 1:3
        fill!(rhs, 0)
        rhs[6 + k] = -1 / (im * η)   # contact traction vs outward Stroh t
        c = F \ rhs
        cp = c[1:3]
        cm = c[4:6]
        # u(0) = Ap c+ + Am <emh> c-
        Û[:, k] = Ap * cp + Am * (emh .* cm)
    end
    return 0.5 .* (Û .+ Û')
end

function _n_coating_compliance(layers, η, n1, n2)
    # recursive: treat everything below the top coating as an equivalent substrate
    # by computing 3 decaying "modes" from the lower stack's Û... simpler: chain
    # transfer of the 3+3 coefficients. For n>2, fold lower layers into an
    # effective As, Bs such that u = As c, t/(iη) = Bs c at the top of the remainder.
    lower = layers[2:end]
    Ûlo = _layered_compliance_η(lower, η, n1, n2)
    # t = iη Ûlo^{-1} u  at the interface, so B_eff = Ûlo^{-1}, A_eff = I
    # (c = u_interface). Then the "substrate" matrices are A=I, B=inv(Ûlo).
    As = Matrix{ComplexF64}(I, 3, 3)
    Bs = inv(Ûlo)
    return _one_coating_compliance(layers[1], As, Bs, η, n1, n2)
end

# ---------------------------------------------------------------------------
# FFT kernels
# ---------------------------------------------------------------------------

@inline _sinc(x) = abs(x) < 1e-14 ? 1.0 : sin(x) / x

"""`Û(q)` of a layered (or homogeneous) half-space."""
layered_compliance(hs::LayeredHalfSpace, qx, qy) = layered_compliance(hs.layers, qx, qy)

"""
    precompute_kernels(nx, ny, hs::LayeredHalfSpace; components=...)

Build Love-layout spatial kernels by IFFT of Stroh `Û(q)` on a grid
`nq ≥ 4 max(2nx,2ny)` (reduces periodization), crop to offsets
`-(n-1):(n-1)`, embed like Pohrt, then `rfft`.

Two-body problems: pass `bodies=(A,B)` or a tuple to add `Û`.
"""
function precompute_kernels(
    nx::Int, ny::Int, hs::LayeredHalfSpace;
    components=instances(InfluenceComponent),
    bodies::Union{Nothing,NTuple{2,LayeredHalfSpace}}=nothing,
    nq::Int=0,
)
    Mx, My = 2nx, 2ny
    hx, hy = hs.hx, hs.hy
    nq = nq <= 0 ? max(256, 4 * max(Mx, My)) : nq
    nq = Int(2^ceil(log2(nq)))
    stacks = bodies === nothing ? (hs,) : bodies
    clear_stroh_cache!()
    cmap = Dict(Kxx=>(1,1), Kxy=>(1,2), Kxz=>(1,3),
                Kyx=>(2,1), Kyy=>(2,2), Kyz=>(2,3),
                Kzx=>(3,1), Kzy=>(3,2), Kzz=>(3,3))
    qx = 2π .* FFTW.fftfreq(nq, 1 / hx)
    qy = 2π .* FFTW.fftfreq(nq, 1 / hy)
    spec = Dict(c => zeros(ComplexF64, nq, nq) for c in components)
    Ûtmp = zeros(ComplexF64, 3, 3)
    @inbounds for j in 1:nq, i in 1:nq
        qxi, qyj = qx[i], qy[j]
        hypot(qxi, qyj) < 1e-18 && continue
        fill!(Ûtmp, 0)
        for st in stacks
            Ûtmp .+= layered_compliance(st, qxi, qyj)
        end
        w = _sinc(qxi * hx / 2) * _sinc(qyj * hy / 2)
        for c in components
            a, b = cmap[c]
            spec[c][i, j] = Ûtmp[a, b] * w
        end
    end
    kernels = Dict{InfluenceComponent,Matrix{ComplexF64}}()
    scratch = zeros(Float64, Mx, My)
    for c in components
        G = real.(ifft(spec[c]))           # wrap origin at (1,1), period nq*hx
        Kwin = zeros(Float64, 2nx - 1, 2ny - 1)
        @inbounds for dj in -(ny - 1):(ny - 1), di in -(nx - 1):(nx - 1)
            ii = di >= 0 ? di + 1 : nq + di + 1
            jj = dj >= 0 ? dj + 1 : nq + dj + 1
            Kwin[di + nx, dj + ny] = G[ii, jj]
        end
        fill!(scratch, 0)
        ContactHalfSpace._embed_kernel!(scratch, Kwin, nx, ny)
        kernels[c] = rfft(scratch)
    end
    return (; nx, ny, Mx, My, kernels, hs=hs)
end

"""Two-body combination: `Û = Û_A + Û_B` (identical grids)."""
function combined_layered(A::LayeredHalfSpace, B::LayeredHalfSpace)
    (A.hx == B.hx && A.hy == B.hy) || error("hx, hy must match")
    # encoded as a pair via a dummy stack; precompute_kernels(::Layered, bodies=)
    return (A, B)
end

function precompute_kernels(
    nx::Int, ny::Int, bodies::Tuple{LayeredHalfSpace,LayeredHalfSpace};
    components=instances(InfluenceComponent),
)
    return precompute_kernels(nx, ny, bodies[1]; components=components, bodies=bodies)
end

"""Self-influence `K_ab(0,0)` from a `prep` (Uzawa penalties)."""
function kernel_self(prep, comp::InfluenceComponent)
    haskey(prep.kernels, comp) || return 0.0
    Kspat = irfft(prep.kernels[comp], prep.Mx)
    return real(Kspat[1, 1])
end

function default_penalties(prep::NamedTuple; frac=0.25)
    kzz = kernel_self(prep, Kzz)
    kxx = haskey(prep.kernels, Kxx) ? kernel_self(prep, Kxx) : kzz
    rn = frac / max(kzz, eps())
    rt = frac / max(kxx, eps())
    return rn, rt
end

function solve_normal_contact(
    gap0::AbstractMatrix{<:Real},
    δ::Real,
    hs::LayeredHalfSpace;
    tol=1e-8,
    maxiter=200,
    prep=nothing,
)
    nx, ny = size(gap0)
    prep = isnothing(prep) ? precompute_kernels(nx, ny, hs; components=(Kzz,)) : prep
    dummy = ElasticHalfSpace(1.0, 0.3; hx=hs.hx, hy=hs.hy)
    return solve_normal_contact(gap0, δ, dummy; tol=tol, maxiter=maxiter, prep=prep)
end

"""Hertz rigid sphere on an isotropic half-space: `a, p0, δ, P, E*`."""
function hertz_rigid_sphere(R, E, ν, a)
    Estar = E / (1 - ν^2)
    δ = a^2 / R
    p0 = 2Estar * a / (π * R)
    P = 2π * p0 * a^2 / 3
    return (; a, p0, δ, P, Estar)
end

"""
Frictionless sphere at prescribed load `P` (Bagault / O'Sullivan: Hertz of the
reference isotropic solid). Reuses `prep`. `P ∝ δ^{3/2}` Newton on the approach.
"""
function solve_sphere_load(gap0, hs::LayeredHalfSpace, P_target, δ0;
                           prep=nothing, rtol=0.02, maxit=8, tol=1e-6)
    nx, ny = size(gap0)
    prep = isnothing(prep) ? precompute_kernels(nx, ny, hs; components=(Kzz,)) : prep
    dummy = ElasticHalfSpace(1.0, 0.3; hx=hs.hx, hy=hs.hy)
    δ = float(δ0)
    sol = solve_normal_contact(gap0, δ, dummy; prep=prep, tol=tol)
    dA = hs.hx * hs.hy
    F = sum(sol.p) * dA
    for _ in 1:maxit
        abs(F - P_target) <= rtol * max(P_target, 1.0) && break
        F <= 0 && (δ *= 1.5; sol = solve_normal_contact(gap0, δ, dummy; prep=prep, tol=tol);
                   F = sum(sol.p) * dA; continue)
        δ *= (P_target / F)^(2 / 3)
        sol = solve_normal_contact(gap0, δ, dummy; prep=prep, tol=tol)
        F = sum(sol.p) * dA
    end
    return sol, δ, F, prep
end

end # module
