"""
    Crack

Unified 2D crack analysis module:

1. **Dual BEM** (Portela–Aliabadi–Rooke / Albuquerque–Sato)
   - Coincident crack faces, discontinuous collocation
   - Displacement BIE (`eq=2`) + traction BIE (`eq=3`)
   - Gmsh + `format2d` with BC type [`CRACK_BC`](@ref) = 5
   - COD stress intensity factors

2. **Cohesive-contact DBEM** (Cordeiro et al. 2024; Alfano–Sacco 2006)
   - Local cohesive stiffness → nonlinear DBEM system
   - Surface conditions: contact, softening, unload/reload, failure
   - Laws: bilinear CZM, PPR, Alfano–Sacco damage+friction
   - Newton + optional single-DOF displacement control

3. **Propagation** (Marcel Sato / UNICAMP)
   - MTS (Erdogan–Sih) and SED (Sih) criteria
   - Paris law (Tanaka ΔK)
   - Tip geometry update
"""
module Crack

using LinearAlgebra
using StaticArrays
using Statistics: mean
using FastGaussQuadrature

export CrackTip, CrackPath, CrackProblem
export sif_from_cod, sif_modeI_quarter
export max_tens_circ, strain_energy_density_angle
export paris_cycles, tanaka_deltaK
export propagation_angle, extend_crack_tip!
export propagate!
export analytical_KI_center_crack

# dual BEM
export DualNode, DualElement, DualMesh
export build_center_crack_mesh, dual_mesh_from_bemdata
export assemble_dual!, apply_bc_dual, solve_dual!
export crack_opening, sif_cod_dual
export dual_to_crack_path
export CRACK_BC

# cohesive-contact DBEM
export CohesiveState, STATE_CONTACT, STATE_SOFTENING, STATE_UNLOAD, STATE_FAILED
export AbstractCohesiveLaw, BilinearCZM, PPRLaw, AlfanoSaccoLaw, FatigueCZM
export CohesiveHistory, CohesivePair, CohesiveDBEMProblem
export build_cohesive_pairs, solve_cohesive_dbem!
export cohesive_tractions, cohesive_openings
export evaluate_surface!, traction_and_stiffness, fatigue_cycle!
export modeI_patch_mesh, modeII_patch_mesh, contact_compression_mesh
export extend_cohesive_process_zone!
export local_to_global_R, opening_local

const Point2D = SVector{2,Float64}

# =============================================================================
# Propagation data structures
# =============================================================================

"""
    CrackTip

One crack tip. `node_upper`/`node_lower` are collocation indices on the two faces
for COD; `tangent` points from the tip into the crack.
"""
mutable struct CrackTip
    id::Int
    node_upper::Int
    node_lower::Int
    pos::Point2D
    tangent::Point2D
    node_upper2::Int
    node_lower2::Int
end

function CrackTip(id, nu, nl, pos, tangent; nu2=nu, nl2=nl)
    t = tangent / (norm(tangent) + eps())
    return CrackTip(id, nu, nl, Point2D(pos), Point2D(t), nu2, nl2)
end

mutable struct CrackPath
    points::Vector{Point2D}
    tips::Vector{CrackTip}
end

mutable struct CrackProblem
    path::CrackPath
    E::Float64
    ν::Float64
    plane_strain::Bool
    C::Float64
    m::Float64
    R_ratio::Float64
    KI::Vector{Float64}
    KII::Vector{Float64}
    theta::Vector{Float64}
    a_hist::Vector{Float64}
    N_cycles::Vector{Float64}
end

function CrackProblem(path::CrackPath; E=1.0, ν=0.3, plane_strain=true,
    C=1e-12, m=3.0, R_ratio=0.0)
    return CrackProblem(path, float(E), float(ν), plane_strain, float(C), float(m),
        float(R_ratio), Float64[], Float64[], Float64[], Float64[], Float64[0.0])
end

# =============================================================================
# Material helpers (shared)
# =============================================================================

function kappa(E, ν, plane_strain::Bool)
    plane_strain && return 3 - 4ν
    return (3 - ν) / (1 + ν)
end

shear_mod(E, ν) = E / (2(1 + ν))

# =============================================================================
# SIF — COD (generic, from displacement vector)
# =============================================================================

function sif_from_cod(tip::CrackTip, u::AbstractVector, nodes::AbstractVector;
    E=1.0, ν=0.3, plane_strain=true, r=nothing)
    iu, il = tip.node_upper, tip.node_lower
    uu = SVector(u[2iu-1], u[2iu])
    ul = SVector(u[2il-1], u[2il])
    Δu = uu - ul
    t̂ = tip.tangent
    n̂ = Point2D(-t̂[2], t̂[1])
    Δun = dot(Δu, n̂)
    Δut = dot(Δu, t̂)
    if r === nothing
        pu = nodes[iu]
        r = norm(pu - tip.pos)
        r < 1e-14 && (r = norm(nodes[tip.node_upper2] - tip.pos))
    end
    r = max(r, 1e-14)
    μ = shear_mod(E, ν)
    κ = kappa(E, ν, plane_strain)
    c = μ / (κ + 1) * sqrt(2π / r)
    return c * Δun, c * Δut
end

function sif_modeI_quarter(E, ν, Δu, L; plane_strain=true)
    Estar = plane_strain ? E / (1 - ν^2) : E
    return Estar * abs(Δu) / 2 * sqrt(π / max(L, eps()))
end

# =============================================================================
# Propagation criteria
# =============================================================================

function max_tens_circ(KI::Real, KII::Real)
    if abs(KI) > 1e-12 && abs(KII) < 1e-12
        return 0.0, float(KI)
    elseif abs(KII) > 1e-12 && abs(KI) < 1e-12
        θ = KII < 0 ? acos(1 / 3) : -acos(1 / 3)
        KIeq = -3 * KII * cos(θ / 2)^2 * sin(θ / 2)
        return θ, KIeq
    else
        Kr = KI / KII
        disc = sqrt(Kr^2 + 8)
        θ1 = 2 * atan(0.25 * Kr + 0.25 * disc)
        θ2 = 2 * atan(0.25 * Kr - 0.25 * disc)
        K1 = KI * cos(θ1 / 2)^3 - 3 * KII * cos(θ1 / 2)^2 * sin(θ1 / 2)
        K2 = KI * cos(θ2 / 2)^3 - 3 * KII * cos(θ2 / 2)^2 * sin(θ2 / 2)
        return KII < 0 ? (θ1, K1) : (θ2, K2)
    end
end

function strain_energy_density_angle(KI, KII; E=1.0, ν=0.3, plane_strain=true)
    κ = kappa(E, ν, plane_strain)
    μ = shear_mod(E, ν)
    B = 1 / (16μ)
    θs = range(-π + 1e-3, π - 1e-3; length=361)
    Sbest, θbest = Inf, 0.0
    for θ in θs
        sθ, cθ = sin(θ), cos(θ)
        a11 = B * (1 + cθ) * (κ - cθ)
        a12 = B * (2 * cθ * sθ - κ * sθ + sθ)
        a22 = B * (κ - κ * cθ + cθ + 3 * cθ^2)
        S = a11 * KI^2 + 2a12 * KI * KII + a22 * KII^2
        if S < Sbest
            Sbest = S
            θbest = θ
        end
    end
    KIeq = sqrt(max(Sbest * 16μ / (2 * (κ - 1)), 0.0))
    return θbest, KIeq, Sbest
end

function propagation_angle(KI, KII; criterion=:MTS, E=1.0, ν=0.3, plane_strain=true)
    c = uppercase(string(criterion))
    if c in ("MTS", "MCT", "MAX_CIRC")
        return max_tens_circ(KI, KII)
    else
        θ, KIeq, _ = strain_energy_density_angle(KI, KII; E=E, ν=ν, plane_strain=plane_strain)
        return θ, KIeq
    end
end

# =============================================================================
# Paris law
# =============================================================================

function tanaka_deltaK(KI, KII, R_ratio)
    dKI = KI * (1 - R_ratio)
    dKII = KII * (1 - R_ratio)
    return sqrt(dKI^2 + 2 * dKII^2)
end

function paris_cycles(C, m, ΔK0, ΔK1, da)
    ΔK0 = max(ΔK0, 1e-30)
    ΔK1 = max(ΔK1, 1e-30)
    return da * 0.5 * (1 / C) * (ΔK0^(-m) + ΔK1^(-m))
end

# =============================================================================
# Geometry update
# =============================================================================

function extend_crack_tip!(path::CrackPath, tip_id::Int, θ_local::Real, da::Real)
    tip = path.tips[tip_id]
    prop0 = -tip.tangent
    c, s = cos(θ_local), sin(θ_local)
    d = Point2D(c * prop0[1] - s * prop0[2], s * prop0[1] + c * prop0[2])
    d = d / (norm(d) + eps())
    newpos = tip.pos + da * d
    push!(path.points, newpos)
    tip.pos = newpos
    tip.tangent = -d
    return newpos
end

function propagate!(prob::CrackProblem, tip_SIFs; da=0.1, criterion=:MTS)
    path = prob.path
    n = length(path.tips)
    θs = zeros(n)
    ΔN = 0.0
    for i in 1:n
        KI, KII = tip_SIFs[i]
        θ, _ = propagation_angle(KI, KII; criterion=criterion,
            E=prob.E, ν=prob.ν, plane_strain=prob.plane_strain)
        θs[i] = θ
        extend_crack_tip!(path, i, θ, da)
        push!(prob.KI, KI)
        push!(prob.KII, KII)
        push!(prob.theta, θ)
        push!(prob.a_hist, crack_length(path))
        if length(prob.KI) >= 2
            ΔK0 = tanaka_deltaK(prob.KI[end-1], abs(prob.KII[end-1]), prob.R_ratio)
            ΔK1 = tanaka_deltaK(KI, abs(KII), prob.R_ratio)
            dN = paris_cycles(prob.C, prob.m, ΔK0, ΔK1, da)
        else
            ΔK1 = tanaka_deltaK(KI, abs(KII), prob.R_ratio)
            dN = paris_cycles(prob.C, prob.m, ΔK1, ΔK1, da)
        end
        ΔN += dN
        push!(prob.N_cycles, prob.N_cycles[end] + dN)
    end
    return θs, ΔN
end

function crack_length(path::CrackPath)
    pts = path.points
    length(pts) < 2 && return 0.0
    s = 0.0
    @inbounds for i in 2:length(pts)
        s += norm(pts[i] - pts[i-1])
    end
    return s
end

function analytical_KI_center_crack(σ, a; W=Inf)
    KI_inf = σ * sqrt(π * a)
    isinf(W) && return KI_inf
    α = π * a / (2W)
    α >= π / 2 && return KI_inf
    return KI_inf * sqrt(1 / cos(α))
end

# =============================================================================
# Dual BEM (assembly, Gmsh mesh, COD) — included plain
# =============================================================================
include("DualCore.jl")

# Cohesive laws + nonlinear DBEM solver (Cordeiro 2024 / Alfano–Sacco 2006)
include("CohesiveLaws.jl")
include("CohesiveDBEM.jl")

end # module
