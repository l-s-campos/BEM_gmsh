# XBEM — Andrade & Leonel (2020) Williams tip enrichment on dual BEM
using Test
using LinearAlgebra
using DrWatson
@quickactivate :BEM
using BEM.Crack

@testset "Williams local: opening on the faces" begin
    μ, κ = 1.0, 2.2   # E=3000, ν=0.2 plane strain: μ=1250, κ=2.2 — scale-free here
    ρ = 0.04
    MI = williams_M_local(μ, κ, ρ, π)
    MIm = williams_M_local(μ, κ, ρ, -π)
    # mode I: ux=0 on faces, uy odd (upper +, lower −)
    @test abs(MI[1, 1]) < 1e-14
    @test MI[2, 1] > 0
    @test MIm[2, 1] ≈ -MI[2, 1] atol=1e-14
    Δv = MI[2, 1] - MIm[2, 1]
    @test Δv ≈ (κ + 1) / μ * sqrt(ρ / (2π)) rtol=1e-12
end

@testset "shifted enrichment vanishes at collocation" begin
    dad = dual_elasticity_problem(; W=5, H=10, a=1, E=3000, ν=0.2, σ=1,
        ndiv_b=4, ndiv_h=6, ndiv_crack=6, ordem=2, nome="xbem_shift")
    assemble_dual!(dad; npg=8, threaded=false)
    tips = williams_tip_positions(dad)
    @test length(tips) == 2
    props = dad.properties
    μ, κ = props.mu, BEM.Crack.kappa(props.E, props.nu, props.plane_strain)
    tip = tips[1]
    e1, e2 = tip_frame(dad, tip)
    enr = BEM.Crack._enriched_elements(dad, tip; n_enr=2)
    @test !isempty(enr)
    el = dad.elements[enr[1]]
    xj = dad.Nodes[el.index]
    ψ_nodes = [williams_global(μ, κ, tip, e1, e2, xj[k]) for k in eachindex(xj)]
    poly = dad.element_type
    # at each collocation ξ of the element, φ = 0
    # collocation = Gauss nodes = poly.nodes for Legendre
    ξcol = poly.nodes
    Nmat, _ = BEM.shapefun(poly, ξcol)
    for q in eachindex(ξcol)
        pg = sum(Nmat[q, k] * xj[k] for k in eachindex(xj))
        ψx = williams_global(μ, κ, tip, e1, e2, pg)
        φ = BEM.Crack._shifted_williams(ψx, view(Nmat, q, :), ψ_nodes)
        @test norm(φ) < 1e-10
    end
end

@testset "XBEM Griffith centre crack" begin
    dad = dual_elasticity_problem(; W=5, H=10, a=1, E=3000, ν=0.2, σ=1,
        ndiv_b=6, ndiv_h=8, ndiv_crack=8, ordem=2, nome="xbem_griffith")
    assemble_dual!(dad; npg=10, threaded=false)
    u, KI, KII = solve_xbem!(dad; n_enr=3, npg=24, n_v=9, threaded=false)
    @test length(KI) == 2
    @test all(isfinite, KI) && all(isfinite, KII)
    @test norm(dad.C_sif) > 1e-8          # polynomial tying, not √ρ (that zeros Cε)
    KIana = analytical_KI_center_crack(1.0, 1.0; W=5.0)
    KIn = 0.5 * (abs(KI[1]) + abs(KI[2]))
    @info "XBEM Griffith" KI KII KIn KIana rel=abs(KIn - KIana) / KIana
    @test KIn > 0
    @test maximum(abs, KII) < 0.05 * KIana
    @test abs(KIn - KIana) / KIana < 0.05
end

@testset "XBEM edge crack vs Civelek–Erdogan" begin
    W, a, σ = 1.0, 0.5, 1.0
    dad = edge_crack_problem(; W=W, a=a, E=1.0, ν=0.3, σ=σ,
        ndiv_b=6, ndiv_h=6, ndiv_crack=8, ndiv_left=4,
        ordem=2, nome="xbem_edge")
    assemble_dual!(dad; npg=10, threaded=false)
    u, KI, KII = solve_xbem!(dad; n_enr=4, npg=24, n_v=9, threaded=false)
    @test length(KI) == 1
    @test all(isfinite, KI) && all(isfinite, KII)
    KIana = analytical_KI_edge_crack(σ, a, W)
    Fn = abs(KI[1]) / (σ * sqrt(π * a))
    Fana = KIana / (σ * sqrt(π * a))
    @info "XBEM edge a/W=0.5" KI=KI[1] KII=KII[1] Fn Fana rel=abs(Fn - Fana) / Fana
    @test abs(KII[1]) < 0.15 * abs(KI[1]) + 0.1
    @test abs(Fn - Fana) / Fana < 0.08
end
