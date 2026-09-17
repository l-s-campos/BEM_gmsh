# Dual BEM: SIF identities + Griffith centre crack.
using Test
using LinearAlgebra
using StaticArrays
using BEM
using BEM.Crack

@testset "SIF criteria" begin
    θ, KIeq = max_tens_circ(2.0, 0.0)
    @test abs(θ) < 1e-12
    @test KIeq ≈ 2.0
    θ45, _ = max_tens_circ(1.0, 1.0)
    @test θ45 ≈ 2 * atan(-0.5) atol=1e-12
    θII, _ = max_tens_circ(0.0, 1.0)
    @test θII ≈ -acos(1 / 3) atol=1e-12
    KIa, KIIa = analytical_sif_inclined_center(1.0, 1.0, π / 4)
    @test KIa ≈ KIIa ≈ 0.5 * sqrt(π)
    @test analytical_KI_center_crack(1.0, 1.0) ≈ sqrt(π)
end

@testset "dual BEM centre crack" begin
    dad = build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, σ=1.0,
        E=3000.0, ν=0.2, n_bottom=4, n_right=6, n_top=4, n_left=6, n_crack=6)
    assemble_dual!(dad; npg=8, threaded=false)
    solve_dual!(dad; threaded=false)
    KI_L, KII_L = sif_cod_dual(dad, dad.tip_nodes[1])
    KI_R, KII_R = sif_cod_dual(dad, dad.tip_nodes[2])
    KIana = analytical_KI_center_crack(1.0, 1.0; W=5.0)
    KIn = 0.5 * (abs(KI_L) + abs(KI_R))
    @test abs(KII_L) < 0.10 * KIana
    @test abs(KIn - KIana) / KIana < 0.10
end

@testset "Kelvin dual twin closed-form matches interp" begin
    # Field normal stays on the integration element; closed-form tensors flip
    # with n_ξ·n_el. Aligning n to n_ξ and also flipping F₋₂ left 2/ρ² in the
    # remainder (‖I_H‖ ~ 10³ vs ~5 on this element).
    dad = build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, σ=1.0,
        E=1.0, ν=0.3, n_bottom=4, n_right=6, n_top=4, n_left=6, n_crack=6,
        nome="twin_laurent")
    BEM._init_quadrature!(dad, 12)
    i = findfirst(==(3), dad.eq_type)
    tw = dad.twin[i]
    @test tw != 0
    el = dad.elements[findfirst(e -> tw in e.index, dad.elements)]
    xj = dad.Nodes[el.index]
    pf = dad.Nodes[i]
    nf = dad.Normal[i]
    f = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
    ncols = 2 * length(el.index)
    hA = zeros(2, ncols); gA = zeros(2, ncols)
    hI = zeros(2, ncols); gI = zeros(2, ncols)
    set_cache!(dad; laurent=:auto)
    integrate_element(dad, el, xj, pf, hA, gA, f; orders=(-1, -2), source=i, twins=dad.twin)
    set_cache!(dad; laurent=:interp)
    integrate_element(dad, el, xj, pf, hI, gI, f; orders=(-1, -2), source=i, twins=dad.twin)
    @test norm(hA) < 50
    @test norm(hI - hA) / max(norm(hA), 1e-16) < 1e-8
    @test norm(gI - gA) / max(norm(gA), 1e-16) < 1e-6
end

@testset "Williams opening" begin
    μ, κ = 1.0, 2.2
    ρ = 0.04
    MI = williams_M_local(μ, κ, ρ, π)
    MIm = williams_M_local(μ, κ, ρ, -π)
    @test abs(MI[1, 1]) < 1e-14
    Δv = MI[2, 1] - MIm[2, 1]
    @test Δv ≈ (κ + 1) / μ * sqrt(ρ / (2π)) rtol=1e-12
end

@testset "Sih–Paris–Irwin tip field" begin
    E, ν = 1000.0, 0.3
    G = E / (2(1 + ν))
    p_iso = lekhnitskii_params(E, E, G, ν)
    ρ = 0.04
    κ = (3 - ν) / (1 + ν)
    @test sih_M_local(p_iso, ρ, π) ≈ williams_M_local(G, κ, ρ, π) rtol=1e-8
    @test sih_M_local(p_iso, ρ, -π) ≈ williams_M_local(G, κ, ρ, -π) rtol=1e-8

    p = lekhnitskii_params(144.8, 11.7, 9.66, 0.21)
    MI = sih_M_local(p, ρ, π)
    MIm = sih_M_local(p, ρ, -π)
    @test MI[2, 1] > 0
    @test MIm[2, 1] ≈ -MI[2, 1] atol=1e-12
    @test abs(MI[1, 1]) < 1e-10          # aligned orthotropy: mode I, ux=0 on faces
    Mcod = MI - MIm
    @test abs(Mcod[1, 1]) < 1e-10
    @test Mcod[2, 1] > 0
    @test abs(det(Mcod)) > 1e-12
    @test (Mcod \ Mcod[:, 1]) ≈ [1.0, 0.0] atol=1e-10

    γ0 = BEM.Crack._lekhnitskii_gamma(p.mi[1], 0.0)
    γp = BEM.Crack._lekhnitskii_gamma(im, π)
    γm = BEM.Crack._lekhnitskii_gamma(im, -π)
    @test γ0 ≈ 1 rtol=1e-12
    @test γp ≈ im atol=1e-12
    @test γm ≈ -im atol=1e-12

    α = π / 5
    pr = lekhnitskii_rotate(p, α)
    μ = p.mi
    μp = (μ .* cos(α) .- sin(α)) ./ (cos(α) .+ μ .* sin(α))
    @test sort(real.(pr.mi)) ≈ sort(real.(μp)) atol=1e-8
    @test sort(imag.(pr.mi)) ≈ sort(imag.(μp)) atol=1e-8
end

@testset "dual BEM anisotropic centre crack" begin
    # Infinite-plate identity: KI = σ√(πa) independent of anisotropy when the
    # crack is along x and remote tension is σ_yy (Sih–Paris–Irwin).
    props = AnisotropicElasticity(lekhnitskii_params(144.8, 11.7, 9.66, 0.21))
    dad = dual_elasticity_problem(; W=5.0, H=10.0, a=1.0, σ=1.0,
        ndiv_b=4, ndiv_h=6, ndiv_crack=6, ordem=2, nome="aniso_cc",
        props=props)
    assemble_dual!(dad; npg=8, threaded=false)
    solve_dual!(dad; threaded=false)
    KI_L, KII_L = sif_cod_dual(dad, dad.tip_nodes[1])
    KI_R, KII_R = sif_cod_dual(dad, dad.tip_nodes[2])
    KIana = analytical_KI_center_crack(1.0, 1.0; W=5.0)
    KIn = 0.5 * (abs(KI_L) + abs(KI_R))
    @test abs(KII_L) < 0.10 * KIana
    @test abs(KII_R) < 0.10 * KIana
    @test abs(KIn - KIana) / KIana < 0.10

    props30 = AnisotropicElasticity(lekhnitskii_params(144.8, 11.7, 9.66, 0.21; θ_deg=30))
    dad30 = dual_elasticity_problem(; W=5.0, H=10.0, a=1.0, σ=1.0,
        ndiv_b=4, ndiv_h=6, ndiv_crack=6, ordem=2, nome="aniso_cc30",
        props=props30)
    assemble_dual!(dad30; npg=8, threaded=false)
    solve_dual!(dad30; threaded=false)
    KI30 = 0.5 * abs(sif_cod_dual(dad30, dad30.tip_nodes[1])[1]) +
           0.5 * abs(sif_cod_dual(dad30, dad30.tip_nodes[2])[1])
    KII30 = 0.5 * abs(sif_cod_dual(dad30, dad30.tip_nodes[1])[2]) +
            0.5 * abs(sif_cod_dual(dad30, dad30.tip_nodes[2])[2])
    @test abs(KI30 - KIana) / KIana < 0.15
    @test KII30 < 0.15 * KIana
end

@testset "XBEM anisotropic vs dual COD" begin
    props = AnisotropicElasticity(lekhnitskii_params(144.8, 11.7, 9.66, 0.21))
    dad = dual_elasticity_problem(; W=5.0, H=10.0, a=1.0, σ=1.0,
        ndiv_b=4, ndiv_h=6, ndiv_crack=6, ordem=2, nome="aniso_xbem",
        props=props)
    assemble_dual!(dad; npg=8, threaded=false)
    u, KI, KII = solve_xbem!(dad; n_enr=2, npg=12, n_v=6, threaded=false)
    KIana = analytical_KI_center_crack(1.0, 1.0; W=5.0)
    KIn = 0.5 * (abs(KI[1]) + abs(KI[2]))
    @test all(isfinite, KI) && all(isfinite, KII)
    @test maximum(abs, KII) < 0.15 * KIana
    @test abs(KIn - KIana) / KIana < 0.15
end

@testset "Hattori 5.3 double-edge off-axis" begin
    # Digitised Direct SIF, Hattori IJNME 2016 Fig. 7. Off-axis HBIE needs npg≳16.
    ref = Dict(40.0 => 1.46, 60.0 => 1.84)
    for β in (40.0, 60.0)
        props = AnisotropicElasticity(lekhnitskii_params(144.8, 11.7, 9.66, 0.21; θ_deg=β))
        dad = double_edge_crack_problem(; W=1.0, H=1.0, a=0.5, σ=1.0,
            ndiv_b=6, ndiv_side=3, ndiv_crack=4, ordem=2,
            nome="h53_$(Int(β))", props=props)
        assemble_dual!(dad; npg=20, threaded=false)
        _, KI, KII = solve_xbem!(dad; n_enr=3, npg=20, n_v=9, threaded=false)
        @test length(KI) == 2
        F = 0.5 * (abs(KI[1]) + abs(KI[2])) / sqrt(π * 0.5)
        @test abs(KI[1] - KI[2]) / max(abs(KI[1]) + abs(KI[2]), 1e-12) < 0.05
        @test abs(F - ref[β]) / ref[β] < 0.08
    end
end

@testset "Erdogan–Sih MTS 45°" begin
    dad = dual_elasticity_problem(; W=5.0, H=10.0, a=1.0, α=π / 4, E=1.0, ν=0.3,
        σ=1.0, ndiv_b=4, ndiv_h=6, ndiv_crack=6, plane_strain=false, ordem=2,
        nome="mts45")
    assemble_dual!(dad; npg=8, threaded=false)
    solve_dual!(dad; threaded=false)
    hist = propagate_dual_mts!(dad; nsteps=1, da=0.2, npg=8, sample=2)
    @test length(hist[1].tips) == 2
    KIa, KIIa = analytical_sif_inclined_center(1.0, 1.0, π / 4)
    θa, _ = max_tens_circ(KIa, KIIa)
    θm = 0.5 * (hist[1].θ[1] + hist[1].θ[2])
    @test sign(hist[1].KII[1]) == sign(KIIa)
    @test abs(θm - θa) < deg2rad(12)

    dadx = dual_elasticity_problem(; W=5.0, H=10.0, a=1.0, α=π / 4, E=1.0, ν=0.3,
        σ=1.0, ndiv_b=4, ndiv_h=6, ndiv_crack=6, plane_strain=false, ordem=2,
        nome="mts45_xbem")
    assemble_dual!(dadx; npg=8, threaded=false)
    _, KIx, KIIx = solve_xbem!(dadx; n_enr=2, npg=8, n_v=6, threaded=false)
    θx = 0.5 * (max_tens_circ(KIx[1], KIIx[1])[1] + max_tens_circ(KIx[2], KIIx[2])[1])
    @test all(sign(k) == sign(KIIa) for k in KIIx)
    @test abs(θx - θa) < deg2rad(12)
end

@testset "Ke 2008 anisotropic MTS" begin
    # Isotropic reduction of Sih–Paris–Irwin hoop (μ → i).
    p_iso = lekhnitskii_params(1000.0, 1000.0, 1000.0 / (2 * 1.3), 0.3)
    θa, _ = max_tens_circ(1.0, 1.0)
    θn, Keq = max_tens_circ(1.0, 1.0, p_iso)
    @test abs(θn - θa) < deg2rad(1)
    @test Keq ≈ (1.0 * cos(θa / 2)^3 - 1.5 * sin(θa) * cos(θa / 2)) rtol=0.05
    p_ge = lekhnitskii_params(48.26, 17.24, 6.89, 0.29)
    @test BEM.Crack._aniso_hoop_keq(1.0, 0.0, p_ge.mi[1], p_ge.mi[2], 0.0) ≈ 1 rtol=1e-8

    # Ke Table V / Gandhi: 45° centre crack, glass-epoxy, a/w=0.2, h/w=2.
    ref = Dict(0.0 => (0.522, 0.507), 90.0 => (0.513, 0.509), 135.0 => (0.532, 0.511))
    for ψ in (0.0, 90.0, 135.0)
        props = AnisotropicElasticity(lekhnitskii_params(48.26, 17.24, 6.89, 0.29; θ_deg=ψ))
        dad = dual_elasticity_problem(; W=1.0, H=2.0, a=0.2, α=π / 4, σ=1.0,
            ndiv_b=8, ndiv_h=12, ndiv_crack=8, ordem=2,
            nome="ke_t5_$(Int(ψ))", props=props)
        assemble_dual!(dad; npg=12, threaded=false)
        solve_dual!(dad; threaded=false)
        tips = williams_tip_positions(dad)
        nodesA = crack_face_nodes(dad; face=2)
        KIs = Float64[]
        KIIs = Float64[]
        for geo in tips
            inode = nodesA[argmin(norm(dad.Nodes[i] - geo) for i in nodesA)]
            KI, KII = sif_cod_dual(dad, inode; sample=2)
            push!(KIs, KI)
            push!(KIIs, KII)
        end
        s = sqrt(π * 0.2)
        F1 = 0.5 * (KIs[1] + KIs[2]) / s
        F2 = 0.5 * (KIIs[1] + KIIs[2]) / s
        @test abs(F1 - ref[ψ][1]) / ref[ψ][1] < 0.10
        @test abs(F2 - ref[ψ][2]) / ref[ψ][2] < 0.10
        e1 = BEM.Crack._crack_ahead(dad, tips[1])
        θ, _ = max_tens_circ(0.5 * (KIs[1] + KIs[2]), 0.5 * (KIIs[1] + KIIs[2]),
            props, e1)
        @test isfinite(θ) && abs(θ) < π
    end
end

@testset "Ke 2008 CSTBD" begin
    # Vertical crack, diametral compression: induced tension opens the crack.
    dad = cstbd_problem(; R=1.0, a=0.3, β=0.0, p=1.0,
        ndiv_outer=20, ndiv_crack=6, ndiv_load=2, ordem=2, nome="cstbd_t0",
        props=Elasticity(1.0, 0.25, 1.0; plane_strain=false))
    assemble_dual!(dad; npg=8, threaded=false)
    solve_dual!(dad; threaded=false)
    tips = williams_tip_positions(dad)
    @test length(tips) == 2
    nodesA = crack_face_nodes(dad; face=2)
    KI, KII = sif_cod_dual(dad, nodesA[argmin(norm(dad.Nodes[i] - tips[1]) for i in nodesA)];
        sample=2)
    @test KI > 0
    @test abs(KII) < 0.25 * KI
    hist = propagate_dual_mts!(dad; nsteps=1, da=0.1, npg=8, sample=2)
    @test abs(hist[1].θ[1]) < deg2rad(15)
end
