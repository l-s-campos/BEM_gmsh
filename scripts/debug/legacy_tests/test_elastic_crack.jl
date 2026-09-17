# Elasticity dual BEM (twin coincident faces) vs finite-width diamond slit
using Test
using LinearAlgebra
using Statistics: mean
using DrWatson
@quickactivate :BEM
using BEM.Crack

const _EC = (
    W=5.0, H=10.0, a=1.0,
    ndiv_b=6, ndiv_h=8, ndiv_crack=8,
    E=3000.0, ν=0.2, σ=1.0,
    ordem=2, npg=10,
)

function _dual_el(; nome="test_el_dual")
    dual_elasticity_problem(;
        W=_EC.W, H=_EC.H, a=_EC.a, E=_EC.E, ν=_EC.ν, σ=_EC.σ,
        ndiv_b=_EC.ndiv_b, ndiv_h=_EC.ndiv_h, ndiv_crack=_EC.ndiv_crack,
        ordem=_EC.ordem, nome=nome, pontointerno=false)
end

function _fw_el(; gap=0.05, nome="test_el_fw")
    finite_width_elasticity_problem(;
        W=_EC.W, H=_EC.H, a=_EC.a, gap=gap, E=_EC.E, ν=_EC.ν, σ=_EC.σ,
        ndiv_b=_EC.ndiv_b, ndiv_h=_EC.ndiv_h, ndiv_crack=_EC.ndiv_crack,
        ordem=_EC.ordem, nome=nome, pontointerno=false)
end

@testset "Kelvin hypersingular = D,S · nξ" begin
    props = Elasticity(1.0, 0.3, 1.0)
    r = Point2D(0.3, 0.4)
    n = Point2D(1.0, 0.0)
    nf = Point2D(0.0, 1.0)
    sk = fundamental_stress(props, r, n)
    kh = fundamental_hyper(props, r, n, nf)
    D, S = sk.D, sk.S
    U = zeros(2, 2)
    T = zeros(2, 2)
    for i in 1:2, k in 1:2
        U[i, k] = nf[1] * D[k, i, 1] + nf[2] * D[k, i, 2]
        T[i, k] = nf[1] * S[k, i, 1] + nf[2] * S[k, i, 2]
    end
    @test kh.U ≈ U atol=1e-14
    @test kh.T ≈ T atol=1e-14
end

@testset "elastic twin pairing + Guiggiani classification" begin
    dad = _dual_el(; nome="test_el_pair")
    @test has_cache(dad, :eq_type)
    @test count(==(2), dad.eq_type) > 0
    @test count(==(3), dad.eq_type) > 0
    nA = crack_face_nodes(dad; face=2)
    @test !isempty(nA)
    i = nA[length(nA) ÷ 2]
    tw = dad.twin[i]
    @test tw != 0
    @test norm(dad.Nodes[i] - dad.Nodes[tw]) < 1e-9
    @test dad.eq_type[tw] == 3
    @test abs(dot(dad.Normal[i], dad.Normal[tw]) + 1) < 1e-8

    elA = dad.elements[dad.crack_face_a[1]]
    iA = elA.index[1]
    @test integral_kind(dad, iA, elA; twins=dad.twin) === :guiggiani_self
    twA = dad.twin[iA]
    el_twin = dad.elements[findfirst(e -> twA in e.index, dad.elements)]
    @test integral_kind(dad, iA, el_twin; twins=dad.twin) === :guiggiani_twin
end

@testset "dual elasticity Feddersen KI" begin
    dad = _dual_el(; nome="test_el_fed")
    assemble_dual!(dad; npg=_EC.npg, threaded=false)
    solve_dual!(dad; npg=_EC.npg, threaded=false)
    @test all(isfinite, dad.u)
    tips = dad.tip_nodes
    @test length(tips) == 2
    KI_L, KII_L = sif_cod_dual(dad, tips[1])
    KI_R, KII_R = sif_cod_dual(dad, tips[2])
    KIana = analytical_KI_center_crack(_EC.σ, _EC.a; W=_EC.W)
    KIn = 0.5 * (abs(KI_L) + abs(KI_R))
    @info "dual elasticity Feddersen" KIn KIana rel=abs(KIn - KIana) / KIana
    @test KIn > 0
    @test abs(KII_L) < 0.25 * KIana
    @test abs(KIn - KIana) / KIana < 0.25
end

@testset "finite-width elasticity opposite face is sinh" begin
    gap = 0.05
    dad = _fw_el(; gap=gap, nome="test_el_fw_kind")
    top = findall(1:dad.n) do i
        p = dad.Nodes[i]
        abs(p[1]) < 0.85 * _EC.a && 0.02 * gap < p[2] < 0.7 * gap
    end
    @test !isempty(top)
    i_top = top[argmin(abs.(getindex.(dad.Nodes[top], 1)))]
    xt = dad.Nodes[i_top][1]
    bot_el = argmin(1:length(dad.elements)) do ie
        el = dad.elements[ie]
        ym = mean(dad.Nodes[j][2] for j in el.index)
        xm = mean(dad.Nodes[j][1] for j in el.index)
        ym < 0 && abs(ym) < gap && abs(xm) < _EC.a ? abs(xm - xt) : Inf
    end
    kind = integral_kind(dad, i_top, dad.elements[bot_el])
    @test kind === :sinh
    chk = check_slit_bc(dad; a=_EC.a, gap=gap)
    @test chk.n_into_hole > chk.n_into_solid
end

@testset "finite-width diamond solves; dual holds the opening" begin
    dad_d = _dual_el(; nome="test_el_cmp_dual")
    assemble_dual!(dad_d; npg=_EC.npg, threaded=false)
    solve_dual!(dad_d; threaded=false)
    nA = crack_face_nodes(dad_d; face=2)
    imid = nA[argmin(abs.(getindex.(dad_d.Nodes[nA], 1)))]
    Δd = abs(crack_opening(dad_d, imid)[2])
    @test Δd > 0

    gap = 0.08
    dad_s = _fw_el(; gap=gap, nome="test_el_cmp_fw")
    assemble_finite_width_elasticity!(dad_s; gap=gap, threaded=false)
    solve(dad_s)
    @test all(isfinite, dad_s.u)
    outer_top = findall(i -> abs(dad_s.Nodes[i][2] - _EC.H) < 1e-8, 1:dad_s.n)
    uy_top = mean(dad_s.u[2i] for i in outer_top)
    # uniaxial plane-strain elongation 2H * σ(1-ν²)/E
    ΔH = 2 * _EC.H * _EC.σ * (1 - _EC.ν^2) / _EC.E
    @test uy_top ≈ ΔH rtol=0.15
    @info "elastic COD dual vs diamond (thin-gap CBIE under-resolves opening)" dual=Δd slit=slit_opening_mid(dad_s; a=_EC.a, gap=gap)[2] uy_top
end
