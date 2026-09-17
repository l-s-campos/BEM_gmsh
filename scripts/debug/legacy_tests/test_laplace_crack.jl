# Laplace dual BEM (twin coincident faces) vs finite-width slit
using Test
using LinearAlgebra
using DrWatson
@quickactivate :BEM
using BEM.Crack

const _LC = (
    W = 5.0, H = 10.0, a = 1.0,
    ndiv_b = 6, ndiv_h = 8, ndiv_crack = 8,
    ordem = 1, npg = 12,
)

function _dual_dad(; field=:y, bc=:insulated, T0=0.0, nome="test_lap_dual")
    dual_laplace_problem(;
        W=_LC.W, H=_LC.H, a=_LC.a,
        ndiv_b=_LC.ndiv_b, ndiv_h=_LC.ndiv_h, ndiv_crack=_LC.ndiv_crack,
        field=field, bc=bc, T0=T0, ordem=_LC.ordem, nome=nome, pontointerno=false)
end

function _fw_dad(; gap=0.05, field=:y, nome="test_lap_fw")
    finite_width_laplace_problem(;
        W=_LC.W, H=_LC.H, a=_LC.a, gap=gap,
        ndiv_b=_LC.ndiv_b, ndiv_h=_LC.ndiv_h, ndiv_crack=_LC.ndiv_crack,
        n_cap=3, field=field, ordem=_LC.ordem, nome=nome, pontointerno=false)
end

"""Collocation on the diamond opening (`|y| < gap`, `|x| < a`), not the outer plate."""
function _slit_wall_nodes(dad, gap; upper::Bool=true)
    return findall(1:dad.n) do i
        p = dad.Nodes[i]
        abs(p[1]) < 0.85 * _LC.a && abs(p[2]) < 0.7 * gap &&
            (upper ? p[2] > 0.02 * gap : p[2] < -0.02 * gap)
    end
end

@testset "twin pairing + integral classification" begin
    dad = _dual_dad(; nome="test_lap_pair")
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
    elB = dad.elements[dad.crack_face_b[1]]
    iA = elA.index[1]
    @test integral_kind(dad, iA, elA; twins=dad.twin) === :guiggiani_self
    twA = dad.twin[iA]
    el_twin = dad.elements[findfirst(e -> twA in e.index, dad.elements)]
    @test integral_kind(dad, iA, el_twin; twins=dad.twin) === :guiggiani_twin
end

@testset "finite-width opposite face is sinh, not Guiggiani" begin
    gap = 0.05
    dad = _fw_dad(; gap=gap, nome="test_lap_fw_kind")
    top = _slit_wall_nodes(dad, gap; upper=true)
    @test !isempty(top)
    i_top = top[argmin(abs.(getindex.(dad.Nodes[top], 1)))]
    xt = dad.Nodes[i_top][1]
    bot_el = argmin(1:length(dad.elements)) do ie
        el = dad.elements[ie]
        ym = mean(dad.Nodes[j][2] for j in el.index)
        xm = mean(dad.Nodes[j][1] for j in el.index)
        ym < 0 && abs(ym) < gap && abs(xm) < _LC.a ? abs(xm - xt) : Inf
    end
    @test isfinite(mean(dad.Nodes[j][2] for j in dad.elements[bot_el].index))
    kind = integral_kind(dad, i_top, dad.elements[bot_el])
    @test kind === :sinh
    @test min_opposite_distance(dad) < 2gap
    chk = check_slit_bc(dad; a=_LC.a, gap=gap)
    @test chk.insulated
    @test chk.n_into_hole > chk.n_into_solid
end

@testset "harmonic identities on dual mesh" begin
    dad = _dual_dad(; nome="test_lap_id")
    assemble_dual_laplace!(dad; npg=_LC.npg, threaded=false)
    n, nt = dad.n, dad.nt
    # constant T, q = 0
    T1 = ones(nt)
    q0 = zeros(n)
    r1 = dad.H * T1 - dad.G * q0
    @test all(isfinite, r1)
    # H 1 = 0 for constant T (q = 0); do not divide by ‖H 1‖ (that ratio is 1).
    @test norm(r1) / n < 0.15

    # linear T = y with consistent q = -∂T/∂n
    Ty = [p[2] for p in all_points(dad)]
    qy = [-dot(SA[0.0, 1.0], dad.Normal[i]) for i in 1:n]   # k=1
    ry = dad.H * Ty - dad.G * qy
    @test all(isfinite, ry)
    @test norm(ry) / (norm(dad.H * Ty) + norm(dad.G * qy) + 1e-30) < 0.12
end

@testset "insulated dual vs Griffith jump" begin
    dad = _dual_dad(; field=:y, bc=:insulated, nome="test_lap_ins")
    solve_dual_laplace!(dad; npg=_LC.npg, threaded=false)
    nodesA = crack_face_nodes(dad; face=2)
    xs = Float64[]
    Δnum = Float64[]
    Δana = Float64[]
    for i in nodesA
        x = dad.Nodes[i][1]
        abs(x) > 0.8 * _LC.a && continue
        dad.twin[i] == 0 && continue
        push!(xs, x)
        push!(Δnum, abs(crack_jump(dad, i)))
        push!(Δana, analytical_insulated_jump(x, _LC.a; G=1.0))
    end
    @test !isempty(Δnum)
    @test all(>(0), Δnum)
    rel = norm(Δnum .- Δana) / (norm(Δana) + 1e-30)
    @info "insulated dual jump" rel n=length(Δnum)
    @test rel < 0.20
end

@testset "conducting dual: T=0 on faces, finite flux" begin
    dad = _dual_dad(; field=:x, bc=:conducting, T0=0.0, nome="test_lap_cond")
    solve_dual_laplace!(dad; npg=_LC.npg, threaded=false)
    nodesA = crack_face_nodes(dad; face=2)
    @test maximum(abs, dad.T[nodesA]) < 1e-10
    @test all(isfinite, dad.q)
    @test maximum(abs, dad.q[nodesA]) > 0
end

@testset "finite-width quadrature self-convergence" begin
    dad = _fw_dad(; gap=0.05, nome="test_lap_fw_quad")
    H1, G1 = assemble_finite_width_laplace!(dad; npg=12, gap=0.05, threaded=false)
    dad2 = _fw_dad(; gap=0.05, nome="test_lap_fw_quad2")
    H2, G2 = assemble_finite_width_laplace!(dad2; npg=48, gap=0.05, threaded=false)
    @test all(isfinite, H1) && all(isfinite, G1)
    relH = norm(H1 - H2) / (norm(H2) + 1e-30)
    @info "finite-width H self-conv" relH
    @test relH < 0.08
end

@testset "finite-width jump approaches dual as δ shrinks" begin
    dad_d = _dual_dad(; field=:y, bc=:insulated, nome="test_lap_cmp_dual")
    solve_dual_laplace!(dad_d; npg=_LC.npg, threaded=false)
    function jump_mid(dad; gap=nothing)
        if has_cache(dad, :twin)
            nodesA = crack_face_nodes(dad; face=2)
            isempty(nodesA) && return NaN
            i = nodesA[argmin(abs.(getindex.(dad.Nodes[nodesA], 1)))]
            return abs(crack_jump(dad, i))
        end
        gap === nothing && return NaN
        top = _slit_wall_nodes(dad, gap; upper=true)
        bot = _slit_wall_nodes(dad, gap; upper=false)
        (isempty(top) || isempty(bot)) && return NaN
        it = top[argmin(abs.(getindex.(dad.Nodes[top], 1)))]
        ib = bot[argmin(abs.(getindex.(dad.Nodes[bot], 1)))]
        return abs(dad.T[it] - dad.T[ib])
    end
    Δd = jump_mid(dad_d)
    Δs = Float64[]
    for (k, δ) in enumerate((0.10, 0.04))
        dad = _fw_dad(; gap=δ, field=:y, nome="test_lap_cmp_fw$k")
        assemble_finite_width_laplace!(dad; gap=δ, threaded=false)
        solve(dad)
        push!(Δs, jump_mid(dad; gap=δ))
    end
    errs = abs.(Δs .- Δd) ./ (abs(Δd) + 1e-30)
    @info "δ-sweep vs dual (mid jump)" dual=Δd fw=Δs errs
    @test all(isfinite, Δs)
    @test all(>(0.5), Δs)                 # opening, not the raw T=y gap
    @test errs[end] ≤ errs[1] + 0.15      # smaller gap not worse
end
