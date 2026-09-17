using LinearAlgebra
using BEM
using .Crack

const W, H, a = 5.0, 10.0, 1.0
const ndiv_b, ndiv_h, ndiv_crack = 6, 8, 8

function jump_mid_dual(dad)
    nodesA = crack_face_nodes(dad; face=2)
    i = nodesA[argmin(abs.(getindex.(dad.Nodes[nodesA], 1)))]
    return abs(crack_jump(dad, i)), dad.Nodes[i][1]
end

function jump_mid_fw(dad, gap)
    top = findall(i -> begin
        p = dad.Nodes[i]
        abs(p[1]) < 0.85a && 0.02gap < p[2] < 0.7gap
    end, 1:dad.n)
    bot = findall(i -> begin
        p = dad.Nodes[i]
        abs(p[1]) < 0.85a && -0.7gap < p[2] < -0.02gap
    end, 1:dad.n)
    (isempty(top) || isempty(bot)) && return NaN, NaN
    it = top[argmin(abs.(getindex.(dad.Nodes[top], 1)))]
    ib = bot[argmin(abs.(getindex.(dad.Nodes[bot], 1)))]
    return abs(dad.T[it] - dad.T[ib]), dad.Nodes[it][1]
end

function opp_kind(dad, gap)
    top = findall(i -> begin
        p = dad.Nodes[i]
        abs(p[1]) < 0.4a && p[2] > 0.02gap && abs(p[2]) < 0.7gap
    end, 1:dad.n)
    isempty(top) && return :none, Inf
    i_top = top[argmin(abs.(getindex.(dad.Nodes[top], 1)))]
    xt = dad.Nodes[i_top][1]
    bot_el = argmin(1:length(dad.elements)) do ie
        el = dad.elements[ie]
        ym = mean(dad.Nodes[j][2] for j in el.index)
        xm = mean(dad.Nodes[j][1] for j in el.index)
        ym < 0 && abs(ym) < gap && abs(xm) < a ? abs(xm - xt) : Inf
    end
    el = dad.elements[bot_el]
    kind = integral_kind(dad, i_top, el)
    _, _, dist = closest_point_1d(dad.element_type, dad.Nodes[el.index], dad.Nodes[i_top])
    return kind, dist
end

println("dual BEM reference…"); flush(stdout)
dad_d = dual_laplace_problem(; W, H, a, ndiv_b, ndiv_h, ndiv_crack,
    field=:y, bc=:insulated, ordem=1, nome="dsw_dual", pontointerno=false)
solve_dual_laplace!(dad_d; npg=12, threaded=false)
Δd, xd = jump_mid_dual(dad_d)
Δana = analytical_insulated_jump(xd, a; G=1.0)
println("dual ΔT=", round(Δd; digits=4), "  x=", round(xd; digits=3),
    "  Griffith=", round(Δana; digits=4),
    "  rel=", round(abs(Δd - Δana) / Δana; digits=4))
flush(stdout)

println()
println(rpad("δ", 8), rpad("kind", 8), rpad("d_opp", 10), rpad("npg*", 6),
    rpad("‖ΔH‖/‖H‖", 12), rpad("ΔT_12", 10), rpad("ΔT_48", 10),
    rpad("err_dual", 10), rpad("into/solid", 12), "insul")
println("-"^96)

for (k, δ) in enumerate((0.20, 0.10, 0.05, 0.02, 0.01, 0.005))
    dad12 = finite_width_laplace_problem(; W, H, a, gap=δ, ndiv_b, ndiv_h, ndiv_crack,
        field=:y, ordem=1, nome="dsw_fw$(k)_12", pontointerno=false)
    chk = check_slit_bc(dad12; a, gap=δ)
    kind, dist = opp_kind(dad12, δ)
    assemble_finite_width_laplace!(dad12; npg=12, gap=δ, threaded=false)
    H12 = copy(dad12.H)
    solve(dad12)
    Δ12, _ = jump_mid_fw(dad12, δ)

    dad48 = finite_width_laplace_problem(; W, H, a, gap=δ, ndiv_b, ndiv_h, ndiv_crack,
        field=:y, ordem=1, nome="dsw_fw$(k)_48", pontointerno=false)
    assemble_finite_width_laplace!(dad48; npg=48, gap=δ, threaded=false)
    relH = norm(H12 - dad48.H) / (norm(dad48.H) + 1e-30)
    solve(dad48)
    Δ48, _ = jump_mid_fw(dad48, δ)
    npg★ = npg_finite_width(δ, mean(el.Length for el in dad12.elements))
    errd = abs(Δ48 - Δd) / (abs(Δd) + 1e-30)
    println(
        rpad(string(δ), 8),
        rpad(string(kind), 8),
        rpad(string(round(dist; sigdigits=4)), 10),
        rpad(string(npg★), 6),
        rpad(string(round(relH; sigdigits=3)), 12),
        rpad(string(round(Δ12; sigdigits=4)), 10),
        rpad(string(round(Δ48; sigdigits=4)), 10),
        rpad(string(round(errd; sigdigits=3)), 10),
        rpad("$(chk.n_into_hole)/$(chk.n_into_solid)", 12),
        chk.insulated,
    )
    flush(stdout)
end
println("done")
