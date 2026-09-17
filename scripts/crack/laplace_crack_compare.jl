# Dual BEM vs finite-width slit (Laplace, insulated centre crack)
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using BEM.Crack

W, H, a = 5.0, 10.0, 1.0
ndiv_b, ndiv_h, ndiv_crack = 8, 12, 12
npg = 16

println("="^60)
println(" Laplace crack: dual BEM vs finite-width slit")
println("="^60)

dad = dual_laplace_problem(; W, H, a, ndiv_b, ndiv_h, ndiv_crack,
    field=:y, bc=:insulated, ordem=1, nome="cmp_dual", pontointerno=false)
solve_dual_laplace!(dad; npg=npg, threaded=false)
nodesA = crack_face_nodes(dad; face=2)
xs = Float64[]
Δd = Float64[]
Δa = Float64[]
for i in nodesA
    dad.twin[i] == 0 && continue
    x = dad.Nodes[i][1]
    push!(xs, x)
    push!(Δd, abs(crack_jump(dad, i)))
    push!(Δa, analytical_insulated_jump(x, a; G=1.0))
end
rel_ana = norm(Δd .- Δa) / (norm(Δa) + 1e-30)
println("dual nodes = ", dad.n, "  face A = ", length(nodesA))
println("  L2 jump vs infinite-plate  rel = ", round(rel_ana; sigdigits=4))

gaps = (0.10, 0.04, 0.015)
println()
println("  x          dual       ana        ", join(("δ=$(δ)" for δ in gaps), "   "))
perm = sortperm(xs)
function fw_jump_at(dad_fw, xq)
    top = findall(i -> dad_fw.Nodes[i][2] > 0, 1:dad_fw.n)
    bot = findall(i -> dad_fw.Nodes[i][2] < 0, 1:dad_fw.n)
    (isempty(top) || isempty(bot)) && return NaN
    it = top[argmin(abs.(getindex.(dad_fw.Nodes[top], 1) .- xq))]
    ib = bot[argmin(abs.(getindex.(dad_fw.Nodes[bot], 1) .- xq))]
    return abs(dad_fw.T[it] - dad_fw.T[ib])
end
Δfw = Vector{Vector{Float64}}()
for (k, δ) in enumerate(gaps)
    dadw = finite_width_laplace_problem(; W, H, a, gap=δ, ndiv_b, ndiv_h, ndiv_crack,
        n_cap=3, field=:y, ordem=1, nome="cmp_fw$k", pontointerno=false)
    assemble_finite_width_laplace!(dadw; gap=δ, threaded=false)
    solve(dadw)
    push!(Δfw, [fw_jump_at(dadw, x) for x in xs])
    err = norm(Δfw[k] .- Δd) / (norm(Δd) + 1e-30)
    println("  δ/a = ", δ / a, "  ‖ΔT_fw − ΔT_dual‖ / ‖ΔT_dual‖ = ", round(err; sigdigits=4))
end

println()
for j in perm
    abs(xs[j]) > 0.85a && continue
    vals = join((string(round(Δfw[k][j]; sigdigits=4)) for k in eachindex(gaps)), "   ")
    println("  ",
        rpad(string(round(xs[j]; digits=3)), 10),
        rpad(string(round(Δd[j]; sigdigits=4)), 11),
        rpad(string(round(Δa[j]; sigdigits=4)), 11),
        vals)
end
println("Done.")
