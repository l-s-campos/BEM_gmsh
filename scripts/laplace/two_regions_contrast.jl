# Perfectly bonded two-region Laplace with contrasting conductivity.
#
#   julia --project=. scripts/laplace/two_regions_contrast.jl
#
# Exact 1-D conduction on the unit square (T(0)=0, T(1)=1, insulated top/bottom):
#   Tif = kR/(kL+kR) at x=0.5; T linear in each slab.

using DrWatson
@quickactivate :BEM
using Statistics
using LinearAlgebra
using BEM.MultiRegion
include(datadir("Laplace", "two_regions.jl"))

function _rel(dad, kL, kR)
    Tex = [ana_two_layer_T(dad.Nodes[i][1], kL, kR) for i in 1:dad.n]
    e = abs.(dad.T[1:dad.n] .- Tex)
    return norm(dad.T[1:dad.n] .- Tex) / max(norm(Tex), 1e-12), median(e), maximum(e)
end

function _jump(prob)
    j = 0.0
    for ip in prob.interfaces
        Ta = prob.regions[ip.reg_a].T[ip.node_a]
        Tb = prob.regions[ip.reg_b].T[ip.node_b]
        j = max(j, abs(Ta - Tb))
    end
    return j
end

function run_case(kL, kR; ndiv=12, npg=12)
    msh = mesh_two_regions(ndiv=ndiv, show=false, nome="two_reg_$(kL)_$(kR)")
    prob = load_two_regions(msh, Laplace(kL), Laplace(kR))
    pair_interfaces!(prob)
    assemble_multiregion(prob; npg=npg)
    solve_multiregion!(prob)
    dadL, dadR = prob.regions
    relL, medL, maxL = _rel(dadL, kL, kR)
    relR, medR, maxR = _rel(dadR, kL, kR)
    if_idx = findall(i -> abs(dadL.Nodes[i][1] - 0.5) < 1e-8, 1:dadL.n)
    Tif_num = median(dadL.T[if_idx])
    Tif_ex = kR / (kL + kR)
    qif = Float64[]
    for ip in prob.interfaces
        push!(qif, prob.regions[ip.reg_a].q[ip.node_a])
        push!(qif, prob.regions[ip.reg_b].q[ip.node_b])
    end
    qex = 2 * kL * kR / (kL + kR)   # |q| = k ∂T/∂x; package q = -k ∂T/∂n
    println("kL=$kL  kR=$kR  pairs=$(length(prob.interfaces))")
    println("  Tif num=$(round(Tif_num; sigdigits=4))  exact=$(round(Tif_ex; sigdigits=4))")
    println("  relL=$(round(relL; sigdigits=3))  medL=$(round(medL; sigdigits=3))  maxL=$(round(maxL; sigdigits=3))")
    println("  relR=$(round(relR; sigdigits=3))  medR=$(round(medR; sigdigits=3))  maxR=$(round(maxR; sigdigits=3))")
    println("  max |Ta-Tb|=$(round(_jump(prob); sigdigits=3))")
    println("  median |q_if|=$(round(median(abs.(qif)); sigdigits=3))  exact |q|=$(round(qex; sigdigits=3))")
    return (; relL, relR, Tif_num, Tif_ex, jump=_jump(prob))
end

println("="^60)
println(" Multi-region Laplace, perfect interface, contrasting k")
println("="^60)
run_case(1.0, 1.0)
run_case(1.0, 4.0)
run_case(4.0, 1.0)
println("done")
