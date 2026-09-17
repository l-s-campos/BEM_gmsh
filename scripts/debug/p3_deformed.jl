# Deformed profile of the 90° anisotropic ring (thesis Fig. 7.7).
#   julia --project=. scripts/debug/p3_deformed.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.8, framestyle=:box,
    grid=false, dpi=160, legendfontsize=9, tickfontsize=9, guidefontsize=11)

include(datadir("Laplace", "Laplace_dad.jl"))

const Ex, Ey, Gxy = 124.04e3, 10.09e3, 6.03e3
const νyx, ηx, ηy = 0.344, 1.255, -0.031
const P, Ri, Ro = 1000.0, 600.0, 900.0   # mm = 0.6 m / 0.9 m
const MSH = datadir("elastico", "anel_aniso_06_09.msh")
const OUT = joinpath(@__DIR__, "p3_deformed.png")

function apply_ty!(dad)
    for i in 1:dad.n
        x, y = dad.Nodes[i][1], dad.Nodes[i][2]
        if x < 1.0
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 0; dad.BV[2i] = 0.0
        elseif y < 1.0 && x > Ri - 1
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = -P
        else
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = 0.0
        end
    end
    return dad
end

function rotC(C, θdeg)
    θ = θdeg * π / 180
    m, n = cos(θ), sin(θ)
    Rot = @SMatrix [m^2 n^2 2m*n; n^2 m^2 -2m*n; -m*n m*n m^2-n^2]
    return inv(Rot) * C * inv(Rot')
end

function solve_p3(props)
    dad = format2d(MSH, props; tipo=2, pontointerno=false)
    apply_ty!(dad)
    assemble!(dad; npg=16, threaded=false)
    solve(dad)
    return dad
end

function element_xy(dad, u; α=1.0)
    ξ = collect(range(-1.0, 1.0; length=17))
    N, _ = BEM.shapefun(dad.element_type, ξ)
    curves = NTuple{2,Vector{Float64}}[]
    for elem in dad.elements
        P = reduce(hcat, dad.Nodes[elem.index])'   # nN × 2
        U = zeros(length(elem.index), 2)
        for (k, i) in enumerate(elem.index)
            U[k, 1] = u[2i-1]
            U[k, 2] = u[2i]
        end
        C = N * (P .+ α .* U)
        push!(curves, (C[:, 1] ./ 10, C[:, 2] ./ 10))  # mm → cm
    end
    return curves
end

function panel!(plt, dad; α, color, label, first)
    for (k, (x, y)) in enumerate(element_xy(dad, dad.u; α=α))
        plot!(plt, x, y; color=color, label=k == 1 ? label : "")
    end
    return plt
end

props0 = AnisotropicElasticity(lekhnitskii_engineering(Ex, Ey, Gxy, νyx;
    η12_1=ηx, η12_2=ηy))
C0 = props0.params.C
dad0 = solve_p3(props0)
dad90 = solve_p3(AnisotropicElasticity(lekhnitskii_params(rotC(C0, 90))))

function umax_cm(dad)
    return maximum(abs, dad.u) / 10
end

@printf("aligned  max|u|=%.2f cm\n", umax_cm(dad0))
@printf("θ=90°    max|u|=%.2f cm\n", umax_cm(dad90))

# Magnification so the peak displacement is ~25% of Ro (visible, not exploding).
α0 = 0.25 * (Ro / 10) / max(umax_cm(dad0), 1e-6)
α90 = 0.25 * (Ro / 10) / max(umax_cm(dad90), 1e-6)

function ring_plot(dad, α, subtitle)
    plt = plot(;
        size=(620, 640),
        xlabel=L"x\ \mathrm{(cm)}",
        ylabel=L"y\ \mathrm{(cm)}",
        aspect_ratio=:equal,
        legend=:topright,
        title=subtitle,
        xlims=(-5, 125),
        ylims=(-55, 105),
    )
    panel!(plt, dad; α=0.0, color=:gray, label="undeformed", first=true)
    panel!(plt, dad; α=α, color=:firebrick, label="deformed ×$(round(α; digits=1))", first=true)
    # clamp and load marks
    plot!(plt, [0, 0], [Ri, Ro] ./ 10; color=:black, lw=4, label="clamp")
    scatter!(plt, [(Ri + Ro) / 20], [0.0];
        marker=:dtriangle, color=:dodgerblue, markersize=10, label=L"t_y=-P")
    return plt
end

p1 = ring_plot(dad0, α0,
    "EPT, \$E_x\$ along \$x\$  (max \$|u|=$(round(umax_cm(dad0); digits=1))\$ cm)")
p2 = ring_plot(dad90, α90,
    "EPT, \$E_x\$ along \$y\$  (max \$|u|=$(round(umax_cm(dad90); digits=1))\$ cm)")

plt = plot(p1, p2; layout=(1, 2), size=(1240, 640),
    plot_title="90° ring (Fernández / Cordeiro §7.2), \$t_y=-1\$ GPa")
savefig(plt, OUT)
println("wrote ", OUT)
