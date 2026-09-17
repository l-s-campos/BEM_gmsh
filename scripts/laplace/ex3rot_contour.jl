# Example 3 rotated K: side-by-side iso DIBEM vs anisotropic FS contours.
using LinearAlgebra
using StaticArrays
using Printf
using Statistics
using Plots
using LaTeXStrings
gr()

include(joinpath(@__DIR__, "orthotropic_dibem_ex1_grid_ex3_rot.jl"))

k1, k2, θdeg = 5.0, 0.5, 30.0
K = rotate_K(k1, k2, deg2rad(θdeg))
nxg = 48
lc = 0.05
path = mesh_square_hole(; nome="ex3rot_contour", lc=lc)

function fill_grid(dad, xs, ys)
    T = fill(NaN, length(xs), length(ys))
    k = 1
    n = dad.n
    Ti = dad.T
    @inbounds for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        point_in_domain(dad, Point2D(x, y)) || continue
        T[ix, iy] = Ti[n + k]
        k += 1
    end
    return T
end

function solve_fs(fs)
    dad = make_dad(path, fs, K)
    internal_grid!(dad, nxg, nxg; d_min=0, layout=:cell)
    apply_ex3_K!(dad, K)
    if fs === :iso
        solve_iso_dibem!(dad, K; hole=true)
    else
        solve_aniso_fs!(dad)
    end
    return dad
end

println("K =\n", K)
dad_iso = solve_fs(:iso)
dad_an = solve_fs(:aniso)
bb = BEM._boundary_bbox(dad_iso)
xs = BEM._axis_nodes(nxg, 0.0, 1.0, :cell)
ys = BEM._axis_nodes(nxg, 0.0, 1.0, :cell)
Tiso = fill_grid(dad_iso, xs, ys)
Tan = fill_grid(dad_an, xs, ys)
vals = filter(isfinite, vcat(vec(Tiso), vec(Tan)))
cl = extrema(vals)
levels = range(cl[1], cl[2]; length=16)
kw = (levels=levels, clims=cl, fill=true, linewidth=0, aspect_ratio=:equal,
    xlabel=L"x", ylabel=L"y", xlims=(0, 1), ylims=(0, 1), color=:viridis)

θ = range(0, 2π; length=80)
hx, hy = 0.5 .+ 0.25 .* cos.(θ), 0.5 .+ 0.25 .* sin.(θ)

out = "/home/lsc/OneDrive/artigos/escritos/2026/DIBEM orto/figs/fig3rot-contour.png"
p1 = contourf(xs, ys, Tiso'; kw..., title="iso DIBEM (IBP)", colorbar=false)
plot!(p1, hx, hy; seriestype=:shape, fillcolor=:white, linecolor=:black, lw=1.1, label="")
p2 = contourf(xs, ys, Tan'; kw..., title="aniso FS", colorbar=true)
plot!(p2, hx, hy; seriestype=:shape, fillcolor=:white, linecolor=:black, lw=1.1, label="")
plt = plot(p1, p2, layout=@layout([a{0.46w} b{0.54w}]), size=(1100, 500),
    left_margin=5Plots.mm, right_margin=6Plots.mm, bottom_margin=5Plots.mm,
    plot_title=L"Example 3  $K=R_{30^\circ}\mathrm{diag}(5,0.5)R_{30^\circ}^T$",
    dpi=220)
savefig(plt, out)
finite = isfinite.(Tiso) .& isfinite.(Tan)
@printf("n=%d ni=%d  max|iso-aniso|=%.3e  RMS=%.3e\n",
    dad_iso.n, dad_iso.ni,
    maximum(abs, Tiso[finite] .- Tan[finite]),
    sqrt(mean(abs2, Tiso[finite] .- Tan[finite])))
println("wrote ", out)
