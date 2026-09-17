# Side-by-side contour: Example 2D discontinuous Dirichlet vs Fourier series.
using LinearAlgebra
using StaticArrays
using Printf
using Statistics
using Plots
using LaTeXStrings
gr()

include(joinpath(@__DIR__, "orthotropic_dibem_ex1_grid_ex3_rot.jl"))

k1, k2 = 2.0, 0.5
K = @SMatrix [k1 0.0; 0.0 k2]
nxg = 40
nel = 80
path = quadrado(; nome="ex2d_contour", ndiv=nel ÷ 4 + 1, show=false, ordem=1)
dad = make_dad(path, :iso, K)
internal_grid!(dad, nxg, nxg; d_min=0, layout=:cell)
apply_dirichlet!(dad, (x, y) -> y >= 1 - 1e-8 ? 1.0 : 0.0)
assemble!(dad; npg=NPG, threaded=true)
DIBEM(dad; rbf=RBF, threaded=true)
solve_anisotropic_ibp!(dad, K; rbf=RBF, npg=NPG, nlocal=min(21, dad.nt - 1))

bb = BEM._boundary_bbox(dad)
xs = BEM._axis_nodes(nxg, bb.xmin, bb.xmax, :cell)
ys = BEM._axis_nodes(nxg, bb.ymin, bb.ymax, :cell)
Ti = dad.T[dad.n+1:dad.n+dad.ni]
@assert length(Ti) == nxg * nxg
Tbem = reshape(Ti, nxg, nxg)
Tana = [example2_discontinuous(x, y; k1=k1, k2=k2) for x in xs, y in ys]
cl = extrema(vcat(vec(Tbem), vec(Tana)))
levels = range(cl[1], cl[2]; length=14)
kw = (levels=levels, clims=cl, fill=true, linewidth=0, aspect_ratio=:equal,
    xlabel=L"x", ylabel=L"y", xlims=(0, 1), ylims=(0, 1), color=:viridis)

out = "/home/lsc/OneDrive/artigos/escritos/2026/DIBEM orto/figs/fig2d-contour.png"
plt = plot(
    contourf(xs, ys, Tbem'; kw..., title="BEM (iso DIBEM)", colorbar=false),
    contourf(xs, ys, Tana'; kw..., title="analytical (Fourier)", colorbar=true),
    layout=@layout([a{0.46w} b{0.54w}]),
    size=(1100, 500),
    left_margin=5Plots.mm,
    right_margin=6Plots.mm,
    bottom_margin=5Plots.mm,
    plot_title="Example 2D  (u=1 on top, u=0 elsewhere)",
    dpi=220,
)
savefig(plt, out)
@printf("ni=%d  max|Tbem-Tana|=%.3e  RMS=%.3e\n",
    dad.ni, maximum(abs, Tbem .- Tana), sqrt(mean(abs2, Tbem .- Tana)))
println("wrote ", out)
