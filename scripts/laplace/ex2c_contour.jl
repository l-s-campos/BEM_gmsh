# Side-by-side contour: Example 2C BEM vs u=(x-y)^2
using LinearAlgebra
using StaticArrays
using Printf
using Statistics
using Plots
using LaTeXStrings
gr()

include(joinpath(@__DIR__, "orthotropic_dibem_ex1_grid_ex3_rot.jl"))

k1, k2 = 1.0, 1.0  # unused; K is off-diagonal
K = @SMatrix [1.0 1.0; 1.0 1.0]
nxg = 40
nel = 80
path = quadrado(; nome="ex2c_contour", ndiv=nel ÷ 4 + 1, show=false, ordem=1)
dad = make_dad(path, :iso, K)
internal_grid!(dad, nxg, nxg; d_min=0, layout=:cell)
apply_dirichlet!(dad, example2_anisotropic)
assemble!(dad; npg=NPG, threaded=true)
DIBEM(dad; rbf=RBF_QUAD, threaded=true)
solve_anisotropic_dibem!(dad, K; rbf=RBF_QUAD, npg=NPG, nlocal=min(21, dad.nt - 1), kiso=1.0)

bb = BEM._boundary_bbox(dad)
xs = BEM._axis_nodes(nxg, bb.xmin, bb.xmax, :cell)
ys = BEM._axis_nodes(nxg, bb.ymin, bb.ymax, :cell)
# internals stored x-fastest, then y (see internal_grid)
Ti = dad.T[dad.n+1:dad.n+dad.ni]
@assert length(Ti) == nxg * nxg
Tbem = reshape(Ti, nxg, nxg)          # Tbem[ix, iy]
Tana = [example2_anisotropic(x, y) for x in xs, y in ys]
cl = extrema(vcat(vec(Tbem), vec(Tana)))
levels = range(cl[1], cl[2]; length=14)
kw = (levels=levels, clims=cl, fill=true, linewidth=0, aspect_ratio=:equal,
    xlabel=L"x", ylabel=L"y", xlims=(0, 1), ylims=(0, 1),
    color=:viridis)

out = "/home/lsc/OneDrive/artigos/escritos/2026/DIBEM orto/figs/fig2c-contour.png"
plt = plot(
    contourf(xs, ys, Tbem'; kw..., title="BEM (iso DIBEM)", colorbar=false),
    contourf(xs, ys, Tana'; kw..., title=L"analytical $(x-y)^2$", colorbar=true),
    layout=@layout([a{0.46w} b{0.54w}]),
    size=(1100, 500),
    left_margin=5Plots.mm,
    right_margin=6Plots.mm,
    bottom_margin=5Plots.mm,
    plot_title="Example 2C",
    dpi=220,
)
savefig(plt, out)
@printf("ni=%d  max|Tbem-Tana|=%.3e  RMS=%.3e\n",
    dad.ni, maximum(abs, Tbem .- Tana), sqrt(mean(abs2, Tbem .- Tana)))
println("wrote ", out)
