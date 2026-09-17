# Plot Example 2D BEM mesh (boundary elements + cell-centred internals).
using LinearAlgebra
using StaticArrays
using Plots
gr()

include(joinpath(@__DIR__, "orthotropic_dibem_ex1_grid_ex3_rot.jl"))

K = @SMatrix [2.0 0.0; 0.0 0.5]
nxg, nel = 40, 80
path = quadrado(; nome="ex2d_mesh", ndiv=nel ÷ 4 + 1, show=false, ordem=1)
dad = make_dad(path, :iso, K)
internal_grid!(dad, nxg, nxg; d_min=0, layout=:cell)

out = "/home/lsc/OneDrive/artigos/escritos/2026/DIBEM orto/figs/fig2d-mesh.png"
plt = plot(; size=(720, 720), dpi=220, aspect_ratio=:equal, framestyle=:box,
    xlabel="x", ylabel="y", xlims=(-0.05, 1.05), ylims=(-0.05, 1.05),
    title="Example 2D mesh  (80 BE, $(dad.ni) internals)", legend=:topright)
for el in dad.elements
    g = isempty(el.geo) ? collect(dad.Nodes[el.index]) : el.geo
    plot!(plt, [p[1] for p in g], [p[2] for p in g];
        color=:black, lw=1.4, label="")
end
ix = [p[1] for p in dad.internalNodes]
iy = [p[2] for p in dad.internalNodes]
scatter!(plt, ix, iy; color=:steelblue, ms=2.2, msw=0, label="internal")
bx = [p[1] for p in dad.Nodes]
by = [p[2] for p in dad.Nodes]
scatter!(plt, bx, by; color=:black, ms=3.5, msw=0, label="collocation")
savefig(plt, out)
println("n=$(dad.n)  ni=$(dad.ni)  nel=$(length(dad.elements))")
println("wrote ", out)
