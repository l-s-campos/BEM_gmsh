using Plots, LaTeXStrings, DelimitedFiles
gr()
default(fontfamily="Computer Modern", linewidth=2.0, framestyle=:box,
    grid=true, gridalpha=0.25, dpi=160, legendfontsize=9, guidefontsize=11,
    tickfontsize=9, size=(760, 480))
outdir = @__DIR__
data, _ = readdlm(joinpath(outdir, "results.csv"), ','; header=true)
methods = String.(data[:, 1])
ns = Int.(data[:, 2])
tasm = Float64.(data[:, 4])
tmv = Float64.(data[:, 5])
err = Float64.(data[:, 7])
style = Dict(
    "dense" => (:black, :circle, "dense"),
    "hmatrix" => (:crimson, :utriangle, "H-matrix"),
    "h2" => (:royalblue, :diamond, "H2 (NNCA)"),
    "fmm" => (:seagreen, :xcross, "FMM"),
)
order = ["dense", "hmatrix", "h2", "fmm"]
nmin, nmax = extrema(ns)
function series(m)
    idx = findall(==(m), methods)
    return ns[idx], tmv[idx], tasm[idx], err[idx]
end

function plot_all()
plt = plot(; xscale=:log10, yscale=:log10, xlabel=L"N", ylabel="matvec time [s]",
    title="2D Laplace " * L"\log|x-y|" * "  (unit square)", legend=:bottomright)
tguide = nothing
nguide = nothing
for m in order
    n, t, _, _ = series(m)
    col, mk, lab = style[m]
    plot!(plt, n, t; marker=mk, color=col, markersize=5, label=lab)
    if tguide === nothing && m != "dense"
        tguide = t[1]
        nguide = Float64(n[1])
    end
end
ng = 10 .^ range(log10(nmin), log10(nmax); length=48)
plot!(plt, ng, tguide .* (ng ./ nguide); ls=:dash, lc=:gray, lw=0.9, label=L"N")
plot!(plt, ng, tguide .* (ng .* log.(ng)) ./ (nguide * log(nguide));
    ls=:dot, lc=:gray, lw=0.9, label=L"N\log N")
plot!(plt, ng, tguide .* (ng ./ nguide) .^ 2; ls=:dashdot, lc=:gray, lw=0.9, label=L"N^2")
savefig(plt, joinpath(outdir, "fig_matvec.png"))

plt_e = plot(; xscale=:log10, yscale=:log10, xlabel=L"N",
    ylabel=L"\|y-y_{\mathrm{ref}}\|/\|y_{\mathrm{ref}}\|",
    title="relative error (vs dense while available, else vs H-matrix)",
    legend=:bottomright, ylims=(1e-16, 1e-2))
for m in ("hmatrix", "h2", "fmm")
    n, _, _, e = series(m)
    mask = e .> 0
    col, mk, lab = style[m]
    plot!(plt_e, n[mask], max.(e[mask], 1e-16);
        marker=mk, color=col, markersize=5, label=lab)
end
savefig(plt_e, joinpath(outdir, "fig_error.png"))

plt_a = plot(; xscale=:log10, yscale=:log10, xlabel=L"N", ylabel="assembly time [s]",
    title="assembly (host)", legend=:bottomright)
for m in order
    n, _, ta, _ = series(m)
    col, mk, lab = style[m]
    plot!(plt_a, n, ta; marker=mk, color=col, markersize=5, label=lab)
end
savefig(plt_a, joinpath(outdir, "fig_assemble.png"))
println("replotted")
end
plot_all()
