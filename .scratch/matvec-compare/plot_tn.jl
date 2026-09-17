using Plots, LaTeXStrings, DelimitedFiles
gr()
default(fontfamily="Computer Modern", linewidth=2.0, framestyle=:box,
    grid=true, gridalpha=0.25, dpi=160, legendfontsize=9, guidefontsize=11,
    tickfontsize=9, size=(760, 480))
outdir = @__DIR__
data, _ = readdlm(joinpath(outdir, "results.csv"), ','; header=true)
methods = String.(data[:, 1])
ns = Int.(data[:, 2])
tmv = Float64.(data[:, 5])
style = Dict(
    "dense" => (:black, :circle, "dense"),
    "hmatrix" => (:crimson, :utriangle, "H-matrix"),
    "h2" => (:royalblue, :diamond, "H2 (NNCA)"),
    "fmm" => (:seagreen, :xcross, "FMM"),
)
plt = plot(; xscale=:log10, xlabel=L"N", ylabel="matvec time / N  [ns]",
    title="Cost per unknown (flat = linear)", legend=:topright,
    ylims=(0, 2500))
for m in ("dense", "hmatrix", "h2", "fmm")
    idx = findall(==(m), methods)
    col, mk, lab = style[m]
    plot!(plt, ns[idx], 1e9 .* tmv[idx] ./ ns[idx];
        marker=mk, color=col, markersize=5, label=lab)
end
savefig(plt, joinpath(outdir, "fig_tn.png"))

# structure from h2_cost.jl (N, r_avg, m2l/N, near/N)
Nn = [1024, 4096, 16384, 65536]
ravg = [13.8, 14.4, 14.6, 14.7]
m2ln = [228.6, 344.1, 418.4, 462.3]
nearn = [121.0, 132.2, 138.1, 141.0]
ilmean_proxy = m2ln ./ (ravg .^ 2)  # not exact
plt2 = plot(; xscale=:log10, xlabel=L"N", ylabel="entries / N",
    title="H2: bounded rank, M2L/N saturating", legend=:topleft)
plot!(plt2, Nn, m2ln; marker=:diamond, color=:royalblue, label="M2L entries / N")
plot!(plt2, Nn, nearn; marker=:circle, color=:gray, label="near entries / N")
plt2b = twinx(plt2)
plot!(plt2b, Nn, ravg; marker=:utriangle, color=:crimson, ylabel="avg rank",
    legend=:bottomright, label="avg rank", ylims=(0, 30))
savefig(plt2, joinpath(outdir, "fig_h2_structure.png"))
println("wrote fig_tn.png fig_h2_structure.png")
