using DrWatson
@quickactivate :BEM
using Plots, DelimitedFiles
get(ENV, "GKSwstype", nothing) === nothing && (ENV["GKSwstype"] = "100")
outdir = projectdir("plots", "schultz_cfp")
raw = readdlm(joinpath(outdir, "journal_misaligned_fvm_field.tsv"), '\t', Float64; skipstart=1)
xs = sort(unique(raw[:, 1])); ys = sort(unique(raw[:, 2]))
nx, ny = length(xs), length(ys)
P = zeros(nx, ny); Th = zeros(nx, ny)
xd = Dict(x => i for (i, x) in enumerate(xs))
yd = Dict(y => j for (j, y) in enumerate(ys))
for k in 1:size(raw, 1)
    i = xd[raw[k, 1]]; j = yd[raw[k, 2]]
    P[i, j] = raw[k, 3]; Th[i, j] = raw[k, 4]
end
cfp = readdlm(joinpath(outdir, "journal_misaligned_field.tsv"), '\t', Float64; skipstart=1)
a1 = heatmap(xs, ys, P'; xlabel="x₁ (m)", ylabel="x₂ (m)", title="p (MPa)  FVM + CFP nodes",
    clims=(0.08, 0.38), colorbar=true, aspect_ratio=:equal)
scatter!(a1, cfp[:, 1], cfp[:, 2]; zcolor=cfp[:, 3], marker=:circle, markersize=3,
    markerstrokewidth=0, label="", clims=(0.08, 0.38))
a2 = heatmap(xs, ys, Th'; xlabel="x₁ (m)", ylabel="x₂ (m)", title="θ  FVM",
    clims=(0.2, 1.0), colorbar=true, aspect_ratio=:equal)
fig = plot(a1, a2; layout=(2, 1), size=(760, 560))
png = joinpath(outdir, "journal_misaligned_field.png")
savefig(fig, png)
println("wrote ", png)
