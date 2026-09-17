# Plot Schultz journal centerline from existing TSVs (no re-solve).
using DrWatson
@quickactivate :BEM
using Plots
using DelimitedFiles

get(ENV, "GKSwstype", nothing) === nothing && (ENV["GKSwstype"] = "100")
const OUTDIR = projectdir("plots", "schultz_cfp")

function load_tsv(path)
    raw = readdlm(path, '\t', Float64; skipstart=1)
    return raw
end

function centerline_cfp(field)
    y0 = 0.015
    bins = Dict{Float64,NTuple{4,Float64}}()
    for i in 1:size(field, 1)
        x, y, p, th = field[i, 1], field[i, 2], field[i, 3], field[i, 4]
        abs(y - y0) <= 0.0036 || continue
        xk = round(x; digits=4)
        dy = abs(y - y0)
        if !haskey(bins, xk) || dy < bins[xk][4]
            bins[xk] = (x, p, th, dy)
        end
    end
    xs = Float64[]; ps = Float64[]; ths = Float64[]
    for xk in sort(collect(keys(bins)))
        x, p, th, _ = bins[xk]
        push!(xs, x); push!(ps, p); push!(ths, th)
    end
    return xs, ps, ths
end

field = load_tsv(joinpath(OUTDIR, "journal_eccentric_field.tsv"))
xs, ps, ths = centerline_cfp(field)
open(joinpath(OUTDIR, "journal_eccentric.tsv"), "w") do io
    println(io, "x_m\tp_MPa\ttheta")
    for i in eachindex(xs)
        println(io, xs[i], '\t', ps[i], '\t', ths[i])
    end
end

fvm = load_tsv(joinpath(OUTDIR, "journal_eccentric_fvm.tsv"))
xf, pf, thf = fvm[:, 1], fvm[:, 2], fvm[:, 3]

ax1 = plot(xf, pf; color=:black, lw=2, label="FVM Elrod",
    xlabel="x₁ (m)", ylabel="p (MPa)",
    title="Eccentric journal — centerline  x₂ = 0.015 m", legend=:topleft)
plot!(ax1, xs, ps; marker=:diamond, markersize=7, label="CFP-DIBEM")
hline!(ax1, [0.08]; color=:gray, linestyle=:dot, label="p_c")
hline!(ax1, [3.2]; color=:red, linestyle=:dash, label="paper ≈ 3.2 MPa")
ax2 = plot(xf, thf; color=:black, lw=2, label="FVM Elrod",
    xlabel="x₁ (m)", ylabel="θ", title="Liquid ratio", ylim=(0, 1.05),
    legend=:bottomleft)
plot!(ax2, xs, ths; marker=:diamond, markersize=7, label="CFP-DIBEM")
fig = plot(ax1, ax2; layout=(2, 1), size=(760, 640))
png = joinpath(OUTDIR, "journal_eccentric.png")
savefig(fig, png)
println("wrote ", png, "  CFP pmax=", maximum(ps), " at x=", xs[argmax(ps)])
