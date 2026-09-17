# Pin-on-disc sliding wear: dense / FFT / FMM / H-matrix.
# Frictionless isotropic Archard, Kzz only. Meshes up to ~1e5 DOFs (N²).
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra
using Printf
using Statistics
using Plots
gr()

const FIG = plotsdir("julia_lerma")
mkpath(FIG)

const METHODS = (:dense, :fft, :fmm, :hmatrix)
# odd N so the Hertz centre sits on a node
const NS = (17, 33, 65, 129, 257, 317)   # ndofs = 289 … 100489
const NSTEPS = 4
const ΔS = 5.0
const TOL = 1e-6
const MAXITER = 250
const DENSE_MAX_NDOFS = 8_000            # dense K is ndofs²; skip beyond ~0.5 GB

function _prep_kwargs(method, N)
    method === :hmatrix && return (; nmax=32, atol=1e-10)
    method === :fmm && return (; eps=1e-8, nmax=-1)
    return NamedTuple()
end

function _run_wear(N, method)
    G = G_from_E(PIN.E, PIN.ν)
    x, hs = square_mesh(N, PIN.L, G, PIN.ν, G, PIN.ν)
    grid = make_grid(x, x, hs, sphere_gap(x, x, PIN.R))
    kw = _prep_kwargs(method, N)
    t0 = time_ns()
    prep = precompute_kernels(N, N, hs; components=(Kzz,), method=method, kw...)
    t_build = (time_ns() - t0) / 1e9
    st = init_state(grid)
    law = isotropic_law(0.0, PIN.i)
    t0 = time_ns()
    hist = sliding_wear_steps!(st, grid, prep, law, PIN.δ, ΔS, NSTEPS;
                               tol=TOL, maxiter=MAXITER)
    t_wear = (time_ns() - t0) / 1e9
    return (; N, ndofs=N * N, method, t_build, t_wear, t_total=t_build + t_wear,
            w=hist.wmax[end], skipped=false)
end

println("warmup N=9 …")
flush(stdout)
for m in METHODS
    _run_wear(9, m)
    @printf "  %s\n" m
    flush(stdout)
end

rows = NamedTuple[]
tsv = joinpath(FIG, "pin_wear_backends.tsv")
open(tsv, "w") do io
    println(io, "method\tN\tndofs\tt_build_s\tt_wear_s\tt_total_s\twmax_mm\tskipped")
end

println("="^78)
@printf "%-10s %6s %8s %10s %10s %10s %12s\n" "method" "N" "ndofs" "t_build" "t_wear" "t_total" "wmax_mm"
println("-"^78)
flush(stdout)

for N in NS
    ndofs = N * N
    for m in METHODS
        if m === :dense && ndofs > DENSE_MAX_NDOFS
            r = (; N, ndofs, method=m, t_build=NaN, t_wear=NaN, t_total=NaN,
                 w=NaN, skipped=true)
            @printf "%-10s %6d %8d %10s %10s %10s %12s  (skip dense)\n" String(m) N ndofs "-" "-" "-" "-"
            flush(stdout)
        else
            r = try
                _run_wear(N, m)
            catch e
                @printf "%-10s %6d %8d  FAILED  %s\n" String(m) N ndofs sprint(showerror, e)
                flush(stdout)
                (; N, ndofs, method=m, t_build=NaN, t_wear=NaN, t_total=NaN,
                 w=NaN, skipped=true)
            end
            if !r.skipped
                @printf "%-10s %6d %8d %10.3f %10.3f %10.3f %12.4e\n" String(m) N r.ndofs r.t_build r.t_wear r.t_total r.w
                flush(stdout)
            end
        end
        push!(rows, r)
        open(tsv, "a") do io
            @printf io "%s\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.8e\t%s\n" r.method r.N r.ndofs r.t_build r.t_wear r.t_total r.w r.skipped
        end
    end
end
println("="^78)
println("wrote ", tsv)

by = Dict(m => filter(r -> r.method === m && !r.skipped && isfinite(r.t_total), rows)
          for m in METHODS)
cols = Dict(:dense=>:black, :fft=>:steelblue, :fmm=>:darkorange,
            :hmatrix=>:seagreen)
mk = Dict(:dense=>:square, :fft=>:circle, :fmm=>:diamond,
          :hmatrix=>:utriangle)

default(linewidth=2, markersize=7, legendfontsize=9, tickfontsize=10,
        guidefontsize=12, grid=true, gridalpha=0.3, framestyle=:box)

nmin = minimum(r.ndofs for r in rows)
nmax = maximum(r.ndofs for r in rows)
plt = plot(size=(720, 540), dpi=150, xscale=:log10, yscale=:log10,
           xlabel="ndofs (N²)", ylabel="time (s)",
           title="Pin-on-disc wear · Kzz backends",
           xlims=(nmin / 1.3, nmax * 1.3))
for m in METHODS
    rs = by[m]
    isempty(rs) && continue
    plot!(plt, [r.ndofs for r in rs], [r.t_total for r in rs];
          color=cols[m], marker=mk[m], label=String(m))
end
fft_rows = by[:fft]
if !isempty(fft_rows)
    nref = [fft_rows[1].ndofs, fft_rows[end].ndofs]
    t1 = fft_rows[1].t_total
    plot!(plt, nref, t1 .* (nref ./ nref[1]), color=:gray, ls=:dash, label="∼ n")
    plot!(plt, nref, t1 .* (nref ./ nref[1]).^2, color=:gray, ls=:dot, label="∼ n²")
end

png = joinpath(FIG, "pin_wear_backends.png")
pdf = joinpath(FIG, "pin_wear_backends.pdf")
savefig(plt, png)
savefig(plt, pdf)
println("wrote ", png)
println("wrote ", pdf)
