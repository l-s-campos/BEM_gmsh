# Isolated Kzz matvec timings (no Uzawa). FFT / H / FMM.
# H is timed serial. FMM is timed at default nmax=400 (eps=1e-8)
# and at nmax=32 / nmax=200 (CHANGELOG sweet spot).
using DrWatson
@quickactivate :BEM
using BEM.Contact
using BEM.HMatrices
using BEM.FMM
using LinearAlgebra
using Printf
using Statistics
using Plots
gr()

const FIG = plotsdir("julia_lerma")
mkpath(FIG)

const NS = (33, 65, 129, 257)
const NWARM = 8
const NRUN = 25

function _hmat_block_stats(H::HMatrix)
    L = collect(leaves(H))
    nfar = count(isadmissible, L)
    nnear = length(L) - nfar
    rmax = 0
    for leaf in L
        isadmissible(leaf) || continue
        rmax = max(rmax, rank(HMatrices.data(leaf)))
    end
    return nfar, nnear, rmax
end

function _stats(op, method)
    if method === :fft
        return (; cr=NaN, rmax=0, extra="fft convolution")
    elseif method === :hmatrix
        nfar, nnear, rmax = _hmat_block_stats(op)
        return (; cr=compression_ratio(op), rmax,
                extra="$(nfar) far, $(nnear) dense")
    elseif method in (:fmm, :fmm32, :fmm200)
        nmax = method === :fmm32 ? 32 : method === :fmm200 ? 200 : -1
        plan = FMM.build_laplace3d_plan(op.F.points; eps=1e-8, nmax=nmax, full_fmm=true)
        return (; cr=NaN, rmax=plan.p,
                extra="p=$(plan.p) nmax=$(plan.nmax) m2l=$(length(plan.m2l_jobs)) p2p=$(length(plan.p2p_jobs)) leaves=$(length(plan.leaf_nodes))")
    else
        error(method)
    end
end

function _assemble(N, method)
    G = G_from_E(PIN.E, PIN.ν)
    _, hs = square_mesh(N, PIN.L, G, PIN.ν, G, PIN.ν)
    if method === :fft
        return build_pohrt_operator(hs, N, N, Kzz; method=:fft), hs
    elseif method === :hmatrix
        return build_pohrt_operator(hs, N, N, Kzz; method=:hmatrix, nmax=32, atol=1e-10), hs
    elseif method === :fmm
        return build_pohrt_operator(hs, N, N, Kzz; method=:fmm, eps=1e-8, nmax=-1), hs
    elseif method === :fmm32
        return build_pohrt_operator(hs, N, N, Kzz; method=:fmm, eps=1e-8, nmax=32), hs
    elseif method === :fmm200
        return build_pohrt_operator(hs, N, N, Kzz; method=:fmm, eps=1e-8, nmax=200), hs
    else
        error(method)
    end
end

function _apply!(y, op, x, method)
    if method === :fft
        fc_forward!(reshape(y, op.nx, op.ny), reshape(x, op.nx, op.ny), Kzz, op)
    elseif method === :hmatrix
        mul!(y, op, x; threads=false)
    else
        mul!(y, op, x)
    end
    return y
end

function _time_apply(op, x, method)
    y = zeros(length(x))
    for _ in 1:NWARM
        _apply!(y, op, x, method)
    end
    t0 = time_ns()
    for _ in 1:NRUN
        _apply!(y, op, x, method)
    end
    t = (time_ns() - t0) / NRUN / 1e9
    return t, y
end

println("Julia threads = ", Threads.nthreads())
println("H-matrix mul! default threads = ", HMatrices.use_threads())
flush(stdout)

# compile
for m in (:fft, :hmatrix, :fmm)
    op, _ = _assemble(17, m)
    _time_apply(op, rand(17 * 17), m)
end

# Threaded H mul! hits ConcurrencyViolationError in RkMatrix._rk_tmp! (shared
# _RK_TMP push! under @threads). Wear and this script use serial H apply.
const METHODS = (:fft, :hmatrix, :fmm, :fmm32, :fmm200)
rows = []
println("="^100)
@printf "%-12s %6s %8s %12s %10s %8s  %s\n" "method" "N" "ndofs" "matvec_s" "cr" "rank" "notes"
println("-"^100)
flush(stdout)

for N in NS
    n = N * N
    x = rand(n)
    ops = Dict{Symbol,Any}()
    hs_ref = nothing
    for m in (:fft, :hmatrix, :fmm, :fmm32, :fmm200)
        op, hs = _assemble(N, m)
        ops[m] = op
        hs_ref = hs
    end
    yref = nothing
    for m in METHODS
        t, y = _time_apply(ops[m], x, m)
        st = _stats(ops[m], m)
        rel = ""
        if m === :fft
            yref = copy(y)
        else
            rel = @sprintf " rel=%.3e" norm(y - yref) / norm(yref)
        end
        extra = st.extra * rel
        @printf "%-12s %6d %8d %12.4e %10.2f %8d  %s\n" String(m) N n t st.cr st.rmax extra
        flush(stdout)
        push!(rows, (; method=m, N, ndofs=n, t, cr=st.cr, rmax=st.rmax, extra))
    end
end
println("="^100)

tsv = joinpath(FIG, "pin_matvec_backends.tsv")
open(tsv, "w") do io
    println(io, "method\tN\tndofs\tmatvec_s\tcr\trank\tnotes")
    for r in rows
        @printf io "%s\t%d\t%d\t%.8e\t%.6f\t%d\t%s\n" r.method r.N r.ndofs r.t r.cr r.rmax r.extra
    end
end
println("wrote ", tsv)

cols = Dict(:fft=>:steelblue, :fmm=>:darkorange, :fmm32=>:orangered,
            :fmm200=>:goldenrod, :hmatrix=>:seagreen)
mk = Dict(:fft=>:circle, :fmm=>:diamond, :fmm32=>:diamond, :fmm200=>:diamond,
          :hmatrix=>:utriangle)
ls = Dict(:fft=>:solid, :fmm=>:solid, :fmm32=>:dash, :fmm200=>:dot,
          :hmatrix=>:solid)
default(linewidth=2, markersize=7, legendfontsize=8, tickfontsize=10,
        guidefontsize=12, grid=true, gridalpha=0.3, framestyle=:box)
plt = plot(size=(760, 560), dpi=150, xscale=:log10, yscale=:log10,
           xlabel="ndofs (N²)", ylabel="matvec time (s)",
           title="Kzz matvec only  ($(Threads.nthreads()) threads)")
for m in METHODS
    rs = filter(r -> r.method === m, rows)
    plot!(plt, [r.ndofs for r in rs], [r.t for r in rs];
          color=cols[m], marker=mk[m], ls=ls[m], label=String(m))
end
n0, n1 = NS[1]^2, NS[end]^2
t0 = filter(r -> r.method === :fft && r.N == NS[1], rows)[1].t
plot!(plt, [n0, n1], t0 .* ([n0, n1] ./ n0), color=:gray, ls=:dash, label="∼ n")
plot!(plt, [n0, n1], t0 .* ([n0, n1] ./ n0).^2, color=:gray, ls=:dot, label="∼ n²")
png = joinpath(FIG, "pin_matvec_backends.png")
savefig(plt, png)
savefig(plt, joinpath(FIG, "pin_matvec_backends.pdf"))
println("wrote ", png)
