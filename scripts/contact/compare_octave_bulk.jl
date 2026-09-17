# Julia twin of dad_Contato_Bulk vs Octave Contato results.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.Contact, BEM.MultiRegion
using LinearAlgebra, Statistics, Printf, DelimitedFiles, Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_contato_bulk.jl"))

const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "octave_bulk")
const MR = BEM.MultiRegion
relerr(a, b) = abs(a - b) / max(abs(b), eps()) * 100

function pin_top_center_ux!(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0) || continue
        ax = abs(dad.Nodes[i][1]); ax < bx && (bx = ax; best = i)
    end
    dad.BC[2best-1] = 0; dad.BV[2best-1] = 0.0
    return best
end

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10, verbose=true,
        common_normal=false, nsteps=1)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg, common_normal=common_normal)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = zeros(ctx.N)
    for cp in pairs; cp.state = 1; cp.ut_lock = 0.0; end
    x, ok = MR._contact_activeset!(prep, pairs, h, x0; tol=tol, maxiter=maxiter,
        verbose=verbose, nsteps=nsteps)
    MR._scatter_contact_solution!(prob, prep, pairs, x)
    return (; ok, nit=0)
end

function summarize(label, prob, par)
    fr = contact_interface_xyτ(prob)
    p = .-fr.tn; q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    w = ones(length(fr.x)); length(fr.x) >= 2 && (w[1] = abs(fr.x[2]-fr.x[1]); w[end] = abs(fr.x[end]-fr.x[end-1]))
    for i in 2:length(fr.x)-1; w[i] = 0.5*abs(fr.x[i+1]-fr.x[i-1]); end
    P = any(cl) ? sum(p[cl].*w[cl]) : 0.0
    Q = any(cl) ? sum(q[cl].*w[cl]) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    util = [abs(fr.tt[i])/(par.μ*abs(fr.tn[i])) for i in eachindex(fr.state)
            if abs(fr.state[i]) != 1 && par.μ*abs(fr.tn[i]) > 1e-12]
    @printf("%-22s  P=%.2f  Q=%+.3f  p0=%.2f (Hertz %.2f%%)  st/sl/op=%d/%d/%d  util=%.3f  |q|max=%.3f\n",
        label, P, Q, p0, relerr(p0, par.p0_H),
        count(==(3), fr.state), count(s -> abs(s)==2, fr.state), count(==(1), fr.state),
        isempty(util) ? 0.0 : mean(util), any(cl) ? maximum(abs, q[cl]) : 0.0)
    return (; fr..., p, q)
end

par = dad_contato_bulk_params()
println("dad_Contato_Bulk twin  P=$(par.P)  a_H=$(round(par.a_H; digits=4))  p0_H=$(round(par.p0_H; digits=2))  μ=$(par.μ)")

oct = readdlm(joinpath(OUT, "octave_bulk_mu.csv"), ','; header=true)
oh, ob = oct[2][1,:], oct[1]
# header row is in oh if readdlm header=true returns (data, header)
# DelimitedFiles: readdlm with header=true → (data, header_strings)
data, hdr = readdlm(joinpath(OUT, "octave_bulk_mu.csv"), ','; header=true)
ox = data[:,1]; otn = data[:,2]; ott = data[:,3]; ost = Int.(data[:,4])
op = .-otn; oq = .-ott
ocl = ost .!= 1
op0 = maximum(op[ocl])
@printf("Octave frictional     p0=%.2f (Hertz %.2f%%)  closed=%d  |q|max=%.3f  util=%.3f\n",
    op0, relerr(op0, par.p0_H), count(ocl), maximum(abs, oq[ocl]),
    mean(abs(ott[i])/(par.μ*abs(otn[i])) for i in eachindex(ost) if ost[i]!=1 && par.μ*abs(otn[i])>1e-12))

println("\n[Julia GL collocation  euclidean gap  geometric n]")
prob, _ = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
    gap=:euclidean, nome="jbulk_m")
pin_top_center_ux!(prob)
let
    dad1, dad2 = prob.regions
    poly = dad1.element_type
    @printf("  collocation nodes=%s\n", poly.nodes)
    cps = sort(prob.contacts; by=cp -> abs(dad1.Nodes[cp.node_a][1]))
    for cp in cps[1:3]
        n1 = dad1.Normal[cp.node_a]; n2 = dad2.Normal[cp.node_b]
        @printf("  pair x=(%.4f,%.4f) n1=(%+.3f,%+.3f) n2=(%+.3f,%+.3f) h=%.5f\n",
            dad1.Nodes[cp.node_a][1], dad2.Nodes[cp.node_b][1],
            n1[1], n1[2], n2[1], n2[2], cp.gap0)
    end
end
@printf("  pad n=%d spec n=%d NPc=%d  gap0∈[%.5f,%.5f]\n",
    prob.regions[1].n, prob.regions[2].n, length(prob.contacts),
    extrema(cp.gap0 for cp in prob.contacts)...)
sol = contato_newton!(prob; common_normal=false, nsteps=50)
println("  ok=$(sol.ok)  (50 Contato load steps)")
jm = summarize("Julia Contato-inc", prob, par)

println("\n[Julia μ=0  GL]")
prob0, _ = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2, μ=0.0,
    gap=:euclidean, nome="jbulk_mu0")
pin_top_center_ux!(prob0)
sol0 = contato_newton!(prob0; common_normal=false, nsteps=50)
println("  ok=$(sol0.ok) it=$(sol0.nit)")
par0 = dad_contato_bulk_params()
summarize("Julia μ=0", prob0, merge(par0, (; μ=1.0)))  # util unused

println("\n[Julia default GL + n_AB]")
prob2, _ = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
    collocation=:legendre, gap=:normal, nome="jbulk_j")
pin_top_center_ux!(prob2)
sol2 = contato_newton!(prob2; common_normal=true)
println("  ok=$(sol2.ok) it=$(sol2.nit)")
jg = summarize("Julia GL+nAB", prob2, par)

default(linewidth=1.6, legendfontsize=8, grid=true, framestyle=:box)
aH, p0H, μ = par.a_H, par.p0_H, par.μ
xa = collect(range(-1.3aH, 1.3aH; length=400))
pH = cattaneo_pressure(xa, aH, p0H)
plt = plot(xa./aH, pH./p0H; color=:black, label="Hertz two-body", xlabel="x/a",
    ylabel="p/p0", title="dad_Contato_Bulk  Octave vs Julia", ylims=(-0.05, 1.2))
scatter!(plt, ox./aH, op./p0H; ms=4, markerstrokewidth=0, color=:darkorange, label="Octave Contato")
scatter!(plt, jm.x./aH, jm.p./p0H; ms=3, markerstrokewidth=0, color=:steelblue, label="Julia GL")
savefig(plt, joinpath(OUT, "compare_p.png"))
pltq = plot(xlabel="x/a", ylabel="q/(f p0)", title="shear", ylims=(-0.5, 0.5))
hline!(pltq, [0.0]; color=:gray, ls=:dot, label=false)
scatter!(pltq, ox./aH, oq./(μ*p0H); ms=4, markerstrokewidth=0, color=:darkorange, label="Octave")
scatter!(pltq, jm.x./aH, jm.q./(μ*p0H); ms=3, markerstrokewidth=0, color=:steelblue, label="Julia GL")
savefig(pltq, joinpath(OUT, "compare_q.png"))
println("plots in $OUT")
println("Done.")
