# 2D anisotropic Poisson with known solution: mesh convergence of S1 and S3.
#
#   ∇·(K ∇u) = -b,  K=diag(5, 0.5),  u = sin(πx) sin(πy)
#   b = π² (kx+ky) u,  q = -n·K∇u
#
# All-Dirichlet on the unit square (quadrado). Error on interior nodes.
using BEM
using LinearAlgebra
using StaticArrays
using Printf
using Plots

K = @SMatrix [5.0 0.0; 0.0 0.5]
ana, bsrc = ana_aniso_poisson_sin(K)
rbf = PHS(3; poly_deg=2)
ndivs = [4, 6, 8, 12, 16]

function interior_rel(dad, ana)
    ni = dad.nt - dad.n
    ni == 0 && return NaN
    Ti = [ana.u(p) for p in dad.internalNodes]
    return norm(dad.T[dad.n+1:dad.nt] .- Ti) / (norm(Ti) + eps())
end

function run_one(ndiv, strat)
    nome = "apoisson_$(strat)_n$(ndiv)"
    dad = format2d(quadrado(ndiv=ndiv, show=false, nome=nome, ordem=2),
        Laplace(1.0); pontointerno=true, tipo=2)
    apply_analytical_bc!(dad, ana)
    assemble!(dad; npg=12, threaded=false)
    DIBEM(dad; rbf=rbf)
    if strat === :S1
        solve_anisotropic_dibem!(dad, K; b=bsrc, rbf=rbf)
    else
        solve_anisotropic_ibp!(dad, K; b=bsrc, rbf=rbf)
    end
    h = 1 / (ndiv - 1)
    err = interior_rel(dad, ana)
    return (ndiv=ndiv, n=dad.n, ni=dad.ni, h=h, err=err)
end

rows = NamedTuple[]
println("anisotropic Poisson  u=sin(πx)sin(πy)  K=diag(5,0.5)")
println("ndiv   n   ni      h         S1          S3")
s1s = []; s3s = []
for nd in ndivs
    r1 = run_one(nd, :S1)
    r3 = run_one(nd, :S3)
    push!(s1s, r1); push!(s3s, r3)
    push!(rows, r1); push!(rows, r3)
    @printf("%4d %5d %5d  %.4f  %.3e  %.3e\n",
        nd, r1.n, r1.ni, r1.h, r1.err, r3.err)
    flush(stdout)
end

function slope(xs)
    length(xs) < 2 && return NaN
    logh = log.([r.h for r in xs])
    loge = log.([r.err for r in xs])
    # last two points
    (loge[end] - loge[end-1]) / (logh[end] - logh[end-1])
end
@printf("observed order (last two h):  S1 %.2f  S3 %.2f\n", slope(s1s), slope(s3s))

outdir = joinpath(@__DIR__, "..", "..", "plots")
mkpath(outdir)
hs1 = [r.h for r in s1s]; e1 = [r.err for r in s1s]
hs3 = [r.h for r in s3s]; e3 = [r.err for r in s3s]
plt = plot(hs1, e1; xscale=:log10, yscale=:log10, marker=:circle, ms=6,
    label="S1 DIBEM Hessian", lw=2, color=:seagreen,
    xlabel="h = 1/(ndiv−1)", ylabel="interior relative L2 error",
    title="2D anisotropic Poisson  u=sin(πx)sin(πy)",
    legend=:bottomright, size=(720, 520), grid=true)
plot!(plt, hs3, e3; marker=:diamond, ms=6, label="S3 IBP + DIBEM", lw=2, color=:purple)
# reference slopes through last S3 point
href = [hs3[1], hs3[end]]
plot!(plt, href, e3[end] * (href ./ hs3[end]).^2; ls=:dash, color=:gray,
    label="O(h²)")
plot!(plt, href, e3[end] * (href ./ hs3[end]).^3; ls=:dot, color=:gray40,
    label="O(h³)")
out = joinpath(outdir, "aniso_poisson_convergence.png")
savefig(plt, out)
println("wrote ", out)

csv = joinpath(outdir, "aniso_poisson_convergence.csv")
open(csv, "w") do io
    println(io, "strategy,ndiv,n,ni,h,err")
    for r in s1s
        @printf(io, "S1,%d,%d,%d,%.6f,%.8e\n", r.ndiv, r.n, r.ni, r.h, r.err)
    end
    for r in s3s
        @printf(io, "S3,%d,%d,%d,%.6f,%.8e\n", r.ndiv, r.n, r.ni, r.h, r.err)
    end
end
println("wrote ", csv)
