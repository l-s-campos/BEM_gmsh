using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

L = 1.5
ndiv = 24
msh = mesh_square_hardwall(; ndiv=ndiv, L=L, nome="ricker_diag_sine", show=false)
dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
H_G_full_direct(dad; npg=10, threaded=false)
DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=12)

σ = 0.05
g = [exp(-(p[1]^2 + p[2]^2) / (2 * σ^2)) / (2π * σ^2) for p in all_points(dad)]
Mg = dad.M * g
ghat = dad.M \ Mg
@printf("||g||=%.3e  ||M*g||=%.3e  ||M\\(M*g)-g||=%.3e  rel=%.3e\n",
    norm(g), norm(Mg), norm(ghat .- g), norm(ghat .- g) / (norm(g) + 1e-30))
@printf("max|g|=%.3e  max|M\\Mg|=%.3e  mean(g)=%.3e  mean(M\\Mg)=%.3e\n",
    maximum(abs, g), maximum(abs, ghat), mean(g), mean(ghat))

sys = build_modal_system(dad)
basis = modal_analysis_mmm(sys)
gf = g[sys.free]
ff = Mg[sys.free]
fbar_M = basis.Φ̃' * (basis.M \ ff)
fbar_g = basis.Φ̃' * gf
@printf("nmodes=%d  ||fbar M\\(Mg)||=%.3e  ||fbar PhiT*g||=%.3e\n",
    length(basis.ω), norm(fbar_M), norm(fbar_g))

println("top |fbar| via M\\(M*g):")
ord = sortperm(abs.(fbar_M); rev=true)
for k in 1:8
    i = ord[k]
    @printf("  mode %4d  omega=%8.3f  |fbar|=%.3e\n", i, basis.ω[i], abs(fbar_M[i]))
end
println("top |fbar| via PhiT*g:")
ord = sortperm(abs.(fbar_g); rev=true)
for k in 1:8
    i = ord[k]
    @printf("  mode %4d  omega=%8.3f  |fbar|=%.3e\n", i, basis.ω[i], abs(fbar_g[i]))
end

println("static |y| = |fbar|/omega^2 via M\\(M*g):")
ord = sortperm(abs.(fbar_M) ./ (basis.ω .^ 2); rev=true)
for k in 1:8
    i = ord[k]
    @printf("  mode %4d  omega=%8.3f  |y|=%.3e\n", i, basis.ω[i], abs(fbar_M[i]) / basis.ω[i]^2)
end

i476 = argmax(abs.(fbar_M))
@printf("dominant mode %d  ||Phi||=%.3e  ||PhiTilde||=%.3e  infPhi=%.3e  infPhiT=%.3e\n",
    i476, norm(basis.Φ[:, i476]), norm(basis.Φ̃[:, i476]),
    norm(basis.Φ[:, i476], Inf), norm(basis.Φ̃[:, i476], Inf))
@printf("mode 1          ||Phi||=%.3e  ||PhiTilde||=%.3e  infPhi=%.3e  infPhiT=%.3e\n",
    norm(basis.Φ[:, 1]), norm(basis.Φ̃[:, 1]),
    norm(basis.Φ[:, 1], Inf), norm(basis.Φ̃[:, 1], Inf))

# L2-like overlap of RIGHT vector with g vs left
ov_R = abs.(basis.Φ' * gf)
ov_L = abs.(basis.Φ̃' * gf)
println("top RIGHT-vector overlap Phi'*g:")
ord = sortperm(ov_R; rev=true)
for k in 1:8
    i = ord[k]
    @printf("  mode %4d  omega=%8.3f  |Phi.g|=%.3e  |PhiT.g|=%.3e\n",
        i, basis.ω[i], ov_R[i], ov_L[i])
end

# centreline of dominant mode
pts = collect(dad.internalNodes)
cl = Int[i for i in eachindex(pts) if abs(pts[i][2]) < 1e-9]
sort!(cl; by=i -> pts[i][1])
n = dad.n
phi = basis.Φ[n+1:end, i476]
println("dominant-mode centreline (every 4th point):")
for k in 1:4:length(cl)
    @printf("  x=%7.3f  phi=%.3e\n", pts[cl[k]][1], phi[cl[k]])
end
