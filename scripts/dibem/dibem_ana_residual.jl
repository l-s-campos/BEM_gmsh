# Discrete BIE residual with analytical u and ü (no time stepper).
# Laplace:  r = H u − G q − M ü
# Elasticity (ν=0): r = H u + M ü − G t
# julia --project=. scripts/dibem_ana_residual.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

rel(a, b) = norm(a - b) / (norm(b) + 1e-30)

function fill_bar_laplace!(u, q, ddu, dad, t; N=400, c=1.0, L=1.0)
    pts = all_points(dad)
    @inbounds for i in eachindex(pts)
        f = bar_sudden_fields(pts[i], t; N=N, c=c, L=L)
        u[i] = f.u
        ddu[i] = f.ddu
    end
    @inbounds for i in 1:dad.n
        f = bar_sudden_fields(dad.Nodes[i], t; N=N, c=c, L=L)
        q[i] = -f.dudx * dad.Normal[i][1]
    end
    return u, q, ddu
end

function fill_bar_elast!(u, tvec, ddu, dad, t; N=400, c=1.0, L=1.0, E=1.0)
    pts = all_points(dad)
    @inbounds for i in eachindex(pts)
        f = bar_sudden_fields(pts[i], t; N=N, c=c, L=L)
        u[2i-1] = f.u
        u[2i] = 0.0
        ddu[2i-1] = f.ddu
        ddu[2i] = 0.0
    end
    @inbounds for i in 1:dad.n
        f = bar_sudden_fields(dad.Nodes[i], t; N=N, c=c, L=L)
        nx = dad.Normal[i][1]
        σxx = E * f.dudx
        tvec[2i-1] = σxx * nx   # σxy = σyy = 0
        tvec[2i] = 0.0
    end
    return u, tvec, ddu
end

function residual_laplace(H, G, M, u, q, ddu)
    Hu = H * u
    Gq = G * q
    Md = M * ddu
    r = Hu .- Gq .- Md
    return (; r, Hu, Gq, Md)
end

function residual_elast(H, G, M, u, tvec, ddu)
    Hu = H * u
    Gt = G * tvec
    Md = M * ddu
    r = Hu .+ Md .- Gt
    return (; r, Hu, Gt, Md)
end

function print_res(tag, res)
    nr, nH, nM = norm(res.r), norm(res.Hu), norm(res.Md)
    nB = hasproperty(res, :Gq) ? norm(res.Gq) : norm(res.Gt)
    den = nH + nM + nB + 1e-30
    @printf("  %-18s  ||r||=%.3e  ||r||/Σ=%.3e  ||Hu||=%.3e  ||M ü||=%.3e  ||B||=%.3e\n",
        tag, nr, nr / den, nH, nM, nB)
    return nr / den
end

function assemble_Ms!(dad; rbf=PHS(3; poly_deg=1))
    H_G_full_direct(dad; npg=8, threaded=false)
    H, G = Matrix(dad.H), Matrix(dad.G)
    Md = DIBEM(deepcopy(dad); method=:dense, rbf=rbf, centers=:collocation)
    Mv = DIBEM(deepcopy(dad); method=:dense, rbf=rbf, centers=:cells)
    Mc = build_cell_mass(deepcopy(dad); npg=8).M
    return H, G, Md, Mv, Mc
end

const TIMES = (0.0, 0.25, 0.5, 1.0, 2.0)

println("="^72)
println(" Analytical u, ü residual  (no time stepper)")
println("="^72)

# ----- Elasticity -----
dadE, metaE = elasticity_bar_sudden(; ndiv=8, n_int=4, ν=0.0)
H, G, Md, Mv, Mc = assemble_Ms!(dadE)
ndof, nb = 2 * dadE.nt, 2 * dadE.n
println("\nElasticity ν=0  n=$(dadE.n) ni=$(dadE.ni)  H u + M ü − G t")
for t in TIMES
    u = zeros(ndof); tv = zeros(nb); ddu = zeros(ndof)
    fill_bar_elast!(u, tv, ddu, dadE, t; N=400, c=metaE.c, L=metaE.L, E=metaE.E)
    @printf("t=%.2f  ||u||=%.3e  ||ü||=%.3e  ||t||=%.3e\n", t, norm(u), norm(ddu), norm(tv))
    print_res("DIBEM colloc", residual_elast(H, G, Md, u, tv, ddu))
    print_res("DIBEM volume", residual_elast(H, G, Mv, u, tv, ddu))
    print_res("cells", residual_elast(H, G, Mc, u, tv, ddu))
end

# ----- Laplace -----
dadL, metaL = wave_problem(:bar_sudden; ndiv=8, n_int=4)
H, G, Md, Mv, Mc = assemble_Ms!(dadL)
println("\nLaplace  n=$(dadL.n) ni=$(dadL.ni)  H u − G q − M ü")
cL = metaL.ana === nothing ? 1.0 : 1.0
for t in TIMES
    u = zeros(dadL.nt); q = zeros(dadL.n); ddu = zeros(dadL.nt)
    fill_bar_laplace!(u, q, ddu, dadL, t; N=400, c=1.0, L=1.0)
    @printf("t=%.2f  ||u||=%.3e  ||ü||=%.3e  ||q||=%.3e\n", t, norm(u), norm(ddu), norm(q))
    print_res("DIBEM colloc", residual_laplace(H, G, Md, u, q, ddu))
    print_res("DIBEM volume", residual_laplace(H, G, Mv, u, q, ddu))
    print_res("cells", residual_laplace(H, G, Mc, u, q, ddu))
end
println("\nDone.")
