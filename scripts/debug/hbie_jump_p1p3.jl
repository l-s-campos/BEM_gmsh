# G' jump 1/2: P1 (must stay accurate) vs P3; G remainder; Burton–Miller mix.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function strip_Gjump!(dad)
    G = copy(dad.G)
    for i in 1:dad.n
        G[2i-1, 2i-1] += 0.5
        G[2i, 2i] += 0.5
    end
    set_cache!(dad; G=G, G_hyper=G)
    return dad
end

# --- G remainder on MAT1 curved arc ---
println("=== U^h remainder (K_D - K_D*) on curved arc ===")
poly = BEM.Legendre(2)
X = [Point2D(600*cos(t), 600*sin(t)) for t in (0.0, π/10, π/5)]
a = poly.nodes[1]
N0, dN0 = BEM.shapefun(poly, a)
_, J0, _, n0 = BEM._geom_1d(poly, X, a)
pf = (N0 * X)[1]
Uh, Th = BEM._lekh_Uh_Th_lead(props.params, n0)
println("  δ         ||Fg-K*||    ||δ(Fg-K*)||")
for δ in (1e-1, 1e-2, 1e-3, 1e-4)
    ξ = a + δ
    N, dN = BEM.shapefun(poly, ξ)
    pg = (N * X)[1]; dx = (dN * X)[1]; J = norm(dx)
    nrm = Point2D(dx[2], -dx[1]) / J
    U, T = fundamental_hyper(props, pg, pf, nrm, n0)
    U = BEM._to_smat(U)
    Fg = zeros(2, 6)
    Ks = zeros(2, 6)
    for j in 1:3
        Fg[:, 2j-1:2j] .= U .* (N[1, j] * J)
        for β in 1:2, α in 1:2
            Ks[α, 2j-2+β] = Uh[α, β] * N0[1, j] / δ
        end
    end
    dF = Fg - Ks
    @printf("  %.0e    %10.3e    %10.3e\n", δ, norm(dF), norm(δ * dF))
end

# --- P1 with/without jump ---
println("\n=== P1 MAT1 HBIE jump ===")
D = inv(props.params.C); Lx, Ly, p = 500.0, 200.0, 100.0
ana = AnalyticalSolution("p1",
    (x; t=0.0) -> SVector(D[1,1]*p*x[1]+D[1,3]*p*x[2], D[1,2]*p*x[2]);
    q = (x, n; t=0.0) -> SVector(p*n[1], 0.0))
msh1 = datadir("elastico", "cordeiro_p1.msh")
dad = format2d(msh1, props; tipo=2, pontointerno=false)
vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1e-6 || dad.Nodes[i][1] > Lx-1e-6]
apply_analytical_bc!(dad, ana, setdiff(1:dad.n, vert))
H_G_hyper(dad; npg=16, threaded=false); solve(dad)
uana = [ana(dad.Nodes[i])[k] for i in 1:dad.n for k in 1:2]
mask = trues(length(uana))
for i in vert; mask[2i-1]=false; mask[2i]=false; end
e(u) = norm(u[mask] .- uana[mask]) / (norm(uana[mask]) + 1e-30)
println("  with jump   relNeu=$(e(dad.u))  max|u|=$(maximum(abs,dad.u))")
strip_Gjump!(dad); solve(dad)
println("  no jump     relNeu=$(e(dad.u))  max|u|=$(maximum(abs,dad.u))")

# --- P3 Burton-Miller ---
println("\n=== P3 Burton-Miller (H+αH')u = (G+αG')t ===")
msh3 = datadir("elastico", "cordeiro_p3.msh")
dad = format2d(msh3, props; tipo=2, pontointerno=false)
assemble!(dad; npg=16, threaded=false)
Hc, Gc = copy(dad.H), copy(dad.G)
solve(dad); uc = copy(dad.u)
H_G_hyper(dad; npg=16, threaded=false)
Hh, Gh = copy(dad.H), copy(dad.G)
println("  ||H||=$(norm(Hc))  ||H'||=$(norm(Hh))  ||G||=$(norm(Gc))  ||G'||=$(norm(Gh))")
for α in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
    H = Hc + α * Hh
    G = Gc + α * Gh
    set_cache!(dad; H=H, G=G)
    solve(dad)
    rel = norm(dad.u .- uc) / (norm(uc) + 1e-30)
    @printf("  α=%8.1e  max|u|=%10.3f  relCBIE=%.3e  cond=%.3e\n",
        α, maximum(abs, dad.u), rel, cond(Matrix(dad.A)))
end
println("done")
