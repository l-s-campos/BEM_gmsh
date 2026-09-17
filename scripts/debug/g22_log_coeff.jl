using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, FastGaussQuadrature, StaticArrays
using BEM.Plate.ThinPlate: ThinPlate as TP
const Point2D = SVector{2,Float64}
D, ν = 1.0, 0.3
L = 1.0
el = Element(; index=[1, 2, 3], Jacobian=fill(L / 2, 3), Length=L, Region=1,
    geo=Point2D[Point2D(0.0, 0.0), Point2D(L / 2, 0.0), Point2D(L, 0.0)])
poly = BEM.Legendre(2)
ξ0 = poly.nodes[1]
pf, _, nf = TP.elem_geom(el, ξ0)
a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
n = 20
ξ, w = gausslegendre(n)
keep = [abs(ξ[i] - a) > 1e-12 for i in eachindex(ξ)]
ξk = ξ[keep]
Nrow, _ = BEM.shapefun(BEM.ArbitraryPolynomial(ξk), a)
Ni = [Nrow[1, i] for i in eachindex(ξk)]
δ = ξk .- a
fi = map(ξk) do x
    pg, J, n̂ = TP.elem_geom(el, x)
    U, _ = TP.plate_kernels(pg, pf, n̂, nf, D, ν)
    Nf, _ = BEM.shapefun(poly, x)
    return [U[2, 2] * Nf[1, j] * J for j in 1:3]
end
Fm1 = sum(Ni[i] * fi[i] * δ[i] for i in eachindex(Ni))
F0_old = sum(Ni[i] * (fi[i] - Fm1 / δ[i]) / log(abs(δ[i])) for i in eachindex(Ni))
F0_ls = let
    sL = 0.0
    sL2 = 0.0
    nused = 0
    sW = zero(Fm1)
    sWL = zero(Fm1)
    for i in eachindex(δ)
        abs(δ[i]) < 1e-10 && continue
        lg = log(abs(δ[i]))
        wi = fi[i] - Fm1 / δ[i]
        sL += lg
        sL2 += lg * lg
        sW += wi
        sWL += wi * lg
        nused += 1
    end
    detA = sL2 * nused - sL * sL
    (nused * sWL - sL * sW) / detA
end
J = L / 2
Na = [BEM.shapefun(poly, a)[1][1, j] for j in 1:3]
F0_true = Na .* (J / (4π * D))
println("a = ", a, "  J = ", J)
println("N(a)     ", Na)
println("Fm1      ", Fm1)
println("F0 interp (Lagrange at a of work/log) ", F0_old)
println("F0 LS     (work ≈ F0 log + b)         ", F0_ls)
println("F0 true   (mn N J / 4πD)              ", F0_true)
