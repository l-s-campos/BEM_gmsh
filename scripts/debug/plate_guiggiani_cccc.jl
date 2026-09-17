# Why fused Guiggiani fails on CCCC (G22 = Mn kernel is log, not 1/r²).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, FastGaussQuadrature, StaticArrays
using BEM.Plate
using BEM.Plate.ThinPlate: ThinPlate as TP

const Point2D = SVector{2,Float64}
const shapefun = BEM.shapefun
const Legendre = BEM.Legendre

D, ν = 1.0, 0.3
L = 1.0
x1, x3 = Point2D(0.0, 0.0), Point2D(L, 0.0)
geo = Point2D[x1, Point2D(L / 2, 0.0), x3]
el = Element(; index=[1, 2, 3], Jacobian=fill(L / 2, 3), Length=L, Region=1, geo=geo)
poly = Legendre(2)
qsi20, w20 = gausslegendre(20)

function pack(el, poly, pf, nf, ξ)
    pg, J, n̂ = TP.elem_geom(el, ξ)
    r = norm(pg - pf)
    r < 1e-14 && return nothing
    U, P = TP.plate_kernels(pg, pf, n̂, nf, D, ν)
    Nf, _ = shapefun(poly, ξ)
    return U, P, Nf, J
end

function g22_analytic(ξ0)
    _, g2 = TP.integraelemsing(x1, x3, D, ν, ξ0, poly)
    return collect(g2)
end

function g22_entry(ξ0; ninterp=20)
    pf, _, nf = TP.elem_geom(el, ξ0)
    a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    nN = 3
    f = ξ -> begin
        samp = pack(el, poly, pf, nf, ξ)
        samp === nothing && return zeros(nN)
        U, _, Nf, J = samp
        return [U[2, 2] * Nf[1, j] * J for j in 1:nN]
    end
    return collect(guiggiani_integral(f, a, 0; laurent=:interp, ninterp=ninterp))
end

function g22_fused(ξ0; ninterp=20, oG=0, oH=-2)
    pf, _, nf = TP.elem_geom(el, ξ0)
    a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    nN = 3
    f = ξ -> begin
        samp = pack(el, poly, pf, nf, ξ)
        samp === nothing && return (zeros(2, 6), zeros(2, 6))
        U, P, Nf, J = samp
        Fg = zeros(2, 6); Fh = zeros(2, 6)
        for j in 1:nN
            NjJ = Nf[1, j] * J
            Fg[:, 2j-1:2j] .= U .* NjJ
            Fh[:, 2j-1:2j] .= P .* NjJ
        end
        return Fg, Fh
    end
    Ig, Ih = guiggiani_GH(f, a; order_G=oG, order_H=oH, laurent=:interp, ninterp=ninterp)
    return Ig[2, 2:2:6], Ih
end

function rel(a, b)
    nb = norm(b)
    return nb < 1e-16 ? norm(a) : norm(a - b) / nb
end

println("="^70)
println("G22 (Mn kernel, log) on a straight element")
for loc in 1:3
    ξ0 = poly.nodes[loc]
    ga = g22_analytic(ξ0)
    ge = g22_entry(ξ0)
    gf, _ = g22_fused(ξ0)
    gf0, _ = g22_fused(ξ0; oG=0, oH=0)
    @printf("  loc=%d ξ0=% .4f\n", loc, ξ0)
    @printf("    analytic  %s\n", string(round.(ga; sigdigits=5)))
    @printf("    per-entry %s   rel=%.3e\n", string(round.(ge; sigdigits=5)), rel(ge, ga))
    @printf("    fused G0/H-2 %s   rel=%.3e\n", string(round.(gf; sigdigits=5)), rel(gf, ga))
    @printf("    fused G0/H0  %s   rel=%.3e\n", string(round.(gf0; sigdigits=5)), rel(gf0, ga))
end

println()
println("="^70)
println("H (P) fused vs per-entry vs analytic (row 1 = Vn,Mn)")
for loc in 1:3
    ξ0 = poly.nodes[loc]
    ha, _ = TP.integraelemsing(x1, x3, D, ν, ξ0, poly)
    pf, _, nf = TP.elem_geom(el, ξ0)
    a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    nN = 3
    h_ent = zeros(2, 6)
    ORDER_P = @SMatrix [-1 0; -2 -1]
    for α in 1:2, β in 1:2
        fP = ξ -> begin
            samp = pack(el, poly, pf, nf, ξ)
            samp === nothing && return zeros(nN)
            _, P, Nf, J = samp
            return [P[α, β] * Nf[1, j] * J for j in 1:nN]
        end
        IP = guiggiani_integral(fP, a, ORDER_P[α, β]; laurent=:interp, ninterp=20)
        for j in 1:nN
            h_ent[α, 2j-2+β] = IP[j]
        end
    end
    _, hf = g22_fused(ξ0)
    @printf("  loc=%d  rel H analytic vs per-entry=%.3e  vs fused-2=%.3e\n",
        loc, rel(vec(h_ent), vec(ha)), rel(vec(hf), vec(ha)))
    @printf("    H12 analytic %s\n", string(round.(ha[1, 2:2:6]; sigdigits=4)))
    @printf("    H12 per-entry %s\n", string(round.(h_ent[1, 2:2:6]; sigdigits=4)))
    @printf("    H12 fused-2   %s\n", string(round.(hf[1, 2:2:6]; sigdigits=4)))
end

println()
println("="^70)
println("CCCC w_c/h vs Levy 0.001263 (Q=17.79)  n_el=6 ni=1")
A, Hh, E, NU = 1.0, 0.01, 1e6, 0.316
q0 = 17.79 * E * Hh^4 / A^4
Dd = E * Hh^3 / (12 * (1 - NU^2))
w_ana = 0.001263 * q0 * A^4 / Dd
props = ThinPlateProps(; E=E, ν=NU, h=Hh, q_c=q0)

function to_mesh(dad)
    corners = TP._plate_corners(dad)
    TP.PlateMesh(dad.elements, dad.element_type, collect(Float64, dad.elem_weight),
        collect(Point2D, dad.Nodes), collect(Point2D, dad.Normal),
        copy(dad.BC), copy(dad.BV), corners, dad.properties;
        internal=collect(Point2D, dad.internalNodes))
end

dad = build_square_plate(; a=A, n_el=6, bc="CCCC", props=props,
    n_internal=1, corner_bc='C', p=2)
for (lab, sing) in (("analytic", :analytic), ("guiggiani fused", :guiggiani))
    mesh = to_mesh(dad)
    assemble_plate!(mesh; npg=12, singular=sing, threaded=false)
    solve_plate!(mesh)
    w = plate_w_int(mesh, 1)
    @printf("  %-18s  w/h=%.4f  e=%.2f%%\n", lab, w / Hh, 100 * abs(w - w_ana) / w_ana)
end
println("done")
