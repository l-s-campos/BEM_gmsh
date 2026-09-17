# Granados & Gallego (EABE 189, 2026) natural maps vs current Euclidean sinh.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

println("=== pole on a straight element (expect ζ0=0, η0=d/J=0.1) ===")
poly1 = BEM.Equispaced(1)
nodes1 = [Point2D(0.0, 0.0), Point2D(2.0, 0.0)]  # J=1, L=2
pf1 = Point2D(1.0, 0.1)
ζ0, η0 = BEM._complex_pole_1d(poly1, nodes1, pf1)
a, _, dist = closest_point_1d(poly1, nodes1, pf1)
@printf("  pole ζ0=%.4f η0=%.4f   euclid a=%.4f d=%.4f  d/L=%.4f  d/J=%.4f\n",
    ζ0, η0, a, dist, dist/2, dist/1)

function kernel_block(dad, i, el; npg, near)
    set_cache!(dad; nearfield=near)
    BEM._init_quadrature!(dad, npg)
    pf = dad.Nodes[i]; nf = dad.Normal[i]
    xj = dad.Nodes[el.index]
    h = zeros(2, 2 * length(el.index)); g = zeros(2, 2 * length(el.index))
    f = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
    BEM.integrate_element(dad, el, xj, pf, h, g, f; orders=(-1, -2), source=i)
    return h, g
end

function nearest_off(dad)
    best = (d=Inf, i=0, el=0)
    for i in 1:dad.n
        pf = dad.Nodes[i]
        for (ie, el) in enumerate(dad.elements)
            i in el.index && continue
            xj = dad.Nodes[el.index]
            _, _, d = closest_point_1d(dad.element_type, xj, pf;
                ξ0=BEM._seed_1d(dad.element_type, xj, pf))
            if d < best.d
                best = (d=d, i=i, el=ie)
            end
        end
    end
    return best
end

msh = datadir("elastico", "p3_cmp_q2.msh")
dad = format2d(msh, props; tipo=2, pontointerno=false)
np = nearest_off(dad)
el = dad.elements[np.el]
ζ0, η0 = BEM._complex_pole_1d(dad.element_type, dad.Nodes[el.index], dad.Nodes[np.i])
println("\n=== nearest off-element pair i=$(np.i) el=$(np.el)  d=$(np.d)  pole=($(ζ0), $(η0)) ===")
href, gref = kernel_block(dad, np.i, el; npg=400, near=:euclid)
println("  ref 1-sinh npg=400  ||H||=$(norm(href))  ||G||=$(norm(gref))")
println("  npg  map         relH        relG")
for npg in (8, 16, 50)
    for near in (:plain, :euclid, :csinh, :sinhsinh, :tangent, :p3c, :tanp3c)
        h, g = kernel_block(dad, np.i, el; npg=npg, near=near)
        @printf("  %3d  %-10s  %10.3e  %10.3e\n", npg, near,
            norm(h .- href)/(norm(href)+1e-30),
            norm(g .- gref)/(norm(gref)+1e-30))
    end
end

function p3(near, npg)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    set_cache!(dad; nearfield=near)
    H_G_hyper(dad; npg=npg, threaded=false); solve(dad)
    rel = norm(dad.u .- uc)/(norm(uc)+1e-30)
    return maximum(abs, uc), maximum(abs, dad.u), rel, cond(Matrix(dad.A))
end

println("\n=== P3 circular HBIE ===")
println("  map        npg     CBIE mm      HBIE mm     rel        cond")
for (near, npg) in (
        (:plain, 50), (:euclid, 16), (:euclid, 50),
        (:csinh, 50), (:sinhsinh, 16), (:sinhsinh, 50),
        (:tangent, 16), (:tangent, 50),
        (:p3c, 16), (:p3c, 50),
        (:tanp3c, 16), (:tanp3c, 50),
    )
    uc, uh, rel, κ = p3(near, npg)
    @printf("  %-10s %3d  %10.3f  %10.3f  %8.3e  %8.2e\n",
        near, npg, uc, uh, rel, κ)
end
println("done")
