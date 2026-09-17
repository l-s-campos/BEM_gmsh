# Johnston / Elliott iterated sinh (IJNME 2008; EABE 2013 product form).
# 1-D: after x = a + b sinh(μs−η), nearest poles at η/μ ± i π/(2μ).
# Second sinh clusters on that pole. niter=1 is our current map.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

# b_pole = π/(2μ)  (Elliott & Johnston 2008, 1-D kernels)
# b_pole = π/(4μ)  (Johnston et al. 2013, 2-D product kernel)
function iterated_nearfield(qsi, w, a, b; niter=1, pole=:pi2)
    T = Float64
    a = T(a); b = max(T(b), T(1e-14))
    fac = pole === :pi4 ? T(π)/4 : T(π)/2
    maps = NTuple{2,T}[]
    aa, bb = a, b
    for _ in 1:niter
        push!(maps, (aa, bb))
        μ = T(0.5) * (asinh((1 + aa) / bb) + asinh((1 - aa) / bb))
        η = T(0.5) * (asinh((1 + aa) / bb) - asinh((1 - aa) / bb))
        μ < T(1e-14) && break
        aa = clamp(η / μ, nextfloat(T(-1)), prevfloat(T(1)))
        bb = max(fac / μ, T(1e-14))
    end
    x = collect(T, qsi)
    J = ones(T, length(qsi))
    for (aa, bb) in Iterators.reverse(maps)
        μ = T(0.5) * (asinh((1 + aa) / bb) + asinh((1 - aa) / bb))
        η = T(0.5) * (asinh((1 + aa) / bb) - asinh((1 - aa) / bb))
        @. J *= bb * μ * cosh(μ * x - η)
        @. x = aa + bb * sinh(μ * x - η)
    end
    return x, collect(T, w) .* J
end

# niter=1 must match BEM._sinhtrans
function check_niter1()
    u, w = BEM.gausslegendre(8)
    a, b = 0.3, 0.05
    x0, w0 = BEM._sinhtrans(u, w, a, b)
    x1, w1 = iterated_nearfield(u, w, a, b; niter=1)
    println("niter=1 vs library sinh:  ||Δx||=$(norm(x1.-x0))  ||Δw||=$(norm(w1.-w0))")
end

function install!(niter, pole)
    @eval BEM function transform(dad, elem, nodes, pf::Point2D; poly=dad.element_type)
        nf = has_cache(dad, :nearfield) ? dad.nearfield : :euclid
        nf === :plain && return dad.qsi, dad.w
        a, _, dist = closest_point_1d(poly, nodes, pf; ξ0=_seed_1d(poly, nodes, pf))
        b = dist / max(elem.Length, eps())
        return Main.iterated_nearfield(dad.qsi, dad.w, a, b;
            niter=$(niter), pole=$(QuoteNode(pole)))
    end
end

function one_element_test(dad)
    # inner-arc collocation vs first outer-arc element (nearly singular in z)
    poly = dad.element_type
    # pick collocation on inner (r≈300) and an outer element
    i_in = findfirst(i -> hypot(dad.Nodes[i][1], dad.Nodes[i][2]) < 350, 1:dad.n)
    el_out = dad.elements[3]  # first outer
    pf = dad.Nodes[i_in]; nf = dad.Normal[i_in]
    xj = dad.Nodes[el_out.index]
    f = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
    function integ(npg, niter, pole)
        BEM._init_quadrature!(dad, npg)
        install!(niter, pole)
        h = zeros(2, 6); g = zeros(2, 6)
        BEM.integrate_element(dad, el_out, xj, pf, h, g, f; source=i_in)
        return h, g
    end
    href, gref = integ(400, 1, :pi2)
    println("\nOff-element inner→outer (src=$i_in el=3)  ref = 1-sinh npg=400")
    println("  ||Href||=$(norm(href))  ||Gref||=$(norm(gref))")
    println("  npg niter pole   relH          relG")
    for npg in (8, 16, 50)
        for (niter, pole) in ((0, :pi2), (1, :pi2), (2, :pi2), (2, :pi4), (3, :pi2))
            if niter == 0
                @eval BEM function transform(dad, elem, nodes, pf::Point2D; poly=dad.element_type)
                    return dad.qsi, dad.w
                end
                BEM._init_quadrature!(dad, npg)
                h = zeros(2, 6); g = zeros(2, 6)
                BEM.integrate_element(dad, el_out, xj, pf, h, g, f; source=i_in)
            else
                h, g = integ(npg, niter, pole)
            end
            @printf("  %3d %5d %-4s  %10.3e  %10.3e\n",
                npg, niter, niter==0 ? "plain" : String(pole),
                norm(h.-href)/(norm(href)+1e-30),
                norm(g.-gref)/(norm(gref)+1e-30))
        end
    end
end

function p3_solve(msh, niter, pole, npg)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    if niter == 0
        @eval BEM function transform(dad, elem, nodes, pf::Point2D; poly=dad.element_type)
            return dad.qsi, dad.w
        end
    else
        install!(niter, pole)
    end
    H_G_hyper(dad; npg=npg, threaded=false); solve(dad)
    rel = norm(dad.u .- uc) / (norm(uc) + 1e-30)
    return maximum(abs, uc), maximum(abs, dad.u), rel, cond(Matrix(dad.A))
end

check_niter1()
msh = datadir("elastico", "p3_cmp_q2.msh")
dad = format2d(msh, props; tipo=2, pontointerno=false)
one_element_test(dad)

println("\nP3 circular mesh  CBIE vs HBIE")
println("niter pole  npg     CBIE mm      HBIE mm     rel        cond")
for (niter, pole, npg) in (
        (0, :pi2, 50), (1, :pi2, 16), (1, :pi2, 50),
        (2, :pi2, 16), (2, :pi2, 50), (2, :pi4, 50),
        (3, :pi2, 50),
    )
    uc, uh, rel, κ = p3_solve(msh, niter, pole, npg)
    @printf("%5d %-4s %3d  %10.3f  %10.3f  %8.3e  %8.2e\n",
        niter, niter==0 ? "pln" : String(pole), npg, uc, uh, rel, κ)
end
println("done")
