# Diagnose cylinder elastoplasticity vs Lamé / von Mises.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf

include(datadir("elastico", "iso", "pressurized_tube.jl"))

function outer_u(dad, b; tol=8.0)
    us = Float64[]
    @inbounds for i in 1:dad.n
        r = hypot(dad.Nodes[i][1], dad.Nodes[i][2])
        abs(r - b) < tol && push!(us, hypot(dad.u[2i-1], dad.u[2i]))
    end
    return us
end
function inner_t(dad, a; tol=8.0)
    ts = Float64[]
    @inbounds for i in 1:dad.n
        r = hypot(dad.Nodes[i][1], dad.Nodes[i][2])
        abs(r - a) < tol && push!(ts, hypot(dad.traction[2i-1], dad.traction[2i]))
    end
    return ts
end

function main()
    a, b, E, ν, σY, p = 100.0, 200.0, 2.1e4, 0.3, 30.0, 19.8
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    ndiv, tipo, nr = 16, 2, 12
    msh = mesh_pressurized_tube(; a=a, b=b, ndiv=ndiv, nr=nr, nome="cyl_diag",
        ordem=tipo)
    dad = format2d(msh, props; pontointerno=true, tipo=tipo)
    apply_radius_pressure!(dad, p; R=a, tol=0.08 * a)
    assemble!(dad, 12)

    Acoef = p * a^2 / (b^2 - a^2)
    uel = b * (1 + ν) / E * (1 - ν) * 2 * Acoef
    pel = σY / sqrt(3) * (1 - (a / b)^2)

    solve(dad)
    @printf("ELASTIC |t|_inner med=%.4f (p=%.2f)\n", median(inner_t(dad, a)), p)
    @printf("ELASTIC u(b) BEM=%.5f  Lamé=%.5f  rel=%.2f%%\n",
        median(outer_u(dad, b)), uel, 100 * abs(median(outer_u(dad, b)) - uel) / uel)

    assemble_plastic_ops!(dad; domain=:cells, npg=10, npg_stress=12, threaded=true)
    pts = dad.plastic_pts
    nc = dad.plastic_nc
    σB0 = dad.plastic_Su * dad.u + dad.plastic_St * dad.traction
    e2 = 0.0
    d2 = 0.0
    qmax = 0.0
    rmin_q = 0.0
    for k in 1:nc
        c = pts[k]
        ρ = hypot(c[1], c[2])
        e = c / ρ
        sxx, syy, txy = σB0[3k-2], σB0[3k-1], σB0[3k]
        σr = sxx * e[1]^2 + syy * e[2]^2 + 2txy * e[1] * e[2]
        σθ = sxx * e[2]^2 + syy * e[1]^2 - 2txy * e[1] * e[2]
        σra = -Acoef * (b^2 / ρ^2 - 1)
        σθa = Acoef * (b^2 / ρ^2 + 1)
        e2 += (σr - σra)^2 + (σθ - σθa)^2
        d2 += σra^2 + σθa^2
        σz = ν * (sxx + syy)
        pb = (sxx + syy + σz) / 3
        J2 = 0.5 * ((sxx - pb)^2 + (syy - pb)^2 + (σz - pb)^2) + txy^2
        q = sqrt(max(3 * J2, 0.0))
        if q > qmax
            qmax = q
            rmin_q = ρ
        end
    end
    @printf("ELASTIC interior σ L2 rel=%.3f  q_max=%.2f at r=%.1f  σY=%.1f  p_el=%.2f\n",
        sqrt(e2 / d2), qmax, rmin_q, σY, pel)

    solve_elastoplastic!(dad, VonMises(σY=σY, H′=0.0); nsteps=16, maxiter=80,
        tol=1e-4, save_history=true, threaded=true, domain=:cells, relax=0.4,
        stress_coupling=:jump, inner=:broyden)
    hist = dad.plastic_history
    println("\nstep load    p      u(b)    nplast nit  resid      maxκ")
    for h in hist
        dad.u = h.u
        ub = median(outer_u(dad, b))
        @printf("%3d  %.3f  %6.2f  %.5f  %6d %3d  %.2e  %.3e\n",
            h.step, h.load, h.load * p, ub, h.nplast, h.niters, h.resid,
            maximum(h.plastic_strain))
    end

    σB = dad.plastic_Su * dad.u + dad.plastic_St * dad.traction
    Fblk = initial_stress_free_term(dad.properties)
    σp_vec = BEM._pack_voigt(dad.plastic_initial_stress)
    δ = 0.0
    for k in 1:nc
        v = Fblk * SVector(σp_vec[3k-2], σp_vec[3k-1], σp_vec[3k])
        σj1 = σB[3k-2] + v[1]
        σj2 = σB[3k-1] + v[2]
        σj3 = σB[3k] + v[3]
        δ += (σj1 - dad.stress[k, 1])^2 + (σj2 - dad.stress[k, 2])^2 +
             (σj3 - dad.stress[k, 3])^2
    end
    @printf("\n||σBIE+F − σ_return|| / ||σ|| = %.4f\n",
        sqrt(δ) / (norm(dad.stress) + 1e-16))
    ana = ana_thick_cylinder_plastic(b; a=a, b=b, p=p, σY=σY, E=E, ν=ν)
    @printf("PLASTIC u(b) BEM=%.5f  VM=%.5f  rel=%.2f%%  c=%.2f\n",
        median(outer_u(dad, b)), ana.u,
        100 * abs(median(outer_u(dad, b)) - ana.u) / ana.u, ana.c)
    @printf("|t|_inner med=%.4f  plastic cells=%d / %d\n",
        median(inner_t(dad, a)), count(>(1e-12), dad.plastic_strain), nc)
    return dad
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
