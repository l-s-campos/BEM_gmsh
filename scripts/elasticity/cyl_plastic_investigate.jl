# Cylinder elastoplasticity: operator identities + coupling/inner sweep vs von Mises.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf, StaticArrays

include(datadir("elastico", "iso", "pressurized_tube.jl"))

const A0, B0, E0, Ν0, ΣY, P0 = 100.0, 200.0, 2.1e4, 0.3, 30.0, 19.8

function outer_u(dad, b; tol=8.0)
    us = Float64[]
    @inbounds for i in 1:dad.n
        r = hypot(dad.Nodes[i][1], dad.Nodes[i][2])
        abs(r - b) < tol && push!(us, hypot(dad.u[2i - 1], dad.u[2i]))
    end
    return us
end

function build(; ndiv=12, nr=nothing, nθ=nothing, tipo=1, progression=1.0,
        domain=:cells, npg=10, npg_stress=12)
    props = Elasticity(E0, Ν0, 1.0; plane_strain=true)
    msh = mesh_pressurized_tube(; a=A0, b=B0, ndiv=ndiv, nr=nr, nθ=nθ,
        nome="cyl_inv", ordem=tipo, progression=progression)
    dad = format2d(msh, props; pontointerno=true, tipo=tipo)
    apply_radius_pressure!(dad, P0; R=A0, tol=0.08 * A0)
    assemble!(dad, 12)
    assemble_plastic_ops!(dad; domain=domain, npg=npg, npg_stress=npg_stress,
        threaded=true)
    return dad
end

function identities(dad)
    nc = dad.plastic_nc
    e1 = zeros(3nc)
    @inbounds for k in 1:nc
        e1[3k - 2] = 1.0
    end
    Qe = dad.plastic_Q * e1
    IE = vcat((BEM._boundary_integral_Estrain(dad, point(dad, i); npg=12)[:, 1]
               for i in 1:dad.nt)...)
    qrel = norm(Qe - IE) / (norm(IE) + eps())
    Se = dad.plastic_Sσ * e1
    IS = vcat((vec(BEM._boundary_integral_Estress(dad, dad.plastic_pts[k]; npg=16,
        free_term=true)[:, 1]) for k in 1:nc)...)
    srel = norm(Se - IS) / (norm(IS) + eps())
    F = initial_stress_free_term(dad.properties)
    dF = 0.0
    nF = 0.0
    @inbounds for k in 1:nc
        blk = dad.plastic_Sσ[3k-2:3k, 3k-2:3k]
        dF += norm(blk - F)
        nF += norm(F)
    end
    @printf("  Q·1 vs ∫Un     rel=%.3e   Sσ·1 vs ∫E  rel=%.3e   ||Sσ_self−F||/||F||=%.3f\n",
        qrel, srel, dF / max(nF, eps()))
    return (; qrel, srel, dFrel=dF / max(nF, eps()))
end

function elastic_report(dad)
    solve(dad)
    Acoef = P0 * A0^2 / (B0^2 - A0^2)
    uel = B0 * (1 + Ν0) / E0 * (1 - Ν0) * 2 * Acoef
    ub = median(outer_u(dad, B0))
    σB = dad.plastic_Su * dad.u + dad.plastic_St * dad.traction
    e2 = d2 = 0.0
    e2b = d2b = 0.0
    @inbounds for k in 1:dad.plastic_nc
        c = dad.plastic_pts[k]
        ρ = hypot(c[1], c[2])
        e = c / ρ
        sxx, syy, txy = σB[3k-2], σB[3k-1], σB[3k]
        σr = sxx * e[1]^2 + syy * e[2]^2 + 2txy * e[1] * e[2]
        σθ = sxx * e[2]^2 + syy * e[1]^2 - 2txy * e[1] * e[2]
        σra = -Acoef * (B0^2 / ρ^2 - 1)
        σθa = Acoef * (B0^2 / ρ^2 + 1)
        e2 += (σr - σra)^2 + (σθ - σθa)^2
        d2 += σra^2 + σθa^2
        if ρ >= A0 + 0.25 * (B0 - A0)
            e2b += (σr - σra)^2 + (σθ - σθa)^2
            d2b += σra^2 + σθa^2
        end
    end
    @printf("  elastic u(b)=%.5f  Lamé=%.5f  rel=%.2f%%   σ L2=%.3f  bulk=%.3f\n",
        ub, uel, 100 * abs(ub - uel) / uel, sqrt(e2 / d2), sqrt(e2b / max(d2b, eps())))
    return ub
end

function sigma_rel(dad; rmin=0.0)
    e2 = d2 = 0.0
    @inbounds for k in 1:dad.plastic_nc
        c = dad.plastic_pts[k]
        ρ = hypot(c[1], c[2])
        ρ < rmin && continue
        e = c / ρ
        sxx, syy, txy = dad.stress[k, 1], dad.stress[k, 2], dad.stress[k, 3]
        σr = sxx * e[1]^2 + syy * e[2]^2 + 2txy * e[1] * e[2]
        σθ = sxx * e[2]^2 + syy * e[1]^2 - 2txy * e[1] * e[2]
        ana = ana_thick_cylinder_plastic(ρ; a=A0, b=B0, p=P0, σY=ΣY, E=E0, ν=Ν0)
        e2 += (σr - ana.σr)^2 + (σθ - ana.σθ)^2
        d2 += ana.σr^2 + ana.σθ^2
    end
    return sqrt(e2 / max(d2, eps()))
end

function run_plastic(dad; coupling=:jump, inner=:picard, nsteps=12, maxiter=60,
        relax=0.4, domain=:cells)
    solve_elastoplastic!(dad, VonMises(σY=ΣY, H′=0.0); nsteps=nsteps,
        maxiter=maxiter, tol=1e-4, save_history=true, threaded=true,
        domain=domain, relax=relax, stress_coupling=coupling, inner=inner)
    ana = ana_thick_cylinder_plastic(B0; a=A0, b=B0, p=P0, σY=ΣY, E=E0, ν=Ν0)
    ub = median(outer_u(dad, B0))
    h = dad.plastic_history[end]
    erru = 100 * abs(ub - ana.u) / abs(ana.u)
    errσ = sigma_rel(dad)
    errb = sigma_rel(dad; rmin=A0 + 0.25 * (B0 - A0))
    finite = all(isfinite, dad.u)
    @printf("  %-6s %-8s  u=%.5f  err_u=%6.2f%%  σ=%.3f bulk=%.3f  nplast=%d  nit=%d  resid=%.2e  fin=%s\n",
        coupling, inner, ub, erru, errσ, errb, h.nplast, h.niters, h.resid, finite)
    return (; coupling, inner, ub, erru, errσ, errb, resid=h.resid, nit=h.niters,
        nplast=h.nplast, finite)
end

function main()
    ana = ana_thick_cylinder_plastic(B0; a=A0, b=B0, p=P0, σY=ΣY, E=E0, ν=Ν0)
    println("analytic u(b)=", ana.u, "  c=", ana.c, "  pel=", ana.pel)
    if !isempty(ARGS) && ARGS[1] == "sweep"
        println("\n=== identities + elastic  ndiv=12  cells ===")
        dad = build(; ndiv=12)
        println("n=$(dad.n)  nt=$(dad.nt)  nc=$(dad.plastic_nc)")
        identities(dad)
        elastic_report(dad)
        println("\n=== coupling × inner  ndiv=12  nsteps=10 ===")
        for coupling in (:jump, :local, :full), inner in (:picard, :broyden)
            d = build(; ndiv=12)
            run_plastic(d; coupling=coupling, inner=inner, nsteps=10, maxiter=50)
        end
        return
    end
    println("\n=== :jump + broyden refine (post Telles-nearfield) ===")
    for (label, kw) in (
            ("ndiv16 nr16", (; ndiv=16, nr=16, nθ=16)),
            ("ndiv16 nr16 p1.15", (; ndiv=16, nr=16, nθ=16, progression=1.15)),
            ("ndiv24 nr20", (; ndiv=24, nr=20, nθ=24)),
            ("quad ndiv16 nr12", (; ndiv=16, nr=12, nθ=16, tipo=2)),
        )
        println(" -- ", label)
        d = build(; kw...)
        println("n=$(d.n)  nc=$(d.plastic_nc)")
        identities(d)
        elastic_report(d)
        run_plastic(d; coupling=:jump, inner=:broyden, nsteps=16, maxiter=80)
        npl = count(>(1e-12), d.plastic_strain)
        @printf("    total plastic cells=%d / %d  maxκ=%.4e\n",
            npl, d.plastic_nc, maximum(d.plastic_strain))
        d = build(; kw...)
        run_plastic(d; coupling=:local, inner=:broyden, nsteps=16, maxiter=80)
        npl = count(>(1e-12), d.plastic_strain)
        @printf("    local plastic cells=%d / %d  maxκ=%.4e\n",
            npl, d.plastic_nc, maximum(d.plastic_strain))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
