# Continue cylinder vs von Mises: grading, nsteps, zone-split σ, :full/:hybrid.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf, StaticArrays

include(datadir("elastico", "iso", "pressurized_tube.jl"))

const A0, B0, E0, Ν0, ΣY, P0 = 100.0, 200.0, 2.1e4, 0.3, 30.0, 19.8

function outer_u(dad, b; tol=8.0)
    us = Float64[]
    @inbounds for i in 1:dad.n
        r = hypot(dad.Nodes[i][1], dad.Nodes[i][2])
        abs(r - b) < tol && push!(us, hypot(dad.u[2i-1], dad.u[2i]))
    end
    return us
end

function build(; ndiv=16, nr=12, nθ=16, tipo=2, progression=1.0, npg=10)
    props = Elasticity(E0, Ν0, 1.0; plane_strain=true)
    msh = mesh_pressurized_tube(; a=A0, b=B0, ndiv=ndiv, nr=nr, nθ=nθ,
        nome="cyl_cont", ordem=tipo, progression=progression)
    dad = format2d(msh, props; pontointerno=true, tipo=tipo)
    apply_radius_pressure!(dad, P0; R=A0, tol=0.08 * A0)
    assemble!(dad, 12)
    assemble_plastic_ops!(dad; domain=:cells, npg=npg, npg_stress=12, threaded=true)
    return dad
end

function zone_sigma(dad; c)
    e2n = d2n = e2p = d2p = e2e = d2e = e2b = d2b = 0.0
    nn = np = ne = nb = 0
    @inbounds for k in 1:dad.plastic_nc
        pt = dad.plastic_pts[k]
        ρ = hypot(pt[1], pt[2])
        e = pt / ρ
        sxx, syy, txy = dad.stress[k, 1], dad.stress[k, 2], dad.stress[k, 3]
        σr = sxx * e[1]^2 + syy * e[2]^2 + 2txy * e[1] * e[2]
        σθ = sxx * e[2]^2 + syy * e[1]^2 - 2txy * e[1] * e[2]
        ana = ana_thick_cylinder_plastic(ρ; a=A0, b=B0, p=P0, σY=ΣY, E=E0, ν=Ν0)
        dσ = (σr - ana.σr)^2 + (σθ - ana.σθ)^2
        dn = ana.σr^2 + ana.σθ^2
        if ρ < A0 + 12
            e2n += dσ; d2n += dn; nn += 1
        elseif ρ < c
            e2p += dσ; d2p += dn; np += 1
        else
            e2e += dσ; d2e += dn; ne += 1
        end
        if ρ >= A0 + 0.25 * (B0 - A0)
            e2b += dσ; d2b += dn; nb += 1
        end
    end
    rel(e, d) = sqrt(e / max(d, eps()))
    return (; near=rel(e2n, d2n), core=rel(e2p, d2p), ring=rel(e2e, d2e),
        bulk=rel(e2b, d2b), nn, np, ne, nb)
end

function runone(dad; coupling=:jump, inner=:broyden, nsteps=16, maxiter=80)
    ana = ana_thick_cylinder_plastic(B0; a=A0, b=B0, p=P0, σY=ΣY, E=E0, ν=Ν0)
    solve(dad)
    uel = median(outer_u(dad, B0))
    Acoef = P0 * A0^2 / (B0^2 - A0^2)
    uana_el = B0 * (1 + Ν0) / E0 * (1 - Ν0) * 2 * Acoef
    solve_elastoplastic!(dad, VonMises(σY=ΣY, H′=0.0); nsteps=nsteps,
        maxiter=maxiter, tol=1e-4, save_history=true, threaded=true,
        domain=:cells, relax=0.4, stress_coupling=coupling, inner=inner)
    h = dad.plastic_history[end]
    ub = median(outer_u(dad, B0))
    z = zone_sigma(dad; c=ana.c)
    npl = count(>(1e-12), dad.plastic_strain)
    @printf("  %-6s %-8s ns=%2d  u=%.5f err=%.2f%%  el_u=%.2f%%  resid=%.2e nit=%d  npl=%d/%d\n",
        coupling, inner, nsteps, ub, 100 * abs(ub - ana.u) / ana.u,
        100 * abs(uel - uana_el) / uana_el, h.resid, h.niters, npl, dad.plastic_nc)
    @printf("         σ  near=%.3f (n=%d)  core=%.3f (n=%d)  ring=%.3f (n=%d)  bulk=%.3f\n",
        z.near, z.nn, z.core, z.np, z.ring, z.ne, z.bulk)
    return (; ub, erru=100 * abs(ub - ana.u) / ana.u, resid=h.resid, z, finite=all(isfinite, dad.u))
end

function print_curve(dad, tag)
    ana(p) = p < 12.99 ? nothing :
        ana_thick_cylinder_plastic(B0; a=A0, b=B0, p=min(p, 0.98 * 24.02),
            σY=ΣY, E=E0, ν=Ν0)
    println("  curve ", tag)
    @printf("    %4s %6s %8s %8s %8s\n", "step", "p", "uBEM", "uVM", "du%")
    for h in dad.plastic_history
        p = h.load * P0
        dad.u = h.u
        ub = median(outer_u(dad, B0))
        a = try
            ana_thick_cylinder_plastic(B0; a=A0, b=B0, p=p, σY=ΣY, E=E0, ν=Ν0)
        catch
            nothing
        end
        if a === nothing
            @printf("    %4d %6.2f %8.5f\n", h.step, p, ub)
        else
            @printf("    %4d %6.2f %8.5f %8.5f %7.2f  npl=%d resid=%.1e nit=%d\n",
                h.step, p, ub, a.u, 100 * abs(ub - a.u) / abs(a.u),
                count(>(1e-12), h.plastic_strain), h.resid, h.niters)
        end
    end
end

function main()
    println("analytic u(b)=",
        ana_thick_cylinder_plastic(B0; a=A0, b=B0, p=P0, σY=ΣY, E=E0, ν=Ν0).u)

    println("\n=== Picard vs Broyden on quad nr12 ===")
    d = build(; tipo=2, ndiv=16, nr=12, nθ=16)
    println("n=$(d.n) nc=$(d.plastic_nc)")
    runone(d; coupling=:jump, inner=:broyden, nsteps=16)
    print_curve(d, "jump broyden ns=16")

    d = build(; tipo=2, ndiv=16, nr=12, nθ=16)
    runone(d; coupling=:jump, inner=:picard, nsteps=16, maxiter=150)
    print_curve(d, "jump picard ns=16")

    d = build(; tipo=2, ndiv=16, nr=12, nθ=16)
    runone(d; coupling=:local, inner=:broyden, nsteps=16)
    print_curve(d, "local broyden ns=16")

    println("\n=== nsteps=20 and 18 jump broyden ===")
    for ns in (18, 20)
        d = build(; tipo=2, ndiv=16, nr=12, nθ=16)
        runone(d; coupling=:jump, inner=:broyden, nsteps=ns)
        print_curve(d, "jump broyden ns=$ns")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
