# DIBEM vs constant cells for the cylinder plastic domain integral.
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

function make(; tipo=2, nr=12, nθ=16)
    props = Elasticity(E0, Ν0, 1.0; plane_strain=true)
    msh = mesh_pressurized_tube(; a=A0, b=B0, ndiv=nθ, nr=nr, nθ=nθ,
        nome="cyl_dibem", ordem=tipo)
    dad = format2d(msh, props; pontointerno=true, tipo=tipo)
    apply_radius_pressure!(dad, P0; R=A0, tol=0.08 * A0)
    assemble!(dad, 12)
    return dad
end

function localized_sigma(dad)
    nc = dad.plastic_nc
    v = zeros(3nc)
    nloc = 0
    @inbounds for k in 1:nc
        ρ = hypot(dad.plastic_pts[k][1], dad.plastic_pts[k][2])
        ρ < 125 && (v[3k-2] = 1.0; v[3k-1] = 1.0; nloc += 1)
    end
    return v, nloc
end

function compare_Q(dadc, dadd)
    nc = dadc.plastic_nc
    e1 = zeros(3nc)
    @inbounds for k in 1:nc
        e1[3k-2] = 1.0
    end
    Qc, Qd = dadc.plastic_Q, dadd.plastic_Q
    @printf("  ||Q||_F  cells=%.3e  dibem=%.3e  ratio=%.2f\n",
        norm(Qc), norm(Qd), norm(Qd) / (norm(Qc) + eps()))
    @printf("  uniform  ||Qd-Qc||/||Qc||=%.3e\n",
        norm(Qd * e1 - Qc * e1) / (norm(Qc * e1) + eps()))
    v, nloc = localized_sigma(dadc)
    @printf("  localized (n=%d/%d)  ||Qc v||=%.3e  ||Qd v||=%.3e  ratio=%.2f\n",
        nloc, nc, norm(Qc * v), norm(Qd * v),
        norm(Qd * v) / (norm(Qc * v) + eps()))
    c = dadd.plastic_dibem_c
    @printf("  dibem c: min=%.3e  max=%.3e  nneg=%d/%d\n",
        minimum(c), maximum(c), count(<(0), c), length(c))
end

function run_pl(dad; domain=:cells, inner=:broyden, remainder=:shepard, nsteps=16)
    solve_elastoplastic!(dad, VonMises(σY=ΣY, H′=0.0); nsteps=nsteps,
        maxiter=80, tol=1e-4, save_history=true, threaded=true,
        domain=domain, relax=0.4, stress_coupling=:jump, inner=inner,
        remainder=remainder)
    ana = ana_thick_cylinder_plastic(B0; a=A0, b=B0, p=P0, σY=ΣY, E=E0, ν=Ν0)
    ub = median(outer_u(dad, B0))
    h = dad.plastic_history[end]
    npl = count(>(1e-12), dad.plastic_strain)
    @printf("  %-6s rem=%-8s  u=%.5f err=%.2f%%  resid=%.2e nit=%d npl=%d/%d fin=%s\n",
        domain, remainder, ub, 100 * abs(ub - ana.u) / ana.u, h.resid, h.niters,
        npl, dad.plastic_nc, string(all(isfinite, dad.u)))
end

function main()
    println("=== Q: cells vs DIBEM (format2d internals = cell centroids) ===")
    dadc = make()
    assemble_plastic_ops!(dadc; domain=:cells, npg=10, npg_stress=12, threaded=true)
    println("n=$(dadc.n)  ni=$(dadc.ni)  nt=$(dadc.nt)  nc=$(dadc.plastic_nc)")
    ξ0 = dadc.plastic_pts[1]
    pint = point(dadc, dadc.n + 1)
    @printf("  first internal vs first plastic_pt  |Δ|=%.3e\n", norm(pint - ξ0))

    dadd = make()
    assemble_plastic_ops!(dadd; domain=:dibem, npg=10, npg_stress=12,
        threaded=true, remainder=:shepard)
    println(" -- DIBEM remainder :shepard  (format2d internals as centres)")
    compare_Q(dadc, dadd)
    coinc = 0
    @inbounds for k in 1:dadd.plastic_nc
        coinc += sum(abs2, point(dadd, dadd.n + k) - dadd.plastic_pts[k]) < 1e-24
    end
    println("  coincident internals/centres = $coinc / $(dadd.plastic_nc)")

    println("\n=== plastic loop  ns=16 jump broyden ===")
    d = make()
    assemble_plastic_ops!(d; domain=:cells, npg=10, npg_stress=12, threaded=true)
    run_pl(d; domain=:cells)

    d = make()
    assemble_plastic_ops!(d; domain=:dibem, npg=10, npg_stress=12,
        threaded=true, remainder=:shepard)
    run_pl(d; domain=:dibem, remainder=:shepard)
    println("  DIBEM step history")
    for h in d.plastic_history
        d.u = h.u
        ub = median(outer_u(d, B0))
        @printf("    step %2d p=%5.2f u=%.5f npl=%3d resid=%.2e nit=%d\n",
            h.step, h.load * P0, ub, count(>(1e-12), h.plastic_strain),
            h.resid, h.niters)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
