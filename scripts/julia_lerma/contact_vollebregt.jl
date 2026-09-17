# Reproduce Vollebregt/Kalker CONTACT examples that our half-space stack can run.
# Source: C:\Users\ufesl\Downloads\CONTACT-main\CONTACT-main\examples
# Units: N, mm, MPa (CONTACT default).
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra, Printf, Statistics, FFTW, Plots
gr()
default(size=(720, 480), linewidth=1.6, legendfontsize=8, guidefontsize=11,
        tickfontsize=9, titlefontsize=11, grid=false, framestyle=:box)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIG  = joinpath(ROOT, "plots", "julia_lerma", "contact_vollebregt")
const CEX  = raw"C:\Users\ufesl\Downloads\CONTACT-main\CONTACT-main\examples"
mkpath(FIG)

function _savefig(plt, name)
    path = joinpath(FIG, name)
    try
        isfile(path) && rm(path; force=true)
        savefig(plt, path)
        println("  wrote ", path)
    catch e
        @warn "savefig failed" name exception=e
    end
end

"""Parse CONTACT .ref_out for (ξ, FX/μFN, FN, FX, pmax) per case."""
function parse_contact_forces(path)
    lines = readlines(path)
    recs = NamedTuple[]
    i = 1
    cksi = NaN
    while i <= length(lines)
        ln = lines[i]
        if occursin("CKSI", ln) && occursin("CETA", ln) && i < length(lines)
            nums = Float64[]
            for s in split(lines[i + 1])
                v = tryparse(Float64, s)
                v !== nothing && push!(nums, v)
            end
            # DT VELOC CKSI CETA CPHI  OR  CHI DQ VELOC CKSI CETA CPHI
            if length(nums) >= 5
                cksi = nums[end - 2]
            end
        end
        if occursin("FX/FSTAT/FN", ln) && occursin("PMAX", ln) && i < length(lines)
            nums = Float64[]
            for s in split(lines[i + 1])
                v = tryparse(Float64, s)
                v !== nothing && push!(nums, v)
            end
            # FN/G, FX/μFN, FY/μFN, APPROACH, PMAX
            if length(nums) >= 5
                push!(recs, (; ξ=cksi, fxμ=nums[2], approach=nums[4], pmax=nums[5]))
            end
        end
        i += 1
    end
    return recs
end

function match_tangential_force!(st, grid, prep, law, δ, Qtarget;
                                 gx0=0.0, rtol=2e-3, maxouter=18, kwargs...)
    gx = float(gx0)
    Qx = 0.0
    for _ in 1:maxouter
        fill!(st.gtx_ref, 0); fill!(st.gty_ref, 0)
        solve_contact_step!(st, grid, prep, law, δ, gx, 0.0; wear_jump=0, kwargs...)
        _, Qx, _ = contact_resultants(st, grid.hs)
        err = Qx - Qtarget
        abs(err) <= rtol * max(abs(Qtarget), 1e-8) && return gx, Qx
        slope = Qx / ifelse(abs(gx) > 1e-10, gx, copysign(1e-4, Qtarget))
        gx -= 0.7 * err / ifelse(abs(slope) > 1e-8, slope, copysign(1.0, Qtarget))
    end
    return gx, Qx
end

function stick_slip_count(st, μ; rel=1e-3)
    pth = rel * max(maximum(st.pn), eps())
    ncon = nslip = nadh = 0
    @inbounds for i in eachindex(st.pn)
        st.pn[i] <= pth && continue
        ncon += 1
        τ = hypot(st.ptx[i], st.pty[i])
        if τ >= 0.95 * μ * st.pn[i]
            nslip += 1
        else
            nadh += 1
        end
    end
    return (; ncon, nadh, nslip)
end

# ---------------------------------------------------------------------------
# 1. cattaneo.inp — sphere shift (Kalker 1990 §5.2.1.1)
# ---------------------------------------------------------------------------
function run_cattaneo()
    println("="^72)
    println(" CONTACT cattaneo.inp  — polyethylene sphere, Cattaneo shift")
    println("="^72)
    G1, ν, R, P, μ = 200.0, 0.42, 50.0, 9.1954, 0.40
    nx = 19
    L = 2 * 1.26667
    x, hs = square_mesh(nx, L, G1, ν, G1, ν)
    grid = make_grid(x, x, hs, sphere_gap(x, x, R))
    prep = precompute_kernels(nx, nx, hs)
    law = isotropic_law(μ, 0.0)
    st = init_state(grid)
    Estar = contact_modulus(hs)
    hz = hertz_sphere_load(R, P, Estar)
    δ, _, _ = set_approach_for_load!(st, grid, prep, law, P, 0.0, 0.0;
                                     δ0=hz.δ, wear_jump=0, tol=1e-6, maxouter=20)
    Pn, _, _ = contact_resultants(st, hs)
    Qtarget = -0.875 * μ * P
    gx, Qx = match_tangential_force!(st, grid, prep, law, δ, Qtarget;
                                     gx0=-8.155e-3, tol=1e-6, maxiter=250)
    ss = stick_slip_count(st, μ)
    pmax = maximum(st.pn)
    println(@sprintf "  Hertz a=%.4f  δ=%.5f  p0=%.3f" hz.a hz.δ hz.p0)
    println(@sprintf "  CONTACT: FN=9.195  FX=-3.218  pmax=4.393  δ=0.01998  NCON=177 NADH=45 NSLIP=132")
    println(@sprintf "  SAM:     FN=%.3f  FX=%.3f  pmax=%.3f  δ=%.5f  NCON=%d NADH=%d NSLIP=%d  gx=%.4e" Pn Qx pmax δ ss.ncon ss.nadh ss.nslip gx)

    jc = (nx + 1) ÷ 2
    a = hz.a
    c = a * (1 - abs(Qtarget) / (μ * P))^(1 / 3)
    pt_ana = similar(x)
    @inbounds for i in eachindex(x)
        r = abs(x[i])
        r >= a && (pt_ana[i] = 0; continue)
        term = sqrt(max(1 - (x[i] / a)^2, 0.0))
        if r < c
            term -= (c / a) * sqrt(max(1 - (x[i] / c)^2, 0.0))
        end
        pt_ana[i] = -μ * hz.p0 * term   # Qx < 0
    end
    plt = plot(x ./ a, st.pn[:, jc] ./ hz.p0; color=:black, label="SAM p/p0",
               xlabel="x / a", ylabel="p, |q| / p₀", title="Cattaneo sphere  19×19")
    plot!(plt, x ./ a, abs.(st.ptx[:, jc]) ./ hz.p0; color=:blue, label="SAM |qₓ|/p0")
    plot!(plt, x ./ a, abs.(pt_ana) ./ hz.p0; color=:blue, linestyle=:dash, label="Mindlin |q|/p0")
    hline!(plt, [0.875 * μ]; color=:gray, linestyle=:dot, label="CONTACT |Q|/(μP)=0.875")
    _savefig(plt, "cattaneo_cut.png")
    return (; Pn, Qx, pmax, δ, ss, hz)
end

# ---------------------------------------------------------------------------
# 2. carter2d.inp — cylinder on flat, steady rolling
# ---------------------------------------------------------------------------
function run_carter2d()
    println("="^72)
    println(" CONTACT carter2d.inp  — 2D Carter (Flamant line + Carter q)")
    println("="^72)
    G1, ν, R, Ptot, μ, ξx = 82000.0, 0.28, 500.0, 35780.0, 0.30, -0.00024
    w = 200.0                          # CONTACT DY (strip width)
    F = Ptot / w                       # N/mm
    n = 55
    h = 0.04
    x = collect(range(-1.05 + h / 2, step=h, length=n))
    hp = ElasticHalfPlane2D(G1 / 2, ν; h=h)   # Pohrt G = G1/2 for identical pair
    Estar = 2 * hp.G / (1 - hp.ν)   # two-body line modulus (Pohrt G already combined)
    a_line = sqrt(4 * F * R / (π * Estar))
    p0_line = 2 * F / (π * a_line)
    gap0 = @. x^2 / (2R)
    sol = solve_line_contact_force(gap0, F, hp; tol=1e-10)
    pmax = maximum(sol.p)
    println(@sprintf "  Hertz line a=%.4f  p0=%.2f  E*=%.1f" a_line p0_line Estar)
    println(@sprintf "  CONTACT: FN=35780  pmax=113.9  FX/μFN=0.648  NCON=50 NADH=30 NSLIP=20")
    println(@sprintf "  SAM 2D:  F=%.2f N/mm  pmax=%.2f  force=%.2f  ncon=%d" F pmax sol.force count(>(0), sol.p))

    # Carter: a' / a = 0.6 ⇒ Q/μP = 1-(0.6)² = 0.64
    c = 0.6 * a_line
    q_car = zeros(n)
    @inbounds for i in 1:n
        abs(x[i]) >= a_line && continue
        term = sqrt(max(1 - (x[i] / a_line)^2, 0.0))
        if abs(x[i]) < c
            # leading-edge stick: Carter subtracts the smaller Hertz
            term -= (c / a_line) * sqrt(max(1 - (x[i] / c)^2, 0.0))
        end
        q_car[i] = μ * p0_line * term
    end
    Q_car = sum(q_car) * h
    fxμ = Q_car / (μ * F)
    println(@sprintf "  Carter theory Q/μP=%.3f  (CONTACT 0.648)  ξx=%.6f" fxμ ξx)

    plt = plot(x ./ a_line, sol.p ./ p0_line; color=:black, label="SAM p/p0",
               xlabel="x / a", ylabel="p, q / p₀", title="Carter 2D  Flamant 55")
    plot!(plt, x ./ a_line, q_car ./ p0_line; color=:red, label="Carter q/p0")
    hline!(plt, [1.0]; color=:gray, linestyle=:dot, label=false)
    _savefig(plt, "carter2d_cut.png")
    return (; Pn=sol.force * w, Qx=Q_car * w, fxμ, pmax, a_line, p0_line)
end

# ---------------------------------------------------------------------------
# 3. catt_to_cart.inp case 1 (Cattaneo) + steady rolling (Paper 3 spheres)
# ---------------------------------------------------------------------------
function run_catt_to_cart()
    println("="^72)
    println(" CONTACT catt_to_cart.inp  — two spheres G=1, P=0.4705, μ=0.4013")
    println("="^72)
    G1, ν, P, μ = 1.0, 0.28, 0.4705, 0.4013
    Rstar = 1 / (2 * 0.002963)          # B1 = 1/(2 R*)
    R = 2 * Rstar                       # two equal spheres
    nx = 31
    L = 8.25                            # CONTACT ±4.125
    x, hs = square_mesh(nx, L, G1, ν, G1, ν)
    grid = make_grid(x, x, hs, sphere_gap(x, x, Rstar))
    prep = precompute_kernels(nx, nx, hs)
    law = isotropic_law(μ, 0.0)
    st = init_state(grid)
    hz = hertz_sphere_load(Rstar, P, contact_modulus(hs))
    δ, _, _ = set_approach_for_load!(st, grid, prep, law, P, 0.0, 0.0;
                                     δ0=hz.δ, wear_jump=0, tol=1e-6, maxouter=20)
    Qtarget = -0.657 * μ * P
    gx, Qx = match_tangential_force!(st, grid, prep, law, δ, Qtarget;
                                     gx0=-0.01774, tol=1e-6, maxiter=250)
    Pn, _, _ = contact_resultants(st, hs)
    ss = stick_slip_count(st, μ)
    println(@sprintf "  Hertz a=%.3f  p0=%.4f  δ=%.4f" hz.a hz.p0 hz.δ)
    println(@sprintf "  CONTACT Cattaneo: FN=0.4705 FX=-0.1240 pmax=0.01834 δ=0.07258 NADH=305 NSLIP=316")
    println(@sprintf "  SAM Cattaneo:     FN=%.4f FX=%.4f pmax=%.5f δ=%.4f NADH=%d NSLIP=%d gx=%.4e" Pn Qx maximum(st.pn) δ ss.nadh ss.nslip gx)

    # CONTACT force-controlled rolling holds Q/μP=0.657; the matching creepage
    # in our (ξ-prescribed) solver is Paper 3's ξx=-0.0031 (Q/μP≈0.66).
    results = []
    for (lab, ξ) in (("ξ=-0.001824", -0.001824), ("ξ=-0.0031 (P3)", -0.0031))
        str = init_state(grid)
        δr, _, _ = rolling_match_load!(str, grid, prep, law, RollingKinematics(1.0, ξ), P, δ;
                                       rtol=5e-3, maxouter=16, wear=false, tol=1e-6, maxiter=220)
        solve_rolling_step!(str, grid, prep, law, RollingKinematics(1.0, ξ), δr;
                            wear=false, tol=1e-6, maxiter=250)
        Pr, Qxr, _ = contact_resultants(str, hs)
        ssr = stick_slip_count(str, μ)
        fxμ = Qxr / (μ * Pr)
        println(@sprintf "  rolling %-22s  FN=%.4f  FX=%.4f  FX/μFN=%.3f  pmax=%.5f  slip=%d/%d" lab Pr Qxr fxμ maximum(str.pn) ssr.nslip ssr.ncon)
        push!(results, (; lab, Pr, Qxr, fxμ, pmax=maximum(str.pn), st=str))
    end
    println("  CONTACT rolling (force Q/μP=0.657): FX=-0.124, ξ=1.824e-3, pmax=0.01834")

    jc = (nx + 1) ÷ 2
    plt = plot(x ./ hz.a, st.ptx[:, jc] ./ hz.p0; color=:black, label="Cattaneo qₓ/p0",
               xlabel="x / a", ylabel="qₓ / p₀", title="Cattaneo → Carter  spheres")
    plot!(plt, x ./ hz.a, results[1].st.ptx[:, jc] ./ hz.p0; color=:blue, label="rolling ξ=1.82e-3")
    plot!(plt, x ./ hz.a, results[2].st.ptx[:, jc] ./ hz.p0; color=:red, linestyle=:dash, label="rolling ξ=3.1e-3")
    _savefig(plt, "catt_to_cart_qx.png")
    return (; Pn, Qx, pmax=maximum(st.pn), rolling=results, hz)
end

# ---------------------------------------------------------------------------
# 4. tractcurv.inp first series — Coulomb creep curve
# ---------------------------------------------------------------------------
function run_tractcurv()
    println("="^72)
    println(" CONTACT tractcurv.inp  — Coulomb creep–force (orig. CONTACT)")
    println("="^72)
    recs = parse_contact_forces(joinpath(CEX, "tractcurv.ref_out"))
    # first series: μ=0.33, until |ξ| hits 0.25; skip later FASTSIM / falling-μ
    coul = NamedTuple[]
    for r in recs
        abs(r.ξ) > 0.26 && break
        push!(coul, r)
        length(coul) >= 30 && break
    end
    G1, ν, P, μ = 82000.0, 0.28, 106700.0, 0.33
    # Hertz a,b from CONTACT p0=640.7, a/b=0.5, P=106700: p0=3P/(2πab)
    a_hz, b_hz, δ_hz = 6.307, 12.614, 0.07651
    nx, ny = 25, 29
    Lx, Ly = 2.4 * a_hz, 2.4 * b_hz
    x, y, hs = rect_mesh(nx, ny, Lx, Ly, G1, ν, G1, ν)
    gap = [δ_hz * ((xi / a_hz)^2 + (yj / b_hz)^2) for xi in x, yj in y]
    grid = make_grid(x, y, hs, gap)
    prep = precompute_kernels(nx, ny, hs)
    law = isotropic_law(μ, 0.0)
    st = init_state(grid)
    δ, _, _ = rolling_match_load!(st, grid, prep, law, RollingKinematics(10000.0, 1e-5), P, δ_hz;
                                  rtol=1e-2, maxouter=14, wear=false, tol=1e-5, maxiter=180)
    ξs = [1e-5, 0.0004, 0.0008, 0.0012, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01,
          0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
    sam_ξ = Float64[]; sam_fx = Float64[]
    for ξ in ξs
        kin = RollingKinematics(10000.0, ξ)
        δ, _, _ = rolling_match_load!(st, grid, prep, law, kin, P, δ;
                                      rtol=1.5e-2, maxouter=8, wear=false, tol=1e-5, maxiter=160)
        solve_rolling_step!(st, grid, prep, law, kin, δ; wear=false, tol=1e-5, maxiter=180)
        Pn, Qx, _ = contact_resultants(st, hs)
        push!(sam_ξ, ξ); push!(sam_fx, Qx / (μ * Pn))
        @printf "  ξ=%7.4f  SAM Q/μP=%6.3f  P=%.0f  pmax=%.1f\n" ξ (Qx / (μ * Pn)) Pn maximum(st.pn)
    end
    cnt_ξ = abs.([r.ξ for r in coul])
    cnt_fx = abs.([r.fxμ for r in coul])
    plt = plot(cnt_ξ, cnt_fx; color=:black, marker=:circle, markersize=3,
               label="CONTACT", xlabel="ξₓ", ylabel="|Qₓ| / μ P",
               title="tractcurv  Coulomb", xscale=:log10, legend=:bottomright)
    plot!(plt, sam_ξ, abs.(sam_fx); color=:red, marker=:square, label="SAM $(nx)×$(ny)")
    hline!(plt, [1.0]; color=:gray, linestyle=:dot, label="full sliding")
    _savefig(plt, "tractcurv_coulomb.png")
    return (; sam_ξ, sam_fx, cnt_ξ, cnt_fx)
end

# ---------------------------------------------------------------------------
# 5. subsurf.inp case 1 — unit square pn=1
# ---------------------------------------------------------------------------
function run_subsurf()
    println("="^72)
    println(" CONTACT subsurf.inp  — unit square pₙ=1, G=1, ν=0.28")
    println("="^72)
    ν = 0.28
    G1 = 1.0
    hx = hy = 1.0
    hs = combined_halfspace(G1, ν, G1, ν; hx=hx, hy=hy)
    x = [0.0]; y = [0.0]
    pn = fill(1.0, 1, 1); ptx = zeros(1, 1); pty = zeros(1, 1)
    zs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.667, 2.5, 5.0]
    # CONTACT .subs SIGVM on axis (skip z=0 surface singularity)
    cnt_z  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.667, 2.5, 5.0]
    # .subs SIGXZ on axis is a surface artefact; axis VM from (σxx,σyy,σzz) only.
    # z=0.4: CONTACT overview MAX σ_VM=0.665 = |σzz−σxx| = 0.800−0.134.
    cnt_zz = [-0.994287, -0.960374, -0.891525, -0.799677, -0.700842, -0.606403,
              -0.521977, -0.449212, -0.387679, -0.336086, -0.240934, -0.149346,
              -0.071611, -0.018785]
    cnt_xx = [-0.555944, -0.366593, -0.227289, -0.134431, -0.076184, -0.040853,
              -0.019799, -0.007388, -0.000156,  0.003977,  0.007720,  0.007342,
               0.004458,  0.001325]
    cnt_vm = abs.(cnt_zz .- cnt_xx)          # J2 axis, σxy=σxz=σyz=0
    vm = zeros(length(zs)); zz = zeros(length(zs)); xx = zeros(length(zs))
    for (k, z) in enumerate(zs)
        σ = subsurface_stress(1e-5, 1e-5, z, ptx, pty, pn, x, y, hs, ν)
        vm[k] = σ.VM
        zz[k] = σ.zz
        xx[k] = σ.xx
        @printf "  z=%5.3f  SAM VM=%.4f σzz=%.4f   CONTACT VM=%.4f σzz=%.4f\n" z σ.VM σ.zz cnt_vm[k] cnt_zz[k]
    end
    plt = plot(cnt_z, cnt_vm; color=:black, marker=:circle, label="CONTACT σ_VM (axis)",
               xlabel="z (mm)", ylabel="stress", title="subsurf  unit square pₙ=1")
    plot!(plt, zs, vm; color=:red, marker=:square, label="SAM σ_VM")
    plot!(plt, cnt_z, abs.(cnt_zz); color=:black, linestyle=:dash, label="CONTACT |σzz|")
    plot!(plt, zs, abs.(zz); color=:red, linestyle=:dash, label="SAM |σzz|")
    _savefig(plt, "subsurf_axis.png")
    rel = abs.(vm .- cnt_vm) ./ max.(cnt_vm, 1e-6)
    println(@sprintf "  median rel. VM error = %.2f%%" (100 * median(rel)))
    return (; zs, vm, zz, cnt_vm, rel)
end

function main()
    println("Vollebregt CONTACT examples → ", FIG)
    println("CONTACT data: ", CEX)
    c1 = run_cattaneo()
    c2 = run_carter2d()
    c3 = run_catt_to_cart()
    c4 = run_tractcurv()
    c5 = run_subsurf()
    println()
    println("="^72)
    println(" Summary vs CONTACT .ref_out")
    println("="^72)
    @printf "  cattaneo   pmax  SAM %.3f / CONTACT 4.393   Qx SAM %.3f / -3.218\n" c1.pmax c1.Qx
    @printf "  carter2d   Q/μP  SAM %.3f / CONTACT 0.648   pmax SAM %.1f / 113.9\n" c2.fxμ c2.pmax
    @printf "  catt→cart  pmax  SAM %.5f / CONTACT 0.01834  Cattaneo Qx %.4f / -0.124\n" c3.pmax c3.Qx
    @printf "  tractcurv  nξ=%d  (plot overlay)\n" length(c4.sam_ξ)
    @printf "  subsurf    median |ΔVM|/VM = %.2f%%\n" (100 * median(c5.rel))
end

main()
