# Paper 1 §4.2 — spherical punch under orthotropic fretting wear
# Juliá & Rodríguez-Tembleque, Int. J. Mech. Sci. (2022) doi:10.1016/j.ijmecsci.2022.107695
#
# Paper mesh is 61×61 on 1.6 mm × 1.6 mm; override with N= env var.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra
using Printf
using Statistics
using FFTW

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

const Nmesh = parse(Int, get(ENV, "N", "41"))
const TOLS  = parse(Float64, get(ENV, "TOL", "1e-6"))
const MAXIT = parse(Int, get(ENV, "MAXIT", "250"))
const DO_VM = get(ENV, "DO_VM", "true") == "true"
const SNAP  = (100, 10_000, 100_000)
const BETAS = (0.0, 60.0, 67.0, 70.0, 80.0, 90.0)

# Fig. 15 colour-bar peaks (mm), read off the paper scales.
const PAPER_WMAX = Dict(
    (0,   100) => 7.0e-5, (60, 100) => 6.0e-5, (70, 100) => 7.0e-5,
    (80,  100) => 7.0e-5, (90, 100) => 7.0e-5,
    (0, 10_000) => 5.0e-3, (60, 10_000) => 4.0e-3, (70, 10_000) => 4.0e-3,
    (80, 10_000) => 3.0e-3, (90, 10_000) => 2.0e-3,
    (0, 100_000) => 3.0e-2, (60, 100_000) => 2.5e-2, (70, 100_000) => 1.6e-2,
    (80, 100_000) => 4.0e-3, (90, 100_000) => 2.0e-3,
)

μ_ellipse(β) = 1 / hypot(cos(β) / FRET.μ1, sin(β) / FRET.μ2)
μ_intens(β)  = hypot(FRET.μ1 * cos(β), FRET.μ2 * sin(β))
δt_incipient(μ, δ, ν) = μ * (2 - ν) / (2 * (1 - ν)) * δ

function slip_stats(st, law; rel=1e-3)
    μ1, μ2, β = law.μ1, law.μ2, law.β
    cβ, sβ = cos(β), sin(β)
    pmax = maximum(st.pn)
    thresh = rel * max(pmax, eps())
    n_contact = n_slip = n_stick = 0
    @inbounds for i in eachindex(st.pn)
        pn = st.pn[i]
        pn <= thresh && continue
        n_contact += 1
        pe1 =  cβ * st.ptx[i] + sβ * st.pty[i]
        pe2 = -sβ * st.ptx[i] + cβ * st.pty[i]
        nμ = hypot(pe1 / μ1, pe2 / μ2)
        if nμ > 0.95 * pn
            n_slip += 1
        else
            n_stick += 1
        end
    end
    return (; n_contact, n_slip, n_stick,
            slip_frac = n_contact == 0 ? 0.0 : n_slip / n_contact)
end

function edge_contact(st; rel=1e-3)
    thresh = rel * max(maximum(st.pn), eps())
    nx, ny = size(st.pn)
    n = 0
    for i in 1:nx
        (st.pn[i, 1] > thresh || st.pn[i, ny] > thresh) && (n += 1)
    end
    for j in 1:ny
        (st.pn[1, j] > thresh || st.pn[nx, j] > thresh) && (n += 1)
    end
    return n
end

function wear_shape(st, x)
    nx, ny = size(st.w)
    ic, jc = (nx + 1) ÷ 2, (ny + 1) ÷ 2
    wmax = maximum(st.w)
    wcen = st.w[ic, jc]
    # radial location of max wear (crown vs centre)
    rmax = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        if st.w[i, j] >= 0.95 * wmax
            r = hypot(x[i], x[j])
            r > rmax && (rmax = r)
        end
    end
    return (; wmax, wcen, r_wmax=rmax, crown = wmax > 0 && wcen / wmax < 0.35)
end

function vm_peak(st, x, hs, ν, a0)
    # coarse xz plane, skip z=0 (surface kernel)
    xs = collect(range(-1.2 * a0, 1.2 * a0; length=17))
    zs = collect(range(0.04 * a0, 1.6 * a0; length=12))
    σ = subsurface_plane(xs, zs, 0.0, st.ptx, st.pty, st.pn, x, x, hs, ν)
    vm, idx = findmax(σ)
    i, k = Tuple(idx)
    return vm, xs[i], zs[k]
end

function stroke!(st, grid, prep, law, δ, gx; wear_jump=1.0)
    niter, Ψ = solve_contact_step!(st, grid, prep, law, δ, gx, 0.0;
                                   wear_jump=wear_jump, tol=TOLS, maxiter=MAXIT)
    commit_tangential_ref!(st)
    return niter, Ψ
end

function snapshot_row(Ncy, βdeg, st, grid, x, hs, hz, a0, p0)
    P, Qx, Qy = contact_resultants(st, hs)
    sl = slip_stats(st, OrthotropicLaw(FRET.μ1, FRET.μ2, FRET.i1, FRET.i2, deg2rad(βdeg)))
    ws = wear_shape(st, x)
    stats = contact_patch_stats(st, hs)
    pmax = maximum(st.pn)
    regime = sl.slip_frac >= 0.85 ? "gross" : sl.slip_frac <= 0.70 ? "partial" : "mixed"
    vm = NaN; xv = NaN; zv = NaN
    if DO_VM
        vm, xv, zv = vm_peak(st, x, hs, FRET.ν, a0)
    end
    return (; N=Ncy, β=βdeg, P, Qx, Qy, pmax, pmax_p0=pmax / p0,
            a=stats.a, a_a0=stats.a / a0, wmax=ws.wmax, wcen=ws.wcen,
            r_wmax=ws.r_wmax, crown=ws.crown, slip_frac=sl.slip_frac,
            n_contact=sl.n_contact, n_slip=sl.n_slip, n_stick=sl.n_stick,
            regime, edge=edge_contact(st), vm_p0=vm / p0, z_vm=zv / a0, x_vm=xv / a0)
end

function print_row(r)
    @printf "  N=%6d  β=%5.1f  %-7s  P=%8.2f  Qx=%8.2f  pmax/p0=%5.2f  a/a0=%5.2f  wmax=%9.2e  wcen/wmax=%5.2f  slip=%5.1f%%  stick=%4d  edge=%d" r.N r.β r.regime r.P r.Qx r.pmax_p0 r.a_a0 r.wmax (r.wmax == 0 ? 0.0 : r.wcen / r.wmax) 100 * r.slip_frac r.n_stick r.edge
    if isfinite(r.vm_p0)
        @printf "  σVM/p0=%5.2f @ z/a0=%4.2f" r.vm_p0 r.z_vm
    end
    println()
end

function run_beta(βdeg, x, hs, grid, prep, hz)
    β = deg2rad(βdeg)
    law = OrthotropicLaw(FRET.μ1, FRET.μ2, FRET.i1, FRET.i2, β)
    st = init_state(grid)
    rows = NamedTuple[]

    # N = 0: normal indent, no wear
    niter, Ψ = solve_contact_step!(st, grid, prep, law, FRET.δ, 0.0, 0.0;
                                   wear_jump=0, tol=TOLS, maxiter=MAXIT)
    commit_tangential_ref!(st)
    push!(rows, snapshot_row(0, βdeg, st, grid, x, hs, hz, hz.a, hz.p0))
    print_row(rows[end])
    if Ψ > 10 * TOLS
        @printf "    warn: indent Ψ=%.2e niter=%d\n" Ψ niter
    end

    # first stroke 0 → +amp (counts toward cycle 1, no jump)
    niter, Ψ = stroke!(st, grid, prep, law, FRET.δ, FRET.amp; wear_jump=1)
    push!(rows, snapshot_row(0, βdeg, st, grid, x, hs, hz, hz.a, hz.p0))  # overwritten conceptually
    # keep going

    Ndone = 0
    gx_sign = -1.0          # next stroke goes to -amp (complete cycle 1)
    w_prev = maximum(st.w)
    hx = hs.hx
    cap = 0.03 * hx         # max wear increment per jumped block

    while Ndone < SNAP[end]
        target = SNAP[searchsortedfirst(collect(SNAP), Ndone + 1)]
        dw = max(maximum(st.w) - w_prev, 0.0)
        # dw is wear of the last *stroke*; a cycle is two strokes
        dw_cycle = max(dw, 1e-16)
        jump_wear = max(1, floor(Int, cap / dw_cycle))
        ΔN = min(jump_wear, target - Ndone)
        # first few cycles explicit
        Ndone < 4 && (ΔN = 1)
        w_prev = maximum(st.w)

        n1, Ψ1 = stroke!(st, grid, prep, law, FRET.δ, gx_sign * FRET.amp; wear_jump=ΔN)
        n2, Ψ2 = stroke!(st, grid, prep, law, FRET.δ, -gx_sign * FRET.amp; wear_jump=ΔN)
        # re-equilibrate pressure on the worn surface
        solve_contact_step!(st, grid, prep, law, FRET.δ, -gx_sign * FRET.amp, 0.0;
                            wear_jump=0, tol=TOLS, maxiter=MAXIT)
        gx_sign = -gx_sign
        Ndone += ΔN
        if max(Ψ1, Ψ2) > 50 * TOLS
            @printf "    warn: N=%d  Ψ=(%.1e, %.1e)  niter=(%d,%d)  ΔN=%d\n" Ndone Ψ1 Ψ2 n1 n2 ΔN
        end
        if Ndone in SNAP
            r = snapshot_row(Ndone, βdeg, st, grid, x, hs, hz, hz.a, hz.p0)
            push!(rows, r)
            print_row(r)
        end
    end
    return rows
end

function main()
    println("="^88)
    println(" Paper 1 §4.2  spherical-punch fretting   mesh ", Nmesh, "×", Nmesh,
            "  L=", FRET.L, " mm")
    println("="^88)

    G = FRET.E / (2 * (1 + FRET.ν))
    x, hs = square_mesh(Nmesh, FRET.L, G, FRET.ν, G, FRET.ν)
    grid = make_grid(x, x, hs, sphere_gap(x, x, FRET.R))
    tprep = time()
    prep = precompute_kernels(Nmesh, Nmesh, hs)
    hz = hertz_sphere(FRET.R, FRET.δ, contact_modulus(hs))
    @printf "Hertz  a0=%.4f mm  P=%.2f N  p0=%.1f MPa  δ=%.3f μm  E*=%.0f MPa\n" hz.a hz.P hz.p0 1e3 * FRET.δ contact_modulus(hs)
    @printf "mesh   hx=%.4f mm  domain=±%.3f mm = ±%.2f a0   kernels %.2fs\n" hs.hx FRET.L / 2 (FRET.L / 2) / hz.a (time() - tprep)
    println()
    println("Mindlin incipient-slip amplitude  δt* = μ (2-ν)/(2(1-ν)) δn")
    @printf "  imposed amp = %.4f mm   amp/δn = %.3f   μ_crit (amp=δt*) = %.3f\n" FRET.amp FRET.amp / FRET.δ (FRET.amp / FRET.δ) * 2 * (1 - FRET.ν) / (2 - FRET.ν)
    println("   β      μ_ellipse   μ_intens   δt*/amp    Mindlin regime")
    for βdeg in BETAS
        β = deg2rad(βdeg)
        μe, μi = μ_ellipse(β), μ_intens(β)
        dte = δt_incipient(μe, FRET.δ, FRET.ν)
        dti = δt_incipient(μi, FRET.δ, FRET.ν)
        # ellipse-in-x is the relevant friction capacity for Qx under x-sliding
        reg = FRET.amp > dte ? "gross" : "partial"
        @printf "  %5.1f   %8.3f    %8.3f    %8.3f     %s (δt_ell=%.4f, δt_int=%.4f)\n" βdeg μe μi dte / FRET.amp reg dte dti
    end
    println()
    println("Paper (Fig. 15 / text): gross slip for small β, partial-slip crown for β ≳ 67°;")
    println("  at N=1e5, pmax > 2 p0 in partial slip; wear fills/widens in gross slip.")
    println()

    t0 = time()
    allrows = NamedTuple[]
    for βdeg in BETAS
        println("── β = ", βdeg, "° ──")
        append!(allrows, run_beta(βdeg, x, hs, grid, prep, hz))
        println()
    end
    @printf "elapsed %.1f s\n" (time() - t0)

    println()
    println("="^88)
    println(" Comparison vs Paper 1 Fig. 15  (w_max, mm)")
    println("="^88)
    @printf "%6s %8s %12s %12s %10s  %s\n" "N" "β" "w_num" "w_paper" "ratio" "regime/notes"
    for r in allrows
        r.N == 0 && continue
        key = (Int(round(r.β)), r.N)
        haskey(PAPER_WMAX, key) || continue
        wp = PAPER_WMAX[key]
        ratio = r.wmax / wp
        note = String[]
        r.crown && push!(note, "crown")
        !r.crown && r.wmax > 0 && push!(note, "filled")
        r.edge > 0 && push!(note, "EDGE")
        r.pmax_p0 > 2 && push!(note, "pmax>2p0")
        @printf "%6d %8.1f %12.3e %12.3e %10.2f  %s %s\n" r.N r.β r.wmax wp ratio r.regime join(note, ",")
    end

    println()
    println("Regime check (paper: β≲60° gross, β≳70° partial, threshold ≈ 67°):")
    for Ncy in SNAP
        @printf "  N=%d " Ncy
        for r in allrows
            r.N == Ncy || continue
            @printf "  β=%g:%s(slip=%.0f%%,crown=%s)" r.β r.regime 100 * r.slip_frac r.crown
        end
        println()
    end
end

main()
