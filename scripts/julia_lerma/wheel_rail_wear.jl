# Wheel wear on the CONTACT Manchester A-2.2 patch.
# CONTACT itself does not evolve profiles; it prints FRIC.POWER [W] for an
# outer Archard loop (user guide §1). Here we apply the same circumferential
# groove as Paper 3 (Ipass = ∫ |pn| ‖s‖_i dx / V) on the WR contact grid.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra, Printf, Statistics, FFTW, Plots
gr()
default(size=(720, 480), linewidth=1.6, legendfontsize=8, guidefontsize=11,
        tickfontsize=9, titlefontsize=11, grid=false, framestyle=:box)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DATA = joinpath(ROOT, "data", "contact", "vollebregt")
const FIG  = joinpath(ROOT, "plots", "julia_lerma", "wheel_rail")
const REP  = raw"C:\Users\ufesl\OneDrive\artigos\escritos\2026\half-space\figures\wheel_rail"
mkpath(FIG); mkpath(REP)

function _savefig(plt, name)
    for dir in (FIG, REP)
        path = joinpath(dir, name)
        try
            isfile(path) && rm(path; force=true)
            savefig(plt, path)
            println("  wrote ", path)
        catch e
            @warn "savefig failed" path exception=e
        end
    end
end

"""CONTACT FRIC.POWER [W]: V ∫ Pt·st dA, with st the shift [mm] / (1e3 dt).
Steady rolling: dt = dx/V, so P_fric = ∫ Pt · (s_vel) dA / 1000  (N·mm/s → W)."""
function fric_power_W(st, grid, kin)
    nx, ny = size(st.pn)
    sx = zeros(nx, ny); sy = similar(sx)
    upwind_slip!(sx, sy, st.ux, st.uy, grid.x, grid.y, kin)
    dA = grid.hs.hx * grid.hs.hy
    acc = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        acc += (st.ptx[i, j] * sx[i, j] + st.pty[i, j] * sy[i, j]) * dA
    end
    return acc / 1000   # N·mm/s → W
end

function setup_patch(trk, ws, rail, wheel; side=:left, dx=0.2, ds=0.2, μ=0.3)
    res = solve_wheel_rail(trk, ws, rail, wheel; side=side, force=true, dx=dx, ds=ds,
                           maxit=8, maxiter=350, rtol=0.025, μ=μ)
    isempty(res.patches) && error("no contact")
    sgn = side === :left ? -1.0 : 1.0
    ws2 = deepcopy(ws); ws2.z = res.z_ws
    m_rail, _, _ = set_rail_marker(trk, rail, sgn)
    m_w, m_ws = set_wheel_markers(ws2, sgn)
    cps = locate_patches(rail, wheel, m_rail, m_w, m_ws, ws2, sgn; dx=dx, ds=ds)
    cp = cps[1]
    wheel_spl = is_varprof(wheel) ? profile_at_theta(wheel, ws2.pitch) : wheel.spl
    m_wheel_trk = marker_2glob(m_w, m_ws)
    x, s, h, pen = undeformed_distance(cp, rail, wheel_spl, m_rail, m_wheel_trk, ws2.nom_radius;
                                       wheel=wheel, pitch=ws2.pitch)
    x, s, h, pen = BEM.Contact.WheelRail._recenter_gap!(cp, rail, wheel_spl, m_rail, m_wheel_trk,
                                                ws2.nom_radius, x, s, h, pen; wheel=wheel, pitch=ws2.pitch)
    x, s, h, pen = BEM.Contact.WheelRail._recenter_gap!(cp, rail, wheel_spl, m_rail, m_wheel_trk,
                                                ws2.nom_radius, x, s, h, pen; wheel=wheel, pitch=ws2.pitch)
    hs = combined_halfspace(82000.0, 0.28, 82000.0, 0.28; hx=cp.dx_eff, hy=cp.ds_eff)
    grid = make_grid(x, s, hs, h)
    prep = precompute_kernels(length(x), length(s), hs)
    st = init_state(grid)
    V, ξx, ξy, φ = creepage_at_patch(cp, ws2, m_w, m_ws, sgn; pen=pen)
    kin = RollingKinematics(V, ξx, ξy, φ)
    law_c = isotropic_law(μ, 0.0)
    niter, Ψ = solve_rolling_step!(st, grid, prep, law_c, kin, pen;
                                   wear=false, tol=1e-6, maxiter=350, johnson=true)
    P, Qx, Qy = contact_resultants(st, hs)
    pw = fric_power_W(st, grid, kin)
    return (; res, st, grid, prep, kin, law_c, pen, P, Qx, Qy, pw, V, ξx, ξy, φ,
            x, s, YCP=sgn * oy(cp.mref))
end

function evolve!(S, iwear, snaps; Fz=10000.0, μ=0.3)
    law_w = isotropic_law(μ, iwear)
    st, grid, prep, kin = S.st, S.grid, S.prep, S.kin
    nx, ny = size(st.pn)
    sx = zeros(nx, ny); sy = similar(sx)
    Ipass = zeros(ny)
    rolling_pass_wear!(Ipass, sx, sy, st, grid, law_w, kin)
    Nrec = Int[0]
    wmax = Float64[0.0]
    fields = Dict{Int,NamedTuple}()
    fields[0] = (; pn=copy(st.pn), w=copy(st.w), Ipass=copy(Ipass))
    @printf "  N=0  P=%.0f  FX=%.1f  FS=%.1f  pmax=%.0f  Ipass=%.3e  Pfric=%.3f W  wmax=0\n" S.P S.Qx S.Qy maximum(st.pn) maximum(Ipass) S.pw
    Ndone = 0
    δ = S.pen
    I0 = copy(Ipass)
    for target in snaps
        left = target - Ndone
        while left > 0
            Icap = max(maximum(I0), 1e-18)
            jn = min(left, 1000, max(1, floor(Int, 2.0e-2 / Icap)))
            niter, Ψ = solve_rolling_step!(st, grid, prep, S.law_c, kin, δ;
                                           wear=false, tol=1e-6, maxiter=300, johnson=true)
            rolling_pass_wear!(I0, sx, sy, st, grid, law_w, kin)
            apply_groove_wear!(st, I0, jn)
            δ, _, _ = rolling_match_load!(st, grid, prep, S.law_c, kin, Fz, δ;
                                          rtol=1e-2, maxouter=8, wear=false,
                                          tol=1e-6, maxiter=300, johnson=true)
            left -= jn
        end
        Ndone = target
        P, Qx, Qy = contact_resultants(st, grid.hs)
        push!(Nrec, Ndone); push!(wmax, maximum(st.w))
        fields[Ndone] = (; pn=copy(st.pn), w=copy(st.w), Ipass=copy(I0))
        @printf "  N=%6d  P=%.0f  FX=%.1f  FS=%.1f  pmax=%.0f  wmax=%.4e  Ipass=%.3e\n" Ndone P Qx Qy maximum(st.pn) maximum(st.w) maximum(I0)
    end
    return (; N=Nrec, wmax, fields, st, δ)
end

rail = read_rail_profile(joinpath(DATA, "MBench_UIC60_v3.prr"))
wheel = read_wheel_profile(joinpath(DATA, "MBench_S1002_v3.prw"))
trk = TrackGeom()
# KTH mild: k/H ≈ 1e-4 / 2940 MPa ≈ 3.4e-8 mm²/N. Accelerated ×30 for visible groove.
const IWEAR = 1.0e-6
snaps = [200, 1000, 4000, 15000]
circ = 2π * 460.0
println("Archard i=", IWEAR, " mm²/N   15000 rev = ", round(15000*circ/1e6; digits=2), " km")

println("="^72)
println(" A-2.2 left Y=0  tread   CONTACT FRIC.POWER = -0.468 W")
ws0 = WheelsetGeom(z=0.198, vs=2000.0, vpitch=-4.34811810, fz=10000.0)
S0 = setup_patch(trk, ws0, rail, wheel; side=:left)
@printf "  ξx=%.3e  ξy=%.3e  φ=%.3e  V=%.1f\n" S0.ξx S0.ξy S0.φ S0.V
H0 = evolve!(S0, IWEAR, snaps)

println("="^72)
println(" A-2.2 left Y=10 mm  flange  CONTACT FRIC.POWER ≈ −350 W (large slip)")
ws10 = WheelsetGeom(y=10.0, z=0.198, roll=-1.126e-2, yaw=0.024, vs=2000.0,
                    vpitch=-4.2436375, fz=10000.0)
S10 = setup_patch(trk, ws10, rail, wheel; side=:left)
@printf "  ξx=%.3e  ξy=%.3e  φ=%.3e  V=%.1f  Pfric=%.2f W\n" S10.ξx S10.ξy S10.φ S10.V S10.pw
H10 = evolve!(S10, IWEAR, snaps)

# figures
plt = plot(xlabel="N (revolutions)", ylabel="w_max [mm]", title="Manchester wheel groove wear",
           legend=:topleft)
plot!(plt, H0.N, H0.wmax; color=:black, marker=:square, label="tread Y=0")
plot!(plt, H10.N, H10.wmax; color=:steelblue, marker=:circle, label="flange Y=10 mm")
_savefig(plt, "wr_wear_wmax.png")

function profile_plot(S, H, title, fname)
    plt = plot(xlabel="s [mm]", ylabel="w [mm]", title=title, legend=:topleft)
    jc = argmin(abs.(S.x))
    cols = [:black, :orange, :green, :blue, :red]
    for (k, Ncy) in enumerate([0, 200, 1000, 4000, 15000])
        haskey(H.fields, Ncy) || continue
        plot!(plt, S.s, H.fields[Ncy].w[jc, :]; color=cols[k], label="N=$Ncy")
    end
    _savefig(plt, fname)
end
profile_plot(S0, H0, "tread Y=0  wear vs s", "wr_wear_tread_w.png")
profile_plot(S10, H10, "flange Y=10 mm  wear vs s", "wr_wear_flange_w.png")

function pn_plot(S, H, title, fname)
    plt = plot(xlabel="s [mm]", ylabel="pₙ [MPa]", title=title, legend=:topright)
    ic = argmin(abs.(S.x))
    cols = [:black, :blue, :red]
    for (k, Ncy) in enumerate([0, 1000, 15000])
        haskey(H.fields, Ncy) || continue
        plot!(plt, S.s, H.fields[Ncy].pn[ic, :]; color=cols[k], label="N=$Ncy")
    end
    _savefig(plt, fname)
end
pn_plot(S0, H0, "tread Y=0  pₙ at x=0", "wr_wear_tread_pn.png")
pn_plot(S10, H10, "flange Y=10 mm  pₙ at x=0", "wr_wear_flange_pn.png")

println("\nCONTACT Y=0 FRIC.POWER = -0.468 W   ours ", round(S0.pw; digits=3), " W")
println("ours Y=10 FRIC.POWER = ", round(S10.pw; digits=1), " W")
println("tread  wmax(15k)=", H0.wmax[end], " mm")
println("flange wmax(15k)=", H10.wmax[end], " mm")
println("done.")
