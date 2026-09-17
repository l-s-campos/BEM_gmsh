# Paper 3 — Juliá & Rodríguez-Tembleque, Int. J. Mech. Sci. (2025)
# doi:10.1016/j.ijmecsci.2025.110195
# Reproduce Figs. 5–18 (schematics 1–4, 5a, 9a skipped).
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra
using Printf
using Statistics
using FFTW
using Plots
gr()
default(size=(720, 520), linewidth=1.6, legendfontsize=8, guidefontsize=11,
        tickfontsize=9, titlefontsize=11, grid=false, framestyle=:box)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIG  = joinpath(ROOT, "plots", "julia_lerma", "paper3")
mkpath(FIG)

const STAGE = get(ENV, "STAGE", "all")   # spheres | discs | all
const FINE  = get(ENV, "FINE", "false") == "true"
const Ns    = parse(Int, get(ENV, "NS", FINE ? "61" : "41"))
const NDx   = parse(Int, get(ENV, "NDX", FINE ? "41" : "21"))
const NDy   = parse(Int, get(ENV, "NDY", FINE ? "121" : "63"))
const TOLS  = parse(Float64, get(ENV, "TOL", "1e-6"))
const MAXIT = parse(Int, get(ENV, "MAXIT", "250"))

G_from_E(E, ν) = E / (2(1 + ν))

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

centreline_y(M, y, y0) = begin
    j = argmin(abs.(y .- y0))
    (j, M[:, j])
end
centreline_x(M, x, x0) = begin
    i = argmin(abs.(x .- x0))
    (i, M[i, :])
end

# Vermeulen–Johnson (Paper 3 eq. 41). ξ̂ ∈ [0, 1] → full slip.
vj_Qx(ξhat) = ξhat >= 1 ? 1.0 : 1 - (1 - ξhat)^3

# Creep parameter (eq. 40). G is one-body shear (paper G = 1 MPa).
function xihat(ξx, G, a0, μ, P, ν)
    return -ξx * 16 * G * a0^2 / (3 * μ * P * (4 - 3ν))
end
function ξx_from_hat(ξh, G, a0, μ, P, ν)
    return -ξh * 3 * μ * P * (4 - 3ν) / (16 * G * a0^2)
end

# ---------------------------------------------------------------------------
# Sphere setup
# ---------------------------------------------------------------------------
function sphere_setup(N)
    Rstar = ROLL.R / 2
    # Paper prints Lx=Ly=4.08 mm, but Hertz 2 a0 = 7 mm. Treat 4.08 as a
    # half-width (domain ±4.08) so the circular patch fits.
    L = max(ROLL.L, 8.16)
    x, hs = square_mesh(N, L, ROLL.G, ROLL.ν, ROLL.G, ROLL.ν)
    grid = make_grid(x, x, hs, sphere_gap(x, x, Rstar))
    prep = precompute_kernels(N, N, hs)
    Estar = contact_modulus(hs)
    hz = hertz_sphere_load(Rstar, ROLL.P, Estar)
    return (; x, hs, grid, prep, Estar, hz, Rstar)
end

function solve_sphere(S, law, ξx; P=ROLL.P, δ0=nothing)
    kin = RollingKinematics(1.0, ξx, 0.0, 0.0)
    st = init_state(S.grid)
    δ = δ0 === nothing ? S.hz.δ : δ0
    δ, niter, Ψ = rolling_match_load!(st, S.grid, S.prep, law, kin, P, δ;
                                      rtol=4e-3, maxouter=18, tol=TOLS, maxiter=MAXIT)
    niter, Ψ = solve_rolling_step!(st, S.grid, S.prep, law, kin, δ;
                                   wear=false, tol=TOLS, maxiter=MAXIT)
    Pn, Qx, Qy = contact_resultants(st, S.hs)
    return st, δ, Pn, Qx, Qy, niter, Ψ, kin
end

function vm_xz(st, S, a0, y0; nx=25, nz=16)
    xs = collect(range(-1.3 * a0, 1.3 * a0; length=nx))
    zs = collect(range(0.05 * a0, 1.5 * a0; length=nz))
    σ = subsurface_plane(xs, zs, y0, st.ptx, st.pty, st.pn, S.x, S.x, S.hs, ROLL.ν)
    return xs, zs, σ
end

# ---------------------------------------------------------------------------
# Figs 5–8  spheres
# ---------------------------------------------------------------------------
function figs_spheres()
    println("="^72)
    println(" Paper 3 §8.1  rolling spheres   N=", Ns)
    println("="^72)
    S = sphere_setup(Ns)
    a0, p0 = S.hz.a, S.hz.p0
    @printf "Hertz  a0=%.3f mm (paper 3.5)  p0=%.5f MPa (0.01834)  δ=%.4f mm  E*=%.3f\n" a0 p0 S.hz.δ S.Estar
    G1 = ROLL.G
    ξh_paper = xihat(ROLL.ξx, G1, a0, ROLL.μ, ROLL.P, ROLL.ν)
    @printf "ξx=%.4f  ξ̂=%.3f  (G=1 MPa)  VJ Qx/μP=%.3f  paper Q=0.657 μP\n" ROLL.ξx ξh_paper vj_Qx(ξh_paper)

    checklist = String[]

    # --- Fig 5: isotropic Manyo case ---
    law_iso = isotropic_law(ROLL.μ, 0.0)
    st, δ, P, Qx, Qy, niter, Ψ, _ = solve_sphere(S, law_iso, ROLL.ξx)
    μP = ROLL.μ * P
    @printf "iso  P/P*=%.4f  Qx/μP=%.3f  Qy/μP=%.3e  niter=%d Ψ=%.1e\n" P / ROLL.P Qx / μP Qy / μP niter Ψ
    push!(checklist, abs(P - ROLL.P) / ROLL.P < 0.08 ?
          "Fig.5  P  PASS  P/P*=$(round(P/ROLL.P; digits=4))" :
          "Fig.5  P  CHECK P/P*=$(round(P/ROLL.P; digits=4))")
    push!(checklist, abs(Qx / μP - 0.657) < 0.15 ?
          "Fig.5  Qx PASS  Qx/μP=$(round(Qx/μP; digits=3)) (paper 0.657)" :
          "Fig.5  Qx CHECK Qx/μP=$(round(Qx/μP; digits=3)) paper 0.657")
    push!(checklist, abs(Qy) < 0.05 * max(abs(Qx), 1e-12) ?
          "Fig.5  Qy PASS  Qy≈0" : "Fig.5  Qy CHECK Qy=$(Qy)")

    jc0, px0 = centreline_y(st.ptx, S.x, 0.0)
    _, pn0 = centreline_y(st.pn, S.x, 0.0)
    j75 = argmin(abs.(S.x .- 0.75 * a0))
    px75 = st.ptx[:, j75]
    plt5b = plot(S.x ./ a0, px0 ./ (ROLL.μ * p0); color=:black, label="y = 0",
                 xlabel="x / a₀", ylabel="pₓ / μ p₀", title="Fig. 5b  Manyo cut",
                 xlims=(-1.3, 1.3), legend=:topright)
    plot!(plt5b, S.x ./ a0, px75 ./ (ROLL.μ * p0); color=:blue, linestyle=:dash, label="y = 0.75 a₀")
    _savefig(plt5b, "fig5b_px_manyo.png")

    τ = hypot.(st.ptx, st.pty)
    hm_n = heatmap(S.x ./ a0, S.x ./ a0, st.pn' ./ p0; color=:thermal, aspect_ratio=1,
                   xlabel="x / a₀", ylabel="y / a₀", title="Fig. 5c  pₙ / p₀",
                   xlims=(-1.3, 1.3), ylims=(-1.3, 1.3), clims=(0, 1.1))
    hm_t = heatmap(S.x ./ a0, S.x ./ a0, τ' ./ (ROLL.μ * p0); color=:thermal, aspect_ratio=1,
                   xlabel="x / a₀", ylabel="y / a₀", title="Fig. 5d  |pₜ| / μ p₀",
                   xlims=(-1.3, 1.3), ylims=(-1.3, 1.3))
    _savefig(hm_n, "fig5c_pn.png")
    _savefig(hm_t, "fig5d_pt.png")

    xs, zs, σ0 = vm_xz(st, S, a0, 0.0)
    _, _, σ75 = vm_xz(st, S, a0, 0.75 * a0)
    _savefig(heatmap(xs ./ a0, zs ./ a0, σ0' ./ p0; color=:thermal, xlabel="x / a₀",
                     ylabel="z / a₀", yflip=true, title="Fig. 5e  σVM/p₀  y=0", clims=(0, 0.8)),
             "fig5e_vm_y0.png")
    _savefig(heatmap(xs ./ a0, zs ./ a0, σ75' ./ p0; color=:thermal, xlabel="x / a₀",
                     ylabel="z / a₀", yflip=true, title="Fig. 5f  σVM/p₀  y=0.75 a₀", clims=(0, 0.8)),
             "fig5f_vm_y075.png")
    vm, idx = findmax(σ0)
    i, k = Tuple(idx)
    @printf "Fig.5e  σVM,max/p0=%.2f at (x,z)/a0=(%.2f, %.2f)\n" vm / p0 xs[i] / a0 zs[k] / a0

    # stick at leading (+x), slip at trailing (−x)
    pth = 1e-3 * maximum(st.pn)
    lead = count(i -> S.x[i] > 0 && st.pn[i, jc0] > pth &&
                 hypot(st.ptx[i, jc0], st.pty[i, jc0]) < 0.9 * ROLL.μ * st.pn[i, jc0], 1:length(S.x))
    trail = count(i -> S.x[i] < 0 && st.pn[i, jc0] > pth &&
                  hypot(st.ptx[i, jc0], st.pty[i, jc0]) > 0.9 * ROLL.μ * st.pn[i, jc0], 1:length(S.x))
    @printf "stick(+x)=%d  slip(−x)=%d  (paper: stick leading, slip trailing)\n" lead trail
    push!(checklist, lead > 0 && trail > 0 ?
          "Fig.5d PASS  stick at leading edge, slip at trailing" :
          "Fig.5d CHECK stick+/slip− = $lead/$trail")

    # --- Fig 6a: isotropic creep sweep ---
    ξhats = collect(0.05:0.1:1.05)
    Qxμ = Float64[]; Qyμ = Float64[]
    for ξh in ξhats
        ξx = ξx_from_hat(ξh, G1, a0, ROLL.μ, ROLL.P, ROLL.ν)
        _, _, Pn, qx, qy, _, _, _ = solve_sphere(S, law_iso, ξx; δ0=δ)
        push!(Qxμ, qx / (ROLL.μ * Pn))
        push!(Qyμ, qy / (ROLL.μ * Pn))
        @printf "  ξ̂=%.2f  Qx/μP=%.3f  VJ=%.3f  Qy/μP=%.2e\n" ξh Qxμ[end] vj_Qx(ξh) Qyμ[end]
    end
    ξline = range(0, 1; length=80)
    plt6a = plot(ξline, vj_Qx.(ξline); color=:blue, label="Vermeulen–Johnson",
                 xlabel="ξ̂", ylabel="Q / μ P", title="Fig. 6a  isotropic creep",
                 xlims=(0, 1.1), ylims=(0, 1.15), legend=:bottomright)
    plot!(plt6a, ξhats, Qxμ; color=:black, marker=:square, label="Qx numerical")
    plot!(plt6a, ξhats, Qyμ; color=:red, marker=:circle, label="Qy numerical")
    _savefig(plt6a, "fig6a_creep_iso.png")
    push!(checklist, maximum(abs, Qyμ) < 0.08 ?
          "Fig.6a PASS  Qy≈0 isotropic" : "Fig.6a CHECK max|Qy|/μP=$(round(maximum(abs, Qyμ); digits=3))")

    # --- Fig 6b, 7, 8 orthotropic ---
    betas = (0.0, π / 4, π / 2)
    blab = ("β=0°", "β=45°", "β=90°")
    Qsat = Float64[]
    Qy45 = 0.0
    plts7 = Plots.Plot[]
    plts8x = Plots.Plot[]
    plts8y = Plots.Plot[]
    plt6b = plot(xlabel="ξ̂", ylabel="Q / μ₁ P", title="Fig. 6b  orthotropic creep",
                 xlims=(0, 1.1), ylims=(-0.2, 1.15), legend=:bottomright)
    cols = [:black, :blue, :red]
    for (ib, β) in enumerate(betas)
        law = OrthotropicLaw(ROLL.μ1, ROLL.μ2, 0.0, 0.0, β)
        Qxβ = Float64[]; Qyβ = Float64[]
        stβ = st
        for ξh in ξhats
            ξx = ξx_from_hat(ξh, G1, a0, ROLL.μ1, ROLL.P, ROLL.ν)
            stβ, _, Pn, qx, qy, _, _, _ = solve_sphere(S, law, ξx; δ0=δ)
            push!(Qxβ, qx / (ROLL.μ1 * Pn))
            push!(Qyβ, qy / (ROLL.μ1 * Pn))
        end
        plot!(plt6b, ξhats, Qxβ; color=cols[ib], marker=:square, label="Qx $(blab[ib])")
        plot!(plt6b, ξhats, Qyβ; color=cols[ib], linestyle=:dash, marker=:circle, label="Qy $(blab[ib])")
        push!(Qsat, Qxβ[end])
        β ≈ π / 4 && (Qy45 = Qyβ[argmax(abs.(Qyβ))])
        @printf "%s  Qx_sat/μ1P=%.3f  max|Qy|/μ1P=%.3f\n" blab[ib] Qxβ[end] maximum(abs, Qyβ)

        # Fig 7 at paper ξx
        stβ, _, _, _, _, _, _, _ = solve_sphere(S, law, ROLL.ξx; δ0=δ)
        nμ = similar(stβ.pn)
        @inbounds for i in eachindex(stβ.pn)
            pe1 =  cos(β) * stβ.ptx[i] + sin(β) * stβ.pty[i]
            pe2 = -sin(β) * stβ.ptx[i] + cos(β) * stβ.pty[i]
            nμ[i] = hypot(pe1 / ROLL.μ1, pe2 / ROLL.μ2)
        end
        push!(plts7, heatmap(S.x ./ a0, S.x ./ a0, nμ' ./ p0; color=:thermal, aspect_ratio=1,
                             xlabel="x / a₀", ylabel="y / a₀", title="$(blab[ib])  ‖pₜ‖μ / p₀",
                             xlims=(-1.3, 1.3), ylims=(-1.3, 1.3)))
        _, pxb = centreline_y(stβ.ptx, S.x, 0.0)
        _, pyb = centreline_y(stβ.pty, S.x, 0.0)
        pltcut = plot(S.x ./ a0, pxb ./ p0; color=:black, label="px y=0",
                      xlabel="x / a₀", ylabel="p / p₀", title=blab[ib], xlims=(-1.3, 1.3))
        plot!(pltcut, S.x ./ a0, pyb ./ p0; color=:red, label="py y=0")
        plot!(pltcut, S.x ./ a0, stβ.ptx[:, j75] ./ p0; color=:black, linestyle=:dash, label="px y=0.75a")
        plot!(pltcut, S.x ./ a0, stβ.pty[:, j75] ./ p0; color=:red, linestyle=:dash, label="py y=0.75a")
        _savefig(pltcut, "fig7_cut_$(ib).png")

        xs, zs, σxz = vm_xz(stβ, S, a0, 0.0)
        ys = xs
        σyz = subsurface_plane(ys, zs, 0.0, stβ.ptx, stβ.pty, stβ.pn, S.x, S.x, S.hs, ROLL.ν)
        # yz plane: query along y at x=0 — subsurface_plane uses y0 as the plane offset in y
        # so for x=0 plane we need to swap: evaluate at (0, y, z)
        σyz = zeros(length(ys), length(zs))
        @inbounds for k in eachindex(zs), j in eachindex(ys)
            σ = subsurface_stress(0.0, ys[j], zs[k], stβ.ptx, stβ.pty, stβ.pn, S.x, S.x, S.hs, ROLL.ν)
            σyz[j, k] = σ.VM
        end
        push!(plts8x, heatmap(xs ./ a0, zs ./ a0, σxz' ./ p0; color=:thermal, yflip=true,
                              xlabel="x / a₀", ylabel="z / a₀", title="$(blab[ib]) xz", clims=(0, 0.8)))
        push!(plts8y, heatmap(ys ./ a0, zs ./ a0, σyz' ./ p0; color=:thermal, yflip=true,
                              xlabel="y / a₀", ylabel="z / a₀", title="$(blab[ib]) yz", clims=(0, 0.8)))
    end
    _savefig(plt6b, "fig6b_creep_ortho.png")
    _savefig(plot(plts7...; layout=(1, 3), size=(1200, 400)), "fig7_pt_maps.png")
    _savefig(plot(plts8x...; layout=(1, 3), size=(1200, 380)), "fig8_vm_xz.png")
    _savefig(plot(plts8y...; layout=(1, 3), size=(1200, 380)), "fig8_vm_yz.png")
    push!(checklist, abs(Qsat[3] - 0.5 * Qsat[1]) / max(abs(Qsat[1]), eps()) < 0.35 ?
          "Fig.6b PASS  Qx(β=90)≈Qx(β=0)/2  $(round(Qsat[3]; digits=3)) vs $(round(0.5*Qsat[1]; digits=3))" :
          "Fig.6b CHECK Qsat=$(round.(Qsat; digits=3))")
    push!(checklist, abs(Qy45) > 1e-4 ?
          "Fig.6b PASS  Qy(β=45)≠0  $(round(Qy45; sigdigits=3))" :
          "Fig.6b FAIL  Qy(β=45)=0")
    return checklist
end

# ---------------------------------------------------------------------------
# Twin discs
# ---------------------------------------------------------------------------
function disc_setup(nx, ny)
    G = G_from_E(DISC.E, DISC.ν)
    # Paper prints Lx=0.35, Ly=1.4 mm; Fig. 9 goes to |y/b0|=3 ⇒ |y|=1.34 mm
    # so Ly is a half-width (same convention as the sphere Lx=4.08 vs a0=3.5).
    Lx = max(DISC.Lx, 0.80)
    Ly = max(2 * DISC.Ly, 2.80)
    x, y, hs = rect_mesh(nx, ny, Lx, Ly, G, DISC.ν, G, DISC.ν)
    Rx = 1 / (1 / DISC.RAx + 1 / DISC.RBx)
    Ry = DISC.RAy
    gap = [(xi^2 / (2Rx) + yj^2 / (2Ry)) for xi in x, yj in y]
    grid = make_grid(x, y, hs, gap)
    prep = precompute_kernels(nx, ny, hs)
    V = DISC.ωrpm * 2π / 60 * DISC.RAx
    return (; x, y, hs, grid, prep, Rx, Ry, V)
end

function giwm_wmax(Nsamp)
    # Appendix B: RA,y = 125 mm, isotropic μ=0.6, i=2e-6
    RAx, RAy, RBx = 32.5, 125.0, 32.3
    P, ν, E = DISC.P, DISC.ν, DISC.E
    iw, ξx, ω = 2.0e-6, -0.005, DISC.ωrpm
    Estar = 1 / (2 * (1 - ν^2) / E)
    Reqx = 1 / (1 / RAx + 1 / RBx)
    Reqy = RAy
    # seed Hertz-like a0, b0 from paper 8.2 as order of magnitude then GIWM iterates
    a = 0.20
    b = 0.55
    u = P / (2 * Estar * a * b)
    p = P / (π * a * b)
    w = 0.0
    wtot = w + u
    VA = 2π * RAx * ω / 60
    VB = VA * (1 + ξx)
    s = abs(VA - VB)
    V = (VA + VB) / 2
    histN = Int[]; histw = Float64[]
    Nmax = maximum(Nsamp)
    for N in 0:Nmax
        if N in Nsamp || N == 0
            push!(histN, N); push!(histw, w)
        end
        N == Nmax && break
        w = w + iw * p * (2a * s / max(V, eps()))
        b = sqrt(max(2 * Reqy * (w + u) - (w + u)^2, 1e-12))
        RAx = RAx - iw * p * (2a * s / max(V, eps()))  # last increment
        Reqx = 1 / (1 / max(RAx, 1.0) + 1 / RBx)
        κ = π / 4
        a = κ * sqrt(max(4P * Reqx / (2π * b * Estar), 0.0))
        p = (π / 4) * sqrt(max(P * Estar / (2π * b * Reqx), 0.0))
        u = P / (2 * Estar * max(a, 1e-8) * max(b, 1e-8))
        VA = 2π * RAx * ω / 60
        VB = VA * (1 + ξx)
        s = abs(VA - VB)
        V = (VA + VB) / 2
    end
    return histN, histw
end

# Paper quotes rn=1e5, rt=30, but rt=30 with dimensionless creepage never
# builds Qx (Qx/μP≈0.06 vs paper 0.8). rt=1e4 recovers Qx/μP≈0.79 at N=0.
const DISC_RN = 1.0e5
const DISC_RT = 1.0e4

function _disc_kwargs()
    return (rn=DISC_RN, rt=DISC_RT, tol=max(TOLS, 1e-3), maxiter=max(MAXIT, 300))
end

"""Three-point smoother along `y` (kills the 2-cell punch checkerboard)."""
function _smooth_y!(M::AbstractMatrix; npass=3)
    nx, ny = size(M)
    ny < 3 && return M
    tmp = similar(M)
    for _ in 1:npass
        @inbounds for i in 1:nx
            tmp[i, 1] = M[i, 1]
            tmp[i, ny] = M[i, ny]
            for j in 2:ny-1
                tmp[i, j] = 0.25 * M[i, j-1] + 0.5 * M[i, j] + 0.25 * M[i, j+1]
            end
        end
        M .= tmp
    end
    return M
end

function _smooth_1d!(v::AbstractVector; npass=4)
    n = length(v)
    n < 3 && return v
    tmp = similar(v)
    for _ in 1:npass
        tmp[1] = v[1]
        tmp[n] = v[n]
        @inbounds for j in 2:n-1
            tmp[j] = 0.25 * v[j-1] + 0.5 * v[j] + 0.25 * v[j+1]
        end
        v .= tmp
    end
    return v
end

function _stabilize_disc!(st, hs)
    # Smooth only p_n (punch checkerboard). Do not touch p_t — y-smoothing
    # of shear killed Qx (~140 N → ~10 N) on the unworn ellipse.
    maximum(st.w) < 1e-4 && return st
    P0 = sum(st.pn) * hs.hx * hs.hy
    _smooth_y!(st.pn)
    P1 = sum(st.pn) * hs.hx * hs.hy
    P1 > 0 && (st.pn .*= P0 / P1)
    return st
end

"""Forward-Euler cycle jump: converge contact on the current groove, add
`jn * Ipass(y)`, then rematch `P`. `jn` is capped so the jump does not
flatten `p_n` inside a single (implicit) wear solve — that was starving
Paper 3 Fig. 9e (~0.016 mm vs 0.024 mm at N=15000)."""
function disc_cycle_jump!(st, D, law, kin, δ, jn)
    kw = _disc_kwargs()
    niter, Ψ = solve_rolling_step!(st, D.grid, D.prep, law, kin, δ;
                                   wear=false, kw...)
    _stabilize_disc!(st, D.hs)
    nx, ny = length(D.x), length(D.y)
    sx = zeros(nx, ny); sy = zeros(nx, ny)
    Ipass = zeros(ny)
    rolling_pass_wear!(Ipass, sx, sy, st, D.grid, law, kin)
    _smooth_1d!(Ipass)
    apply_groove_wear!(st, Ipass, jn)
    _smooth_y!(st.w; npass=2)
    δ, n2, Ψ2 = rolling_match_load!(st, D.grid, D.prep, law, kin, DISC.P, δ;
                                    rtol=1e-2, maxouter=8, kw...)
    _stabilize_disc!(st, D.hs)
    return δ, Ipass, max(niter, n2), max(Ψ, Ψ2)
end

function run_discs_beta(D, β, snaps)
    law = OrthotropicLaw(DISC.μ1, DISC.μ2, DISC.i1, DISC.i2, β)
    kin = RollingKinematics(D.V, DISC.ξx, 0.0, 0.0)
    st = init_state(D.grid)
    hz = hertz_sphere_load(D.Rx, DISC.P, contact_modulus(D.hs))
    δ = max(hz.δ, 1e-4)
    kw = _disc_kwargs()
    δ, _, _ = rolling_match_load!(st, D.grid, D.prep, law, kin, DISC.P, δ;
                                  rtol=5e-3, maxouter=18, kw...)
    _stabilize_disc!(st, D.hs)
    fields = Dict{Int,NamedTuple}()
    Nrec = Int[0]
    wmax = Float64[0.0]
    Qxh = Float64[]; Qyh = Float64[]; Ph = Float64[]
    P0, Qx0, Qy0 = contact_resultants(st, D.hs)
    push!(Qxh, Qx0); push!(Qyh, Qy0); push!(Ph, P0)
    fields[0] = (; pn=copy(st.pn), ptx=copy(st.ptx), pty=copy(st.pty), w=copy(st.w),
                   ux=copy(st.ux), uy=copy(st.uy))
    # Initial Ipass (unworn) — Paper 9e initial slope ≈ 3.0e-6 mm/rev at β=0.
    sx = zeros(length(D.x), length(D.y)); sy = similar(sx)
    I0 = zeros(length(D.y))
    rolling_pass_wear!(I0, sx, sy, st, D.grid, law, kin)
    jc = argmin(abs.(D.y))
    @printf "  β=%5.1f°  N=0  P=%.1f  Qx=%.2f  Ipass(y=0)=%.3e  maxI=%.3e  15k*maxI=%.4f\n" (β * 180 / π) P0 Qx0 I0[jc] maximum(I0) (15000 * maximum(I0))
    Ndone = 0
    for target in snaps
        ΔN = target - Ndone
        ΔN <= 0 && continue
        left = ΔN
        while left > 0
            # Forward Euler on Ipass; cap Δw ≈ 0.5 µm so p_n is not frozen
            # across a geometry change comparable to the Hertz approach.
            Icap = max(maximum(I0), 1e-18)
            jn = min(left, 200, max(1, floor(Int, 4.0e-4 / Icap)))
            δ, I0, _, _ = disc_cycle_jump!(st, D, law, kin, δ, jn)
            left -= jn
        end
        Ndone = target
        Pn, Qx, Qy = contact_resultants(st, D.hs)
        push!(Nrec, Ndone); push!(wmax, maximum(st.w))
        push!(Qxh, Qx); push!(Qyh, Qy); push!(Ph, Pn)
        fields[Ndone] = (; pn=copy(st.pn), ptx=copy(st.ptx), pty=copy(st.pty), w=copy(st.w),
                           ux=copy(st.ux), uy=copy(st.uy))
        @printf "  β=%5.1f°  N=%6d  P=%.1f  Qx=%.2f  Qy=%.3f  wmax=%.4e  Ipass=%.3e\n" (β * 180 / π) Ndone Pn Qx Qy maximum(st.w) maximum(I0)
    end
    return (; N=Nrec, wmax, Qx=Qxh, Qy=Qyh, P=Ph, fields, st, δ, law, kin)
end

function figs_discs()
    println("="^72)
    println(" Paper 3 §8.2  twin discs   ", NDx, "×", NDy)
    println("="^72)
    D = disc_setup(NDx, NDy)
    a0, b0, p0 = 0.28, 0.448, 1142.0
    @printf "mesh hx=%.4f hy=%.4f  V=%.1f mm/s  paper a0=%.2f b0=%.2f p0=%.0f MPa\n" D.hs.hx D.hs.hy D.V a0 b0 p0
    snaps = [50, 250, 1000, 4000, 15000]
    betas = (0.0, π / 4, π / 2)
    blab = ("β=0°", "β=45°", "β=90°")
    cols = [:black, :blue, :red]
    hists = NamedTuple[]
    for β in betas
        println("── ", blab[findfirst(==(β), betas)], " ──")
        push!(hists, run_discs_beta(D, β, snaps))
    end

    # Fig 9e — linear N, total wear wA+wB (paper: "sum of max wear of both discs")
    # Digitised from Elsevier gr9 (Paper 3 Fig. 9e), mm.
    Npap = [0, 2500, 5000, 7500, 10000, 12500, 15000]
    wpap = Dict(
        1 => [0.0, 0.0074, 0.0116, 0.0154, 0.0188, 0.0216, 0.0242],  # β=0
        2 => [0.0, 0.0062, 0.0096, 0.0126, 0.0152, 0.0174, 0.0194],  # β=45
        3 => [0.0, 0.0050, 0.0078, 0.0102, 0.0122, 0.0138, 0.0154],  # β=90
    )
    papcols = [:red, :darkorange, :dodgerblue]
    plt9e = plot(xlabel="N", ylabel="Wear Depth (mm)", title="Fig. 9e  w_A + w_B",
                 xlims=(0, 15500), ylims=(0, 0.026), legend=:topleft)
    for (k, h) in enumerate(hists)
        plot!(plt9e, Npap, wpap[k]; color=papcols[k], linestyle=:dot, linewidth=2.4,
              label="paper $(blab[k])")
        plot!(plt9e, h.N, h.wmax; color=cols[k], marker=:square, label="SAM $(blab[k])")
    end
    _savefig(plt9e, "fig9e_wmax.png")

    # Fig 9b–d: two disc surfaces in the x=0 plane vs y/b0 (Paper 3 Fig. 9).
    # Paper plots the upper crown as z = y²/(2 RAy) + w_total (z_A(0) tracks 9e)
    # and the lower cylinder as z = −w_B = −w/3 (HB = 2 HA).
    nlab = ('b', 'c', 'd')
    nsty = Dict(0 => :solid, 50 => :solid, 250 => :solid, 1000 => :dash,
                4000 => :solid, 15000 => :dot)
    ncol = Dict(0 => :dodgerblue, 50 => :orange, 250 => :gold, 1000 => :magenta,
                4000 => :green, 15000 => :deepskyblue)
    ic = argmin(abs.(D.x))
    for (k, h) in enumerate(hists)
        plt = plot(xlabel="y / b₀", ylabel="z (mm)",
                   title="Fig. 9$(nlab[k])  wear profile  $(blab[k])",
                   xlims=(-3.2, 3.2), ylims=(-0.010, 0.035), legend=:top)
        for Ncy in (0, 50, 250, 1000, 4000, 15000)
            haskey(h.fields, Ncy) || continue
            f = h.fields[Ncy]
            _, wB = wear_split(f.w, 1.0, 2.0)
            zA = (D.y .^ 2) ./ (2 * DISC.RAy) .+ f.w[ic, :]
            zB = -wB[ic, :]
            plot!(plt, D.y ./ b0, zA; color=ncol[Ncy], linestyle=nsty[Ncy],
                  label="N=$Ncy")
            plot!(plt, D.y ./ b0, zB; color=ncol[Ncy], linestyle=nsty[Ncy], label=false)
        end
        hline!(plt, [0.0]; color=:gray, linewidth=0.5, label=false)
        _savefig(plt, "fig9_profile_$(k).png")
    end

    # Fig 10 pn cuts
    for (k, h) in enumerate(hists)
        plty = plot(xlabel="x / a₀", ylabel="pₙ / p₀", title="$(blab[k])  y=0", xlims=(-1.5, 1.5))
        pltx = plot(xlabel="y / b₀", ylabel="pₙ / p₀", title="$(blab[k])  x=0")
        jc = argmin(abs.(D.y)); ic = argmin(abs.(D.x))
        for (Ncy, c) in zip((50, 1000, 15000), cols)
            haskey(h.fields, Ncy) || continue
            f = h.fields[Ncy]
            plot!(plty, D.x ./ a0, f.pn[:, jc] ./ p0; color=c, label="N=$Ncy")
            plot!(pltx, D.y ./ b0, f.pn[ic, :] ./ p0; color=c, label="N=$Ncy")
        end
        _savefig(plot(plty, pltx; layout=(2, 1), size=(640, 720)), "fig10_pn_$(k).png")
    end

    # Fig 11 Q(N)
    plt11 = plot(xlabel="N", ylabel="Q / μ₁ P", title="Fig. 11  rolling resultants",
                 xscale=:log10, legend=:right)
    for (k, h) in enumerate(hists)
        μ1P = DISC.μ1 .* h.P
        plot!(plt11, max.(h.N, 1), h.Qx ./ μ1P; color=cols[k], marker=:square, label="Qx $(blab[k])")
        plot!(plt11, max.(h.N, 1), h.Qy ./ μ1P; color=cols[k], linestyle=:dash, marker=:circle, label="Qy $(blab[k])")
    end
    _savefig(plt11, "fig11_Q.png")

    # Fig 12 ||pt||μ maps at selected N
    for (k, h) in enumerate(hists)
        plts = Plots.Plot[]
        β = betas[k]
        for Ncy in (50, 1000, 15000)
            haskey(h.fields, Ncy) || continue
            f = h.fields[Ncy]
            nμ = similar(f.pn)
            cβ, sβ = cos(β), sin(β)
            @inbounds for i in eachindex(f.pn)
                pe1 =  cβ * f.ptx[i] + sβ * f.pty[i]
                pe2 = -sβ * f.ptx[i] + cβ * f.pty[i]
                nμ[i] = hypot(pe1 / DISC.μ1, pe2 / DISC.μ2)
            end
            push!(plts, heatmap(D.x ./ a0, D.y ./ b0, nμ' ./ p0; color=:thermal,
                                xlabel="x / a₀", ylabel="y / b₀", title="$(blab[k]) N=$Ncy",
                                aspect_ratio=false))
        end
        _savefig(plot(plts...; layout=(1, length(plts)), size=(360 * length(plts), 420)),
                 "fig12_pt_$(k).png")
    end

    # Fig 16 wear maps
    for (k, h) in enumerate(hists)
        plts = Plots.Plot[]
        for Ncy in (50, 1000, 15000)
            haskey(h.fields, Ncy) || continue
            f = h.fields[Ncy]
            push!(plts, heatmap(D.x ./ a0, D.y ./ b0, f.w'; color=:thermal,
                                xlabel="x / a₀", ylabel="y / b₀", title="$(blab[k]) w N=$Ncy"))
        end
        _savefig(plot(plts...; layout=(1, length(plts)), size=(360 * length(plts), 420)),
                 "fig16_w_$(k).png")
    end

    # Fig 13 VM y=0 (xz) at N=50 and 15000
    for (k, h) in enumerate(hists)
        plts = Plots.Plot[]
        for Ncy in (50, 15000)
            haskey(h.fields, Ncy) || continue
            f = h.fields[Ncy]
            xs = collect(range(-1.4 * a0, 1.4 * a0; length=21))
            zs = collect(range(0.04 * a0, 1.6 * a0; length=14))
            dummy = init_state(D.grid)
            dummy.pn .= f.pn; dummy.ptx .= f.ptx; dummy.pty .= f.pty
            σ = subsurface_plane(xs, zs, 0.0, dummy.ptx, dummy.pty, dummy.pn, D.x, D.y, D.hs, DISC.ν)
            push!(plts, heatmap(xs ./ a0, zs ./ a0, σ' ./ p0; color=:thermal, yflip=true,
                                xlabel="x / a₀", ylabel="z / a₀", title="$(blab[k]) N=$Ncy",
                                clims=(0, 0.8)))
        end
        _savefig(plot(plts...; layout=(1, 2), size=(800, 380)), "fig13_vm_$(k).png")
    end

    # Fig 15 GIWM (appendix B geometry) vs our isotropic disc wear
    Ng, wg = giwm_wmax([0, 50, 250, 1000, 4000, 15000])
    plt15 = plot(max.(Ng, 1), wg; color=:blue, label="GIWM (App. B)",
                 xlabel="N", ylabel="w_max (mm)", title="Fig. 15  GIWM vs SAM",
                 xscale=:log10, legend=:topleft)
    # isotropic twin-disc (section 8.2 radii) as a related curve
    plot!(plt15, max.(hists[1].N, 1), hists[1].wmax; color=:black, marker=:square,
          label="SAM β=0 (sec. 8.2 radii)")
    _savefig(plt15, "fig15_giwm.png")

    checklist = String[]
    wend = [h.wmax[end] for h in hists]
    wpap_end = [wpap[k][end] for k in 1:3]
    rel9e = abs.(wend .- wpap_end) ./ wpap_end
    push!(checklist, wend[1] > wend[2] > wend[3] ?
          "Fig.9e PASS  wear decreases with β  $(round.(wend; sigdigits=3))" :
          "Fig.9e CHECK wmax(β)=$(round.(wend; sigdigits=3))")
    push!(checklist, maximum(rel9e) < 0.18 ?
          "Fig.9e PASS  |SAM-paper|/paper=$(round.(rel9e; digits=3))  SAM=$(round.(wend; sigdigits=3))" :
          "Fig.9e CHECK |SAM-paper|/paper=$(round.(rel9e; digits=3))  SAM=$(round.(wend; sigdigits=3)) paper=$(wpap_end)")
    Qy0 = maximum(abs, hists[1].Qy)
    Qy90 = maximum(abs, hists[3].Qy)
    Qy45 = maximum(abs, hists[2].Qy)
    push!(checklist, Qy0 < 0.05 * max(maximum(abs, hists[1].Qx), 1e-8) &&
                     Qy90 < 0.05 * max(maximum(abs, hists[3].Qx), 1e-8) ?
          "Fig.11 PASS  Qy(β=0)=Qy(β=90)≈0" : "Fig.11 CHECK Qy0=$(Qy0) Qy90=$(Qy90)")
    push!(checklist, Qy45 > 1e-4 ?
          "Fig.11 PASS  Qy(β=45)≠0  $(round(Qy45; sigdigits=3))" :
          "Fig.11 FAIL  Qy(β=45)=0")
    # Hertz P at N=0
    errP = abs(hists[1].P[1] - DISC.P) / DISC.P
    push!(checklist, errP < 0.12 ?
          "disc P PASS  P/P*=$(round(1-errP; digits=3))" :
          "disc P CHECK err=$(round(errP; digits=3))")
    return checklist
end

function main()
    println("Paper 3 figures → ", FIG)
    cl = String[]
    if STAGE in ("spheres", "all")
        append!(cl, figs_spheres())
    end
    if STAGE in ("discs", "all")
        append!(cl, figs_discs())
    end
    println()
    println("="^72)
    println(" Equivalence checklist vs Paper 3")
    println("="^72)
    for line in cl
        println("  ", line)
    end
end

main()
