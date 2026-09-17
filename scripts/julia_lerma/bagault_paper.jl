# Reproduce Bagault, Nélias, Baietto, Ovaert, IJSS 50 (2013) 743–754.
# Rigid sphere, frictionless, load P = Hertz of the reference isotropic solid.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra, Printf, FFTW, Plots
gr()
default(size=(720, 440), linewidth=1.7, legendfontsize=8, guidefontsize=11,
        tickfontsize=9, titlefontsize=11, grid=false, framestyle=:box)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIG = joinpath(ROOT, "plots", "julia_lerma", "layered")
mkpath(FIG)

const ν = 0.30
const Eref = 1.0
const aH = 1.0                 # Hertz radius of the isotropic reference
const Rpar = 10.0              # parametric studies (Figs 3, 5–13): R/a = 10
const N = 32
const nq = 128
const L = 2.6 * aH

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

hz = hertz_rigid_sphere(Rpar, Eref, ν, aH)
hx = 2L / N
x = collect(range(-L + hx / 2, L - hx / 2; length=N))
gap_par = [(xi^2 + yj^2) / (2 * Rpar) for xi in x, yj in x]
pHertz(r) = hz.p0 * sqrt(max(1 - (r / aH)^2, 0.0))

println(@sprintf("Hertz ref  a=%.3f  R=%.3f  p0=%.4f  P=%.4f  δ=%.4f  ν=%.2f",
                 aH, Rpar, hz.p0, hz.P, hz.δ, ν))

function run_case(hs; gap=gap_par, P=hz.P, δ0=hz.δ)
    hs = with_grid(hs, hx, hx)
    prep = precompute_kernels(N, N, hs; components=(Kzz,), nq=nq)
    sol, δ, F, _ = solve_sphere_load(gap, hs, P, δ0; prep=prep, rtol=0.02, maxit=7)
    return sol, δ, F
end

function midline(sol, along=:x)
    # along=:x → p(x, y=0); :y → p(x=0, y)
    imid = argmin(abs.(x))
    if along === :x
        return x, sol.p[:, imid]
    else
        return x, sol.p[imid, :]
    end
end

function pmax_of(sol)
    return maximum(sol.p)
end

iso_sub() = isotropic_halfspace(Eref, ν; hx=hx, hy=hx)
iso_coat(Ec, Zc) = isotropic_coated(Ec, ν, Zc, Eref, ν; hx=hx, hy=hx)
function ortho_coat(E1, E2, E3, Zc; θm=0.0)
    G = Eref / (2(1 + ν))
    C = orthotropic_C(E1, E2, E3, ν, ν, ν, G, G, G)
    θm != 0 && (C = rotate_C_about_x(C, θm))
    Cs = cubic_almost_isotropic(Eref, ν)
    return LayeredHalfSpace([Layer(C; thickness=Zc), Layer(Cs; thickness=Inf)]; hx=hx, hy=hx)
end
function ortho_sub(E1, E2, E3, Zc)
    G = Eref / (2(1 + ν))
    Cs = orthotropic_C(E1, E2, E3, ν, ν, ν, G, G, G)
    Cc = cubic_almost_isotropic(Eref, ν)
    return LayeredHalfSpace([Layer(Cc; thickness=Zc), Layer(Cs; thickness=Inf)]; hx=hx, hy=hx)
end

# =====================================================================
println("\n=== Fig. 1  isotropic coating, R = 10 h  (O'Sullivan & King) ===")
# h = aH so R1 = 10 aH, Hertz a/R = 0.1 (thick coating → (Ec/Es)^{2/3} asymptote)
h1 = aH
R1 = 10 * h1
hz1 = hertz_rigid_sphere(R1, Eref, ν, aH)
gap1 = [(xi^2 + yj^2) / (2 * R1) for xi in x, yj in x]
plt1 = plot(xlabel="x / a_Hertz", ylabel="p / p_Hertz",
            title="Fig. 1  isotropic coating  (R = 10 h)",
            xlims=(0, 1.6), ylims=(0, 2.6))
plot!(plt1, range(0, 1; length=80), pHertz.(range(0, 1; length=80)) ./ hz.p0;
      color=:black, label="Hertz")
fig1_peaks = Dict{Float64,Float64}()
for (ratio, col) in ((0.25, :pink), (0.5, :red), (1.0, :black), (2.0, :blue), (4.0, :cyan))
    hs = isotropic_coated(ratio * Eref, ν, h1, Eref, ν; hx=hx, hy=hx)
    sol, δ, F = run_case(hs; gap=gap1, P=hz1.P, δ0=hz1.δ)
    xv, pv = midline(sol, :x)
    # positive x only, like the paper
    mask = xv .>= -1e-12
    plot!(plt1, xv[mask] ./ aH, pv[mask] ./ hz1.p0; color=col,
          label="Ec/Es=$(ratio)")
    pk = maximum(sol.p) / hz1.p0
    fig1_peaks[ratio] = pk
    println(@sprintf("  Ec/Es=%4.2f  pmax/pH=%.3f  F/P=%.3f  (paper ~ %.2f)",
                     ratio, pk, F / hz1.P,
                     ratio == 4 ? 2.35 : ratio == 2 ? 1.52 : ratio == 1 ? 1.00 :
                     ratio == 0.5 ? 0.72 : 0.55))
end
_savefig(plt1, "bagault_fig1.png")

# =====================================================================
println("\n=== Fig. 3  homogeneous orthotropic E3 = 2 E1 = 2 E2 ===")
Co = bagault_orthotropic(Eref, 2Eref, ν)
sol3, _, F3 = run_case(homogeneous(Co; hx=hx, hy=hx))
solI, _, _ = run_case(iso_sub())
xv, pv = midline(sol3, :y)
xi, pi_ = midline(solI, :y)
plt3 = plot(xlabel="y / a_Hertz", ylabel="p / p_Hertz",
            title="Fig. 3  orthotropic E3 = 2 E2", xlims=(-1.2, 1.2), ylims=(0, 1.5))
rh = range(-1, 1; length=80)
plot!(plt3, rh, pHertz.(abs.(rh)) ./ hz.p0; ls=:dash, color=:black, label="Hertz")
plot!(plt3, xi ./ aH, pi_ ./ hz.p0; color=:gray, label="isotropic SAM")
plot!(plt3, xv ./ aH, pv ./ hz.p0; color=:blue, label="SAM E3=2E")
println(@sprintf("  pmax/pH=%.3f  F/P=%.3f  (paper ~1.33)", maximum(sol3.p) / hz.p0, F3 / hz.P))
_savefig(plt3, "bagault_fig3.png")

# =====================================================================
println("\n=== Fig. 5  Ec1 on isotropic substrate, Zc = 0.5 a ===")
Zc = 0.5 * aH
plt5a = plot(xlabel="y / a_Hertz", ylabel="p / p_Hertz",
             title="Fig. 5a  plane x=0  (vary Ec1)", xlims=(-2, 2), ylims=(0, 1.25))
plt5b = plot(xlabel="x / a_Hertz", ylabel="p / p_Hertz",
             title="Fig. 5b  plane y=0  (vary Ec1)", xlims=(-2, 2), ylims=(0, 1.25))
cols5 = [:cyan, :blue, :black, :green, :red, :magenta]
for (k, ratio) in enumerate((0.25, 0.5, 1.0, 2.0, 4.0, 6.0))
    hs = ortho_coat(ratio * Eref, Eref, Eref, Zc)
    sol, _, F = run_case(hs)
    ya, pa = midline(sol, :y)
    xa, pb = midline(sol, :x)
    plot!(plt5a, ya ./ aH, pa ./ hz.p0; color=cols5[k], label="Ec1=$(ratio) Es")
    plot!(plt5b, xa ./ aH, pb ./ hz.p0; color=cols5[k], label="Ec1=$(ratio) Es")
    println(@sprintf("  Ec1/Es=%4.2f  pmax/pH=%.3f  F/P=%.3f", ratio, pmax_of(sol) / hz.p0, F / hz.P))
end
_savefig(plt5a, "bagault_fig5a.png")
_savefig(plt5b, "bagault_fig5b.png")

# =====================================================================
println("\n=== Fig. 6  contact area  Ec1 = Es vs 6 Es ===")
plt6 = plot(xlabel="X / a_Hertz", ylabel="Y / a_Hertz",
            title="Fig. 6  contact area", xlims=(-1.6, 1.6), ylims=(-1.6, 1.6),
            aspect_ratio=1)
for (ratio, col, lab) in ((1.0, :black, "Ec1 = Es"), (6.0, :blue, "Ec1 = 6 Es"))
    hs = ortho_coat(ratio * Eref, Eref, Eref, Zc)
    sol, _, _ = run_case(hs)
    thresh = 0.02 * maximum(sol.p)
    Z = Float64.(sol.p .> thresh)
    contour!(plt6, x ./ aH, x ./ aH, Z'; levels=[0.5], color=col, linewidth=2,
             label=lab, colorbar=false)
    println(@sprintf("  Ec1/Es=%4.1f  ncon=%d", ratio, count(>(thresh), sol.p)))
end
_savefig(plt6, "bagault_fig6.png")

# =====================================================================
println("\n=== Fig. 7  Ec3 on isotropic substrate, Zc = 0.5 a ===")
plt7 = plot(xlabel="y / a_Hertz", ylabel="p / p_Hertz",
            title="Fig. 7  plane x=0  (vary Ec3)", xlims=(-2, 2), ylims=(0, 1.85))
cols7 = [:cyan, :blue, :black, :green, :red, :magenta]
for (k, ratio) in enumerate((0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
    hs = ortho_coat(Eref, Eref, ratio * Eref, Zc)
    sol, _, F = run_case(hs)
    ya, pa = midline(sol, :y)
    plot!(plt7, ya ./ aH, pa ./ hz.p0; color=cols7[k], label="Ec3=$(ratio) Es")
    println(@sprintf("  Ec3/Es=%4.2f  pmax/pH=%.3f  F/P=%.3f", ratio, pmax_of(sol) / hz.p0, F / hz.P))
end
_savefig(plt7, "bagault_fig7.png")

# =====================================================================
println("\n=== Fig. 8  orientation θm  (Zc = 0.5 a, isotropic substrate) ===")
plt8 = plot(xlabel="θm [deg]", ylabel="pmax / pmax isotropic",
            title="Fig. 8  coating orientation", xlims=(0, 90), ylims=(0.8, 1.55))
θs = 0:10:90
for (ratio, mk) in ((0.5, :diamond), (0.75, :utriangle), (1.0, :circle),
                    (1.5, :rtriangle), (2.0, :star), (3.0, :hexagon), (5.0, :utriangle))
    pks = Float64[]
    for th in θs
        hs = ortho_coat(Eref, Eref, ratio * Eref, Zc; θm=deg2rad(th))
        sol, _, _ = run_case(hs)
        push!(pks, pmax_of(sol) / hz.p0)
    end
    plot!(plt8, collect(θs), pks; marker=mk, label="Ec3=$(ratio) Es")
    println(@sprintf("  Ec3/Es=%4.2f  pmax/pH @0°=%.3f  @90°=%.3f", ratio, pks[1], pks[end]))
end
_savefig(plt8, "bagault_fig8.png")

# =====================================================================
println("\n=== Fig. 9  Zc sweep, Ec3 = 2 Es, isotropic substrate ===")
plt9 = plot(xlabel="y / a_Hertz", ylabel="p / p_Hertz",
            title="Fig. 9  Ec3 = 2 Es, vary Zc", xlims=(-2, 2), ylims=(0, 1.55))
Zcs = [0.0, 0.2, 0.5, 1.0, 1.5, 4.0]   # 4 a ≈ coating-only
labs = ["0", "0.2", "0.5", "1", "1.5", "∞"]
cols9 = [:cyan, :blue, :black, :green, :red, :magenta]
for (k, Z) in enumerate(Zcs)
    hs = Z <= 0 ? iso_sub() : ortho_coat(Eref, Eref, 2Eref, Z * aH)
    sol, _, F = run_case(hs)
    ya, pa = midline(sol, :y)
    plot!(plt9, ya ./ aH, pa ./ hz.p0; color=cols9[k], label="Zc=$(labs[k]) a")
    println(@sprintf("  Zc/a=%4.1f  pmax/pH=%.3f", Z, pmax_of(sol) / hz.p0))
end
_savefig(plt9, "bagault_fig9.png")

# =====================================================================
println("\n=== Fig. 10  pmax vs Zc  (a) orthotropic coating  (b) isotropic coating ===")
Zgrid = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
plt10a = plot(xlabel="Zc / a_Hertz", ylabel="pmax / pmax isotropic",
              title="Fig. 10a  orthotropic coating", xlims=(0, 4), ylims=(0.4, 3.1))
plt10b = plot(xlabel="Zc / a_Hertz", ylabel="pmax / pmax isotropic",
              title="Fig. 10b  isotropic coating", xlims=(0, 4), ylims=(0.4, 3.1))
for ratio in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    pks = Float64[]
    for Z in Zgrid
        hs = Z <= 0 ? iso_sub() : ortho_coat(Eref, Eref, ratio * Eref, Z * aH)
        sol, _, _ = run_case(hs)
        push!(pks, pmax_of(sol) / hz.p0)
    end
    plot!(plt10a, Zgrid, pks; label="Ec3=$(ratio) Es")
    println(@sprintf("  ortho Ec3/Es=%4.2f  Z=0 → %.3f   Z=4 → %.3f", ratio, pks[1], pks[end]))
end
for ratio in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    pks = Float64[]
    for Z in Zgrid
        hs = Z <= 0 ? iso_sub() : iso_coat(ratio * Eref, Z * aH)
        sol, _, _ = run_case(hs)
        push!(pks, pmax_of(sol) / hz.p0)
    end
    plot!(plt10b, Zgrid, pks; label="Ec=$(ratio) Es")
    println(@sprintf("  iso   Ec/Es =%4.2f  Z=0 → %.3f   Z=4 → %.3f  (Ec/Es)^(2/3)=%.3f",
                     ratio, pks[1], pks[end], ratio^(2 / 3)))
end
_savefig(plt10a, "bagault_fig10a.png")
_savefig(plt10b, "bagault_fig10b.png")

# =====================================================================
println("\n=== Fig. 11–12  isotropic coating on orthotropic substrate, Zc = 0.5 a ===")
plt11 = plot(xlabel="y / a_Hertz", ylabel="p / p_Hertz",
             title="Fig. 11  vary Es1  (isotropic coating)", xlims=(-2, 2), ylims=(0, 1.25))
plt12 = plot(xlabel="y / a_Hertz", ylabel="p / p_Hertz",
             title="Fig. 12  vary Es3  (isotropic coating)", xlims=(-2, 2), ylims=(0, 1.7))
for (k, ratio) in enumerate((0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
    hs = ortho_sub(ratio * Eref, Eref, Eref, Zc)
    sol, _, F = run_case(hs)
    ya, pa = midline(sol, :y)
    plot!(plt11, ya ./ aH, pa ./ hz.p0; color=cols7[k], label="Es1=$(ratio) Ec")
    println(@sprintf("  Es1/Ec=%4.2f  pmax/pH=%.3f", ratio, pmax_of(sol) / hz.p0))
end
for (k, ratio) in enumerate((0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
    hs = ortho_sub(Eref, Eref, ratio * Eref, Zc)
    sol, _, F = run_case(hs)
    ya, pa = midline(sol, :y)
    plot!(plt12, ya ./ aH, pa ./ hz.p0; color=cols7[k], label="Es3=$(ratio) Ec")
    println(@sprintf("  Es3/Ec=%4.2f  pmax/pH=%.3f", ratio, pmax_of(sol) / hz.p0))
end
_savefig(plt11, "bagault_fig11.png")
_savefig(plt12, "bagault_fig12.png")

# =====================================================================
println("\n=== Fig. 13  Zc sweep, isotropic coating, Es3 = 2 Ec ===")
plt13 = plot(xlabel="y / a_Hertz", ylabel="p / p_Hertz",
             title="Fig. 13  Es3 = 2 Ec, vary Zc", xlims=(-2, 2), ylims=(0, 1.7))
Z13 = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
cols13 = [:cyan, :blue, :black, :green, :red, :magenta]
for (k, Z) in enumerate(Z13)
    hs = ortho_sub(Eref, Eref, 2Eref, Z * aH)
    sol, _, F = run_case(hs)
    ya, pa = midline(sol, :y)
    plot!(plt13, ya ./ aH, pa ./ hz.p0; color=cols13[k], label="Zc=$(Z) a")
    println(@sprintf("  Zc/a=%4.1f  pmax/pH=%.3f", Z, pmax_of(sol) / hz.p0))
end
_savefig(plt13, "bagault_fig13.png")

println("\ndone. figures in ", FIG)
