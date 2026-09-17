# Paper Example 2 geometry: 12×6 bar, 4 periods, DRM f=r.
# Tests: (1) particular-solution sign vs paper M=ρ(Gη-Hψ)F⁻¹
#         (2) Δt = 1, 0.5, 0.25 (paper used 1).
# julia --project=. scripts/plot_bar_rect12x6.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box,
    grid=false, dpi=160, size=(720, 400), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

const Lx, Ly = 12.0, 6.0
const Tper = 4 * Lx          # 4L/c, c=1
const tf = 4 * Tper          # 4 periods = 192
const PROBE = Point2D(Lx, Ly / 2)

function mesh_rect(; Lx=12.0, Ly=6.0, ndivx=13, ndivy=7, ordem=2, P=1.0, nome="bar12x6")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 0.5
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(l1, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l3, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l2, ndivy)
    gmsh.model.mesh.setTransfiniteCurve(l4, ndivy)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [l1, l3], -1, "1;0;1;0")       # top/bottom traction-free
    gmsh.model.addPhysicalGroup(1, [l2], -1, "1;$P;1;0")          # right P
    gmsh.model.addPhysicalGroup(1, [l4], -1, "0;0;0;0")           # left clamped
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function make_dad()
    msh = mesh_rect()
    props = Elasticity(E=1.0, nu=0.0, rho=1.0; plane_stress=true)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    set_internal_grid!(dad; nx=3, ny=3, x=(0, Lx), y=(0, Ly), pad=1.0)
    attach_analytical!(dad, ana_bar_sudden(; N=400, c=1.0, L=Lx))
    return dad
end

function drm_mass!(dad; flip::Bool, kernel::Symbol=:r, C::Real=0.01)
    ker = Val(kernel)
    nu = effective_nu(dad.properties)
    Gmod = shear_modulus(dad.properties)
    nt, n = dad.nt, dad.n
    ndof, nb = 2nt, 2n
    pts = all_points(dad)
    s = flip ? -1.0 : 1.0
    F = zeros(ndof, ndof)
    Ψ = zeros(ndof, ndof)
    for j in 1:nt, i in 1:nt
        rvec = pts[i] - pts[j]
        F[2i-1:2i, 2j-1:2j] .= BEM._drm_phi(norm(rvec), ker, C) * I(2)
        Ψ[2i-1:2i, 2j-1:2j] .= s .* BEM._drm_u(rvec, nu, Gmod, ker, C)
    end
    ε = 1e-12 * (sum(abs, F) / max(ndof^2, 1) + 1)
    for k in 1:ndof
        F[k, k] += ε
    end
    η = zeros(nb, ndof)
    for j in 1:nt, i in 1:n
        rvec = dad.Nodes[i] - pts[j]
        η[2i-1:2i, 2j-1:2j] .= s .* BEM._drm_t(rvec, dad.Normal[i], nu, Gmod, ker, C)
    end
    H, G = Matrix(dad.H), Matrix(dad.G)
    # paper: M = ρ(Gη − HΨ)F⁻¹ ;  we store M_bem so that H u − G t = M_bem ü
    M_bem = (H * Ψ - G * η) / F
    M_paper = (G * η - H * Ψ) / F   # ρ=1
    set_cache!(dad; M = M_bem)
    return (; M_bem, M_paper, F, Ψ, η)
end

function houbolt_condensed!(dad, Δt, tf; mass_sign=1.0)
    sys = build_modal_system(dad)
    M = mass_sign .* sys.M
    K, f0, free = sys.K, sys.f0, sys.free
    ndof, nb = BEM._neq(dad)
    t = collect(0:Δt:tf)
    nT = length(t)
    Y = zeros(length(free), nT)
    V = zeros(length(free))
    Ae = factorize(M .+ (Δt^2) .* K)
    for i in 2:min(3, nT)
        a = Ae \ (f0 .- K * (Y[:, i - 1] .+ Δt .* V))
        Y[:, i] .= Y[:, i - 1] .+ Δt .* V .+ (Δt^2) .* a
        V .+= Δt .* a
    end
    if nT >= 4
        Mt = M ./ Δt^2
        A = factorize(K .+ 2 .* Mt)
        for i in 4:nT
            rhs = f0 .+ Mt * (5 .* Y[:, i - 1] .- 4 .* Y[:, i - 2] .+ Y[:, i - 3])
            Y[:, i] .= A \ rhs
        end
    end
    U = zeros(ndof, nT)
    q = zeros(nb, nT)
    for j in 1:nT
        U[free, j] .= Y[:, j]
        BEM._pin_known!(view(U, :, j), dad)
        BEM._scatter_step!(dad, view(U, :, j), view(q, :, j))
    end
    set_cache!(dad; u=U, T=U, traction=q, q=q, time=t)
    return U, t, sys
end

function probe_ux(U, dad)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - PROBE) for p in pts)
    return U[2 * (ip - 1) + 1, :]
end

rel(a, b) = norm(a .- b) / (norm(b) + eps())
nneg(A) = count(<( -1e-8), real.(eigvals(A)))

function _run_rect_paper_plots()
dad0 = make_dad()
@printf("elements=%d n=%d ni=%d nt=%d  Lx=%.0f Ly=%.0f  T=%.0f tf=%.0f\n",
    length(dad0.elements), dad0.n, dad0.ni, dad0.nt, Lx, Ly, Tper, tf)
H_G_full_direct(dad0; npg=12, threaded=false)

ana = ana_bar_sudden(; N=400, c=1.0, L=Lx)

println("\n========== sign / equivalence (Δt=1, paper) ==========")
# flip=true  → current library: Ψ=-û, M_bem = (Gη-Hû)/F = M_paper
# flip=false → Ψ=+û,  M_bem = (Hû-Gη)/F = -M_paper  → Hu−Gt = −M_paper ü
# condensed always integrates M̄ü + K̄u = f (K flipped to tr>0).
runs_sign = []
for flip in (true, false)
    dad = deepcopy(dad0)
    drm = drm_mass!(dad; flip=flip)
    sys = build_modal_system(dad)
    @printf("flip=%-5s  tr(M_bem)=% .3e  nneg(M_bem)=%d  tr(M_paper)=% .3e  nneg(M̄)=%d  tr(K)=%.3e\n",
        flip, tr(drm.M_bem), nneg(drm.M_bem), tr(drm.M_paper), nneg(sys.M), tr(sys.K))
    U, t, _ = houbolt_condensed!(dad, 1.0, tf)
    ua = [ana.u(PROBE; t=ti) for ti in t]
    ux = probe_ux(U, dad)
    @printf("  Houbolt condensed  finite=%s  max=%.3f  rel=%.3e\n",
        all(isfinite, ux), maximum(abs, filter(isfinite, ux); init=0.0),
        all(isfinite, ux) ? rel(ux, ua) : NaN)
    push!(runs_sign, (; flip, t, ux, ua))
end

println("\n========== Δt sweep (flip=true = current / M_paper) ==========")
dts = (1.0, 0.5, 0.25)
runs_dt = []
dadM = deepcopy(dad0)
drm_mass!(dadM; flip=true)
for Δt in dts
    dad = deepcopy(dadM)
    U, t, _ = houbolt_condensed!(dad, Δt, tf)
    ua = [ana.u(PROBE; t=ti) for ti in t]
    ux = probe_ux(U, dad)
    r = all(isfinite, ux) ? rel(ux, ua) : NaN
    @printf("  Δt=%4.2f  nT=%4d  finite=%s  max=%.3f  rel=%.3e\n",
        Δt, length(t), all(isfinite, ux),
        maximum(abs, filter(isfinite, ux); init=0.0), r)
    push!(runs_dt, (; Δt, t, ux, ua, r))
end

# also no-flip dt=1 already in runs_sign
mkpath(joinpath(projectdir(), "plots"))

pltS = plot(runs_sign[1].t, runs_sign[1].ua; color=:black, ls=:dash, label="1D series")
labs = Dict(true => "current (Ψ=-û, M=M_paper)", false => "no flip (Ψ=+û, M=-M_paper)")
cols = Dict(true => :steelblue, false => :darkorange)
for r in runs_sign
    ux = copy(r.ux)
    ux[.!isfinite.(ux) .| (abs.(ux) .> 80)] .= NaN
    plot!(pltS, r.t, ux; color=cols[r.flip], label=labs[r.flip])
end
plot!(pltS; xlabel=L"t", ylabel=L"u_x(L, H/2)", ylim=(-2, 28),
    title="12×6 bar  DRM \$f=r\$  4 periods  \$\\Delta t=1\$")
outS = joinpath(projectdir(), "plots", "bar12x6_sign")
savefig(pltS, outS * ".png"); savefig(pltS, outS * ".pdf")
println("wrote ", outS * ".png")

pltD = plot(runs_dt[1].t, runs_dt[1].ua; color=:black, ls=:dash, label="1D series")
colsdt = [:steelblue, :darkorange, :seagreen]
for (k, r) in enumerate(runs_dt)
    ux = copy(r.ux)
    ux[.!isfinite.(ux) .| (abs.(ux) .> 80)] .= NaN
    plot!(pltD, r.t, ux; color=colsdt[k],
        label=latexstring("\\Delta t=$(r.Δt)"))
end
plot!(pltD; xlabel=L"t", ylabel=L"u_x(L, H/2)", ylim=(-2, 28),
    title="12×6 bar  current DRM  4 periods")
outD = joinpath(projectdir(), "plots", "bar12x6_dt")
savefig(pltD, outD * ".png"); savefig(pltD, outD * ".pdf")
println("wrote ", outD * ".png")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    _run_rect_paper_plots()
end
