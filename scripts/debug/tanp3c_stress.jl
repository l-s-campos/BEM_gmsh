# Hunt conditions where default :tanp3c loses to :sinhsinh or goes NaN.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays, FastGaussQuadrature

const NPG = 12
u0, w0 = gausslegendre(NPG)

function apply_map(nf, ζ0, η0; n=NPG)
    u, w = gausslegendre(n)
    nf === :tanp3c && return BEM._tanp3ctrans(u, w, ζ0, η0)
    nf === :tangent && return BEM._tangenttrans(u, w, ζ0, η0)
    nf === :sinhsinh && return BEM._sinhtrans_iterated(u, w, ζ0, η0; niter=2)
    nf === :csinh && return BEM._sinhtrans(u, w, ζ0, η0)
    return collect(u), collect(w)
end

function Iquad(x, w, f)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += w[i] * f(x[i])
    end
    return s
end

R(ξ, ζ0, η0) = hypot(ξ - ζ0, η0)
I1true(ζ0, η0) = asinh((1 - ζ0) / η0) - asinh((-1 - ζ0) / η0)
I2true(ζ0, η0) = (atan((1 - ζ0) / η0) + atan((1 + ζ0) / η0)) / η0

function report_geom(label, ζ0, η0; nref=80)
    println("\n-- $label  ζ0=$(ζ0)  η0=$(η0) --")
    xref, wref = apply_map(:sinhsinh, ζ0, η0; n=nref)
    kernels = (
        ("1/R",   ξ -> 1 / R(ξ, ζ0, max(η0, 0)), η0 > 0 ? I1true(ζ0, η0) : log((abs(ζ0)+1)/(abs(ζ0)-1))),
        ("1/R²",  ξ -> 1 / R(ξ, ζ0, max(η0, 0))^2, η0 > 0 ? I2true(ζ0, η0) : 2/(ζ0^2-1)),
        ("log R", ξ -> log(R(ξ, ζ0, max(η0, 1e-300)))),
        ("1/R⁴",  ξ -> 1 / R(ξ, ζ0, max(η0, 0))^4),
    )
    @printf("  %-6s  %8s %10s %10s %10s %10s\n", "map", "n", "in[-1,1]", "sumw-2", "1/R² rel", "log rel")
    for nf in (:tanp3c, :tangent, :sinhsinh, :csinh, :plain)
        x, w = apply_map(nf, ζ0, η0)
        ok = all(-1.001 .<= x .<= 1.001) && all(isfinite, x) && all(isfinite, w)
        dw = sum(w) - 2
        I2 = Iquad(x, w, kernels[2][2])
        I2t = kernels[2][3]
        Ilog = Iquad(x, w, kernels[3][2])
        Ilogr = Iquad(xref, wref, kernels[3][2])
        rel2 = abs(I2 - I2t) / max(abs(I2t), 1e-16)
        rell = abs(Ilog - Ilogr) / max(abs(Ilogr), 1e-16)
        flag = ok ? " " : " FAIL"
        @printf("  %-6s  %8d %10s %10.2e %10.2e %10.2e%s\n",
            String(nf), length(x), string(ok), dw, rel2, rell, flag)
    end
end

# Geometric pole placements that assembly actually hits.
report_geom("interior pole", 0.0, 0.1)
report_geom("close interior", 0.25, 1e-4)
report_geom("very close interior", 0.0, 1e-8)
report_geom("offset interior", 0.75, 1e-3)
report_geom("endpoint adjacent ζ0=1", 1.0, 1e-2)
report_geom("endpoint adjacent tiny", 1.0, 1e-6)
report_geom("endpoint ζ0=-1", -1.0, 1e-4)
report_geom("just outside (Case 2)", 1.05, 1e-4)
report_geom("just outside (Case 3)", -1.2, 6e-5)
report_geom("collinear left η0=0", -4.0, 0.0)
report_geom("collinear right η0=0", 4.58, 0.0)
report_geom("almost collinear", -3.5, 1e-12)

function solve_near(maker, props, ana; near, npg=12, nome="x")
    dad = maker(nome, props)
    ana !== nothing && attach_analytical!(dad, ana)
    set_cache!(dad; nearfield=near)
    assemble!(dad, npg; threaded=false)
    solve(dad)
    return dad
end

println("\n=== BIE: tanp3c vs sinhsinh ===")
function bie_row(label, dad_t, dad_s; flux=false)
    eT = has_cache(dad_t, :analytical) ? rel_error(dad_t) : NaN
    eS = has_cache(dad_s, :analytical) ? rel_error(dad_s) : NaN
    dH = norm(dad_t.H - dad_s.H) / max(norm(dad_s.H), 1e-30)
    dG = norm(dad_t.G - dad_s.G) / max(norm(dad_s.G), 1e-30)
    @printf("  %-28s  tanp3c %.3e  sinh %.3e  relH %.2e  relG %.2e",
        label, eT, eS, dH, dG)
    if flux
        @printf("  q_t %.3e  q_s %.3e", rel_error_flux(dad_t), rel_error_flux(dad_s))
    end
    println()
    return eT, eS
end

# Laplace T=x linear
anaL = ana_laplace_linear(; direction=SA[1.0, 0.0])
mkL1(nome, p) = format2d(quadrado(ndiv=10, show=false, nome=nome), p; pontointerno=true)
dt = solve_near(mkL1, Laplace(1.0), anaL; near=:tanp3c, nome="st_L1t")
ds = solve_near(mkL1, Laplace(1.0), anaL; near=:sinhsinh, nome="st_L1s")
bie_row("Laplace T=x linear", dt, ds; flux=true)

# Laplace T=x quadratic
mkL2(nome, p) = format2d(quadrado(ndiv=8, show=false, nome=nome, ordem=2), p;
    pontointerno=true, tipo=2)
dt = solve_near(mkL2, Laplace(1.0), anaL; near=:tanp3c, nome="st_L2t")
ds = solve_near(mkL2, Laplace(1.0), anaL; near=:sinhsinh, nome="st_L2s")
bie_row("Laplace T=x quadratic", dt, ds; flux=true)

# Elasticity patch
anaE = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
function mkE(nome, p)
    dad = format2d(quadrado_elasticity(ndiv=8, show=false, nome=nome), p; pontointerno=false)
    apply_analytical_bc!(dad, anaE)
    return dad
end
dt = solve_near(mkE, Elasticity(1.0, 0.3, 1.0), anaE; near=:tanp3c, nome="st_Et")
ds = solve_near(mkE, Elasticity(1.0, 0.3, 1.0), anaE; near=:sinhsinh, nome="st_Es")
bie_row("elasticity patch", dt, ds)

# Helmholtz (no ana): compare fields
mkH(nome, p) = format2d(quadrado(ndiv=8, show=false, nome=nome), p; pontointerno=false)
dt = solve_near(mkH, Helmholtz(; ω=1.0, c=1.0), nothing; near=:tanp3c, nome="st_Ht")
ds = solve_near(mkH, Helmholtz(; ω=1.0, c=1.0), nothing; near=:sinhsinh, nome="st_Hs")
@printf("  %-28s  finite T %s  relT %.2e  relH %.2e\n", "Helmholtz ω=1",
    string(all(isfinite, dt.T) && all(isfinite, ds.T)),
    norm(dt.T - ds.T) / max(norm(ds.T), 1e-30),
    norm(dt.H - ds.H) / max(norm(ds.H), 1e-30))

dt = solve_near(mkH, Helmholtz(; ω=8.0, c=1.0), nothing; near=:tanp3c, npg=20, nome="st_H8t")
ds = solve_near(mkH, Helmholtz(; ω=8.0, c=1.0), nothing; near=:sinhsinh, npg=20, nome="st_H8s")
@printf("  %-28s  finite T %s  relT %.2e  relH %.2e\n", "Helmholtz ω=8",
    string(all(isfinite, dt.T) && all(isfinite, ds.T)),
    norm(dt.T - ds.T) / max(norm(ds.T), 1e-30),
    norm(dt.H - ds.H) / max(norm(ds.H), 1e-30))

# HBIE Laplace
function hbie(near, nome)
    dad = format2d(quadrado(ndiv=6, show=false, nome=nome), Laplace(1.0); pontointerno=false)
    set_cache!(dad; nearfield=near)
    H, G = H_G_hyper(dad; npg=12, threaded=false)
    return H, G, all(isfinite, H) && all(isfinite, G)
end
Ht, Gt, okt = hbie(:tanp3c, "st_hbt")
Hs, Gs, oks = hbie(:sinhsinh, "st_hbs")
@printf("  %-28s  finite %s  relH %.2e  relG %.2e\n", "Laplace HBIE",
    string(okt && oks),
    norm(Ht - Hs) / max(norm(Hs), 1e-30),
    norm(Gt - Gs) / max(norm(Gs), 1e-30))

# Dual BEM centre crack
using BEM.Crack
function crack(near)
    dad = build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, σ=1.0,
        E=3000.0, ν=0.2, n_bottom=4, n_right=6, n_top=4, n_left=6, n_crack=6,
        nome="st_cr_$(near)")
    set_cache!(dad; nearfield=near)
    assemble_dual!(dad; npg=8, threaded=false)
    solve_dual!(dad; threaded=false)
    KI_L, KII_L = sif_cod_dual(dad, dad.tip_nodes[1])
    KI_R, KII_R = sif_cod_dual(dad, dad.tip_nodes[2])
    KIana = analytical_KI_center_crack(1.0, 1.0; W=5.0)
    KIn = 0.5 * (abs(KI_L) + abs(KI_R))
    return KIn, KIana, abs(KII_L) / KIana
end
Kt, ana, k2t = crack(:tanp3c)
Ks, _, k2s = crack(:sinhsinh)
@printf("  %-28s  tanp3c KI rel %.3e  sinh %.3e  KII/KI t %.3e s %.3e\n",
    "dual centre crack", abs(Kt-ana)/ana, abs(Ks-ana)/ana, k2t, k2s)

# Anisotropic default must stay zsinh
pars = lekhnitskii_engineering(124.04, 10.09, 6.03, 0.334; η12_1=1.255, η12_2=-0.031)
@printf("  anisotropic default = %s (expect zsinh)\n",
    string(default_nearfield(AnisotropicElasticity(pars))))

println("\ndone")
