# Tran, Lee, Nguyen-Van, Nguyen-Xuan, Abdel Wahab,
# Int. J. Non-Linear Mech. 72 (2015) 42–52 — static cases, BEM not IGA-HSDT.
#
# Thin isotropic: Kirchhoff DIBEM + von Kármán (`LargePlate`).
# Laminates: Wang FSDT + Lekhnitskii membrane + von Kármán (`LaminatedShell`).
# Soft SS (w=0, M free) vs paper hard SS (β_n=0). Wang drops B (symmetric ESL).
#
#   julia --project=. scripts/plates/tran2015/run_static.jl
#   julia --project=. scripts/plates/tran2015/run_static.jl fig2 table1
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
try
    using Plots
catch
end
using BEM.Plate

const OUT = @__DIR__
const NEL = 6
const NINT = 25
const NEL_P = 4
const NINT_P = 9
const NPG = 6
const NSUB = 4
const RBF = PHS(2; poly_deg=1)

# Paper “Present” (IGA-HSDT) for comparison.
const T1_P = [1, 2, 3, 6, 10, 15]
const T1_PRESENT = [0.1669, 0.323, 0.4562, 0.761, 1.0487, 1.2989]
const T1_ANAL = [0.169, 0.323, 0.457, 0.761, 1.035, 1.279]

const T2_P = [50, 100, 150, 200, 250]
const T2_PRESENT = Dict(
    40 => [0.2936, 0.4643, 0.5798, 0.6683, 0.7407],
    20 => [0.3126, 0.4807, 0.5928, 0.6784, 0.7486],
    10 => [0.3609, 0.5179, 0.6213, 0.7005, 0.7659])

# Table 3 HSDT nonlinear [0/90/90/0] and [0/90/0] (PDF Table 3).
const T3_P = [50, 100, 200, 300]
const T3_0900_HSDT_L = Dict(4 => [0.947, 1.894, 3.787, 5.681],
    10 => [0.357, 0.715, 1.430, 2.144],
    20 => [0.253, 0.506, 1.012, 1.518],
    100 => [0.217, 0.434, 0.868, 1.303])
const T3_0900_HSDT_NL = Dict(4 => [0.7198, 1.1214, 1.6555, 2.0447],
    10 => [0.3474, 0.6501, 1.1148, 1.4612],
    20 => [0.2504, 0.4872, 0.8960, 1.2255],
    100 => [0.2159, 0.4243, 0.7993, 1.1146])
const T3_090_HSDT_NL = Dict(4 => [0.7262, 1.1284, 1.6606, 2.0472],
    10 => [0.3462, 0.6478, 1.1116, 1.4586],
    20 => [0.2494, 0.4849, 0.8921, 1.2190],
    100 => [0.2158, 0.4238, 0.7969, 1.1101])
const T3_090_HSDT_L = Dict(4 => [0.961, 1.922, 3.844, 5.765],
    10 => [0.356, 0.712, 1.425, 2.137],
    20 => [0.252, 0.504, 1.008, 1.513],
    100 => [0.217, 0.434, 0.868, 1.303])
const T3_0900_FSDT_L = Dict(4 => [0.856, 1.712, 3.423, 5.135],
    10 => [0.331, 0.662, 1.324, 1.986],
    20 => [0.245, 0.490, 0.980, 1.470],
    100 => [0.216, 0.432, 0.865, 1.297])
const T3_0900_FSDT_NL = Dict(4 => [0.6791, 1.0788, 1.6111, 1.9877],
    10 => [0.3236, 0.6121, 1.0667, 1.4100],
    20 => [0.2428, 0.4734, 0.8763, 1.2024],
    100 => [0.2150, 0.4226, 0.7967, 1.1117])
const T3_090_FSDT_L = Dict(4 => [0.889, 1.778, 3.556, 5.335],
    10 => [0.334, 0.669, 1.338, 2.006],
    20 => [0.246, 0.491, 0.982, 1.473],
    100 => [0.216, 0.432, 0.865, 1.297])
const T3_090_FSDT_NL = Dict(4 => [0.6948, 1.0788, 1.6316, 2.0078],
    10 => [0.3264, 0.6162, 1.0713, 1.4154],
    20 => [0.2432, 0.4737, 0.8752, 1.1999],
    100 => [0.2149, 0.4222, 0.7945, 1.1074])

const LEVY_Q = [17.79, 38.3, 63.4, 95.0, 134.9, 184.0, 245.0, 318.0, 402.0]
const LEVY_W = [0.237, 0.471, 0.695, 0.912, 1.121, 1.323, 1.521, 1.714, 1.902]

isotropic_A(E, ν, h) = begin
    C = E * h / (1 - ν^2)
    @SMatrix [C ν*C 0; ν*C C 0; 0 0 E*h / (2 * (1 + ν))]
end

q_from_P(P, E2, a, h) = P * E2 * h^4 / a^4

function matIII(E2=1.0)
    E1 = 25 * E2
    return (E1, E2, 0.25, 0.5 * E2, 0.5 * E2, 0.2 * E2)
end
function matIV(E2=1.0)
    E1 = 40 * E2
    return (E1, E2, 0.25, 0.6 * E2, 0.6 * E2, 0.5 * E2)
end

function plies_cross(angles, h, mat)
    E1, E2, ν12, G12, G13, G23 = mat
    t = h / length(angles)
    return [(E1, E2, ν12, G12, Float64(θ), t) for θ in angles], G13, G23
end

function pct(a, b)
    (isnan(a) || b == 0) && return NaN
    return 100 * (a - b) / b
end

# ---------------------------------------------------------------------------
# LaminatedShell von Kármán helper
# ---------------------------------------------------------------------------

function make_shell(a, h, angles, mat; bc="SSSS", mem_bc=:navier_ss,
        n_el=NEL, n_int=NINT, q=1.0)
    pl, G13, G23 = plies_cross(angles, h, mat)
    props = laminate_fsdt_props(pl; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=1.0)
    A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=G13, G23=G23)
    mesh = build_square_fsdt(; a=a, n_el=n_el, bc=bc, props=props, n_internal=n_int)
    shell = LaminatedShell(mesh, A, FlatShell(); mem_bc=mem_bc)
    assemble_laminated_shell!(shell; npg=NPG, nsub=NSUB, rbf=RBF,
        rbf_grad=PHS(3; poly_deg=1))
    return shell
end

function w_center(shell)
    return abs(fsdt_w_int(shell.plate, 1))
end

function solve_lin!(shell)
    solve_laminated_shell!(shell)
    return w_center(shell)
end

function set_plate_q!(mesh, q)
    if mesh isa FSDTMesh
        mesh.q = q
    else
        set_cache!(mesh; fsdt_q=q, q=q)
    end
    return q
end

"""Pagano load `q₀ sin(πx/a) sin(πy/a)` as DIBEM `Mw * q_pts`."""
function apply_sine_pressure!(shell, a, q0)
    mesh = shell.plate
    pts = Point2D[BEM.Plate._plate_nodes(mesh); BEM.Plate._plate_internal(mesh)]
    q_pts = [q0 * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
    isempty(shell.Mw) && apply_shell_coupling!(shell)
    return set_plate_q!(mesh, shell.Mw * q_pts)
end

function solve_nl!(shell; nsteps, λ_max=1.0, nonlinear=:newton)
    res = solve_laminated_shell!(shell; large=true, nsteps=nsteps, λ_max=λ_max,
        nonlinear=nonlinear, e_relax=0.4, maxiters=8, atol=1e-8)
    w = abs.(res.w_center)
    for i in eachindex(w)
        (isfinite(w[i]) && w[i] < 50 * BEM.Plate._plate_props(shell.plate).h) ||
            (w[i] = NaN)
    end
    return abs.(res.λ), w
end

"""Linear + NL `w/h` at load parameters `Ps` (`q_c` set to `Pmax`)."""
function laminate_curve(a, h, angles, mat, Ps; bc="SSSS", mem_bc=:navier_ss,
        n_el=NEL, n_int=NINT, E2=mat[2], nsteps::Int=0, nonlinear=:newton,
        linear_sine::Bool=false)
    Pmax = maximum(Ps)
    q = q_from_P(Pmax, E2, a, h)
    shell = make_shell(a, h, angles, mat; bc=bc, mem_bc=mem_bc,
        n_el=n_el, n_int=n_int, q=q)
    q_uni = copy(BEM.Plate._plate_q(shell.plate))
    if linear_sine
        apply_sine_pressure!(shell, a, q)
    end
    wL = solve_lin!(shell)
    wL_h = [(P / Pmax) * wL / h for P in Ps]
    if !linear_sine
        set_plate_q!(shell.plate, q_uni)
    end
    ns = nsteps > 0 ? nsteps : max(10, 2 * length(Ps))
    λs, wcs = solve_nl!(shell; nsteps=ns, λ_max=1.0, nonlinear=nonlinear)
    wnl = [interp_wh(λs, wcs, P / Pmax) / h for P in Ps]
    return (w_lin=wL_h, w_nl=wnl, shell=shell)
end

const NEL_U = 3
const NINT_U = 9
const NPG_U = 4
const NSUB_U = 4

"""Hsu–Hwu 5-DOF von Kármán when `B≠0`; Wang 3-DOF shell if `B≈0` (singular Z)."""
function unsym_curve(a, h, angles, mat, Ps; bc="SSSS", n_el=NEL_U, n_int=NINT_U,
        E2=mat[2])
    Pmax = maximum(Ps)
    q = q_from_P(Pmax, E2, a, h)
    pl, G13, G23 = plies_cross(angles, h, mat)
    A, B, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=G13, G23=G23)
    if norm(B) < 1e-8 * (norm(A) * h + eps())
        println("    B≈0 → Wang 3-DOF + membrane")
        return laminate_curve(a, h, angles, mat, Ps; bc=bc, mem_bc=:navier_ss,
            n_el=n_el, n_int=n_int, E2=E2)
    end
    props = laminate_unsym_props(pl; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=1.0,
        nθ=8)
    mesh = build_square_fsdt(; a=a, n_el=n_el, bc=bc, props=props, n_internal=n_int)
    assemble_fsdt!(mesh; npg=NPG_U, nsub=NSUB_U)
    dibem_fsdt!(mesh; npg=NPG_U)
    solve_fsdt!(mesh)
    wL = abs(fsdt_w_int(mesh, 1))
    wL_h = [(P / Pmax) * wL / h for P in Ps]
    nsteps = 20
    res = solve_fsdt!(mesh; large=true, nsteps=nsteps, λ_max=1.0,
        nonlinear=:picard, e_relax=0.4, maxiters=10)
    wcs = abs.(res.w_center)
    for i in eachindex(wcs)
        (isfinite(wcs[i]) && wcs[i] < 20 * h) || (wcs[i] = NaN)
    end
    wnl = [interp_wh(res.λ, wcs, P / Pmax) / h for P in Ps]
    return (w_lin=wL_h, w_nl=wnl, mesh=mesh)
end

function interp_wh(λs, wcs, λt)
    keep = [i for i in eachindex(λs) if isfinite(wcs[i])]
    isempty(keep) && return NaN
    λk, wk = λs[keep], wcs[keep]
    i = searchsortedlast(λk, λt)
    i <= 0 && return wk[1]
    i >= length(λk) && return wk[end]
    Δ = λk[i + 1] - λk[i]
    t = Δ == 0 ? 0.0 : (λt - λk[i]) / Δ
    return (1 - t) * wk[i] + t * wk[i + 1]
end

function internals_square(a, nint)
    pts = Point2D[Point2D(a / 2, a / 2)]
    nint <= 1 && return pts
    g = ceil(Int, sqrt(nint))
    xs = range(a / (g + 1), a * g / (g + 1); length=g)
    for y in xs, x in xs
        hypot(x - a / 2, y - a / 2) < 1e-12 && continue
        push!(pts, Point2D(x, y))
        length(pts) >= nint && break
    end
    return pts
end

function internals_disk(R, nint)
    pts = Point2D[Point2D(0.0, 0.0)]
    nint <= 1 && return pts
    g = ceil(Int, sqrt(nint))
    xs = range(-R, R; length=g + 2)[2:(end - 1)]
    for y in xs, x in xs
        hypot(x, y) < 0.85 * R || continue
        hypot(x, y) < 1e-12 && continue
        push!(pts, Point2D(x, y))
        length(pts) >= nint && break
    end
    return pts
end

function gmsh_disk(R, ndiv, ordem, nome, phys, folder)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 2 * π * R / (4 * max(ndiv - 1, 3))
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p = [gmsh.model.geo.addPoint(R * cos(k * π / 2), R * sin(k * π / 2), 0.0, lc)
         for k in 0:3]
    arcs = [gmsh.model.geo.addCircleArc(p[i], c, p[mod1(i + 1, 4)]) for i in 1:4]
    cl = gmsh.model.geo.addCurveLoop(arcs)
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for a in arcs
        gmsh.model.mesh.setTransfiniteCurve(a, ndiv)
    end
    gmsh.model.addPhysicalGroup(1, arcs, -1, phys)
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir(folder, nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function print_row(io, cols...)
    for (i, c) in enumerate(cols)
        i > 1 && print(io, "  ")
        if c isa AbstractFloat
            @printf(io, "%10.4f", c)
        else
            print(io, rpad(string(c), 10))
        end
    end
    println(io)
end

# =============================================================================
# Fig 2 — clamped isotropic square (Levy / NACA 847), Kirchhoff BEM
# =============================================================================

function run_fig2()
    println("\n=== Fig 2  CCCC isotropic square  Kirchhoff von Kármán BEM ===")
    a, h, E, ν = 1.0, 0.01, 1.0e6, 0.316
    Qmax = 400.0
    q0 = q_from_P(Qmax, E, a, h)
    props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
    internal = internals_square(a, NINT)
    plate = build_square_plate(; a=a, n_el=NEL, bc="CCCC", props=props,
        n_internal=length(internal), internal=internal, corner_bc='C', p=2)
    assemble_plate!(plate; npg=NPG)
    dibem_plate!(plate; npg=NPG, rbf=RBF, apply_load=true)

    include(datadir("Laplace", "Laplace_dad.jl"))
    msh = quadrado_elasticity(; ndiv=NEL + 1, show=false, nome="tran2015_pe",
        Lx=a, Ly=a, ordem=1)
    dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false);
        pontointerno=true)
    fill!(dad_pe.BC, 1); fill!(dad_pe.BV, 0.0)
    @inbounds for i in 1:dad_pe.n
        p = dad_pe.Nodes[i]
        if p[1] < 1e-8 * a || p[1] > a * (1 - 1e-8)
            dad_pe.BC[2i - 1] = 0; dad_pe.BV[2i - 1] = 0.0
        end
        if p[2] < 1e-8 * a || p[2] > a * (1 - 1e-8)
            dad_pe.BC[2i] = 0; dad_pe.BV[2i] = 0.0
        end
    end
    set_internal_nodes!(dad_pe, plate.internalNodes)
    H_G_full_direct(dad_pe; npg=NPG, threaded=false)
    dibem_elasticity!(dad_pe; npg=NPG, rbf=RBF)
    prob = build_large_plate_problem(plate, dad_pe; npg_plate=NPG, npg_pe=NPG,
        plate_dibem=true, rbf=RBF)

    D = bending_stiffness(props)
    w_lin_h = abs(linear_wmax_reference(prob; λ=17.79 / Qmax)) / h
    w_levy_lin = 0.001263 * q_from_P(17.79, E, a, h) * a^4 / (D * h)
    @printf("  linear Q=17.79: BEM w/h=%.4f  Levy 0.001263→%.4f  e=%.2f %%\n",
        w_lin_h, w_levy_lin, pct(w_lin_h, w_levy_lin))

    Ps = collect(50.0:50.0:400.0)
    λ_path = [Ps; LEVY_Q] ./ Qmax
    res = solve_large_plate!(prob; λ_path=λ_path, e_relax=0.4,
        abstol=1e-6, reltol=1e-6, maxiters=20, nonlinear=:newton)
    wcs = abs.(res.w_center) ./ h
    nP = length(Ps)
    println("  P=q a⁴/(E h⁴)     BEM w/h")
    for i in 1:nP
        @printf("  %8.0f        %8.4f\n", Ps[i], wcs[i])
    end
    println("  Levy Table 5:")
    @printf("  %8s  %9s  %9s  %8s\n", "Q", "BEM", "Levy", "e %")
    for (k, Q) in enumerate(LEVY_Q)
        i = nP + k
        e = pct(wcs[i], LEVY_W[k])
        @printf("  %8.2f  %9.4f  %9.3f  %8.2f\n", Q, wcs[i], LEVY_W[k], e)
    end

    # Kirchhoff σ̄_x(centre, z=h/2) from RBF Hessian.
    w = extract_w_field(prob, res.u_final)
    wxx = prob.Fx * (prob.Fx * w)
    wyy = prob.Fy * (prob.Fy * w)
    k0 = 1
    σ = -E * (h / 2) / (1 - ν^2) * (wxx[k0] + ν * wyy[k0])
    σbar = σ * a^2 / (E * h^2)
    @printf("  σ̄_x centre top at P=400: %.3f\n", σbar)

    plt = plot(Ps, wcs[1:nP]; lw=2, marker=:circle, label="BEM Kirchhoff VK",
        xlabel=raw"$\bar P = q a^4/(E h^4)$", ylabel=raw"$\bar w = w_c/h$",
        title="Fig 2a  CCCC isotropic square", legend=:topleft)
    scatter!(plt, LEVY_Q, LEVY_W; marker=:square, label="Levy NACA R-740")
    savefig(plt, joinpath(OUT, "fig2_w.png"))
    open(joinpath(OUT, "fig2.csv"), "w") do io
        println(io, "P,w_h")
        for i in 1:nP
            println(io, "$(Ps[i]),$(wcs[i])")
        end
    end
    return wcs
end

# =============================================================================
# Table 1 — clamped circular plate
# =============================================================================

function run_table1()
    println("\n=== Table 1  CCCC circular  Kirchhoff von Kármán BEM ===")
    R, ν, E = 1.0, 0.3, 1.0e7
    h = R / 50
    Pmax = 15.0
    q = q_from_P(Pmax, E, R, h)
    props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q)
    internal = internals_disk(R, NINT)
    msh = gmsh_disk(R, 7, 2, "tran2015_circ_plate", "0;0;0;0", "Laplace")
    plate = formatdata(msh, props; tipo=2, pontointerno=false)
    set_internal_nodes!(plate, internal)
    prepare_plate!(plate; corner_bc='C')
    assemble_plate!(plate; npg=NPG)
    dibem_plate!(plate; npg=NPG, rbf=RBF, apply_load=true)

    msh_pe = gmsh_disk(R, 7, 1, "tran2015_circ_pe", "0;0;0;0", "elastico")
    dad_pe = format2d(msh_pe, Elasticity(E, ν, 1.0; plane_strain=false);
        pontointerno=true)
    set_internal_nodes!(dad_pe, internal)
    H_G_full_direct(dad_pe; npg=NPG, threaded=false)
    dibem_elasticity!(dad_pe; npg=NPG, rbf=RBF)
    prob = build_large_plate_problem(plate, dad_pe; npg_plate=NPG, npg_pe=NPG,
        plate_dibem=true, rbf=RBF)

    λ_path = T1_P ./ Pmax
    res = solve_large_plate!(prob; λ_path=λ_path, e_relax=0.4,
        abstol=1e-6, reltol=1e-6, maxiters=20, nonlinear=:newton)
    wcs = abs.(res.w_center) ./ h
    println("  P      BEM w/h   Present   Anal.    e_anal %")
    open(joinpath(OUT, "table1.csv"), "w") do io
        println(io, "P,BEM,Present,Anal")
        for i in eachindex(T1_P)
            @printf("  %4.0f   %8.4f  %8.4f  %8.4f  %8.2f\n",
                T1_P[i], wcs[i], T1_PRESENT[i], T1_ANAL[i], pct(wcs[i], T1_ANAL[i]))
            println(io, "$(T1_P[i]),$(wcs[i]),$(T1_PRESENT[i]),$(T1_ANAL[i])")
        end
    end
    return wcs
end

# =============================================================================
# Fig 3 — Zaghloul & Kennedy
# =============================================================================

function run_fig3()
    println("\n=== Fig 3  Zaghloul–Kennedy  LaminatedShell BEM ===")
    a = 0.3048
    # (a) SSSS1 orthotropic, Material I
    h = 3.51e-3
    matI = (20.684e9, 8.825e9, 0.32, 2.551e9, 2.551e9, 2.551e9)
    qmax = 14e3
    shell = make_shell(a, h, (0.0,), matI; bc="SSSS", mem_bc=:navier_ss,
        n_el=NEL, n_int=NINT, q=qmax)
    λs, wcs = solve_nl!(shell; nsteps=14, λ_max=1.0)
    q_kPa = λs .* 14
    w_cm = wcs .* 100
    open(joinpath(OUT, "fig3a.csv"), "w") do io
        println(io, "q_kPa,w_cm")
        for i in eachindex(q_kPa)
            println(io, "$(q_kPa[i]),$(w_cm[i])")
        end
    end
    plt = plot(q_kPa, w_cm; lw=2, marker=:circle, label="BEM FSDT VK",
        xlabel="load (kPa)", ylabel="w_c (cm)",
        title="Fig 3a  SSSS orthotropic (Mat. I)")
    savefig(plt, joinpath(OUT, "fig3a.png"))
    @printf("  (a) q=14 kPa  w_c=%.3f cm\n", w_cm[end])

    # (b) CCCC [0/90/90/0] Material II
    h2 = 2.44e-3
    matII = (12.604e9, 12.627e9, 0.2395, 2.155e9, 2.155e9, 2.155e9)
    shell2 = make_shell(a, h2, (0.0, 90.0, 90.0, 0.0), matII; bc="CCCC",
        mem_bc=:clamped, n_el=NEL, n_int=NINT, q=qmax)
    λs2, wcs2 = solve_nl!(shell2; nsteps=14, λ_max=1.0)
    w_cm2 = wcs2 .* 100
    open(joinpath(OUT, "fig3b.csv"), "w") do io
        println(io, "q_kPa,w_cm")
        for i in eachindex(λs2)
            println(io, "$(λs2[i]*14),$(w_cm2[i])")
        end
    end
    plt = plot(λs2 .* 14, w_cm2; lw=2, marker=:circle, label="BEM FSDT VK",
        xlabel="load (kPa)", ylabel="w_c (cm)",
        title="Fig 3b  CCCC [0/90/90/0] (Mat. II)")
    savefig(plt, joinpath(OUT, "fig3b.png"))
    @printf("  (b) q=14 kPa  w_c=%.3f cm\n", w_cm2[end])
    return nothing
end

# =============================================================================
# Table 2 — SSSS2 [0/90/90/0] Material III
# =============================================================================

function run_table2()
    println("\n=== Table 2  SSSS2 [0/90/90/0]  Material III ===")
    a, E2 = 1.0, 1.0
    mat = matIII(E2)
    open(joinpath(OUT, "table2.csv"), "w") do io
        println(io, "L_h,P,BEM_lin,BEM_nl,Present,e_pct")
        println("  L/h    P    BEM lin   BEM NL   Present    e %")
        for Lh in (40, 20, 10)
            h = a / Lh
            r = laminate_curve(a, h, (0.0, 90.0, 90.0, 0.0), mat, T2_P;
                bc="SSSS", mem_bc=:clamped, n_el=NEL, n_int=NINT, E2=E2,
                nsteps=10, nonlinear=:newton)
            for (i, P) in enumerate(T2_P)
                e = pct(r.w_nl[i], T2_PRESENT[Lh][i])
                @printf("  %4d  %4d  %8.4f  %8.4f  %8.4f  %7.2f\n",
                    Lh, P, r.w_lin[i], r.w_nl[i], T2_PRESENT[Lh][i], e)
                println(io, "$Lh,$P,$(r.w_lin[i]),$(r.w_nl[i]),$(T2_PRESENT[Lh][i]),$e")
            end
        end
    end
end

# =============================================================================
# EX2-SSSS2 — like Table 2 (immovable membrane) + sine load, thickness sweep
# =============================================================================

const EX2SSSS2_P = [50, 150, 250]
const EX2SSSS2_LH = (4, 10, 20, 100)
const FEM_EX2SSSS2 = joinpath(homedir(),
    "Projects/fenics experimentos/von_karman_plate/tran2015/ex2ssss2.csv")

function run_ex2ssss2()
    n_el, n_int = 8, 49
    println("\n=== EX2-SSSS2  [0/90/90/0] Mat.III  SSSS2  sine  n_el=$n_el n_int=$n_int ===")
    a, E2 = 1.0, 1.0
    mat = matIII(E2)
    angs = (0.0, 90.0, 90.0, 0.0)
    fem = Dict{Tuple{Int,Int},NTuple{2,Float64}}()
    if isfile(FEM_EX2SSSS2)
        for row in eachline(FEM_EX2SSSS2)
            startswith(row, "L_h") && continue
            isempty(strip(row)) && continue
            parts = split(row, ',')
            length(parts) >= 4 || continue
            Lh, P = parse(Int, parts[1]), parse(Int, parts[2])
            fem[(Lh, P)] = (parse(Float64, parts[3]), parse(Float64, parts[4]))
        end
        println("  FEniCS csv: $FEM_EX2SSSS2")
    else
        println("  no FEniCS csv yet (run tran2015_fenics.py ex2ssss2)")
    end
    open(joinpath(OUT, "ex2ssss2.csv"), "w") do io
        println(io, "L_h,P,BEM_lin,BEM_nl,FEM_lin,FEM_nl,e_lin,e_nl")
        println("  L/h    P    BEM lin   BEM NL   FEM lin   FEM NL   e_lin%  e_nl%")
        for Lh in EX2SSSS2_LH
            h = a / Lh
            r = laminate_curve(a, h, angs, mat, EX2SSSS2_P;
                bc="SSSS", mem_bc=:clamped, n_el=n_el, n_int=n_int, E2=E2,
                nsteps=10, nonlinear=:newton, linear_sine=true)
            for (i, P) in enumerate(EX2SSSS2_P)
                fl, fn = get(fem, (Lh, P), (NaN, NaN))
                el, en = pct(r.w_lin[i], fl), pct(r.w_nl[i], fn)
                @printf("  %4d  %4d  %8.4f  %8.4f  %8.4f  %8.4f  %7.2f  %7.2f\n",
                    Lh, P, r.w_lin[i], r.w_nl[i], fl, fn, el, en)
                println(io, "$Lh,$P,$(r.w_lin[i]),$(r.w_nl[i]),$fl,$fn,$el,$en")
            end
        end
    end
end

# =============================================================================
# Table 3 + Fig 5 — SSSS1 [0/90/90/0] and [0/90/0]
# =============================================================================

function run_table3()
    println("\n=== Table 3  SSSS1  Material III  linear / NL ===")
    a, E2 = 1.0, 1.0
    mat = matIII(E2)
    stacks = (
        ("[0/90/90/0]", (0.0, 90.0, 90.0, 0.0),
            T3_0900_HSDT_L, T3_0900_FSDT_L, T3_0900_HSDT_NL, T3_0900_FSDT_NL),
        ("[0/90/0]", (0.0, 90.0, 0.0),
            T3_090_HSDT_L, T3_090_FSDT_L, T3_090_HSDT_NL, T3_090_FSDT_NL),
    )
    open(joinpath(OUT, "table3.csv"), "w") do io
        println(io, "stack,L_h,P,BEM_lin_sine,HSDT_lin,FSDT_lin,e_lin_HSDT,e_lin_FSDT,BEM_nl,HSDT_nl,FSDT_nl,e_nl_HSDT,e_nl_FSDT")
        for (name, angs, hL, fL, hN, fN) in stacks
            println("  $name")
            println("  sine q₀ sin(πx/a) sin(πy/a) for linear and NL (Pagano / paper HSDT)")
            println("  L/h    P   BEM lin  HSDT lin FSDT lin  e_F%   BEM NL  HSDT NL  FSDT NL  e_F%")
            for Lh in (4, 10, 20, 100)
                h = a / Lh
                r = laminate_curve(a, h, angs, mat, T3_P;
                    bc="SSSS", mem_bc=:navier_ss, n_el=NEL, n_int=NINT, E2=E2,
                    nsteps=12, nonlinear=:newton, linear_sine=true)
                for (i, P) in enumerate(T3_P)
                    elH = pct(r.w_lin[i], hL[Lh][i])
                    elF = pct(r.w_lin[i], fL[Lh][i])
                    enH = pct(r.w_nl[i], hN[Lh][i])
                    enF = pct(r.w_nl[i], fN[Lh][i])
                    @printf("  %4d %4d  %8.4f  %8.4f %8.4f %6.1f  %8.4f  %8.4f %8.4f %6.1f\n",
                        Lh, P, r.w_lin[i], hL[Lh][i], fL[Lh][i], elF,
                        r.w_nl[i], hN[Lh][i], fN[Lh][i], enF)
                    println(io, "$name,$Lh,$P,$(r.w_lin[i]),$(hL[Lh][i]),$(fL[Lh][i]),$elH,$elF,$(r.w_nl[i]),$(hN[Lh][i]),$(fN[Lh][i]),$enH,$enF")
                end
            end
        end
    end
end

function run_fig5()
    println("\n=== Fig 5  [0/90/90/0] load–deflection vs L/h ===")
    a, E2 = 1.0, 1.0
    mat = matIII(E2)
    Ps = collect(0:25:300)
    Ps[1] = 1.0
    plt = plot(xlabel=raw"$\bar P$", ylabel=raw"$\bar w$",
        title="Fig 5  [0/90/90/0] SSSS1  BEM FSDT VK", legend=:topleft)
    for Lh in (4, 10, 20, 100)
        h = a / Lh
        r = laminate_curve(a, h, (0.0, 90.0, 90.0, 0.0), mat, Ps;
            bc="SSSS", mem_bc=:navier_ss, n_el=NEL_P, n_int=NINT_P, E2=E2)
        plot!(plt, Ps, r.w_nl; lw=2, marker=:circle, markersize=3, label="L/h=$Lh")
    end
    savefig(plt, joinpath(OUT, "fig5.png"))
end

# =============================================================================
# Fig 6 — through-thickness σ_x at centre (FSDT from N, M)
# =============================================================================

function run_fig6()
    println("\n=== Fig 6  [0/90/90/0] L/h=10  σ_x(z) at centre ===")
    a, Lh, E2 = 1.0, 10, 1.0
    h = a / Lh
    mat = matIII(E2)
    pl, G13, G23 = plies_cross((0.0, 90.0, 90.0, 0.0), h, mat)
    A, _, D, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=G13, G23=G23)
    E1, E2m, ν12, G12, _, _ = mat
    ν21 = ν12 * E2m / E1
    den = 1 - ν12 * ν21
    Q11_0, Q22_0, Q12_0 = E1 / den, E2m / den, ν12 * E2m / den
    plt = plot(xlabel=raw"$\bar\sigma_x = \sigma_x h^2/(q a^2)$",
        ylabel="z/h", title="Fig 6a  axial stress at centre", legend=:left)
    for P in (50, 100, 200, 300)
        q = q_from_P(P, E2, a, h)
        shell = make_shell(a, h, (0.0, 90.0, 90.0, 0.0), mat; bc="SSSS",
            mem_bc=:navier_ss, n_el=NEL, n_int=NINT, q=q)
        solve_laminated_shell!(shell; large=true, nsteps=8, λ_max=1.0,
            nonlinear=:newton, maxiters=10)
        rq = shell_resultants(shell; method=:rbf)
        ic = shell.plate.n + 1
        N = [rq.Nx[ic], rq.Ny[ic], 0.0]
        M = [rq.Mx[ic], rq.My[ic], 0.0]
        ε0 = A \ N
        κ = D \ M
        zs, σs = Float64[], Float64[]
        for z in range(-h / 2, h / 2; length=41)
            # 0° outer plies use Q11 of 0°; 90° inner use swapped Q22 as Q11
            zrel = z / h
            Q11, Q12 = abs(zrel) > 0.25 ? (Q11_0, Q12_0) : (Q22_0, Q12_0)
            εx = ε0[1] + z * κ[1]
            εy = ε0[2] + z * κ[2]
            σx = Q11 * εx + Q12 * εy
            push!(zs, zrel)
            push!(σs, σx * h^2 / (q * a^2))
        end
        plot!(plt, σs, zs; lw=2, label="P=$P")
        @printf("  P=%3d  σ̄_x(h/2)=%.3f  σ̄_x(-h/2)=%.3f\n", P, σs[end], σs[1])
    end
    savefig(plt, joinpath(OUT, "fig6.png"))
end

# =============================================================================
# Fig 7–9  Material IV  SSSS1  L/h=10
# =============================================================================

function run_fig7()
    println("\n=== Fig 7  [0/90]_N  Hsu–Hwu 5-DOF von Kármán  L/h=10 ===")
    a, Lh, E2 = 1.0, 10, 1.0
    h = a / Lh
    mat = matIV(E2)
    Ps = collect(0:25:300); Ps[1] = 1.0
    plt = plot(xlabel=raw"$\bar P$", ylabel=raw"$\bar w$",
        title="Fig 7  [0/90]_N  Hsu–Hwu 5-DOF VK", legend=:topleft)
    open(joinpath(OUT, "fig7.csv"), "w") do io
        println(io, "N,P,w_lin,w_nl")
        for N in 1:5
            angs = Tuple(repeat([0.0, 90.0], N))
            r = unsym_curve(a, h, angs, mat, Ps; bc="SSSS", E2=E2)
            plot!(plt, Ps, r.w_nl; lw=2, marker=:circle, markersize=3, label="N=$N")
            for (i, P) in enumerate(Ps)
                println(io, "$N,$P,$(r.w_lin[i]),$(r.w_nl[i])")
            end
            @printf("  N=%d  P=300  w_lin/h=%.4f  w_nl/h=%.4f\n",
                N, r.w_lin[end], r.w_nl[end])
        end
    end
    savefig(plt, joinpath(OUT, "fig7.png"))
end

function run_fig8()
    println("\n=== Fig 8  [-θ/θ/-θ/θ]  Hsu–Hwu 5-DOF von Kármán ===")
    a, Lh, E2 = 1.0, 10, 1.0
    h = a / Lh
    mat = matIV(E2)
    Ps = collect(0:25:300); Ps[1] = 1.0
    plt = plot(xlabel=raw"$\bar P$", ylabel=raw"$\bar w$",
        title="Fig 8  [-θ/θ/-θ/θ]  Hsu–Hwu 5-DOF VK", legend=:topleft)
    open(joinpath(OUT, "fig8.csv"), "w") do io
        println(io, "theta,P,w_lin,w_nl")
        for θ in (0, 15, 30, 45)
            angs = (-Float64(θ), Float64(θ), -Float64(θ), Float64(θ))
            r = unsym_curve(a, h, angs, mat, Ps; bc="SSSS", E2=E2)
            plot!(plt, Ps, r.w_nl; lw=2, marker=:circle, markersize=3,
                label="θ=$θ")
            for (i, P) in enumerate(Ps)
                println(io, "$θ,$P,$(r.w_lin[i]),$(r.w_nl[i])")
            end
            @printf("  θ=%2d  P=300  w_lin/h=%.4f  w_nl/h=%.4f\n",
                θ, r.w_lin[end], r.w_nl[end])
        end
    end
    savefig(plt, joinpath(OUT, "fig8.png"))
end

function run_fig9()
    println("\n=== Fig 9  symmetric vs antisymmetric  Hsu–Hwu / Wang ===")
    a, Lh, E2 = 1.0, 10, 1.0
    h = a / Lh
    mat = matIV(E2)
    thetas = 0:15:90
    plt = plot(xlabel="fibre angle θ", ylabel=raw"$\bar w$",
        title="Fig 9  P=100,200,300  (anti: Hsu–Hwu; sym: Wang if B=0)",
        legend=:topright)
    Ps = [100, 200, 300]
    open(joinpath(OUT, "fig9.csv"), "w") do io
        println(io, "kind,theta,P,w_lin,w_nl")
        curves = Dict{String,Dict{Int,Vector{Float64}}}()
        for (kind, maker) in (("anti", θ -> (-Float64(θ), Float64(θ),
                                             -Float64(θ), Float64(θ))),
                              ("sym", θ -> (-Float64(θ), Float64(θ),
                                            Float64(θ), -Float64(θ))))
            curves[kind] = Dict(P => Float64[] for P in Ps)
            for θ in thetas
                r = unsym_curve(a, h, maker(θ), mat, Ps; bc="SSSS", E2=E2)
                for (i, P) in enumerate(Ps)
                    push!(curves[kind][P], r.w_nl[i])
                    println(io, "$kind,$θ,$P,$(r.w_lin[i]),$(r.w_nl[i])")
                end
                @printf("  %s θ=%2d  P=300 w_lin/h=%.4f  w_nl/h=%.4f\n",
                    kind, θ, r.w_lin[end], r.w_nl[end])
            end
        end
        for P in Ps
            plot!(plt, collect(thetas), curves["anti"][P]; lw=2, marker=:circle,
                label="anti P=$P")
            plot!(plt, collect(thetas), curves["sym"][P]; lw=2, marker=:cross,
                label="sym P=$P")
        end
    end
    savefig(plt, joinpath(OUT, "fig9.png"))
end

# =============================================================================
# Driver
# =============================================================================

const ALL = ["fig2", "table1", "fig3", "table2", "table3", "fig5", "fig6",
    "fig7", "fig8", "fig9"]
const JOBS = Dict(
    "fig2" => run_fig2,
    "table1" => run_table1,
    "fig3" => run_fig3,
    "table2" => run_table2,
    "ex2ssss2" => run_ex2ssss2,
    "ex2scsc" => run_ex2ssss2,
    "table3" => run_table3,
    "fig5" => run_fig5,
    "fig6" => run_fig6,
    "fig7" => run_fig7,
    "fig8" => run_fig8,
    "fig9" => run_fig9)

function main(args)
    todo = isempty(args) ? ALL : args
    println("Tran 2015 static — BEM (Kirchhoff / Wang FSDT + membrane VK)")
    println("  n_el=$NEL  n_int=$NINT  (parametric n_el=$NEL_P n_int=$NINT_P)")
    println("  cases: $(join(todo, ", "))")
    t0 = time()
    for name in todo
        haskey(JOBS, name) || (println("skip unknown $name"); continue)
        t1 = time()
        try
            JOBS[name]()
        catch e
            @error "case $name failed" exception = (e, catch_backtrace())
        end
        @printf("  [%s] %.1f s\n", name, time() - t1)
    end
    @printf("\nTotal %.1f s   output %s\n", time() - t0, OUT)
end

main(ARGS)
