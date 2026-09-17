# Diagnose why Crisfield on LaminatedShell did not turn λ.
# Run: julia --project=. .scratch/shell_snap_diag.jl
using LinearAlgebra
using Printf
using StaticArrays
using BEM
using BEM.Plate

function setup(; κsign=+1, n_el=2, n_internal=4)
    E1, E2, ν12 = 25.0, 1.0, 0.25
    G12 = 0.5 * E2
    a, h, q, ρ = 1.0, 0.02, 1.0, 1.0
    rise = 2h
    R = a^2 / (8 * rise)
    plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=q, ρ=ρ)
    A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
    As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
    mesh = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props,
        n_internal=n_internal)
    geom = SphericalShell(κsign * R)
    shell = LaminatedShell(mesh, A, geom; mem_bc=:clamped)
    assemble_laminated_shell!(shell; npg=4, nsub=3)
    gold = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=κsign / R,
        κ2=κsign / R, A=A, D=D, As=As)
    return (; shell, a, h, R, q, A, D, As, gold, κsign, rise)
end

function onedof_qw(w; Kb, A11, κ, c)
    N = A11 * (κ * w + 0.5 * c * w^2)
    return Kb * w + N * κ + N * c * w
end

function diagnose(P; nsteps=16)
    shell = P.shell
    a, h, R = P.a, P.h, P.R
    κ = P.κsign / R
    println("=== κsign=", P.κsign, "  R=", R, "  rise=", P.rise, "  h=", h, " ===")
    println("Navier w=", P.gold.w, "  Nx=", P.gold.Nx, "  Ny=", P.gold.Ny)
    # 1-mode analog: q = Kb w + N κ + N w_xx, w_xx = -c w → + N c w
    D11 = P.D[1, 1]
    A11 = P.A[1, 1]
    c = (π / a)^2
    Kb = D11 * c^2
    println("1-mode Kb=", Kb, " Aκ²=", A11 * κ^2)
    ws = range(0, 4h; length=9)
    qs = [onedof_qw(w; Kb=Kb, A11=A11, κ=κ, c=c) for w in ws]
    println("1-mode q(w/h) [no u-relief]:")
    for (w, qv) in zip(ws, qs)
        @printf "  w/h=%5.2f  q=%12.4e\n" w / h qv
    end
    dq = diff(qs)
    imax = findfirst(<(0), dq)
    println("1-mode fold (dq<0) index: ", imax === nothing ? "none" : imax)

    sys = BEM.Plate._laminated_shell_system(shell)
    tknown = BEM.Plate._laminated_apply_bc!(sys, shell.BVm)
    H, G, q0 = sys.H, sys.G, sys.q
    Gt = G * tknown
    xlin = H \ (q0 .+ Gt)
    wlin = BEM.Plate._shell_w_center(shell, sys, xlin)
    println("linear BEM w(λ=1)=", wlin)

    res = solve_laminated_shell!(shell; large=true, nonlinear=:arclength,
        nsteps=nsteps, λ_max=0.0, maxiters=8, atol=1e-4)
    println("n=", length(res.λ), "  λmax=", maximum(res.λ),
        "  argmax=", argmax(res.λ), "  w_end=", res.w_center[end])
    dλ = diff(res.λ)
    println("λ: ", res.λ)
    println("w: ", res.w_center)
    println("dλ:", dλ)
    # N at centre from current shell fields
    nt = BEM.Plate._nt(shell.plate)
    n = BEM.Plate._n(shell.plate)
    ic = n + 1
    rr = BEM.Plate._rbf_resultants(shell)
    println("RBF Nxx(center)=", rr.Nx[ic], " Nyy=", rr.Ny[ic], " w=", rr.w[ic])
    turned = argmax(abs.(res.λ)) < length(res.λ) && res.λ[end] < maximum(res.λ) - 1e-12
    println("λ turned: ", turned)
    return res
end

println("==== +κ (test geometry) ====")
Pplus = setup(; κsign=+1)
diagnose(Pplus; nsteps=12)

println("\n==== −κ (snap-capable sign) ====")
Pminus = setup(; κsign=-1)
diagnose(Pminus; nsteps=12)
