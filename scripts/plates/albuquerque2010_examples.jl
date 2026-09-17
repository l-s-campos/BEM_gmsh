# Albuquerque & Aliabadi, CMAME 199 (2010) 2663–2668 — three numerical examples.
# 60 quadratic BE (15 per edge) and formatdata cell-centroid internals (15×15 = 225).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

const N_EL = 15
const N_INT = 225   # n_div = 15 cell centres of the same partition
const NPG, NSUB = 8, 6
const RBF = PHS(2; poly_deg=1)
const RBFG = PHS(3; poly_deg=1)

function make_shell(; a, h, R, q, plies, bc="SSSS", mem_bc=:navier_ss,
        G13=nothing, G23=nothing)
    G12 = plies[1][4]
    g13 = G13 === nothing ? G12 : G13
    g23 = G23 === nothing ? 0.2 * plies[1][2] : G23
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=g13, G23=g23, q_c=q, ρ=1.0)
    A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=g13, G23=g23)
    mesh = build_square_fsdt(; a=a, n_el=N_EL, bc=bc, props=props, n_internal=N_INT)
    geom = !isfinite(R) || R == 0 ? FlatShell() : SphericalShell(R)
    shell = LaminatedShell(mesh, A, geom; mem_bc=mem_bc)
    assemble_laminated_shell!(shell; npg=NPG, nsub=NSUB, rbf=RBF, rbf_grad=RBFG)
    return shell, A, D, AT, mesh
end

function set_qpts!(shell, qpts)
    q = shell.Mw * qpts
    mesh = shell.plate
    mesh isa BEMdata ? set_cache!(mesh; fsdt_q=q, q=q) : (mesh.q = q)
    return q
end

function navier_sin_w(a, q, κ, A, D, As)
    α = π / a
    A11, A22, A12, A66 = A[1, 1], A[2, 2], A[1, 2], A[3, 3]
    D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
    A44, A55 = As[2, 2], As[1, 1]
    K = zeros(5, 5)
    K[1, 1] = A11 * α^2 + A66 * α^2
    K[1, 2] = (A12 + A66) * α * α
    K[1, 3] = -(A11 * κ + A12 * κ) * α
    K[2, 2] = A22 * α^2 + A66 * α^2
    K[2, 3] = -(A12 * κ + A22 * κ) * α
    K[3, 3] = A55 * α^2 + A44 * α^2 + (A11 * κ + A12 * κ) * κ + (A12 * κ + A22 * κ) * κ
    K[3, 4] = A55 * α
    K[3, 5] = A44 * α
    K[4, 4] = D11 * α^2 + D66 * α^2 + A55
    K[4, 5] = (D12 + D66) * α * α
    K[5, 5] = D22 * α^2 + D66 * α^2 + A44
    K[2, 1] = K[1, 2]; K[3, 1] = K[1, 3]; K[3, 2] = K[2, 3]
    K[4, 3] = K[3, 4]; K[5, 3] = K[3, 5]; K[5, 4] = K[4, 5]
    W = (K \ [0.0, 0.0, q, 0.0, 0.0])[3]
    return W
end

function print_cl(shell; nd=x -> x, label="w")
    cl = shell_centreline(shell; dir=:x, method=:rbf)
    @printf("  %8s %12s\n", "x/a", label)
    xs = [p[1] for p in BEM.Plate._plate_nodes(shell.plate)]
    a = maximum(xs) - minimum(xs)
    for i in eachindex(cl.s)
        @printf("  %8.3f %12.5e\n", cl.s[i] / a, nd(cl.w[i]))
    end
end

# ---------------------------------------------------------------------------
println("="^72)
println(" Albuquerque & Aliabadi 2010  — 60 BE (15/side) + cell-centre internals")
mesh0 = build_square_fsdt(; a=1.0, n_el=N_EL, n_internal=N_INT)
@printf("  n_el/edge=%d  n_BE=%d  n_nodes=%d  n_internal=%d\n",
    N_EL, length(mesh0.elements), mesh0.n, mesh0.ni)
println("="^72)

# 3.1  Square spherical cross-ply [0/90]s
# ---------------------------------------------------------------------------
println("\n## 3.1  SS spherical [0/90]s  a/h=100  uniform q")
E1, E2, ν12 = 25.0, 1.0, 0.25
G12 = 0.5 * E2
a, h, q0 = 1.0, 0.01, 1.0
ndw(w) = 1e3 * w * E2 * h^3 / (q0 * a^4)
TABLE2 = Dict(Inf => 6.8331, 100.0 => 6.7772, 50.0 => 6.6148, 20.0 => 5.6618,
    10.0 => 3.7208, 5.0 => 1.5358, 2.0 => 0.2844, 1.0 => 0.0715)
TABLE3 = Dict(Inf => 4.3368, 10.0 => 2.4030, 5.0 => 1.0279, 1.0 => 0.0532)
plies01 = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
@printf("  %8s %10s %10s %8s\n", "R/a", "ŵ BEM", "ŵ Reddy", "e%")
for Ra in (Inf, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0)
    R = isfinite(Ra) ? Ra * a : Inf
    shell, A, D, AT, mesh = make_shell(; a=a, h=h, R=R, q=q0, plies=plies01)
    solve_laminated_shell!(shell)
    ŵ = ndw(fsdt_w_int(mesh, 1))
    gold = TABLE2[Ra]
    @printf("  %8s %10.4f %10.4f %7.2f\n",
        isinf(Ra) ? "Inf" : string(Int(Ra)), ŵ, gold, 100 * abs(ŵ - gold) / gold)
end

println("\n## 3.1  sinusoidal q")
@printf("  %8s %10s %10s %8s\n", "R/a", "ŵ BEM", "ŵ Reddy", "e%")
for Ra in (Inf, 10.0, 5.0, 1.0)
    R = isfinite(Ra) ? Ra * a : Inf
    κ = isfinite(R) ? 1 / R : 0.0
    shell, A, D, AT, mesh = make_shell(; a=a, h=h, R=R, q=0.0, plies=plies01)
    pts = [BEM.Plate._plate_nodes(mesh); BEM.Plate._plate_internal(mesh)]
    set_qpts!(shell, [q0 * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts])
    solve_laminated_shell!(shell)
    ŵ = ndw(fsdt_w_int(mesh, 1))
    gold = TABLE3[Ra]
    @printf("  %8s %10.4f %10.4f %7.2f\n",
        isinf(Ra) ? "Inf" : string(Int(Ra)), ŵ, gold, 100 * abs(ŵ - gold) / gold)
end

# 3.2  Orthotropic single ply
# ---------------------------------------------------------------------------
println("\n## 3.2  Orthotropic spherical  a=0.254 m  h=0.0127 m  q=2.07 MPa")
a2, h2, q2 = 0.254, 0.0127, 2.07e6
E2o = 6.895e9
E1o = 2 * E2o
ν12o = 0.3
G12o = E2o / (2 * (1 - ν12o))
plies2 = [(E1o, E2o, ν12o, G12o, 0.0, h2)]
w0C, w0S = 8.423e-3, 27.01e-3
for (bc, mem, w0, tag) in (("CCCC", :clamped, w0C, "clamped"),
        ("SSSS", :navier_ss, w0S, "SS"))
    println("  --- ", tag, "  w0=", w0, " m")
    @printf("  %8s %12s %10s\n", "R/a", "w_c (m)", "w/w0")
    for Ra in (10.0, 5.0)
        shell, _, _, _, mesh = make_shell(; a=a2, h=h2, R=Ra * a2, q=q2,
            plies=plies2, bc=bc, mem_bc=mem, G13=G12o, G23=G12o)
        solve_laminated_shell!(shell)
        wc = fsdt_w_int(mesh, 1)
        @printf("  %8.0f %12.5e %10.4f\n", Ra, wc, wc / w0)
        print_cl(shell; nd=w -> w / w0, label="w/w0")
    end
end

# 3.3  Angle-ply [45/-45]s
# ---------------------------------------------------------------------------
println("\n## 3.3  SS spherical [45/-45]s  R/a=5  uniform q")
plies45 = [(E1, E2, ν12, G12, θ, h / 4) for θ in (45.0, -45.0, -45.0, 45.0)]
shell3, _, _, _, mesh3 = make_shell(; a=a, h=h, R=5a, q=q0, plies=plies45)
solve_laminated_shell!(shell3)
wc3 = fsdt_w_int(mesh3, 1)
@printf("  w_c=%.6e  ŵ=%.4f\n", wc3, ndw(wc3))
print_cl(shell3; nd=ndw, label="ŵ")
println("\ndone")
