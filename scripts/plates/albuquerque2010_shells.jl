# Albuquerque & Aliabadi, CMAME 199 (2010) 2663–2668
# Symmetric laminated composite shallow shells (Kirchhoff + RIM).
# We use Wang FSDT + Lekhnitskii membrane + DIBEM (a/h=100 ⇒ shear small).
# Table 2: uniform q, ATPS f = r²log r; Table 3: sinusoidal q.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

const E1, E2, ν12 = 25.0, 1.0, 0.25
const G12 = 0.5 * E2
const A_LEN, H, Q0 = 1.0, 0.01, 1.0

# Paper Table 2 / 3 (Reddy [21]), ŵ = 10³ w E₂ h³ / (q₀ a⁴)
const TABLE2_REDDY = Dict(
    Inf => 6.8331, 100.0 => 6.7772, 50.0 => 6.6148, 20.0 => 5.6618,
    10.0 => 3.7208, 5.0 => 1.5358, 2.0 => 0.2844, 1.0 => 0.0715)
const TABLE3_REDDY = Dict(
    Inf => 4.3368, 100.0 => 4.3021, 50.0 => 4.2015, 20.0 => 3.6104,
    10.0 => 2.4030, 5.0 => 1.0279, 2.0 => 0.2054, 1.0 => 0.0532)
const TABLE2_MESH3 = Dict(
    Inf => 6.7991, 100.0 => 6.7437, 50.0 => 6.5824, 20.0 => 5.6352,
    10.0 => 3.7044, 5.0 => 1.5290, 2.0 => 0.2833, 1.0 => 0.0712)

ndw(w) = 1e3 * w * E2 * H^3 / (Q0 * A_LEN^4)

function navier_sin(x, y; a, q, κ1, κ2, A, D, As)
    α, β = π / a, π / a
    A11, A22, A12, A66 = A[1, 1], A[2, 2], A[1, 2], A[3, 3]
    D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
    A44, A55 = As[2, 2], As[1, 1]
    K = zeros(5, 5)
    K[1, 1] = A11 * α^2 + A66 * β^2
    K[1, 2] = (A12 + A66) * α * β
    K[1, 3] = -(A11 * κ1 + A12 * κ2) * α
    K[2, 2] = A22 * β^2 + A66 * α^2
    K[2, 3] = -(A12 * κ1 + A22 * κ2) * β
    K[3, 3] = A55 * α^2 + A44 * β^2 +
              (A11 * κ1 + A12 * κ2) * κ1 + (A12 * κ1 + A22 * κ2) * κ2
    K[3, 4] = A55 * α
    K[3, 5] = A44 * β
    K[4, 4] = D11 * α^2 + D66 * β^2 + A55
    K[4, 5] = (D12 + D66) * α * β
    K[5, 5] = D22 * β^2 + D66 * α^2 + A44
    K[2, 1] = K[1, 2]; K[3, 1] = K[1, 3]; K[3, 2] = K[2, 3]
    K[4, 3] = K[3, 4]; K[5, 3] = K[3, 5]; K[5, 4] = K[4, 5]
    W = (K \ [0.0, 0.0, q, 0.0, 0.0])[3]
    return W * sin(α * x) * sin(β * y)
end

function set_qpts!(shell, qpts)
    mesh = shell.plate
    q = shell.Mw * qpts
    if mesh isa BEMdata
        set_cache!(mesh; fsdt_q=q, q=q)
    else
        mesh.q = q
    end
    return q
end

function run_one(; R_over_a, n_el, n_int, load=:uniform, npg=6, nsub=4)
    κ = isfinite(R_over_a) && R_over_a > 0 ? 1 / (R_over_a * A_LEN) : 0.0
    plies = [(E1, E2, ν12, G12, θ, H / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
    qc = load === :uniform ? Q0 : 0.0
    props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=qc, ρ=1.0)
    A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
    As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
    mesh = build_square_fsdt(; a=A_LEN, n_el=n_el, bc="SSSS", props=props,
        n_internal=n_int)
    geom = κ == 0 ? FlatShell() : SphericalShell(1 / κ)
    shell = LaminatedShell(mesh, A, geom; mem_bc=:navier_ss)
    assemble_laminated_shell!(shell; npg=npg, nsub=nsub, rbf=PHS(2; poly_deg=1),
        rbf_grad=PHS(3; poly_deg=1))
    if load === :sin
        pts = [BEM.Plate._plate_nodes(mesh); BEM.Plate._plate_internal(mesh)]
        qpts = [Q0 * sin(π * p[1] / A_LEN) * sin(π * p[2] / A_LEN) for p in pts]
        set_qpts!(shell, qpts)
    end
    solve_laminated_shell!(shell)
    w = fsdt_w_int(shell.plate, 1)
    if load === :sin
        wser = navier_sin(A_LEN / 2, A_LEN / 2; a=A_LEN, q=Q0, κ1=κ, κ2=κ,
            A=A, D=D, As=As)
    else
        wser = navier_ss_laminate_shell(A_LEN / 2, A_LEN / 2; a=A_LEN, q=Q0,
            κ1=κ, κ2=κ, A=A, D=D, As=As).w
    end
    return (w=w, ŵ=ndw(w), wser=wser, ŵser=ndw(wser))
end

function print_table(title, Reddy, rows)
    println("\n## ", title)
    @printf("  %8s %10s %10s %10s %8s\n", "R/a", "ŵ BEM", "ŵ Navier", "ŵ Reddy", "e%")
    for (Ra, r) in rows
        gold = Reddy[Ra]
        e = 100 * abs(r.ŵ - gold) / gold
        Ra_s = isinf(Ra) ? "Inf" : string(Ra)
        @printf("  %8s %10.4f %10.4f %10.4f %7.2f\n", Ra_s, r.ŵ, r.ŵser, gold, e)
    end
end

println("="^72)
println(" Albuquerque & Aliabadi 2010  laminated shallow shells")
println(" FSDT+DIBEM  PHS2 (r²log r)  a/h=100  [0/90]s  Reddy SS-1")
println("="^72)

ratios = [Inf, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]

println("\n# Table 2  uniform q  mesh 1 (12 quad BE, 9 internals)")
t2m1 = Tuple{Float64,NamedTuple}[]
for Ra in ratios
    @printf("  assembling R/a=%s ...\n", isinf(Ra) ? "Inf" : string(Ra))
    r = run_one(; R_over_a=Ra, n_el=3, n_int=9, load=:uniform)
    push!(t2m1, (Ra, r))
end
print_table("Table 2 mesh 1  uniform", TABLE2_REDDY, t2m1)

println("\n# Table 2  uniform q  mesh 2 (20 quad BE, 25 internals)")
t2m2 = Tuple{Float64,NamedTuple}[]
for Ra in ratios
    @printf("  assembling R/a=%s ...\n", isinf(Ra) ? "Inf" : string(Ra))
    r = run_one(; R_over_a=Ra, n_el=5, n_int=25, load=:uniform)
    push!(t2m2, (Ra, r))
end
print_table("Table 2 mesh 2  uniform", TABLE2_REDDY, t2m2)

println("\n# Table 3  sinusoidal q  mesh 2")
t3 = Tuple{Float64,NamedTuple}[]
for Ra in (Inf, 10.0, 5.0, 1.0)
    @printf("  assembling R/a=%s ...\n", isinf(Ra) ? "Inf" : string(Ra))
    r = run_one(; R_over_a=Ra, n_el=5, n_int=25, load=:sin)
    push!(t3, (Ra, r))
end
print_table("Table 3 mesh 2  sinusoidal", TABLE3_REDDY, t3)

if "--fine" in ARGS
    println("\n# Table 2  uniform  n_el=4, 81 internals (Useche 9.6.1 cloud)")
    t2f = Tuple{Float64,NamedTuple}[]
    for Ra in (Inf, 10.0, 5.0, 1.0)
        @printf("  assembling R/a=%s ...\n", isinf(Ra) ? "Inf" : string(Ra))
        r = run_one(; R_over_a=Ra, n_el=4, n_int=81, load=:uniform, npg=8, nsub=6)
        push!(t2f, (Ra, r))
    end
    print_table("Table 2  16 BE + 81 DIBEM", TABLE2_REDDY, t2f)
end

# Last run (PHS2 r²log r, FSDT+DIBEM vs Reddy [21] / paper Table 2 mesh 3):
# mesh 1 (12 BE, 9 int)  Inf 0.6%   R/a=10  85%   R/a=1  93%  (too stiff)
# mesh 2 (20 BE, 25 int) Inf 3.6%   R/a=10  53%   R/a=1  67%
# 16 BE + 81 int         Inf 0.6%   R/a=10   9%   R/a=1  17%
# Paper Kirchhoff+RIM mesh 3 stays <1% down to R/a=1. Our coupling needs
# the 81-centre cloud; coarse DIBEM misses u,v relief (same as Useche 9.6.1).
