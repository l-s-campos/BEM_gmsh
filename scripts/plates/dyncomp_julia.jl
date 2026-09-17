# MATLAB Dynamic_Composite_Plate / test/EjemploDin_10.m (static) vs Julia DIBEM.
# SS square [0,1]^2, [0/90/90/0] Wang FSDT, q=1. MATLAB: 2 quadratic els/edge.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, DelimitedFiles
using BEM.Plate

println("="^72)
println(" Dynamic_Composite_Plate  EjemploDin_10 static  vs Julia DIBEM")
println("="^72)

E1, E2, ν12 = 4e6, 2e6, 0.25
G12, G13, G23 = 1e6, 1e6, 5e5
a, h, q, ρ = 1.0, 0.1, 1.0, 4000.0
plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=ρ, nθ=8)
gold = navier_ss_fsdt_MQ(a / 2, a / 2, props; a=a, q=q)
D11, D22, D12, D66 = props.D[1, 1], props.D[2, 2], props.D[1, 2], props.D[3, 3]
A44, A55 = props.AT[1, 1], props.AT[2, 2]

@printf("  SS [0,1]^2  [0/90/90/0]  h=%.3f  q=%.1f  ρ=%.0f\n", h, q, ρ)
@printf("  D11=%.6f  D22=%.6f  D12=%.6f  D66=%.6f\n", D11, D22, D12, D66)
@printf("  A44=%.1f  A55=%.1f\n", A44, A55)
@printf("  Navier  w_c=%.6e  Mx=%.4e  My=%.4e\n", gold.w, gold.Mx, gold.My)

function run_case(; n_el, n_internal=1, rbf=PHS(2; poly_deg=1), npg=8, nsub=6, method=:dibem)
    mesh = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props,
        n_internal=n_internal, p=2)
    assemble_fsdt!(mesh; npg=npg, nsub=nsub)
    if method === :dibem
        dibem_fsdt!(mesh; npg=npg, rbf=rbf)
    elseif method === :rim
        BEM.Plate._rim_q_wang!(mesh; npg=npg, nρ=8)
    else
        error("method")
    end
    solve_fsdt!(mesh)
    wc = fsdt_w_int(mesh, 1)
    wmaxb = maximum(abs(fsdt_w(mesh, i)) for i in eachindex(mesh.nodes))
    return mesh, wc, wmaxb
end

rows = Tuple{String,Int,Float64,Float64,Float64}[]
println("\n  Julia Wang FSDT + domain load")
@printf("  %-22s %4s %12s %10s %10s\n", "method", "n_el", "w_c", "vs Navier", "max|w|_Γ")

for (tag, n_el, method, rbf) in (
        ("DIBEM PHS2 r²ln r", 1, :dibem, PHS(2; poly_deg=1)),
        ("DIBEM PHS2 r²ln r", 2, :dibem, PHS(2; poly_deg=1)),
        ("DIBEM PHS3 r³", 2, :dibem, PHS(3; poly_deg=2)),
        ("Wang polar RIM Q", 2, :rim, PHS(2; poly_deg=1)),
        ("DIBEM PHS2 r²ln r", 4, :dibem, PHS(2; poly_deg=1)),
    )
    mesh, wc, wmaxb = run_case(; n_el=n_el, method=method, rbf=rbf)
    rel = 100 * abs(wc - gold.w) / abs(gold.w)
    @printf("  %-22s %4d %12.4e %9.2f %% %10.2e\n", tag, n_el, wc, rel, wmaxb)
    push!(rows, (tag, n_el, wc, rel, wmaxb))
    if n_el == 2 && method === :dibem && rbf isa typeof(PHS(2; poly_deg=1))
        @printf("    ||H||=%.4e  ||G||=%.4e  ||M||=%.4e  ||q||=%.4e\n",
            norm(mesh.H), norm(mesh.G), norm(mesh.M), norm(mesh.q))
        @printf("    n_node=%d  n_int=%d  centre (%.3f, %.3f)\n",
            length(mesh.nodes), length(mesh.internal),
            mesh.internal[1][1], mesh.internal[1][2])
    end
end

octfile = joinpath(@__DIR__, "dyncomp_octave.txt")
if isfile(octfile)
    kv = Dict{String,Float64}()
    for line in eachline(octfile)
        k, v = split(line, '=')
        kv[k] = parse(Float64, v)
    end
    wo = kv["w_center"]
    println("\n  Octave DomainTermsRIM (EjemploDin_10 static, n_el=2)")
    @printf("  w_c=%.6e  vs Navier %.2f %%  vs Julia DIBEM n_el=2 %.2f %%\n",
        wo, 100 * abs(wo - gold.w) / abs(gold.w),
        100 * abs(wo - rows[2][3]) / abs(rows[2][3]))
    @printf("  max|w|_Γ=%.3e  ||Q||=%.4e  ||H||=%.4e\n",
        kv["max_abs_w_boundary"], kv["normQ"], kv["normH"])
    @printf("  Octave D11=%.6f  (Julia %.6f)\n", kv["D11"], D11)
else
    println("\n  Octave results not yet in dyncomp_octave.txt")
end
println("Done.")
