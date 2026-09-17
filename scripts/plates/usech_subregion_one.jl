# MATLAB Static_Cracked_Laminated_Plate_Subregion / prueba.inp — region 1 only.
# Cantilever [0,1]×[0,1], E=70e3, ν=0, h=1.5, q=-1, clamp x=0.
# MATLAB: 1 discontinuous quadratic element per edge.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, DelimitedFiles
using BEM.Plate

E, ν, h, q0 = 70e3, 0.0, 1.5, -1.0
Lx, Ly = 1.0, 1.0
D = E * h^3 / (12 * (1 - ν^2))
κGh = shear_stiffness(FSDTProps(; E=E, ν=ν, h=h))
# FSDT beam, uniform pressure, width Ly (line load q0*Ly), tip x=Lx
w_bend = q0 * Ly * Lx^4 / (8 * D)
w_shear = q0 * Ly * Lx^2 / (2 * κGh)
w_beam = w_bend + w_shear

println("="^72)
println(" Useche MATLAB subregion prueba.inp — region 1 only (Julia FSDT)")
println("="^72)
@printf("  [0,1]×[0,1]  E=%.0f  ν=%.0f  h=%.1f  q=%.1f  clamp x=0\n", E, ν, h, q0)
@printf("  D=%.4f  κGh=%.1f  λ=√10/h=%.4f\n", D, κGh, sqrt(10) / h)
@printf("  FSDT beam w_tip = %.6e  (bend %.6e, shear %.6e)\n", w_beam, w_bend, w_shear)

function run_one(n_el)
    props = FSDTProps(; E=E, ν=ν, h=h, q_c=q0)
    mesh = build_rect_fsdt(; Lx=Lx, Ly=Ly, n_el=n_el, bc="FFFC", props=props,
        n_internal=max(1, n_el * n_el), p=2)
    assemble_fsdt!(mesh; npg=10, nsub=4)
    dibem_fsdt!(mesh; npg=8)
    solve_fsdt!(mesh)
    itip = argmin(i -> begin
            p = mesh.nodes[i]
            abs(p[1] - Lx) > 0.2 ? Inf : abs(p[2] - Ly / 2)
        end, eachindex(mesh.nodes))
    wt = fsdt_w(mesh, itip)
    wint = fsdt_w_int(mesh, 1)
    return mesh, itip, wt, wint
end

for n_el in (1, 4)
    mesh, itip, wt, wint = run_one(n_el)
    pt = mesh.nodes[itip]
    @printf("\n  n_el=%d  n_node=%d  n_int=%d\n",
        n_el, length(mesh.nodes), length(mesh.internal))
    @printf("  tip (%.3f, %.3f)  w=%.6e  vs beam %.1f %%\n",
        pt[1], pt[2], wt, 100 * abs(wt - w_beam) / abs(w_beam))
    @printf("  interior nearest centre  w=%.6e\n", wint)
    println("  free-end w(y):")
    for i in eachindex(mesh.nodes)
        p = mesh.nodes[i]
        abs(p[1] - Lx) < 0.2 || continue
        @printf("    y=%.3f  w=%.4e\n", p[2], fsdt_w(mesh, i))
    end
end

oct = joinpath(@__DIR__, "usech_subregion_one_octave.csv")
if isfile(oct)
    A = readdlm(oct, ',')
    println("\n  Octave region-1 collocation w:")
    wR = Float64[]
    for k in 1:size(A, 1)
        x, y, wx, wy, w = A[k, :]
        @printf("    (%.3f, %.3f)  w=%.4e\n", x, y, w)
        abs(x - 1) < 1e-9 && push!(wR, w)
    end
    if !isempty(wR)
        @printf("  Octave mean w(x=1)=%.6e  Julia n_el=1 tip vs Octave %.1f %%\n",
            sum(wR) / length(wR),
            100 * abs(run_one(1)[3] - sum(wR) / length(wR)) / abs(sum(wR) / length(wR)))
    end
end
println("Done.")
