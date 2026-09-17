# Useche 8.6.1 / MATLAB Ejemplo_Laminado.m
# Cantilever [0/90/90/0] 10×5, h=0.1, end shear Vz=-100.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

cantilever_w_tip(q, L, D11, A55) = q * L^3 / (3 * D11) + q * L / A55

E1, E2, ν12 = 4e6, 2e6, 0.25
G12, G13, G23 = 1e6, 1e6, 5e5
Lx, Ly, h = 10.0, 5.0, 0.1
Vz = -100.0
plies = [(E1, E2, ν12, G12, θ, 0.025) for θ in (0.0, 90.0, 90.0, 0.0)]
props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G13, G23=G23, ρ=4000.0, q_c=0.0, nθ=10)
D11, D22, D12, D66 = props.D[1, 1], props.D[2, 2], props.D[1, 2], props.D[3, 3]
A44, A55 = props.AT[1, 1], props.AT[2, 2]
w_beam = cantilever_w_tip(abs(Vz), Lx, D11, A55)
w_bend = abs(Vz) * Lx^3 / (3 * D11)
w_shear = abs(Vz) * Lx / A55

println("="^72)
println(" Useche 8.6.1  cantilever [0/90]s end shear  (Wang FSDT BEM)")
println(" MATLAB Dynamic_Composite_Plate/test/Ejemplo_Laminado.m")
println("="^72)
@printf("  10×5×0.1  [0/90/90/0]  Vz=%.0f  (force/length)\n", Vz)
@printf("  D11=%.4f  D22=%.4f  D12=%.4f  D66=%.4f\n", D11, D22, D12, D66)
@printf("  A44=%.1f  A55=%.1f  (Octave AT=62500)\n", A44, A55)
@printf("  FSDT beam w_tip = %.6e  (bending %.6e, shear %.2f %%)\n",
    w_beam, w_bend, 100 * w_shear / w_beam)

# MATLAB: 8 els on L=10, 4 on L=5; clamp x=0; Vz on x=L.
# bc = bottom,right,top,left
mesh = build_rect_fsdt(; Lx=Lx, Ly=Ly, n_el=(8, 4, 8, 4), bc="FFFC",
    vn=(0.0, Vz, 0.0, 0.0), props=props, n_internal=4, p=2)
@printf("  mesh n_node=%d  n_int=%d  n_el=%d\n",
    length(mesh.nodes), length(mesh.internal), length(mesh.elements))
assemble_fsdt!(mesh; npg=10, nsub=8)
solve_fsdt!(mesh)

# free-end collocation nearest (L, b/2)
itip = argmin(i -> begin
        p = mesh.nodes[i]
        abs(p[1] - Lx) > 0.25 ? Inf : abs(p[2] - Ly / 2)
    end, eachindex(mesh.nodes))
w_tip = fsdt_w(mesh, itip)
pt = mesh.nodes[itip]
@printf("  BEM tip node (%.3f, %.3f)  w=%.6e  vs beam %.3f %%\n",
    pt[1], pt[2], w_tip, 100 * abs(abs(w_tip) - w_beam) / w_beam)

# ∫ Qn dΓ per edge (Qn is z-shear). No domain load ⇒ sum should be ~0.
wi = mesh.elem_weight
Qn_edge = zeros(4)
L_edge = zeros(4)
for el in mesh.elements
    e = el.Region
    for (a, ja) in enumerate(el.index)
        wJ = el.Jacobian[a] * wi[a]
        Qn_edge[e] += mesh.t[3ja] * wJ
        L_edge[e] += wJ
    end
end
@printf("  edge length ΣJw  bottom=%.3f right=%.3f top=%.3f left=%.3f\n",
    L_edge[1], L_edge[2], L_edge[3], L_edge[4])
@printf("  ∫Qn  bottom=%.2f right=%.2f top=%.2f left=%.2f  sum=%.2f  (Vz*Ly=%.1f)\n",
    Qn_edge[1], Qn_edge[2], Qn_edge[3], Qn_edge[4], sum(Qn_edge), Vz * Ly)

println("  tip-edge w(y):")
for i in eachindex(mesh.nodes)
    p = mesh.nodes[i]
    abs(p[1] - Lx) < 0.25 || continue
    @printf("    y=%.3f  w=%.4e\n", p[2], fsdt_w(mesh, i))
end

if !isempty(mesh.internal)
    println("  internals:")
    for k in eachindex(mesh.internal)
        p = mesh.internal[k]
        @printf("    (%.3f, %.3f)  w=%.6e\n", p[1], p[2], fsdt_w_int(mesh, k))
    end
end
# one refinement (MATLAB is 8×4)
mesh2 = build_rect_fsdt(; Lx=Lx, Ly=Ly, n_el=(16, 8, 16, 8), bc="FFFC",
    vn=(0.0, Vz, 0.0, 0.0), props=props, n_internal=4, p=2)
assemble_fsdt!(mesh2; npg=10, nsub=8)
solve_fsdt!(mesh2)
itip2 = argmin(i -> begin
        p = mesh2.nodes[i]
        abs(p[1] - Lx) > 0.15 ? Inf : abs(p[2] - Ly / 2)
    end, eachindex(mesh2.nodes))
w2 = fsdt_w(mesh2, itip2)
@printf("  refined n_el=(16,8,16,8) tip (%.3f, %.3f) w=%.6e  vs beam %.2f %%\n",
    mesh2.nodes[itip2][1], mesh2.nodes[itip2][2], w2,
    100 * abs(abs(w2) - w_beam) / w_beam)
println("  Fig 8.4  deflection / rotation along y=b/2 (internals + tip)")
rq = fsdt_resultants(mesh)
nb = length(mesh.nodes)
@printf("  %8s %12s %12s %12s\n", "x", "w", "ψx", "ψy")
keep = findall(i -> abs(rq.pts[i][2] - Ly / 2) < 0.4, eachindex(rq.pts))
ord = sort(keep; by=i -> rq.pts[i][1])
for i in ord
    p = rq.pts[i]
    (i > nb || abs(p[1] - Lx) < 0.3 || abs(p[1]) < 0.3) || continue
    @printf("  %8.3f %12.4e %12.4e %12.4e\n", p[1], rq.w[i], rq.ψx[i], rq.ψy[i])
end
println("  book: BEM <1% vs Reddy on 6–16 elements (Fig. 8.4); beam is 1-D.")
println("Done.")
