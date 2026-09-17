# Center-cracked finite plate — Gmsh + format2d + dual BEM on BEMdata
# Prefer `dual_elasticity_problem` / `build_center_crack_mesh` from `BEM.Crack`.

"""
    mesh_center_crack(; kwargs...) -> path

Deprecated wrapper: use [`BEM.Crack.mesh_center_crack`](@ref).
"""
mesh_center_crack(; kwargs...) = BEM.Crack.mesh_center_crack(; kwargs...)

"""
    solve_center_crack_dual(; kwargs...) -> (dad, KI_L, KII_L, KI_R, KII_R, KI_ana)

Gmsh → `format2d` (BC type 5) → dual BEM → COD SIFs.
"""
function solve_center_crack_dual(; W=5.0, H=10.0, a=1.0,
    n_bottom=6, n_right=12, n_top=6, n_left=12, n_crack=12,
    E=3000.0, ν=0.2, σ=1.0, plane_strain=true, npg=12, ordem=2)

    dad = build_center_crack_mesh(; W, H, a, n_bottom, n_right, n_top, n_left,
        n_crack, E, ν, σ, plane_strain, ordem)
    assemble_dual!(dad; npg=npg, threaded=false)
    solve_dual!(dad; threaded=false)

    tL, tR = dad.tip_nodes[1], dad.tip_nodes[2]
    KI_L, KII_L = sif_cod_dual(dad, tL; sample=2)
    KI_R, KII_R = sif_cod_dual(dad, tR; sample=2)
    KI_ana = analytical_KI_center_crack(σ, a; W=W)
    return dad, KI_L, KII_L, KI_R, KII_R, KI_ana
end
