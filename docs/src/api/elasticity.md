# Elasticity

Kelvin kernels, 3D anisotropic (Ting–Lee), local ``(n,t)`` frame, thermoelasticity.

2-D GPU: `assemble!(dad; method=:gpu)` / `DIBEM(dad; method=:gpu)` (KernelAbstractions).

```@docs
Elasticity
AnisotropicElasticity
lekhnitskii_params
lekhnitskii_rotate
AnisotropicElasticity3D
aniso3d_isotropic
aniso3d_cubic
solve_local
solve_thermoelastic!
solve_heterogeneous!
ana_elasticity_patch
ana_aniso3d_patch
recover_strain_stress!
VonMises
solve_elastoplastic!
assemble_plastic_ops!
ana_thick_cylinder_plastic
apply_radius_pressure!
export_vtk
```

2-D constant-cell plasticity (initial stress, von Mises):

```julia
dad = format2d(mesh, Elasticity(E, ν, 1.0; plane_strain=true); pontointerno=true)
assemble!(dad)
mat = VonMises(σY=240.0, H′=0.0)
solve_elastoplastic!(dad, mat; nsteps=8)
# dad.stress, dad.plastic_strain  at cell centroids
```

Needs a Gmsh surface mesh (`format2d` stores `dad.cells`). Domain integral
`:cells` (default) or `:dibem` (RBF centres = cell centroids). Traction-free
inner radius: [`apply_radius_pressure!`](@ref). Analytic tube:
[`ana_thick_cylinder_plastic`](@ref).


3D anisotropy:

```julia
props = aniso3d_cubic(230e3, 135e3, 117e3)   # C11, C12, C44
dad = format3d(mesh, props; pontointerno=false)
assemble!(dad); solve(dad)
recover_strain_stress!(dad)
export_vtk(dad, "cube.vtk")
```

Linear triangles: `mesh_unit_cube(; recombine=false)` then `format3d` (collapsed quads).
3D DRM: `build_drm_matrices(dad)` after `assemble!` (particular solution ``R+R^3``).
