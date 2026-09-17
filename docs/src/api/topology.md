# Topology optimization

`using BEM.Topology`. Loops → BEM; Pacheco motion; Amstutz / HJ level-set;
Portela (2012) dual-BEM shape design (elasticity + Laplace analogue).
**3-D** is density-only on a fixed surface mesh: DIBEM-SIMP / DT-ρ
([`heat_cube_3d`](@ref), [`cantilever_cube_3d`](@ref)); explicit Γ motion
stays 2-D.

```@docs
BEM.Topology
BEM.Topology.pacheco_inverted_v
BEM.Topology.coelho_cantilever
BEM.Topology.portela_plate_hole
BEM.Topology.portela_heat_hole
BEM.Topology.heat_cube_3d
BEM.Topology.cantilever_cube_3d
BEM.Topology.bemdata_from_loops
BEM.Topology.topological_derivative
BEM.Topology.thermal_conductance
BEM.Topology.elastic_compliance
BEM.Topology.isotropic_3d_DT
BEM.Topology.solve_topology!
BEM.Topology.move_boundary!
BEM.Topology.move_boundary_standin!
BEM.Topology.move_boundary_jump!
BEM.Topology.PachecoOptions
BEM.Topology.solve_dibem_simp!
BEM.Topology.solve_dt_density!
BEM.Topology.density_from_dt
BEM.Topology.DibemSimpOptions
BEM.Topology.PortelaOptions
BEM.Topology.solve_portela!
BEM.Topology.prepare_design_dual!
BEM.Topology.strain_energy_density
BEM.Topology.export_vtk_density
BEM.Topology.export_vtk_isosurface
BEM.Topology.marching_cubes
BEM.Topology.cut_density_3d!
BEM.Topology.density_grid_3d
BEM.Topology.bemdata_from_iso
BEM.Topology.extract_closed_cavities
```
