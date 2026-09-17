# Crack (dual BEM)

`using BEM.Crack`. Displacement BIE + traction BIE on coincident faces.
SIFs from crack-opening displacement. Isotropic (`Elasticity`, Kelvin) or
anisotropic (`AnisotropicElasticity`, Lekhnitskii). XBEM enrichment is
Williams or Sih–Paris–Irwin. Growth uses the Erdogan–Sih MTS criterion
(isotropic) or Ke’s anisotropic hoop-stress MTS (`LekhnitskiiParams` in
the ligament frame).

```@docs
BEM.Crack
BEM.Crack.CRACK_BC
BEM.Crack.prepare_crack!
BEM.Crack.assemble_dual!
BEM.Crack.solve_dual!
BEM.Crack.sif_cod_dual
BEM.Crack.analytical_KI_center_crack
BEM.Crack.sih_M_local
BEM.Crack.assemble_xbem!
BEM.Crack.solve_xbem!
BEM.Crack.max_tens_circ
BEM.Crack.analytical_sif_inclined_center
BEM.Crack.extend_dual_crack_tip!
BEM.Crack.propagate_dual_mts!
```
