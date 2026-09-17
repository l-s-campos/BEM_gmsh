# Recipes

Copy-paste workflows. These match the CI families in `test/`. Activate first:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using BEM
```

---

## 1. Steady Laplace on the unit square

```julia
dad = format2d(quadrado(ndiv=20, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))  # T = x
assemble!(dad, 20)
solve(dad)
@show rel_error(dad)
```

H-matrix path: `assemble!(dad; method=:hmatrix)`.

2-D Laplace on the GPU (KernelAbstractions; `using CUDA` first):

```julia
using CUDA
assemble!(dad; method=:gpu, T=Float32)
DIBEM(dad; method=:gpu)              # optional domain mass (default Float64)
solve(dad)
gpu_float_support()
```

---

## 2. Elasticity patch

```julia
dad = format2d(quadrado_elasticity(ndiv=12, show=false), Elasticity(1.0, 0.3, 1.0))
ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
apply_analytical_bc!(dad, ana)
assemble!(dad, 12)
solve(dad)
@show rel_error(dad)
```

---

## 2b. Constant-cell plasticity (thick cylinder)

```julia
include(datadir("elastico", "iso", "pressurized_tube.jl"))
props = Elasticity(2e5, 0.3, 1.0; plane_strain=true)
dad = format2d(mesh_pressurized_tube(; ndiv=8, nome="cyl_pl"), props)
apply_radius_pressure!(dad, 120.0; R=50.0, tol=4.0)
assemble!(dad, 12)
solve_elastoplastic!(dad, VonMises(σY=240.0); nsteps=6)
```

---

## 3. DIBEM mass + wave (Houbolt)

```julia
dad = format2d(quadrado(ndiv=16, show=false), Laplace(1.0); pontointerno=true)
assemble!(dad, 12)
dibem!(dad; rbf=PHS(3; poly_deg=1))   # dad.cache.M
solve_Houbolt(dad, 0.01, 1.0)
```

---

## 4. Local BEM Poisson

```julia
dad = format2d(quadrado(ndiv=16, show=false), Laplace(1.0); pontointerno=true)
for i in 1:dad.n
    dad.BC[i] = 0
    dad.BV[i] = dad.Nodes[i][1]^2 + dad.Nodes[i][2]^2   # ∇²u = 4
end
solve_local_bem!(dad, 4.0)
```

---

## 5. SBM Laplace

```julia
dad = format2d(quadrado(ndiv=12, show=false), Laplace(1.0))
apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
d = solve_sbm_laplace(dad)
@show sbm_rel_error(d, ana_laplace_linear(; direction=SA[1.0, 0.0]))
```

---

## 6. Modal transient (MMM)

```julia
# after assemble! + dibem! and free-DOF setup:
U, t, basis = solve_mmm!(dad, 0.01, 2.0; nmodes=12)
@show basis.ω[1:min(4, end)]
```

Demo: `scripts/transient/mmm_membrane_demo.jl`.

---

## 7. Dual BEM centre crack

```julia
using BEM.Crack
dad = build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, σ=1.0, E=3000.0, ν=0.2)
assemble_dual!(dad; npg=8, threaded=false)
solve_dual!(dad; threaded=false)
KI, KII = sif_cod_dual(dad, dad.tip_nodes[1])
```

Isotropic Reissner plate (Useche 10.5.1), same Portela split. Unsymmetric
Hsu–Hwu (`UnsymFSDTProps`) uses EABE 156 `T*` complete solutions on face B.
`assemble_fsdt!` calls Dual when the mesh has twins.

```julia
using BEM.Plate
props = FSDTProps(; E=2.1e5, ν=0.3, h=0.5)
mesh = build_rect_fsdt_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
    ndiv_b=8, ndiv_h=8, ndiv_crack=16)
assemble_fsdt_dual!(mesh; npg=10, nsub=8)   # or assemble_fsdt!(mesh)
solve_fsdt!(mesh)
K1b, K2b, K3b, _, _, _ = sif_ctod_fsdt(mesh; tip=:right)
F = K1b / sqrt(π * 0.2)                    # Table 10.1, Mo=1

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props5 = laminate_unsym_props(plies; Ks=5/6, G13=1e6, G23=5e5, nθ=8)
mesh5 = build_rect_fsdt_crack(; W=1.0, H=2.0, a=0.2, props=props5, Mo=1.0)
assemble_fsdt_dual!(mesh5)                 # Hsu T* HBIE on face B
solve_fsdt!(mesh5)
```

---

## 8. Contact (Cattaneo / Hertz)

```julia
using BEM.Contact
par = loyola_cattaneo_params()
x = collect(range(-1.5par.a, 1.5par.a; length=201))
p = cattaneo_pressure(x, par.a, par.p0)
```

---

## 9. Lubrication pad (Guiggiani 2020, Laplace + DIBEM)

```julia
film = film_h2(; a=2.0, hi=2.0)
msh = mesh_guiggiani_pad(show=false)
dad = format2d(msh, Laplace(1.0); pontointerno=false)
internal_grid!(dad, 13, 9; d_min=0.02, layout=:cell)
assemble!(dad; npg=12)
solve_reynolds_dibem!(dad, film)
p_nd = reynolds_pressure.(dad.T, Ref(film))   # p ho²/(μ U L)
```

Figs. 2–3 (films + infinite bearing) and Fig. 5 (finite pad):
`scripts/laplace/guiggiani_lubrication.jl`.

Mass-conserving cavitation (Profito 2015 §4.1, Elrod–Adams `p–θ`):

```julia
c = profito_single_slider()
p, θ = solve_elrod_1d(c.h, c.x[2]-c.x[1]; U=c.U, rheo=c.rheo,
    pleft=c.pleft, pright=c.pright, opt=c.opt)
```

`scripts/laplace/profito_cavitation.jl`.

---

## 10. Topology (Pacheco inverted-V)

```julia
using BEM.Topology
d = pacheco_inverted_v(; ne=8, nint=8, degree=1)
dad = bemdata_from_loops(d)
assemble!(dad; npg=8, threaded=false)
solve(dad)
@show thermal_conductance(dad)
```

Demos: `scripts/topology/topology_compare.jl`,
`scripts/topology/dibem_simp_compare.jl` (heat),
`scripts/topology/dibem_simp_elasticity.jl` (Coelho plane-stress),
`scripts/topology/dibem_simp_3d.jl` (cube heat / box cantilever).

```julia
using BEM.Topology
d = coelho_cantilever(; ne=8, nint=10, degree=1)
opt = DibemSimpOptions(; volfrac=0.35, n_simp=8, cut=true, pacheco=true, verbose=false)
d, dad, ρ, hist = solve_dibem_simp!(d, opt)
@show n_holes(d) design_area(d) elastic_compliance(dad)
```

3-D density on a fixed mesh (no iso-cut / Pacheco):

```julia
using BEM.Topology
dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
assemble!(dad; npg=8, threaded=false)
opt = DibemSimpOptions(; volfrac=0.45, n_simp=4, cut=true, pacheco=false, ngrid=21)
dad, ρ, hist = solve_dibem_simp!(dad, opt)
export_vtk_density(dad, ρ, "heat_cube_simp.vtk")
has_cache(dad, :simp_iso) && export_vtk_isosurface(dad.simp_iso, "heat_cube_iso.vtk")
# closed interior iso-components → cavities, BEM rebuilt (bemdata_from_iso)
```

---

## 11. Thin plate Navier `w_max`

```julia
using BEM, BEM.Plate
props = ThinPlateProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0)
msh = quadrado_plate(; a=1.0, ndiv=7, ordem=2, bc="SSSS")
dad = formatdata(msh, props; tipo=2, pontointerno=false)
set_internal_nodes!(dad, [SVector(0.5, 0.5)])
prepare_plate!(dad; corner_bc='F')
assemble!(dad); solve(dad)
@show plate_w_int(dad, 1)
```

Orthotropic / laminated Kirchhoff uses the full ``D_{ij}`` (Lekhnitskii ``μ``), not a smeared isotropic ``D``:

```julia
using BEM.Plate
props = aniso_thin_plate_props(; D11=17.3, D22=1.16, D12=0.35, D66=0.05, q_c=1e4, h=0.01)
dad = build_square_plate(; a=1.0, n_el=6, bc="SSSS", props=props, corner_bc='F')
assemble!(dad); solve(dad)
@show plate_w_int(dad, 1)
```

---

## 12. Unsymmetric FSDT CBIE vs HBIE (Useche 8.3.1)

Same `[0/90]` SS square, Hsu–Hwu 5×5 (EABE 156 `T*`). CBIE is the
displacement BIE. HBIE is a single traction BIE on every boundary node;
interior `w` is Somigliana after solve. Interior moments: constitutive
of source-derivatives of that CBIE (complete solutions), not `nξ·∇P`.

```julia
using BEM.Plate
plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5/6, G13=1e6, G23=5e5, q_c=1.0, nθ=8)
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)
mesh = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=props, n_internal=9)
assemble_fsdt!(mesh; singular=:guiggiani); dibem_fsdt!(mesh); solve_fsdt!(mesh)
wc_c = fsdt_w_int(mesh, 1)
meshH = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=props, n_internal=9)
assemble_unsym_fsdt_hbie!(meshH); solve_fsdt!(meshH)
wc_h = fsdt_w_int(meshH, 1)
```

`scripts/plates/usech_831_cbie_hbie.jl` prints Navier / CBIE / HBIE.
