# Cohesive-contact Dual BEM

Nonlinear cohesive crack / interface formulation on top of the dual BEM
([`DualMesh`](@ref)), following:

| Paper | Role |
|-------|------|
| **Cordeiro et al. (2024)** TAEM 130:104249 | DBEM + local cohesive stiffness; surface conditions (contact, softening, unload/reload, failure); PPR; DOF control |
| **Alfano & Sacco (2006)** IJNME 68:542–582 | Damage + friction on the damaged fraction of the REA |
| **Távara et al. (2013)** Comput Mech 51:535–551 | CZM in BEM / arc-length context |

## Surface conditions (Cordeiro Algorithm 1)

At each twin collocation pair, openings ``\delta_n,\delta_t`` select:

1. **Contact** — ``\delta_n \le 0``: penalty normal stiffness, no tensile traction  
2. **Softening** — inside cohesive envelope and past historic peak  
3. **Unload/reload** — inside envelope but below historic peak (linear return)  
4. **Complete failure** — beyond final openings: residual ``\eta``-stiffness only  

## Quick start

```julia
using BEM
using BEM.Crack

mesh, top = modeI_patch_mesh(; L=0.1, n_coh=4, E=32e9, ν=0.2)
assemble_dual!(mesh)

# top Dirichlet uy
uy = 3e-5
load_dofs = Int[]; load_u = Float64[]
for e in top, loc in 1:3
    j = mesh.elements[e].fis[loc]
    push!(load_dofs, 2j); push!(load_u, uy)
    mesh.elements[e].bc_type[loc, 2] = 0
    mesh.elements[e].bc_val[loc, 2] = uy
end

law = BilinearCZM(; σn=4e6, σt=3e6, Gn=100.0, Gt=200.0, δn0=5e-7)
# or:  PPRLaw(; Γn=100, Γt=200, σn=4e6, σt=3e6, α=5, β=1.6)
# or:  AlfanoSaccoLaw(; kn=1e12, kt=1e12, σn=3e6, μ=0.3)

prob = CohesiveDBEMProblem(mesh, law; kn_pen=1e13, tol=1e-5)
prob.load_dofs = load_dofs
prob.load_ū = load_u

u_hist, λ_hist = solve_cohesive_dbem!(prob; nsteps=15, λ_end=1.0)

cohesive_openings(prob)    # Vector of (δn, δt)
cohesive_tractions(prob)   # Vector of (tn, tt)
[cp.hist.state for cp in prob.pairs]
```

### Contact (compression)

Yes — contact is a first-class surface state. Under ``\delta_n \le 0`` the law
switches to penalty contact (optional Coulomb ``\mu``):

```julia
law = BilinearCZM(; σn=4e6, Gn=100, δn0=5e-7, μ=0.3, kt_contact=1e11)
mesh, top = contact_compression_mesh(; gap0=0.0)
# prescribe uy < 0 on top …
solve_cohesive_dbem!(prob; nsteps=10)
@assert any(cp.hist.state == STATE_CONTACT for cp in prob.pairs)
```

Multi-region type-4 pairs with the same laws:

```julia
states = solve_cohesive_contact!(prob_multi, AlfanoSaccoLaw(; μ=0.4); δ=1e-4, nsteps=8)
```

### Continuation

```julia
# single-DOF displacement control
solve_cohesive_dbem!(prob; method=:dof, control_dof=uy_dof, Δu_control=1e-6, nsteps=40)

# spherical arc-length (Crisfield)
solve_cohesive_dbem!(prob; method=:arclength, Δs=1e-4, ψ=0.0, nsteps=40, λ_end=1.0)
```

### Fatigue + process-zone growth

```julia
law = FatigueCZM(; base=BilinearCZM(...), C=1e-4, m=2.0)
# after each cycle peak:
fatigue_cycle!(law, cp.hist, δn_peak, δt_peak)

# extend cohesive zone ahead of tip, then reassemble:
extend_cohesive_process_zone!(mesh, tip_node, da; n_new=2)
assemble_dual!(mesh)
prob = CohesiveDBEMProblem(mesh, law)  # rebuild pairs
```

## Laws

_See source docstrings in `src/` (HTML `@docs` disabled in lightweight build)._

## Problem API

_See source docstrings in `src/` (HTML `@docs` disabled in lightweight build)._

## Sign convention (calibrated)

Crack-face mesh normals are **solid-outward** (into the gap). Cohesive pairs use an
opening normal

``\hat n = -n_{+}^{\mathrm{out}} \approx n_{-}^{\mathrm{out}}``

so that under mode I

``\delta_n = \hat n\cdot(u^{+} - u^{-}) > 0``.

BEM tractions on the ``+`` face are ``t^{+} = -R\,t_{\mathrm{loc}}`` (tension pulls faces together).
Magnitudes on the Gmsh center-crack plate satisfy ``\langle\delta_n\rangle = O(\sigma a/E)``.

## Gmsh center-crack demo

```bash
julia --project=. scripts/cohesive_gmsh_modeI.jl
```

```text
linear dual reference  ⟨δn⟩ ~ 2.3e-4   (σ=1, E=3000, a=1)
load σ=0.5 .. 4.0      ⟨δn⟩ scales, soft → peak tn≈σn
unload σ=1.0           STATE_UNLOAD
reload σ=4.0           back to SOFTENING
```

## From Gmsh dual cracks

```julia
dad = format2d("crack.msh", Elasticity(E,ν,1.0))
mesh = dual_mesh_from_bemdata(dad)
assemble_dual!(mesh)
prob = CohesiveDBEMProblem(mesh, PPRLaw(...))
solve_cohesive_dbem!(prob; nsteps=20)
```

Crack faces must be paired as twins (`node.twin`) — already done by
`dual_mesh_from_bemdata` / `build_center_crack_mesh` for type-5 BC faces.
