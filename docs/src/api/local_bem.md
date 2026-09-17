# Local BEM

Compact-kernel regularized fundamental solution (``u_i^*=∂_r u_i^*=0`` at
``r=r_i``). Volume terms via DIBEM / compact RIM on ``∂(Ω ∩ B)``.

2-D and 3-D Laplace / Poisson (`format3d` cube: compact RIM on all of ``Γ``
with primitive ``Ψ(\min(R,r_i))``, cone cubature of ``Ω ∩ B``). Elastic local
BEM remains 2-D.

```@docs
assemble_local_bem!
solve_local_bem!
```
