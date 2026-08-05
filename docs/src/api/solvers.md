# Solvers

_See source docstrings in `src/` (HTML `@docs` disabled in lightweight build)._

| Function | Equation | Notes |
|----------|----------|-------|
| `solve` | ``A x = b`` | dense LU or H-mat GMRES; elasticity `frame=:global\|:local` |
| `solve_local` | ``\hat H \hat u = \hat G \hat t + p`` | elasticity in nodal (n,t) frame |
| `solve_Houbolt` | 2nd-order wave Houbolt | needs `DIBEM` |
| `solve_transient` | ``M\dot T = \ldots`` | DiffEq / Tsit5 |
| `solve_transient_o2` | ``M\ddot u + A u = b`` | `SecondOrderODEProblem` |
| `solve_mmm!` | modal ``\ddot y_i+\omega_i^2 y_i=\bar f_i`` | MMM (Prodonoff–Zepka) |
| `solve_mmc!` | same, left = ``\Phi^+`` | classical modal, non-symmetric |

Always call `DIBEM(dad)` before the transient solvers.

## Elasticity — local (n, t) frame (Leonardo 2026 §4.7)

Instead of applying BCs in global ``(x_1,x_2)``, rotate each node to its outward
normal / tangent basis:

```math
\begin{Bmatrix}u_1\\u_2\end{Bmatrix}
=
\begin{bmatrix}n_1 & -n_2\\ n_2 & n_1\end{bmatrix}
\begin{Bmatrix}u_n\\u_t\end{Bmatrix}
= R\,\hat u,
\qquad
\hat H = H R_{\mathrm{block}},\quad
\hat G = G R_{\mathrm{block}}.
```

Boundary conditions then refer to ``(u_n,u_t,t_n,t_t)`` — ideal for rollers,
symmetry planes, and frictional contact.

```julia
H_G_full_direct(dad)
# dad.BC / dad.BV: dof 2i-1 = normal, dof 2i = tangent
# e.g. roller: u_n=0 (Dirichlet), t_t=0 (Neumann)
solve(dad; frame=:local)   # or solve_local(dad)
# dad.u, dad.traction          — global
# dad.u_local, dad.traction_local — (n,t)
```

Helpers: `node_rotation2d`, `transform_HG_local`, `bc_global_to_local!`,
`global_to_local_field`, `local_to_global_field`.

Rigid-body diagonal terms of ``H`` are built in the global frame during assembly
(before the local map), as required in §4.7–4.8.

Test: `test/test_local_frame_elasticity.jl`.

## Método Modal Modificado (MMM)

Reference: Áquila Santos thesis §4.5 (Prodonoff & Zepka, 1983).

For non-symmetric BEM operators ``\bar M,\bar K``:

```math
D=\bar M^{-1}\bar K,\quad
(D-\omega^2 I)\varphi=0,\quad
(D^{\mathsf T}-\omega^2 I)\tilde\varphi=0,\quad
\tilde\varphi_i^{\mathsf T}\varphi_i=1.
```

```julia
H_G_full_direct(dad); DIBEM(dad)
sys   = build_modal_system(dad)
basis = modal_analysis_mmm(sys; nmodes=20)
U, t, basis = solve_mmm!(dad, Δt, tf; nmodes=20, select=:amplitude, nkeep=10)
# classical non-symmetric alternative:
U, t, basis = solve_mmc!(dad, Δt, tf; nmodes=20)
```

Mode selection (§4.6): ``A_i=|\bar f_i/\omega_i^2|`` via `mode_amplitudes` / `select_modes_amplitude`.

Demo: `julia --project=. scripts/mmm_membrane_demo.jl`
