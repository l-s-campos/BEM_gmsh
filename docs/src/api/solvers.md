# Solvers

```@docs
solve
solve_Houbolt
solve_transient
solve_transient_o2
```

| Function | Equation | Notes |
|----------|----------|-------|
| `solve` | ``A x = b`` | dense LU or H-mat GMRES |
| `solve_Houbolt` | 2nd-order wave Houbolt | needs `DIBEM` |
| `solve_transient` | ``M\dot T = \ldots`` | DiffEq / Tsit5 |
| `solve_transient_o2` | ``M\ddot u + A u = b`` | `SecondOrderODEProblem` |
| `solve_mmm!` | modal ``\ddot y_i+\omega_i^2 y_i=\bar f_i`` | MMM (Prodonoff–Zepka) |
| `solve_mmc!` | same, left = ``\Phi^+`` | classical modal, non-symmetric |

Always call `DIBEM(dad)` before the transient solvers.

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
