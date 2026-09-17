# Changelog

## Unreleased

### Plotting: Plots.jl only

- Dropped `CairoMakie` / `GLMakie` / `WGLMakie`. `Plots` is a hard dependency
  (`plot_geo`, `plot_hmatrix`). Demo/paper scripts that used Makie `Figure`/`Axis`
  now use Plots.

### Lubrication (Guiggiani 2020) via Laplace FS + DIBEM

- Reynolds equation after `𝒫 = p h^{3/2}` is `∇²𝒫 + f 𝒫 = g`. Laplace
  fundamentals plus DIBEM mass treat the domain terms (no Helmholtz /
  Klein–Gordon kernel, no particular integral required).
- Films `h1`–`h5` and the linear wedge (`film_h1` … `film_linear`,
  `guiggiani_films`). Infinite bearing: `infinite_bearing_pressure`.
- Finite pad (rounded rectangle, Fig. 4): `mesh_guiggiani_pad`,
  `solve_reynolds_dibem!`. Cross-check for `h2`:
  `solve_reynolds_particular!` (Guiggiani eq. 38).
- Script: `scripts/laplace/guiggiani_lubrication.jl` (Figs. 2, 3, 5).
- Full-film Reynolds as heterogeneous DIBEM: `∇·(h³ ∇p) = 6μU ∂h/∂x`
  (`solve_reynolds_het!`). `solve_heterogeneous!` takes `source=f`.
  Geometry `dM` (`dM^α_ij = S_j ∂u*/∂x_α`) is cached; each `K` update is
  `A p = dM diag(∇K) p` with the `[p(X)−p(ξ)]` row-sum, not a new FS loop.
- Schultz et al. 2025 Interpretation I (characteristic fixed point for
  JFO): `solve_reynolds_cfp!` uses DIBEM as `P(θ)` and the analytic
  characteristic (8) as `C(p)`. Script:
  `scripts/laplace/schultz_cfp_dibem.jl`.
- SLIPPY-style iterative semi-system EHL: `solve_semi_system!`
  (heterogeneous DIBEM Reynolds + Pohrt–Li FFT on interiors). Script:
  `scripts/laplace/slippy_semi_system.jl` (ball-on-flat, Azam/SLIPPY numbers).
- Periodic-in-`x` BEM for unfolded journals: `mark_periodic_x!` pairs
  `x=0` with `x=L` (`p` continuous, `qn_L + qn_R = 0`). Used by CFP
  (`periodic_x=true`).
- Mass-conserving cavitation (Elrod–Adams `p–θ` / JFO) on structured
  grids: `solve_elrod_1d!`, `solve_elrod_2d!`, `solve_elrod_radial!`.
  Dowson–Higginson, Barus, optional Eyring. Profito et al. 2015 §4.1
  cases (`profito_single_slider`, `profito_journal`, …).
  Script: `scripts/laplace/profito_cavitation.jl`.

### Interior collocation grid

- `internal_grid(dad, nx, ny)` / `internal_grid(dad, nx, ny, nz)` — Cartesian
  interior poles (MATLAB `gera_p_in`): even–odd in/out on the BEM surface,
  clearance `d_min * L_max` from `Γ`. `layout=:gera|:cell|:cheb`.
  `internal_grid!` writes them through `set_internal_nodes!`. Alias `gera_p_in`.
  `point_in_domain(dad, p)` is the in/out test (holes included).
- `internal_layer(dad; δ, every, fill)` — few-pole strategy: inward offset of
  boundary collocation at `O(h_Γ)` plus an optional coarse cell-centred core.
  Use this when `n_int ≪ n_edge`; a coarse lattice on a fine `Γ` leaves IBP
  stencils with no interior neighbours.

### Rank-k H² LU preconditioner

- `h2_lu_prec(A; rank=2)` weighted-truncates an H² operator to rank `k`
  and returns `lu` (H-LU or nested) for `gmres_h(...; Pl=...)`. Near
  leaves stay dense.

### Chebyshev interpolative H²

- `assemble_h2(K, tree; method=:cheb, order=p)` builds nested Chebyshev
  interpolation bases (Hackbusch–Börm) and M2L as the kernel on those
  nodes. `K` must be a `KernelMatrix`. Default `order` is 6 in 2-D and 4
  in 3-D. After assembly, H2Lib-style weighted truncation (nested QR of
  `L2P` plus QR of stacked `W_s S'`, then SVD of `Vhat Z'`) to `rtol`
  (`rtol=0` keeps the full Chebyshev rank).
  Apply is the existing NNCA M2M / M2L / L2L / near.

### HSS ID and ULV

- `interpolative_decomp` is early-terminating GEQP3 (not a post-truncated
  full QR). Tall, large Kid (`m ≥ 2n` and `m n ≥ 80_000`) uses a Gaussian
  row-sketch then the same ID (`sketch=:auto`; `sketch=false` disables it).
  Wide proxy samples stay on RRQR so HSS ranks are not cut. Same contract:
  `A[:,rd] ≈ A[:,sk] * T` with `max|T| ≤ Tmax`.
- ULV stores packed Householder / compact-WY `Q` (`ULVQ`) instead of dense
  `m×m` factors. Apply is `lmul!` / `rmul!` with a complement-then-range
  permutation. Factor + solve accuracy vs the HSS matrix is unchanged
  (`residH ~ 1e-14`). `_ulv_BV` copies `V[rows,:]` so `B*V'` is BLAS;
  complementary QR is `qr!` of a dense transpose.
- `assemble_hss` IDs same-level boxes in parallel (`threads=true`; BLAS
  pinned to 1 thread in that loop). Optional `recompress=true` QR-proper
  + relative tsvd of sibling `B` after the sweep (hm-toolbox).

### Laplace FMM apply

- 3D M2L is in-place (`_equiv_charges!` + `form_local3d!(; Ptab=ws.Ptab)`), so
  apply no longer `similar`s charges or a Legendre table per pair.
- 2D and 3D M2L thread over `m2l_groups` (one writer per target box) with
  per-spawn translation workspaces (`trans_tls`, `let tid=tid`). `@threads`
  + `threadid()` disagrees with dense on Julia 1.13 (pool ids are not
  `1:nthreads`).
- Default 3D leaf size is Flatiron `lndiv` again (`200` at `1e-8`). A small
  `nmax` (H²-scale 16) makes a deeper octree but multiplies O(K²) M2L pairs;
  pass `nmax=` to override. L2P/L2L use a per-target associated-Legendre table
  (`ylgndrfw`-style), not a scalar `P_n^m` per (n,m).
- 3D spherical moments use FMM3D `ylgndru` packing:
  `Y_n^m = sqrt((n-m)!/(n+m)!) P_n^m` (Condon–Shortley P). Same-size
  cubic-octree M2L is FMM3D plane-wave (`mpoletoexp` / diagonal shift /
  `exptolocal`). Mixed-size pairs stay equivalent-sphere.
- P2M / M2M / L2L / L2P thread over boxes by level (FMM3D OpenMP-over-ibox).
  Legendre uses cached `ylgndruf` recurrence weights. Plane-wave Fourier
  maps use tabulated `e^{imα}` (`ftophys` even/odd 2 Re / 2i Im) and
  FMM3D `rlscini` λ-polynomials (`rlsc`); M2X/X2L cap Fourier modes at
  `nfour` per λ.
- Expansion order is already a call-site keyword: `p=` on
  `build_laplace3d_plan`, `fmm_laplace3d_matrix`, and `fmm_kelvin3d_matrix`
  (`nothing` → `laplace3d_nterms(eps)`, capped at 16).
- 3D P2P `1/r` uses 64-source L1 tiles, 4-target unroll, and `@fastmath`
  `1/sqrt` (LLVM rsqrt). Same-leaf jobs skip the diagonal in a 4×4 block
  so the long ranges stay branch-free.

### EX2-SSSS2 (sine, immovable membrane)

- Like Tran Table 2 (`[0/90/90/0]` Mat. III, SSSS2 `u=v=0` on all edges)
  but sine pressure and thickness sweep. FEniCS twin
  `tran2015_fenics.py ex2ssss2`. Run `run_static.jl ex2ssss2`.

### Membrane SSSS1 traction from `N_vk`

- Navier / SSSS1 free-normal DOFs had `t_L = 0`, so `(N_L+N_vk)·n = N_vk·n ≠ 0`.
  Physical BC is total `N_nn = 0`. On traction-known membrane DOFs set
  `t_L = −N_vk·n` (`Gm t_vk`; equivalent to `t_total = 0`). RBF `N` on `Γ`
  is then projected so `n·N·n = 0` before `N∇w` (plate `Γ` sees the BC).
  SSSS2 `:clamped` has no free membrane traction; Table 2 is unchanged.

### Von Kármán geometric load (one IBP)

- FSDT / laminated-shell geometric term is one divergence theorem,
  `∫_Ω U* ∇·(N∇w) = ∫_Γ U*(N∇w)·n - ∫_Ω ∇U* · (N∇w)`, not RBF
  `div(N∇w)` through `Mw`. Volume maps `Mx, My` (`-∂U*_w/∂X c`) plus
  unmixed `G t` with `Vn = n·(N∇w)`. Remainder `Mx 1 + Γ(e_x) = 0` so
  a constant flux (zero divergence) is exact. Kirchhoff `LargePlate`
  still uses `M*(Fx vx + Fy vy)` (Table 1).

### FSDT DIBEM `M` for non-uniform `q`

- Uniform remainder `Mw 1 = ID(:,3)` is exact (rel 1e-16). `Mw` matches
  polar RIM of `U* q` for smooth `q` (sine ~9 %, quadratic ~1 %). RBF
  `div(N∇w)` is a high-frequency field (`qg` at a sine-w centre should
  be 0); that is why `Mw*qg` was the wrong geometric operator.

### IBP `Γ` term + adaptive Newton

- `Γ` uses the unmixed `G` so SS unknown `Vn` is not overwritten.
- Shell `nonlinear=:newton` skips Picard warm-start after the first
  step, uses Armijo, and halves `Δλ` on failure instead of freezing `w`.

### Laminated-shell Newton–Armijo

- `solve_laminated_shell!(; large=true, nonlinear=:newton)` iterates
  Newton with Armijo line search on the von Kármán residual (not a
  single Jacobian step). Load-control Picard can stall at large `λ`;
  SS plates stiffen monotonically so Newton is the right alternative
  (`:arclength` / `:wcontrol` for snap-through).

### Laminated-shell von Kármán continuation

- Picard no longer mixes the linear predictor into the next load step, and
  no longer replaces a NaN iterate by `x_lin`. That silent fallback made
  Tran Table 2 `L/h=10`, `P̄=250` report the linear `w/h=2.59` as
  nonlinear (paper 0.77). Continue from the last accepted state; revert
  to it if `|w|` exceeds `8 w_lin`.

### Tran 2015 Figs 7–9 (Hsu–Hwu)

- `scripts/plates/tran2015/run_static.jl` fig7/fig8/fig9 use
  `solve_fsdt!(; large=true)` on `UnsymFSDTProps` when `B≠0`. Symmetric
  angle-ply (`B=0`) stays on Wang 3-DOF (`wang_kernels` singular at `B=0`).

### Hsu–Hwu 5-DOF von Kármán

- `solve_unsym_fsdt_large!` / `solve_fsdt!(mesh; large=true)` on
  `UnsymFSDTProps`: linear Hsu–Hwu `H,G` plus DIBEM loads from
  `½∇w⊗∇w` (`N_vk=A ε_NL`, couples `M_vk=B ε_NL`) and
  `∇·(N ∇w)`. Picard or Newton. 3-DOF Wang large deflection stays on
  `LaminatedShell`.

### Tran 2015 static plates (BEM)

- `scripts/plates/tran2015/run_static.jl` reruns the static cases of Tran
  et al. (Int. J. Non-Linear Mech. 72, 2015) with BEM, not IGA-HSDT:
  Kirchhoff DIBEM + von Kármán for the thin isotropic square (Levy /
  Fig. 2); Wang FSDT + Lekhnitskii membrane + von Kármán
  (`LaminatedShell`) for the circular plate and laminates (Tables 1–3,
  Figs 3–9). Soft SS and Wang `B=0` differ from the paper’s hard-SS
  Reddy HSDT.

### Interpolant Guiggiani quadrature cache

- Gauss–Legendre nodes, quadrature weights, and barycentric `wᵢ` for a
  given `ninterp` (default 20) are built once. `_interp_rule` only
  evaluates `Nᵢ(a)` and `ξᵢ-a`. The unused `ArbitraryPolynomial`
  derivative matrix (`Dmat` / `shapefun`) is no longer formed per
  singular pair.

### Guiggiani log coefficient (Kirchhoff `G₂₂`)

- Interpolant `F₀` for `order=0` is a least-squares fit
  `f-F₋₁/(ξ-a) ≈ F₀ log|ξ-a| + b` on the sample nodes. Evaluating the
  Lagrange interpolant of `(f)/log|ξ-a|` **at** the collocation was
  unstable (off-centre Gauss nodes): Kirchhoff `G₂₂∼log` was 27% off
  (wrong sign on the self term) and CCCC `w_max` 15% stiff. Richardson
  rays were already fine. CCCC Guiggiani now matches Levy 0.001263 to
  `<0.1%`. `singular=:auto` still uses analytic `integraelemsing` on
  straight edges.

### Kirchhoff on-element CPV (clamped)

- `assemble!(dad::BEMdata{<:ThinPlate})` defaults to `singular=:auto`:
  analytic `integraelemsing` on straight edges (BEM atual `calc_HeG`).
  `singular=:guiggiani` is valid for CCCC after the log-`F₀` fix.

### Kirchhoff plate DIBEM

- `dibem_plate!` ports BEM atual `Monta_M_RIMd` onto
  `BEMdata{<:AbstractThinPlate}`: PHS Gram `F`, RIM `IF`/`ID`, remainder
  `M 1_w = ID`, packed on w-columns (dummy internal slope stays unused).
  `plate_Mw` is the collocation map used by laminated-shell coupling
  (`scripts/plates/albuquerque2010_thinplate.jl`). Particular-integral
  `plate_q` is unchanged unless `apply_load=true`.

### GPU apply for H-matrix and NNCA H²

- `gpu(H)` / `gpu(A)` packs an assembled `HMatrix` / `NNCAMatrix` for
  KernelAbstractions matvec (`device=:cuda` needs `using CUDA`; `:cpu` is
  the KA host backend). `assemble_hmatrix` / `assemble_h2` /
  `H_G_Hmat` / DIBEM `:hmatrix`/`:h2` take `device=:host|:cuda|:cpu`.
  Assembly stays on the CPU. Host `x`/`y` still copy once per apply;
  `CuArray` (or other `A.backend`) vectors permute and apply on the device.
  Packed apply uses one thread per output row of each leaf (dense / Rk /
  NNCA M2M·M2L·L2L·near), not one thread per leaf.

### Fundamental solutions

- `_R2` / `_R`; 2-D Laplace `U = -log(R²)/(4πk)` (no `sqrt`).
  [`fundamental_U`](@ref) / [`fundamental_T`](@ref) for a single layer
  (Helmholtz skips the unused Hankel; Kelvin 2-D/3-D `U` skips `T`).
  `LaplaceDqKernel` / `LaplaceDuKernel` / DIBEM `U*` use those, with
  batched `getblock!` / NNCA row-col fills.

### H² LU

- Nested LR (H2Lib `lrdecomp_h2matrix`) on square NNCA: `h2node(A)` keeps
  nested couplings, `lu(A; method=:nested)` factorizes the H² block tree
  without expanding to an H-matrix. `method=:hmatrix` remains the
  convert-then-H-LU fallback.
- Nested `h2_addmul!` follows H2Lib `addmul_h2matrix`: k×k Gram when all
  three blocks are nested uniforms; otherwise a structured Rk of `αAB`
  (nested apply / thin H²×k) is projected onto `C`'s nested bases or
  injected by local rkupdate. PartialACA of the virtual `C+αAB` is last
  resort only.
- Formatted nested triangular solves: `L \\ B` / `B / U` apply `L` (resp.
  `U^{-T}`) to the cluster basis `V` (k columns). If `L\\V` still lives
  in `span(V)`, keep a nested k×k coupling; otherwise store the exact
  thin Rk `(L\\V) S V_c'` (`s_full`, no SVD of `S`). Schur inject into a
  nested `C` Galerkin-projects when `XY'` already lives in `C`'s bases.
- Börm–Reimer `rkupdate`: R2 expands nested bases and rewrites couplings;
  R3 weighted recompress via [`prepare_h2_weights`](@ref). Use
  `h2_rkupdate!(...; method=:nested)` for a global `XY'` update.

### Rectangular NNCA

- `assemble_h2(K, rowtree, coltree)` is a dual-tree H² of size
  `(n_row)×(n_col)`. Share a root `container` for equal-size boxes.
  `H_G_Hmat(...; format=:H2)` uses it for rectangular `Du` (all
  collocation × boundary). Square `assemble_h2(K, tree)` is unchanged.
  GPU wrap of a rectangular operator stays on the CPU.

### NNCA assembly

- Same-level ACA is threaded (siblings only read finer children).
  `KernelMatrix` is copied into tree-local order so ACA/M2L/near do not
  gather through `PermutedMatrix` on every sample.
- ACA reuses a thread-local workspace (no per-pivot `copy`).
  2-D/3-D `KernelMatrix` row/col/block fills gather points to SoA and
  `@simd` the kernel.
- M2L and near are packed CSR (`m2l_data` / `near_data`), not
  `Dict{Int,Matrix}`. Symmetric M2L is not used: it is wrong for
  nonsymmetric kernels.

### Laminated shell geometry and large deflection

- Donnell `κ_αβ` come from [`ShellGeometry`](@ref): `SphericalShell`,
  `CylindricalShell`, `HeightGraph` (`Hess z`), `FlatShell`, or
  `ConstantCurvature`. `LaminatedShell(plate, A, κ1, κ2)` still wraps
  uniform κ. Coupling uses nodal `κ` (uniform geom recovers Useche scalars).
- `solve_laminated_shell!(; large=true)` adds von Kármán `½∇w⊗∇w` and
  `N:∇∇w` on the Donnell operators. Geometric load uses the same DIBEM
  `Mw`/`Mm` as linear `q` and `Hw`. `nonlinear=:picard|:newton` is load
  control; `:arclength` is spherical Crisfield on `R(x,λ)` (bordered
  corrector, displacement arc `ψ=0`). First step is the linear tangent;
  later predictors use the last accepted increment (Ramm), not `J\\q` at
  the fold. `:wcontrol` increments crown `w` so a load limit is a regular
  point (snap-through when the path is single-valued in `w`).
- [`shell_navier_centre`](@ref) compares linear BEM crown `w` to 5-DOF
  Navier on the same mesh. Useche 9.6.1 (`n_el=4`, 81 centres) is the
  accuracy gate (`relerr < 15%`); coarse continuation tests are solver
  smokes only. After that gate, 9.6.1 applies von Kármán at
  `λ w_lin ~ h/2`.


### Removed HODLR, HSS, and ScalarizedMatrix

- Dropped `HODLRMatrix` / `HSSMatrix` (`:hodlr`, `:hss`, `:hbs`),
  `ScalarizedMatrix`, `expand_tree`, and `assemble_hss_fmm`. Remaining
  structured formats: H-matrix, BLR, NNCA H².
- DIBEM / `H_G_Hmat` compression is `:hmatrix`, `:h2`, or `:fmm`.

### Block NNCA and tensor H-matrices

- [`assemble_h2`](@ref) on `SMatrix{p,p}` kernels is block NNCA: FMM
  interaction lists and ACA skeletons stay on points (Frobenius pivots);
  `L2P`/`M2L`/near are `(p n)×(p r)` scalar blocks.
- Elasticity DIBEM `:hmatrix` assembles an H-matrix of `SMatrix{d,d}` on the
  point tree. `:h2` is block NNCA.
- Tensor H-matrix / NNCA matvecs accept `Vector{SVector{p}}` and a flat
  length-`p n` vector. Tensor H-matrix apply stays in the flat layout at
  the leaves (no `Vector{SVector}` pack/unpack on the DIBEM path).

### NNCA H² (SAFRAN-LAB port) and quad/oct H-matrix trees

- New [`assemble_h2`](@ref) / `NNCAMatrix` is Gujjula–Ambikasaran NNCA
  (geometric quad/oct tree, FMM interaction list, full-IL ACA, nested
  L2P, dense M2L). Experiment 1 at N=25 600 matches SAFRAN C++ apply
  (~21 ms, rank ~21, ε~10⁻⁸).
- H-matrix assembly defaults to `DyadicSplitter(tight=false)` (quadtree /
  octree). `PrincipalComponentSplitter` remains available.
- `RkMatrix` scratch pool is lock-grown (fixes threaded `mul!`).
- `:h2` restored on Laplace DIBEM, `H_G_Hmat`, Pohrt, `HalfSpaceBEM`.

### Thick plates (FSDT) from `formatdata` / Dual BEM on `BEMdata`

- `FSDT <: AbstractFSDT <: Vectorial`. Field DOFs (`n_dof=3`) are separate
  from geometry (`dad.dimension=2`). `format2d` / `H_G_full_direct` pack
  by `n_dof`. Prefer `FSDTProps` when `using BEM.Plate`.
- Square/rect plates: `quadrado_fsdt` → [`formatdata`](@ref) →
  `assemble!` → `solve`. `build_square_fsdt` / `build_rect_fsdt` are that
  pipeline for 3-DOF Reissner/Wang. Unsymmetric 5-DOF stays `FSDTMesh`.
- Dual-BEM cracks live in the Gmsh file: `quadrado_fsdt(..., crack=a)`
  embeds coincident twins tagged `"5;2"` / `"5;3"` repeated `ndof` times,
  then `formatdata` → `prepare_crack!` → `assemble!`. On-element HBIE keeps
  the Taylor HFP (not Guiggiani). `quadrado_plate` uses 4-token
  `"5;2;5;2"` / `"5;3;5;3"`. Hsu–Hwu 5-DOF, DIBEM, Houbolt, XBEM, and
  laminated shells use the same `BEMdata{<:AbstractFSDT}` stack
  (`ndof=5` SS-1 tokens in `quadrado_fsdt`).

### Kirchhoff plates from `formatdata` / Dual BEM on `BEMdata`

- Square plates: `quadrado_plate` → [`formatdata`](@ref)/`format2d` →
  [`prepare_plate!`](@ref) (corners) → `assemble!` → `solve`.
  `build_square_plate` is that pipeline. `formatdata` is an alias of `format2d`.
- Dual-BEM cracks: `mesh_center_crack` → `formatdata(msh, ThinPlate)` →
  `prepare_crack!` → `assemble!` (CBIE/HBIE) on `BEMdata{<:ThinPlate}`
  (no `PlateMesh` lift).

### Kirchhoff plates are `BEMdata{<:ThinPlate}`

- `ThinPlate <: Vectorial` (like elasticity). `build_square_plate` returns
  `BEMdata{<:ThinPlate}`. `assemble!` / `H_G_full_direct` is the same
  collocation loop as 2-D elasticity (Gauss nodes, far lumping, sinh,
  Guiggiani), with source-normal kernels, ``\\tfrac12 I`` jump, then plate
  extras (distributed `q`, dummy internal slope, corner ``R_c``).
  `solve(dad)` applies mixed BCs (tractions scaled by ``D``).
  `ThinPlateProps` is an alias of `ThinPlate`. Dual BEM still uses `PlateMesh`.

### Thin-plate assembly matches Laplace

- `assemble_plate!` uses the same far/near/on-element split as
  `H_G_full_direct`: Gauss–Legendre collocation nodes, nodal lumping when
  `r > near_factor L` (default `1.5`), Euclidean sinh near-field, on-element
  fused `guiggiani_GH`. `threaded=true` by default. `singular=:auto` picks
  analytic CPV/HFP on straight edges. Straight `elem_geom` is the linear
  endpoint map (cached `Equispaced` only for curves). `PlateMesh` is
  parametric in `props` and `element_type`. `apply_bc_plate` copies `H`
  once. Distributed-load `q` is skipped when `q_a=q_b=q_c=0`.
  `assemble!(mesh::PlateMesh)` is an alias of `assemble_plate!`.

### Scalar `near_factor`

- `assemble!` / `H_G_full_direct` on Laplace and Helmholtz accept
  `near_factor` (`Inf` = no far lumping), same as vectorial H/G. Also
  Dual Laplace and `H_G_hyper`.
- Default is `:auto` → [`auto_near_factor`](@ref): Laplace `1.5`;
  Helmholtz `1.5` if `κ L_max ≤ 0.6` (≳10 points per wavelength), else
  `Inf`. Override with a `Real`. Chosen value is cached as `dad.near_factor`.

### Analytic DIBEM M′ (Burton–Miller)

- `dibem_hyper_mass` uses one analytic ∂/∂nξ of the RIM of `U*` per
  boundary source instead of two finite-difference RIM sweeps (`2n` IDs).
  Remainder still `M′1 = ID′` on HBIE rows. Requires internal points;
  interior rows copy CBIE DIBEM `M` (no `nξ`). Tests: `test/helmholtz.jl`.

### Scalar H/G thread-local buffers

- Dense scalar `assemble!` (Laplace / Helmholtz) reuses per-thread `hloc`/`gloc`
  like vectorial H/G, and caches element nodes. Same buffers on Laplace Dual
  and `H_G_hyper`.

### 3-D Helmholtz and hypersingular kernels

- 3-D Helmholtz fundamental solution ``G=e^{iκR}/(4πR)``,
  ``H=∂G/∂n_x``. 2-D kernels unchanged (Hankel).
- `fundamental_hyper` for Helmholtz in 2-D and 3-D, and Laplace in 3-D
  (κ→0 check: Helmholtz ``G`` matches Laplace ``k=1``, ``H`` matches
  ``-``Laplace because ``q=∂u/∂n``).
- `H_G_hyper` on `BEMdata{<:Helmholtz}` in 2-D and 3-D (`ComplexF64`,
  jump ``G'_{ii}-=1/2``). Laplace HBIE stays 2-D.
- 3-D surface Guiggiani honors `laurent=:interp|:richardson|:auto`
  (ray interpolant of ``F(ρ)=φρ``). 2-D Helmholtz gets closed-form
  leading Laurent tensors (`:auto`). Tests: `test/helmholtz.jl`.

### Half-space contact acceleration (H / FMM)

- Pohrt–Li `precompute_kernels(...; method=:hmatrix|:fmm)` uses the same
  `assemble_hmatrix` path as Laplace DIBEM. `:fmm` (`Kzz`) is
  `FMM.fmm_laplace3d_matrix` (same octree plan as DIBEM) scaled by
  ``4A/E*`` in the far field, with Love `influence_coeff` on a near
  stencil — HalfSpaceBEM `lfmm3d` + exact near panels, not point 1/r
  plus Love self. FFT remains the default on a uniform grid.
- Tests: `test/contact.jl`.

### Half-space contact (Pohrt–Li + Juliá Lerma)

- Two-body Kalker combination (`combined_halfspace`) and coupling `K` on
  `ElasticHalfSpace`. Coupled `fc_displacements`. `Kxz`/`Kyz` use
  principal-value `atan(y/x)` so the kernels stay odd (atan2 leaked a
  uniform slide).
- Layered/anisotropic Fourier compliance (`LayeredAniso`, Bagault 2013).
- Uzawa / Alart–Curnier with elliptic Coulomb and orthotropic Archard
  wear (`OrthotropicUzawa`), subsurface Love/Johnson stress
  (`SubsurfaceStress`), Kalker rolling (`RollingContact`), planar
  wheel–rail (`WheelRail`).
- Juliá Lerma Ch. 3 drivers in `scripts/julia_lerma/`; notes in
  `_research/julia_lerma_2025/`. Profiles in `data/contact/vollebregt/`.
- Still loaded as `using BEM.Contact` (not reexported by `using BEM`).
- Tests: `test/contact.jl`, `test/test_julia_lerma_contact.jl`,
  `test/test_julia_lerma_convergence.jl`, `test/test_layered_aniso.jl`,
  `test/test_wheel_rail.jl`.

### GPU 2-D Laplace assembly

- `src/Laplace/Assembly_GPU.jl`: KernelAbstractions far-field kernel for dense
  collocation `H` and `G` (Julia, not CUDA-C). Near/singular pairs reuse
  `integrate_element` on the host (`near=:cpu`, default). `assemble!(dad;
  method=:gpu, T=Float32)`. Solve stays on LinearSolve.jl.
- `gpu_float_support()` reports device formats after `using CUDA`.
- Compare: `scripts/laplace/gpu_vs_cpu_assembly.jl`.
- `src/Laplace/DIBEM_GPU.jl`: same stack for dense DIBEM `F`, `D`, and far
  RIM (`IF`, `ID`). Near Gauss stays on the host; `Fc=IF` and `M` are host
  Float64. `DIBEM(dad; method=:gpu)` / `DIBEM_gpu`. PHS1–7 or `FundamentalRBF`.
  Compare: `scripts/dibem/gpu_vs_cpu_dibem.jl`.
- 2-D isotropic elasticity: `src/Elasticity/Assembly_GPU.jl` Kelvin far-field
  `H`,`G` and pairwise DIBEM `F`,`D`. `assemble!(dad; method=:gpu)` /
  `DIBEM(dad; method=:gpu)`. Compare: `scripts/elasticity/gpu_vs_cpu_assembly.jl`.

### Constant-cell elastoplasticity (2-D)

- Initial-stress BEM with piecewise-constant Gmsh cells (Telles / Gao & Davies),
  von Mises, linear isotropic hardening. Interior cell stress is Hooke of
  ``∂/∂p`` of the displacement edge integral of ``U`` (centroid off the
  edges); Gao–Davies ``E_{ijkl}`` is a check only. Domain integral
  `domain=:dibem` interpolates ``σ^p`` with RBFs at **cell centroids**
  (remainder on the nearest centre so a uniform field reproduces
  ``∫_Γ U n``). Inner iteration is damped Picard or inverse Broyden;
  default stress coupling is the free term `:jump` (`:full` `Sσ` is
  consistent for uniform ``σ^p`` but the iteration diverges). Thick-cylinder
  vs von Mises (quadratic, `nr=12`, 16 steps): ``u(b)`` ~4% (elastic floor
  ~2%), elastic-ring ``σ`` ~6%. Plastic-core ``σ`` stays ~35% — constant
  cells jump ring-by-ring (load–displacement sawtooth about the closed
  form). Picard and Broyden follow the same path. Plastic `domain=:dibem`
  uses format2d internals as centres; near-field cell-edge ``Q``, far
  ``c_k E``. Same ~4% cylinder ``u(b)`` as cells — not a substitute for
  8-node cells. A global PHS remainder on a plastic jump diverges.
  `solve_elastoplastic!`, `ana_thick_cylinder_plastic`.

### 3-D surface DIBEM (`nearfield=:dibem`)

- Off-element 3-D face integrals can use a 2-D DIBEM interpolant on the
  parent square: PHS3 + linear polynomial at vertices + edge Gauss,
  with the interior spike recovered by an analytic polar `ID` of the
  leading Laplace kernels (`1/R`, `(r·n)/R³`). Used when `d/L < 0.05`;
  far faces stay on `:auto` (tensor sinh). On-element stays Guiggiani.
  Opt-in: `set_cache!(dad; nearfield=:dibem)`. Default remains
  `:tanp3c` (polar + Granados eq. 41).
- Tests: parent-square kernels in `test/core.jl`; cube interior
  approach in `test/laplace.jl`. Script:
  `scripts/debug/surface_dibem_3d.jl`.

### Anisotropic Kirchhoff (Shi–Bezine)

- `AnisoThinPlateProps` / `aniso_thin_plate_props` take the full Voigt
  ``D_{ij}`` (including ``D_{16},D_{26}``) and Lekhnitskii ``μ`` with
  Im>0. Kernels, corners, particular integral, and CPV/HFP match
  `placa_fina` in BEM.jl atual. Isotropic ``μ₁=μ₂=i`` stays on
  `ThinPlateProps` — do not smear ``D=√(D_{11} D_{22})``.
- `assemble_plate!` dispatches on `mesh.props`. Useche 7.5.1 / 7.5.2
  in `scripts/plates/usech_ch78.jl` run the anisotropic BEM.

### FSDT / Reissner plate BEM

- `BEM.Plate` FSDT (`FSDT.jl`): Vander Weeën 3×3 kernels (MATLAB
  `kernelsHGC.m`), same `Element` / GL collocation as Kirchhoff.
  Soft SS is `w=0`, moments free. Domain load and rotary inertia via
  DRM/DIBEM (`uqchp` particular solutions, MATLAB `calcQ_drm` /
  `calcM_drm`). Dynamics: Houbolt (`solve_fsdt_houbolt!`).
  Compare: `scripts/plates/fsdt_vs_matlab.jl`.
- `build_rect_fsdt` for mixed-BC rectangles (`vn` = MATLAB `Vz` on a free
  edge). Useche 8.6.1 cantilever `[0/90]s` end shear:
  `scripts/plates/usech_861_cantilever.jl`.
- `build_circle_fsdt` (straight-chord disk). Useche 9.6 Wang FSDT + DIBEM
  (SS spherical Donnell `kmem`, clamped circular): `scripts/plates/usech_96_dibem.jl`.
- Ch.9 laminated shallow shells (`LaminatedShell`): Wang plate (3 DOF) +
  Lekhnitskii membrane (2 DOF), curvature by DIBEM (`Ha`, `Hw`, `Hc`)
  instead of MATLAB RIM. Coupled mass (9.17) is
  ``M=\mathrm{diag}(M_\mathrm{plate}, I_0 M_\mathrm{mem})`` with ``I_1=0``.
  Houbolt `solve_laminated_shell_houbolt!` (default `mass=:raw`). 9.6.1
  Fig. 9.3 `N,M` along centre-lines: `shell_resultants(; method=:ss1)`
  scales 19-term Reddy by BEM/Navier `w`. Raw RBF `∇u` does not cancel
  `u,x+κw`. `scripts/plates/usech_96_dibem.jl` prints the x1-axis.
  SS sphere: `scripts/plates/usech_96_dibem.jl` (static, 81 centres) and
  `scripts/plates/usech_96_houbolt.jl` (step ``q(t)``, 25 centres — a
  denser PHS mass is indefinite and Houbolt diverges).
- Reissner Dual BEM for cracked plates (Useche 10.2 / Dirgantara): same
  Portela–Aliabadi–Rooke layout as in-plane `assemble_dual_elasticity!`
  (Gmsh twins, CBIE on outer+face A, HBIE on face B, ½I free terms).
  `build_rect_fsdt_crack`, `assemble_fsdt_dual!`, `sif_ctod_fsdt`.
  10.5.1 centre crack bending+tension: `scripts/plates/usech_1051.jl`.
  CTOD `r` is the distance to the CAD element end (`ξ=±1`), not to the
  inset collocation (MATLAB `La=5Le/6`, `Lb=Le/2` on `CoordCHP`).
  Reissner XBEM (`assemble_fsdt_xbem!`, `solve_fsdt_xbem!`): same extra
  columns + tip tying as in-plane Andrade–Leonel, with Hui–Zehnder /
  Dolbow (2000) eq. (36) for `(ψ, w)`. Extra DOFs are Dolbow
  `(K1,K2,K3)`; Table 10.1 `F = K1/(Mo√a)` (`K1b=√π K1`).
  Tip tying matches in-plane `xbem_tying`: Lagrange in `ρ`, not `√ρ`
  (`Cε` is the extrapolated Hui–Zehnder opening; `√ρ` made `Cε=0` and
  dropped `K` from the tying row). Default `n_v=3` (one tip element);
  plate tying mixes `ψ~√ρ` and `w~ρ^{3/2}`, so `n_v=9` Runge-oscillates.
  Extra `Hε`: skip self/twin (shifted `φ=0` at the source); Telles of
  `Tφ` on other crack pairs.
- Kirchhoff Dual BEM (`assemble_plate_dual!`) on `PlateMesh` with
  `eq_type`/`twin` (not a flag on Reissner Dual/XBEM). Outer+face A:
  CBIE. Face B: traction BIE (`Vn_ξ, Mn_ξ`) — same Dual split as
  Useche 10.2 / `assemble_fsdt_dual!` (not a w-BIE + Mn mix).
  Self/twin: Taylor of kernel × N (four Laurent terms `F_{-4}…F_{-1}`)
  + Telles remainder + analytic HFP, matching MATLAB `CalcGtes` /
  `_hbie_sing!`. Collinear neighbours use the same Telles peak. ½I
  free terms on `G` (self+twin). HBIE rows are O(1/L³) vs CBIE O(1):
  `solve_plate!` row-equilibrates `A`. `sif_ctod_plate` default
  `:band` is the median `K1(ρ)` on `0.3a–0.85a`. Kirchhoff XBEM
  (`solve_plate_xbem!`) was removed; plate extra-DOF SIFs stay on
  Reissner `solve_fsdt_xbem!`. Compare Dual vs Reissner:
  `scripts/plates/kirchhoff_xbem_compare.jl`.
  Dolbow 5.2 angled centre crack (`α` on `build_rect_fsdt_crack`):
  `scripts/plates/dolbow_52_angled.jl`.
- Unsymmetric FSDT Portela Dual BEM (`assemble_unsym_fsdt_dual!`, dispatched
  from `assemble_fsdt_dual!` / `assemble_fsdt!` when the mesh has twins):
  outer+face A Hsu–Hwu CBIE, face B EABE 156 complete-solution traction BIE
  (`W=T*(ξ,x;n_ξ)`, `S=L_{nξ}(T*,∇_ξ T*)`). Self/twin Guiggiani.
  `build_rect_fsdt_crack` accepts `UnsymFSDTProps` (5 DOF, edge moment on
  `Hy`).
- Hsu–Hwu traction BIE (`unsym_hbie_kernels`): EABE 156 `T* = (Tx,Ty,Hx,Hy,Qn)`
  from constitutive of `(v, v,ρ)`. `W = T*(ξ,x;n_ξ)` (Betti swap).
  `S = L_{nξ}(T*, ∇_ξ T*)` (complete solutions: differentiate the
  displacement BIE, then the same `T*`). `θ`-integral is Guiggiani HFP
  of `1/ρ²` at the two `ω ⊥ r` poles (Richardson rays). On-element
  assembly uses Richardson rays. Interior moments: `unsym_interior_t`
  is that complete-solution traction, not `nξ·∇P`.
- Unsymmetric FSDT CBIE vs HBIE on the same mesh (`assemble_unsym_fsdt!`
  `bie=:cbie/:hbie`, `singular=:guiggiani/:telles`). Self-element
  `guiggiani_GH`: CBIE interpolant `(0,-1)`, HBIE Richardson rays
  `(-1,-2)`. Single HBIE on the whole boundary (`eq_type=3`); interiors
  are Somigliana after solve. Domain load `q_i = q_c ∫ W_{i5} dΩ` (RIM
  of `W`, all 5 traction rows). Interior moments from a CBIE solve use
  EABE 156 complete solutions (`unsym_interior_t` vs `unsym_interior_t_fd`),
  `scripts/plates/unsym_interior_hbie_moments.jl`.
- Unsymmetric FSDT fundamentals (Useche 8.3.1 / Hsu–Hwu 2023): 5×5
  `unsym_fsdt_kernels` on `(u,β,w)` with full ABD (`UnsymFSDTProps`,
  `laminate_unsym_props`). `F(ρ)` is the Jordan particular integral of
  `F'−JF=ρ^{-2}I` with even homogeneous `β=−3/2` (`f3=−½ρ²ln|ρ|`), not
  the book OCR of (8.38–8.42). Tiny-`B` `U_ww`, `P_ww`, and the 3×3
  bending block match Wang. Symmetric `B=0` stays on `wang_kernels`.
- Symmetric-laminate Wang kernels (`wang_kernels`, MATLAB `KernelP.m`):
  `LaminateFSDTProps` with ABD `D` and ContsLam `AT=[A44 A45; A45 A55]`.
  `laminate_fsdt_props(plies)` from `(E1,E2,ν12,G12,θ,t)`. Assembly
  dispatches on props. Uniform pressure via polar RIM of `U*`; DRM mass
  stays isotropic-only.
- FSDT/Wang singularity maps (`map=` on `wang_kernels` and
  `assemble_fsdt!`): `:telles` (default), `:gauss`, `:sinh`, `:sinhsinh`,
  `:power`. Telles wins the θ-integral at MATLAB `nθ=10–12` and
  near-element ξ vs refined Gauss (`scripts/plates/fsdt_map_compare.jl`).
- Maxima radial primitives of FSDT `U*` (`scripts/plates/fsdt_radial_maxima.mac`):
  Reissner `∫ Az z dz` / `∫ U ρ dρ` and Wang `∫ φ^{(n)} ρ dρ`. DIBEM
  (`dibem_fsdt!`) follows Laplace: PHS Gram on **boundary collocation +
  domain-cell centroids** (`format2d` `pontointerno`), polynomial `c`,
  `M_ij=U* c_j`, `ID` from the Maxima RIM primitive. MATLAB DRM remains
  `method=:drm`. Compare: `scripts/plates/fsdt_vs_matlab.jl`.

### Plate elements share `Element` / GL collocation

- Kirchhoff `PlateMesh` stores `Vector{Element}` (CAD `geo`, collocation
  `index`, Jacobian, length) instead of a private `PlateElement`.
- Collocation is Gauss–Legendre (`Legendre(p)` field, `Equispaced(p)`
  geometry), same split as `format2d`. Mesh BCs are `BC`/`BV` (length `2n`).
- Closed-form CPV/HFP (`integraelemsing`) uses moments of the field
  interpolant at the actual source `ξ0` (matches the old `ξ=±2/3`
  formulae to machine precision for `N` at those nodes).
- Default on-element assembly is Guiggiani; Laurent tails by Richardson
  **per kernel entry** (`P` has mixed `1/r²`, `1/r`, `log`). Fused
  `order_H=-2` leaves `M_n∼log` unsubtracted (~1e-4). Per-entry matches
  closed-form to ~1e-6 on a straight element
  (`scripts/debug/plate_sing_compare.jl`). `singular=:analytic` keeps
  the closed form.
- `solve_large_plate!(; nonlinear=:newton|:picard|:anm)`. `:anm` is
  Cochelin ANM in the load `λ` (cubic series, one Jacobian per step)
  plus Padé of that series, with Taylor fallback if Padé blows up.
  Jacobian of `R(x,λ)` is ForwardDiff (`_ad_jacobian`); Newton and ANM
  share it. Pass `alg=` to use NonlinearSolve instead.

### Contato bulk twin (`dad_Contato_Bulk`)

- Gmsh transfinite count is elements+1 (MATLAB `MALHA`), so 45 contact
  elements have a collocation at \(x=0\).
- `NOS_RES` pins: pad top-centre and specimen bottom-centre \(u_x=0\).
- `dad_contato_bulk` collocation is Gauss–Legendre again (`:legendre`).
  Nodal lumping (`near_factor=2`) then matches full integration:
  50 steps, 15 stick, \(p_0=203.85\) (Hertz 0.20%), \(q_{\max}=0.71\)
  vs Octave \(p_0=203.56\), \(q_{\max}=0.69\). Lumping failed at
  \(ξ=±2/3\) because GL weights were applied at non-GL nodes.
- Optional on-element `singular=:telles` (Contato `calc_gh`). Same
  3-node H,G block as Guiggiani / Octave after n–t rotation.
- Discontinuous elements store CAD `geo` nodes; kernels use Equispaced
  geometry and the field polynomial for `N` (Contato `calc_RNorm` vs
  `calc_fforma_d`).
- Kernel \(n=\mathrm{tan2normal}(dx)\) is flipped to match stored outward
  `dad.Normal` when `_orient_normals_outward!` reversed a face. Far
  lumping already used stored \(n\); near/on-element \(T\) did not, so
  the specimen compressed the wrong way and the 1-stick pad did not wrap.

### Scalar Guiggiani keyword

- Vector (Laplace) `_integrate_singular_guiggiani!` accepts unused `source=`
  so `integrate_element` can pass the collocation index (elasticity already did).

### DIBEM Ricker wavelet (hard-wall acoustics)

- `solve_Newmark` / `solve_Houbolt` accept a time-dependent domain source
  `force` in ``H u - G q = M(\ddot u - f)``.
- `solve_mmm!` `f` may be `f(t)` or a `(n_free, nT)` matrix.
- `filter_mmm_ghosts` drops DIBEM modes whose left eigenvector exploded
  (`‖Φ̃‖_∞` vs the first physical modes). Those ghosts are high-`k` sines
  and were the whole-domain sine on the Ricker centreline.
- Square mesh origin `(x0,y0)` and `mesh_square_hardwall` (all-Neumann
  ``(-L,L)²``). Script: `scripts/transient/ricker_wavelet_dibem.jl`
  (same continuum problem as the Trixi Ricker driver; default stepper MMM
  because raw DIBEM `M` is indefinite).

### Isotropic MTS crack growth (Erdogan–Sih)

- Dual BEM centre-crack mesh takes inclination `α` to the x-axis
  (`β = π/2 - α` to the tensile axis). [`max_tens_circ`](@ref) kinks
  interior tips; [`extend_dual_crack_tip!`](@ref) appends coincident
  type-5 twins; [`propagate_dual_mts!`](@ref) loops solve → SIF →
  extend. Isotropic COD now uses the local ``u^+-u^-`` face (same
  as the anisotropic path) so ``K_{II}`` has the same sign at both
  tips. Compare Erdogan & Sih, *J. Basic Eng.* 85 (1963): dual COD and
  XBEM SIFs (`scripts/crack/erdogan_sih_mts.jl`).
- Anisotropic MTS (Ke, Chen, Ku & Chen, *IJNAMG* 33, 2009): maximise
  Sih–Paris–Irwin hoop ``σ_θ`` in the ligament frame. Degenerate
  ``μ_1≈μ_2`` falls back to Erdogan–Sih. [`propagate_dual_mts!`](@ref)
  accepts `AnisotropicElasticity`. Compare Gandhi / Sollero Table V
  (`scripts/crack/ke2008_aniso.jl`). CSTBD marble paths (Figs. 18–24)
  in `scripts/crack/ke2008_cstbd.jl`.

### SST factors inside Guiggiani

- On-element assembly is always [`guiggiani_GH`](@ref). Cordeiro SST
  tensors (`_aniso_sst_star`) are the Laurent coefficients when they
  exist; otherwise Richardson. [`sst_GH`](@ref) is that path with
  orders `(-1,-2)`. Twin sign from ``n_ξ·n_{el}``.

### Anisotropic dual BEM / XBEM

- `assemble_dual_elasticity!` / `solve_dual!` accept
  `AnisotropicElasticity` (Lekhnitskii `fundamental` /
  `fundamental_hyper`). Crack HBIE keeps ``G_{ii}-=I/2`` (+twin), not
  solid ``c_{ij}=0``.
- Sih–Paris–Irwin tip field `sih_M_local` in the ligament frame:
  ``μ'=(μ\\cosα-\\sinα)/(\\cosα+μ\\sinα)`` via `lekhnitskii_rotate`.
  COD uses ``K=L^{-1}Δu`` with ``L=M(π)-M(-π)``. Degenerate ``μ_1≈μ_2``
  falls back to Williams.
- XBEM columns / tying / `solve_xbem!` use that field. Extra HBIE
  columns stay Guiggiani ``Sφ∼1/r`` (not SST). Compare aligned
  Griffith ``K_I=σ√(πα)`` and off-axis mixed COD in `test/crack.jl`.
- Dual-BEM coincident twins: closed-form / SST leading tensors of
  ``T^h=n_ξ·S`` were derived with the element normal; the twin has
  ``n_ξ=-n``. Guiggiani / SST now flip those tensors when a kernel
  probe disagrees (isotropic KI ~1.5% on the coarse Griffith mesh).
- Hattori, Alatawi & Trevelyan (IJNME 2016) §§5.2–5.3:
  `scripts/crack/hattori2016_52_53.jl`. Double-edge mesh
  (`mesh_double_edge_crack`); Williams tips are valence-1 crack
  ends (mouths dropped). Right-crack twins follow 180° rotation
  (off-axis fibres are not reflection-symmetric). Dual HBIE never
  1-point-lumps ``1/r^2``; SST/Guiggiani twin leading tensors flip
  on ``n_ξ·n_{el}<0``. Off-axis HBIE needs ``n_{pg}≳16`` (SST remainder);
  with 20 points §5.3 at 40°/60° matches Direct SIF to ~1%.

### Multi-region elasticity (perfect interface)

- `assemble_multiregion` / `solve_multiregion!` for 2-D
  `Elasticity` / `AnisotropicElasticity`: ``u_a=u_b``, ``t_a+t_b=0``.
  Compare Cordeiro 2015 §7.3 (`scripts/debug/viga_7_3.jl`).


### Granados–Gallego near-field maps

- Off-element `transform` can use the complex pole ``ξ_0=ζ_0+iη_0`` of
  ``z(ξ)=x_1+ix_2`` (Granados & Gallego, EABE 189, 2026) instead of
  Euclidean ``d/L``. `dad.nearfield=:csinh` / `:sinhsinh` / `:tangent` /
  `:p3c` (complete cubic, JACM 12, 2026) / `:tanp3c` (tangent then p3c
  on the tangent ``±π/2`` poles; split at ``B`` when ``|ζ_0|<1``).
  Default is `:tanp3c` (eq. 41 tangent, then p3c on the ``±π/2``
  poles): holds hypersingular interior points to mesh accuracy down to
  ``d/L\\sim10^{-6}``. Maps follow eqs. (40)–(41) piecewise: ``η_0≠0``
  uses sinh (40) or tangent (41); collinear real poles (``|ζ_0|>1`` and
  ``η_0\\lesssim10^{-8}``, including atan-saturated ``η_0\\sim10^{-12}``)
  use Möbius ``ξ=ζ_0-A/(ξ̃-B)`` (41) or the exponential (40). `:csinh`
  / `:sinhsinh` remain available for ``1/r``. Anisotropic elasticity
  defaults to `:zsinh`. Override with `dad.nearfield`.
  Compare `scripts/debug/granados_vs_sinh.jl` and
  `scripts/debug/tanp3c_stress.jl`.
  Off-element polynomial subtraction (`nearfield=:interp`) was removed.

- 3-D `transform_surface` honours `dad.nearfield=:plain/:tensor/:polar/:auto`
  / `:tanp3c` / `:tangent`. Polar sinh skips empty sectors (edge/vertex
  projections) and uses ``n_ρ=n_θ\\ge n``. `:tanp3c` is polar + Granados
  eq. (41) on each ray (pole at ``ρ=0``): essentially exact on ``z/R^3``,
  weaker on ``1/R`` (radial sinh). `:auto` still polar+sinh when
  ``d/L<0.05``. Sweep: `scripts/debug/nearfield_approach_3d.jl`.



### Lekhnitskii characteristic polynomial

- Cordeiro P3 geometry is ``R_i=0.6\,\mathrm{m}``, ``R_o=0.9\,\mathrm{m}``
  (Fig. 7.8 path 94.25 / 124.25 / 265.62 / 295.62 cm), load ``t_y=-P``.
  MAT1 Poisson is Fernández/Cordeiro 2015 ``ν_{yx}=0.344`` (``D_{12}=-ν/E_x``).
  Plane-strain ``D_{13}=-ν_{31}/E_1`` so the published ``ν_{zx}=0.40`` stays PD.

- `lekhnitskii_params` now solves Cordeiro/Lekhnitskii eq. (10)
  ``a_{11}μ^4-2a_{16}μ^3+(2a_{12}+a_{66})μ^2-2a_{26}μ+a_{22}=0``
  (``z=x_1+μ x_2``). The previous companion matrix used the reciprocal
  quartic ``a_{22}μ^4+⋯+a_{11}``, which is invisible for isotropy
  (``μ=i``) and wrong for MAT1. Compare P3 HBIE on the on-circle mesh.

### Guiggiani / Lekhnitskii (Cordeiro & Leonel 2020)

- Closed-form Laurent coefficients for 2-D anisotropic elasticity CBIE
  (`AnisotropicElasticity`): log tensor of `U` is `2 Re(A q̄ᵀ)`; traction
  `F₋₁ = s · 2 Re(A ḡᵀ) φ` (Cordeiro & Leonel, EABE 119, 2020).
  `lekhnitskii_engineering` builds the 3×3 compliance including
  `η₁₂,₁`, `η₁₂,₂` and optional plane-strain reduction.
  `applyBC` / `solve` accept `AnisotropicElasticity`. Traction BIE via
  `fundamental_hyper` + `H_G_hyper` (orders `(-1,-2)`, `G_{ii} -= I/2`,
  rigid-body row-sum so ``c_{ij}=0``, paper Table 7). HBIE Laurent
  coefficients are the paper’s SST tensors (eqs. 34–37:
  ``U^h∼1/((ξ-ξ_0)J_0)``, ``T^h∼1/((ξ-ξ_0)^2 J_0^2)`` with frozen
  ``J_0,n`` and ``φ^{**}=φ_0+φ_{,ξ}(ξ-ξ_0)``). On-element HBIE is
  [`guiggiani_GH`](@ref) with those tensors as Laurent coefficients
  (``ξ-a=sρ``); [`sst_GH`](@ref) is that path with orders `(-1,-2)`.
  Richardson at parent ``h=10^{-3}`` recovers ``F_{-2}`` and
  ``F_{-1}`` on straight elements; the earlier HBIE failure was
  1-point far lumping of ``1/r^2``. Default `npg=50`. Compare script
  `scripts/elasticity/cordeiro2020_aniso.jl` (CBIE and HBIE, paper
  problems 1–3).
  2-D Kelvin / Laplace HBIE now have the same closed-form ``F_{-2}``:
  ``T^h=μ I/(2π(1-ν)R^2)`` so ``F_{-2}=μ I φ/(2π(1-ν)J)``; Laplace
  ``∂T/∂n_ξ→-1/(2π R^2)``. 3-D Kelvin `fundamental_hyper` contracts
  `D,S` with the collocation normal.

### HARA removed

- Hierarchical Adaptive Randomized Approximation (`hara`, `hara_h2`,
  `hara_product`, `assemble_h2_fmm`) is gone. H² assembly stays
  `assemble_h2` (proxy / NCA / NNCA). FMM still builds HSS via
  `assemble_hss_fmm`. Generic matvec wrappers
  (`AbstractMatvecSampler`, `FunctionSampler`, `KernelMatvecSampler`)
  remain in `src/Hmat/arith/matvec_sampler.jl`.

### 3-D topology (density-only)

- Laplace / isotropic elasticity topological derivatives on 3-D BEM:
  insulating spherical hole `DT = (3/2) k |∇T|²`; elasticity
  `isotropic_3d_DT` (Novotny spherical cavity). `boundary_grad_T` /
  `interior_grad_T` and Kelvin `fundamental_stress` are 3-D.
  `interior_grad_T` now returns the physical `∇T` (the BIE contraction
  was sign-flipped in 2-D as well; `DT = c k |∇T|²` is unchanged).
- Heterogeneous Laplace `∇·(K∇u)=0` and heterogeneous elasticity
  `solve_heterogeneous!` accept 3-D meshes (`_grad_field` is
  dimension-generic).
- DIBEM-SIMP / DT-ρ on a **fixed** 3-D surface mesh:
  `solve_dibem_simp!(dad::BEMdata, opt)`. Constructors `heat_cube_3d`,
  `cantilever_cube_3d`; VTK `export_vtk_density`.
  `opt.cut` in 3-D extracts a volume-matched iso-surface
  (`cut_density_3d!`, marching tetrahedra). Closed interior components
  become traction-free cavities and the BEM mesh is rebuilt
  (`bemdata_from_iso`: original outer surface + collapsed linear
  triangles), the 3-D analogue of `bemdata_from_loops`. Open iso-surfaces
  that hit Γ are skipped. Pacheco node motion stays 2-D.
- Tests in `test/topology.jl`; demo `scripts/topology/dibem_simp_3d.jl`.

### Laplace FMM apply (Flatiron FMM2D/FMM3D)

- Cached 2D/3D Laplace apply now threads P2M, P2P, and L2P **by target
  box** (OpenMP-style, `julia -t auto`). 3D Gumerov M2L is also threaded;
  2D binomial M2L stays serial (concurrent `m2l!` disagrees with dense on
  Julia 1.13 even with per-target temps). FMM2D/FMM3D still rebuild the
  tree every call; they were matching us because Fortran OpenMP used every
  core while our cached apply was serial.
- 3D leaf size follows Flatiron `lndiv` (200 at 1e-6, **400 at 1e-8**).
  The N=10k gap vs FMM3D was ~10⁵ Gumerov y-rotations on a depth-3 octree,
  not Julia P2P: a tight `1/√r²` kernel is ~0.8 Gpair/s and **faster**
  than FMM3D `l3ddir`. `nmax=200` drops apply ~270 ms → ~23 ms.
- Default FMM tree is `DyadicSplitter` (quadtree / octree), matching
  Flatiron `pts_tree2d` / `pts_tree3d`. Pass `GeometricSplitter` or an
  existing `tree=` to keep the old binary longest-axis clustering (H² sharing).
- Charge-only P2P uses SoA coordinates and `log(r²)/2` (2D) / `1/√r²` (3D);
  skip-self only on the same leaf.
- Gumerov Wigner `T(θ)` is prefilled at plan build (cache cap 4096).
- Plane-wave M2L is **not** ported: FMM3D’s diagonal form needs a cubic octree
  with 6-face interaction lists and Norman quadratures for same-size boxes.
  We keep Gumerov O(p³) rotate-and-z-translate, which is valid on a general
  `ClusterTree`.

### JuMP first-order shape step (vs DT stand-in)

- `PachecoOptions.motion = :jump` solves the same DT / area subproblem with
  JuMP. `jump_mode=:linear` (default) is the linearized MMFD LP
  (`min −s·ℓ∘vn` s.t. `ℓ·vn ≤ ΔA`, HiGHS). `jump_mode=:mma` is NLopt
  `LD_MMA` with BEM evaluations (`jump_maxeval`). Compare against the
  closed-form stand-in in `scripts/topology/dt_standin_compare.jl`.

### DT stand-in motion (vs Pacheco `move_boundary!`)

- `PachecoOptions.motion = :standin` uses the Portela first-order step on
  **topological derivative**: `vn ∝ (DT − λ)`, both ways (or
  `standin_inward` for recede-only), area projected to `Atarget`. Default
  `:quantile` is unchanged (inward, lowest-DT quantile).
  `move_boundary_standin!`. Compare script
  `scripts/topology/dt_standin_compare.jl`.

### Portela 2012 dual-BEM shape design (Laplace / elasticity)

- Explicit-boundary **shape** optimizer (`solve_portela!`): traction-free /
  insulated `Γ_d`, material-derivative `δΨ0 = −∫ W vn dΓ` with hoop energy
  `W = σ_I²/(2E')` (elasticity) or `½ k|∇T|²` (Laplace). Area via `∫ vn`.
  Velocity `vn ∝ (W − λ)` both ways (constant-`W` free boundary); no
  nucleation. Dual collocation: HBIE on `Γ_d`, CBIE elsewhere
  (`state=:dual`, fallback `:cbie`). Radial (`param=:radial`) or vertex-normal
  design variables. MMFD-lite line search; anti-fold clamp reused from
  Pacheco. Constructors `portela_plate_hole` (quarter plate, Banichuk hole)
  and `portela_heat_hole`. Compare script
  `scripts/topology/portela_compare.jl`; paper check
  `scripts/topology/portela_plate_hole.jl`.

### DIBEM-SIMP topology (Laplace / elasticity)

- Nodal density `ρ` at DIBEM collocation/internal points.
  Laplace: `K = Kmin + (K0-Kmin) ρ^p`. Elasticity: `E = Emin + (E0-Emin) ρ^p`
  with `Emin = Kmin E0`, `ν` fixed; state `solve_heterogeneous!`
  (Picard DIBEM body force `b=(∇ln k)·σ_ref`, Kelvin H,G at solid E0).
  SIMP OC: self-adjoint FGM sensitivity (`|∇T|² ∂K/∂ρ` / `(σ:ε) ∂E/∂ρ` of
  the heterogeneous state), volume of the Heaviside-projected density
  (Xu 2010 / Wang–Lazarov–Sigmund 2011), `p`/`β` continuation. After the
  iso-cut, Pacheco is **shape-only** (`nucleate_every = ∞`).
  Default `n_simp=40` (continuation) and polish `maxiter=40`.
  Node motion clamps the inward step so local triangles cannot invert;
  a self-intersecting pass is reverted (`_design_folded`).
- After `n_simp` OC steps, `cut_low_density!` turns a **volume-matched**
  `ρ` iso (`match_area`: grid solid ≈ `volfrac`) into traction-free holes.
  Pacheco then drives `A` into `[0.97, 1.08] Ap` (`match_volume!`: adaptive
  `pct`/`vmax`, revert if `A` drops below `Ap`).
- `solve_dibem_simp!`, `DibemSimpOptions`. `method=:dt` / `solve_dt_density!`
  builds `ρ` from the topological derivative (`DT = k|∇T|²` or plane-stress
  DT) with a volume threshold, then the same iso-cut + Pacheco polish.
- Tests in `test/topology.jl`; heat: `scripts/topology/dibem_simp_compare.jl`;
  Coelho: `scripts/topology/dibem_simp_elasticity.jl`.

### Guiggiani Laurent coefficients (Laplace / Kelvin)

- On-element CBIE no longer Richardson-extrapolates `F_{-1}, F_{-2}` for
  Laplace and isotropic elasticity. Closed forms from Marczak (2002) /
  Guiggiani (1998) as in CILAMCE 2016 cap-iso live in
  [`laurent_coefficients`](@ref) (geometry method); Richardson is the
  fallback. `guiggiani_GH` / `guiggiani_GH_surface` call that when
  `props, poly, nodes` are passed. Potential flux `F_{-1}=F_{-2}=0`;
  2-D Kelvin traction
  `F_{-1}=-((1-2ν)/(4π(1-ν)))(n_α t_β-n_β t_α)φ`, `F_{-2}=0`; 3-D polar
  `F_{-1}=-((1-2ν)/(8π(1-ν)))(n_α A_β-n_β A_α)φ J/A³`. Hypersingular
  paths keep extrapolation.

### Multi-region Laplace solvers (Kane)

- `solve_multiregion!(prob; strategy=)` for perfect-interface Laplace:
  `:dense` (filled global matrix), `:noncondensing` (blocked zone factors +
  `n_if` Schur; Kane–Saigal 1990), `:condense` (zone condensation onto
  shared `(T_if, q_if)`, size `2 n_if`).
- `multiregion_ndof`. Tests in `test/multiregion.jl`; compare script
  `scripts/laplace/two_regions_strategies.jl`.

### Core cleanup

- Deleted unused `Parallel.jl` (assembly already uses `Threads.@threads`).
- Helmholtz is `LaplaceLike`: dense `assemble!` / `applyBC` / `solve` work
  with complex kernels (`kernel_eltype`).
- `format2d` / `format3d` use the refcounted Gmsh session; `polygon_area`
  and `point_in_polygon` are shared Core helpers.
- Geometric properties use Green identities (no radial Gauss of polynomials).
- Scalar DIBEM factored `M` is `ColWeightedOp` (`DibemFactoredOperator` alias).
- Removed `dibem_phs3` / `DIBEM_phs3.jl`. Use `DIBEM(dad; rbf=PHS(3; poly_deg=1))`
  (optional `rim=:full_gauss` on dense Laplace).
- Transient ODE reduction (`reduced_heat_system`, …) lives in `Laplace/Solver.jl`.
- Dropped dead aliases: `integraelem`, `MixedBCOperator`, `applyBC_Hmat`,
  `plot_nodes`, Schur/H-LU name duplicates.
- Folded Gmsh session helpers (`with_gmsh`, `gmsh_ensure!`, `gmsh_release!`)
  into `Input.jl`; removed `GmshSession.jl`.

### Anisotropic Laplace (heat)

- `AnisotropicLaplace(K)` 2D/3D Green's function (`n·K⁻¹ r` flux kernel).
  `OrthotropicLaplace` uses the same kernels (H kernel corrected).
- Strategy 1: `solve_anisotropic_dibem!` — isotropic `Laplace(1)` + DIBEM
  residual ``f^*=(1/k)∇·(ΔK∇u)``. Package identity is
  ``H u - G q = M∇²u`` with ``∇²u=-b/k-f*``, so ``L=H+M A_f``.
  RBF interpolant now includes the monomial Hessian (``poly_deg=2``),
  so quadratic anisotropic fields ``x^2/k_x-y^2/k_y`` are reproduced.
  On domains with holes, derivatives use a local RBF-FD stencil
  (`nlocal=21`) and Neumann ``q_\\mathrm{iso}`` uses shape-function
  ``∇_Γ`` (a global PHS Hessian was O(100) on the plate-with-hole).
- Strategy 3: `solve_anisotropic_ibp!` — one IBP on ``∇·(ΔK∇u)``, then
  DIBEM. Volume ``∫∇Φ·(ΔK∇u)`` uses the Loeffler operator with kernel
  ``∇Φ`` (same ``c`` from `int(rbf,x,xj)`, analytic RIM of ``∇Φ``).
  No Hessian of ``u``. Boundary ``∇_Γ`` uses the field polynomial and
  the stored geometry Jacobian (linear and quadratic elements).
  Plate comparison uses quadratic (`ordem=2`, `tipo=2`).
  `format2d` orients 1-D normals from 2-D cells (quadratic hole curves).
- FEniCS plate-with-hole comparison: `placa_furo_orto`, CSV
  `data/Laplace/fenics_ortho_plate_Tright.csv`.

### 3D anisotropic elasticity, triangles, DRM, VTK

- Ting–Lee / Barnett–Lothe 3D Green’s function (`AnisotropicElasticity3D`,
  cubic/hcp/trigonal/full `C`). Isotropic limit matches Kelvin.
- `format3d` linear triangles (Gmsh type 2) as collapsed 4-node quads.
  `mesh_cube` / `mesh_unit_cube(; recombine=false)`.
- Boundary `σ,ε` from `∇u` + traction: `recover_strain_stress!`.
- ASCII VTK POLYDATA: `export_vtk`.
- 3D DRM mass `û=(R+R³)I` (BESLE) for isotropic and anisotropic elasticity.

### Heterogeneous Laplace (DIBEM, Barcelos–Loeffler)

- `src/Laplace/Heterogeneous.jl`: `∇·(K ∇u)=0` with Poisson FS. DIBEM
  interpolates `[u(X)-u(ξ)] ∇K·∇u*` (EABE 131, 2021; Laplace case 2019).
- Homogeneous `K` recovers `H u = G q`. API: `solve_heterogeneous!`,
  `heterogeneous_K_operator`. Tests in `test/dibem.jl`; script
  `scripts/dibem/heterogeneous_laplace.jl`.

### JSON 1.x

- The old `JSON = "0.21.4"` pin was not a BEM API need (`src/` never imports
  JSON). It was locked by **PlotlyJS → JSExpr**, which still requires JSON 0.21.
- PlotlyJS is removed entirely. Plotting uses Plots.jl with the default **GR**
  backend.
- Compat is now `JSON = "1"`.

### Documentation (didactic reorg, phase 6)

- Documenter binds `modules = [BEM, BEM.Crack, …]` and uses `@docs`.
- Recipes match CI families; EN + pt-BR narrative pages; API signatures
  stay English.

### Tests (didactic reorg, phase 4)

- CI is `test/runtests.jl` plus one file per family (`core.jl`, `laplace.jl`,
  …): smoke + one analytic check.
- Former `test/test_*.jl` suites live under `scripts/debug/legacy_tests/`
  (not CI).

### Submodules (didactic reorg, phase 3)

- Specialist physics is no longer dumped into `using BEM`. Use
  `BEM.Crack`, `BEM.Contact`, `BEM.Plate`, `BEM.Topology`,
  `BEM.MultiRegion`, `BEM.HMatrices`, `BEM.FMM`.
- `BEM.Contact` wraps the five half-space / Cattaneo / mortar modules.
- `BEM.Plate` wraps thin plate, large deflection, buckling, and shells.
- Laplace / elasticity types stay in `BEM` (the name `Laplace` is the
  problem type; it cannot also be a submodule).

### Scripts categorized (didactic reorg, phase 5)

- All runnable scripts live under `scripts/<family>/` (intro, laplace,
  elasticity, dibem, transient, sbm_drm, meshless, crack, contact, plates,
  topology, hmat_fmm, papers, debug, profile). Index: `scripts/README.md`.

### Public API (didactic reorg, phase 2)

- Day-1 meshes: `quadrado`, `quadrado_elasticity`, `placa_com_furo` live in
  `BEM.Examples` and are reexported. First example no longer `include`s `data/`.
- New student names: `assemble!(dad)` (dense) and
  `assemble!(dad; method=:hmatrix)`. `H_G_full_direct` / `H_G_Hmat` remain.

### Package hygiene (didactic reorg, phase 1)

- Default environment no longer pulls `Macchiato`, `RadialBasisFunctions`,
  `WhatsThePoint`, `DifferentialEquations`, CUDA, KernelAbstractions,
  Polyester, Infiltrator, Revise, BenchmarkTools, Documenter, or Test.
  Transients use `OrdinaryDiffEq` (`Rodas5P`). Meshless comparisons run
  under `scripts/meshless/`.
- `using BEM` no longer reexports DrWatson, Infiltrator, TimerOutputs, or
  Plots. Mesh builders that write via `datadir` now `using DrWatson: datadir`.
- Root scratch files (`tmp_drm_plot.jl`, `test_drm_constant.jl`, backup
  Laurent test) moved to `scripts/debug/`.

### XBEM (Andrade & Leonel 2020)

- `src/Crack/XBEM.jl`: shifted first-order Williams tip enrichment on the
  existing dual BEM. Extra DOFs are `(KI, KII)` per tip; crack-face polar
  angle is `±π` from the face normal (coincident twins share `x`).
- Tip tying `u⁺=u⁻` at the geometric tip (Andrade eqs. 43–45): Lagrange in
  ρ on a macro-element (`n_v=9`). Extra HBIE columns use Guiggiani
  (`Sφ ∼ 1/r`). SIFs are read from `solve_xbem!`.
- Edge-crack mesh (`mesh_edge_crack` / `edge_crack_problem`) and
  Civelek–Erdogan 1982 factors (Andrade Table 1).
- Tests: `test/test_xbem.jl` (Griffith ~0.4% vs Feddersen; edge a/W=0.5
  ~0.8% vs Civelek–Erdogan). Heaviside mouth enrichment is not in this
  pass.

### Elastic cracks on BEMdata (dual BEM + diamond slit)

- Kelvin `fundamental_hyper`: contract stress kernels `D,S` with the
  collocation normal (same blocks as the old DualMesh `kelvin_DS`).
- Shared `prepare_crack!` for Laplace (`"5;2"`/`"5;3"`) and elasticity
  (`"5;2;5;2"`/`"5;3;5;3"`). Traction-free rewrite on type-5 faces.
- `assemble_dual_elasticity!` / `solve_dual!` / `sif_cod_dual` on `BEMdata`.
- Finite-width diamond for elasticity (`finite_width_elasticity_problem`),
  sinh on the opposite face, same orientation fix as Laplace.
- Cohesive-contact DBEM (`CohesiveDBEMProblem`) now takes
  `BEMdata{<:Elasticity}`; nodal `G` (`2n×2n`) drops the element-traction map.
- Removed `DualMesh` / `DualCore.jl`.
- Tests: `test/test_elastic_crack.jl`, `test/test_laplace_crack.jl`.

### Topology optimization (Laplace + plane-stress elasticity)

- `src/Topology/`: explicit boundary loops → BEM (`bemdata_from_loops`);
  Gmsh is only used for an optional *initial* mesh (`extract_loops`).
  `BoundarySegment.bc` / `value` have length 1 (Laplace) or 2 (elasticity).
- Laplace DT ``k|\nabla T|^2`` (homogeneous Neumann hole) from boundary
  ``q`` + tangential `Dmat` and interior `fundamental_grad`.
- Plane-stress DT (Novotny / Coelho 2021)
  ``2/(1+ν)\,σ:ε + (3ν-1)/(2(1-ν²))\,(\operatorname{tr}σ)(\operatorname{tr}ε)``
  from boundary traction + tangential strain and interior Kelvin ``D,S``.
  Objective is compliance ``J=\int t\cdot u\,dΓ``.
- Pacheco 2020: inward-normal node motion, DT iso-curve nucleation (iter 1
  and later). Dirichlet and non-zero load patches stay frozen.
  Elasticity also nucleates when ``\min(DT_Γ)\cdot 20 > \min(DT_Ω)``.
- Four heat-conductor examples (`pacheco_problem(1:4)`) and Coelho cases
  3–5 (`coelho_problem(3:5)`: cantilever, clamped–clamped, clamp+roller).
- Level-set: Amstutz (``φ = DT-τ``) and Hamilton–Jacobi. HJ uses
  ``φ_t - v_n|∇φ|=0`` for ``Ω=\{φ>0\}`` (``v_n=DT-λ``), masks the void,
  reinitializes, then shifts ``φ`` to the area target.
- Tests: `test/test_topology.jl`. Compare: `scripts/topology_compare.jl`,
  `scripts/topology_elasticity.jl`.

### Elastodynamics sweep (cell / DRM / DIBEM)

- Analytics copied to `data/elastico/iso/analytical_elastodynamics.jl` with
  Phase 0 tests (`test/test_analytical_elastodynamics.jl`).
- Driver `scripts/elastodynamics_sweep.jl` writes JSON.jl JSONL + JSON.
  Short beams use `L/h = 4`. Internals are cell centroids.

### Particular-solution RBF-BEM (DRM) polynomial source

- `particular_from_coeffs` now adds closed-form particulars for the CPD
  polynomial part of \(f\) (2D: \(r^2/4\), \(x r^2/8\), \(y r^2/8\)).
  Default `PHS(3; poly_deg=1)` was parking constants in \(β\) so \(u_p\) and
  \(q_p\) vanished and Poisson flux RMSE was \(O(1)\).

### Local BEM (compact kernel + DIBEM volume)

- 3D Laplace / Poisson: same compact kernel on `format3d`. Volume RIM on
  all of ``Γ`` with ``Ψ(\min(R,r_i))``; ``Ω ∩ B`` cubature by radial cones
  (no spherical-cap clipping). Cube ``u=|x|²`` (ndiv=2): interior rel
  ``5.5\times 10^{-3}``, flux ``1.1\times 10^{-2}``. Elastic local BEM is
  still 2-D.
- `src/Laplace/LocalBEM.jl`: 2D Poisson / Laplace with the compactly supported
  regularized fundamental solution (``u_i^*=∂_r u_i^*=0`` at ``r=r_i``).
- Domain integrals ``∫_{Ω^i} u`` and ``∫_{Ω^i} f u_i^*`` via DIBEM / compact
  radial integration on ``Γ`` (no cells). Default RBF: ``PHS(3)`` (poly_deg=2).
- Local boundary ``Γ ∩ B(y_i,r_i)`` by clipping each edge to the disk
  (`clip_element_to_ball`); H/G quadrature only on those segments.
- DIBEM volume RIM on ``∂(Ω ∩ B)``: clipped ``Γ`` plus interior circular
  arcs ``Ψ(r_i)Δθ`` (full disk is ``2π Ψ(r_i)``, no far-``Γ`` loop).
- Local average ``∫_{Ω^i} u`` is per-ball CPD (``φ=r^3`` + poly on nodes in
  ``B(y_i,r_i)``; RIM of each neighbour ``φ_j`` on ``∂(Ω^i)``), so Laplace
  linears/quadratics give BIE-accurate flux.
- Source ``M``: default `source=:local` is the same per-ball CPD for
  ``∫ f u_i^*``; `source=:global` keeps DIBEM lumping. Both are cached.
- Poisson source sign from Green: ``A_u u - G q_n = -M f`` (the notes' last
  display had a plus; Laplace \(f=0\) had hidden it).
- API: `assemble_local_bem!`, `solve_local_bem!`; kernels `local_u_star`,
  `local_du_dn`, `radial_integral_local_ustar`.
- Tests: `test/test_local_bem.jl`. Demo: `scripts/local_bem_poisson.jl`.
- 2D isotropic elastic local BEM (`Elasticity/LocalBEM.jl`): compact Kelvin
  with companion so ``U^*=T^*=0`` on ``r=r_i``; Navier extra
  ``α_0∫u+α_2∫r²u+β∫r(r·u)`` via the same local CPD. Patch test.


### SBM-DRM transient diffusion (Kovářík et al. 2017)

- `src/Laplace/SBM_DRM.jl`: singular boundary method + dual reciprocity for
  ``∂t u = κ Δu`` (Poisson path, Euler / Houbolt).
- Chen–Gu OIFs (from `SBM.jl`) for Dirichlet/Neumann diagonals — empirical
  ``u_{ii}`` is unstable when coupled to DRM residuals.
- Default RBF: **PHS2** ``φ = r²\log r``, ``Φ = r⁴/16(\log r - 1/4)``.
- API: `solve_sbm_drm`, `sbm_drm_setup`, `sbm_drm_step!` (`basis=` kwarg).
- Compare script: `scripts/sbm_drm_vs_dibem.jl` (Examples 1–2:
  SBM-DRM vs BEM-DRM vs BEM-DIBEM, all PHS2).
- Tests: `test/test_sbm.jl` (SBM-DRM heat).




### Singular Boundary Method (BEMdata)

- Laplace and 2D elastic SBM take a formatted `BEMdata` only (`sbm_from_bemdata`).
- Collocation is `format2d` GL points; OIFs on the parent BEM element (Guiggiani).
- ``L_m =`` parent-element length / local collocation count (`Length/2` for linear).
- Meshless node/normal constructor removed.
- API: `solve_sbm_laplace`, `solve_sbm_elasticity`, `assemble_sbm!`,
  `origin_intensity_factors!`, `compare_sbm_bem`.
- Tests: `test/test_sbm.jl`, `test/test_sbm_elasticity.jl`.



### Mesh order = element order

- `format2d(...; tipo=nothing)` (default): field degree follows the Gmsh 1-D
  mesh order (`mesh.setOrder` / line element type).
- Explicit `tipo` reorders the mesh with `gmsh.model.mesh.setOrder(tipo)` so
  geometry and collocation degree stay matched.

### Continuous elements removed

- Dropped `continuous=true` / `discretization=:continuous`, Clenshaw–Curtis
  path, and double-node corner handling. Discontinuous Lagrange remains.



### IGA removed

- Removed isogeometric / Bézier path: `format2d(...; discretization=:iga)`,
  `Bezier.jl`, and related tests.


### FMM / HMatrices refactor

- `ClusterTree.node_id` + `nnodes` / `assign_node_ids!` (geometry metadata only).
- `Laplace2DFMMPlan`: optional `tree=`, node-id expansion vectors, cached M2L/P2P lists.
- `build_laplace3d_plan` + `lfmm3d(...; plan=)` for repeated 3D matvecs.
- `lfmm2d` / `rfmm2d` accept `plan=` fast path (real charges, self, pot/grad).
- Kernels use `node_id` (not `objectid`) for leaf maps / KIFMM / dualtree job bins.
- **Layout:** `FMM/{core,kernels,operators}/`, `Hmat/{tree,compress,formats,arith}/`.
- Admissibility: `Hmat/tree/admissibility.jl` (`FMMStrongAdmissibility`, …).
- HSS `_sample_mul` multi-RHS; `FMMKernelMatrix <: AbstractKernelMatrix`.
- Details: `_research/fmm_hmat_refactor_plan.md`. Tests: `test/test_fmm_plan_refactor.jl`.

### Contact / multibody

- Frictional solvers: `:activeset` (default), `:ssn` (Alart–Curnier),
  `:proj_gnm` / `:proj_newton` (RT & Abascal 2013),
  `:gnmls` (RT & Abascal, Comput. Struct. 2010: single Λ, frozen slip,
  quasi-complementarity, Pang line search).
- Contact frame: optional common normal ``n_{AB}=(n_A E_A-n_B E_B)/‖·‖``
  (`common_normal=true`). Default **off** to match Contato MATLAB (geometric
  `n`). Gap default ``‖x_B-x_A‖`` (`gap=:euclidean`, `calc_gap_subreg`).
  Active-set verifies from `x=0` (do not force all-stick).
- Public projections: `project_contact_traction`, `project_contact_multipliers`.
- Split multibody contact out of `SubRegions.jl`:
  - `ContactCommon.jl` — prep, kinematics, verify, scatter, BIE+constraint assemble
  - `ContactActiveSet.jl` — Contato active-set
  - `ContactSSN.jl` — Alart–Curnier SSN
  - `ContactProjectedNewton.jl` — projected GNM / accelerated Newton
  - `ContactFriction.jl` — public API + solver dispatch

### Integration

- Removed Dumont near/singular path (`src/Core/Dumont.jl`) and Lekhnitskii SST.
- 2D policy: **Guiggiani** on-element, **sinh** nearly singular, plain GL far.
- `Integration.jl` cut to assembly spine: dropped `singular` /
  `singular_laurent` / `guiggiani_line` / shape-Laurent / legacy 3D `transform`.
- Fused on-element `guiggiani_GH` (one `fundamental` per sample); Laurent
  orders from `singularity_orders(::Problem)` default `(0,-1)`.
- Tests: `test/test_singular_integration.jl`, `test/test_laurent_singular.jl`.



### FMM / H-matrices

- `assemble_hss_fmm` is a thin façade over `HMatrices.assemble_hss(...; method=:fmm)`.
- Martinsson HSS-from-matvec lives in `src/Hmat/formats/hss.jl` (`_hss_compress_fmm!`).
- Removed legacy in-tree FMM HSS types and tree copies (use `HMatrices` trees).
- `PartialACA` accepts `ExpandedClusterTree` (vector / block DOF kernels).
- Convenience `assemble_hss_fmm(points; kernel=...)` default kernel is
  `:laplace2d` (was `:laplace3d`); supports `:kelvin2d` via `expand_tree`.

### Laplace

- Removed duplicate Laplace-local copies of assembly / BC / analytical helpers
  that already live under `Core/`:
  - `src/Laplace/Analytical.jl` → use `Core/Analytical.jl`
  - `src/Laplace/Assembly_full.jl` → use `Core/Assembly_full.jl`
  - `src/Laplace/Boundary_conditions.jl` → use `Core/Boundary_conditions.jl`
- `Laplace/Assembly_H.jl` and `Laplace/Solver.jl` thinned accordingly.

### Elasticity / thermoelasticity

- `dibem_elasticity!` is defined in `Elasticity/Domain.jl` only (removed duplicate
  from `Thermoelasticity.jl`).

### Half-plane contact

- `fc_inverse_2d` uses mean-removed log kernel (rank-safe).
- Added `solve_line_contact_force` (force-controlled Polonsky–Keer style).

### Docs

- `docs/make.jl` renames a locked `docs/build` out of the way before Documenter
  runs (Windows / OneDrive `EACCES` workaround); cleans trash synchronously.
