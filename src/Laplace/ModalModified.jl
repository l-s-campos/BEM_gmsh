# =============================================================================
# Método Modal Modificado (MMM) — Prodonoff & Zepka (1983)
# Thesis: Áquila Santos, Ch.4 §4.5 (TESE_AQUILA_BANCAdef.pdf)
#
# Non-symmetric BEM mass/stiffness (MECID / DIBEM):
#   D = M̄⁻¹ K̄
#   Right:  (D − ω² I) φ  = 0
#   Left:   (Dᵀ − ω² I) φ̃ = 0
#   Bi-normalisation: φ̃ᵢᵀ φᵢ = 1
#   u = Φ y,   ÿ + Λ y = f̄,   f̄ = Φ̃ᵀ M̄⁻¹ f
# Time integration of modal ODEs: Houbolt (thesis App. A)
# Mode selection (§4.6): Aᵢ = |f̄ᵢ / ωᵢ²| for constant forcing
# Also provides MMC (§4.4): classical modal via Φ⁻¹ for non-symmetric D.
# =============================================================================

export ModalSystem, build_modal_system
export modal_analysis_mmm, modal_analysis_mmc
export solve_mmm!, solve_mmc!
export select_modes_amplitude, mode_amplitudes, mode_relative_amplitudes
export recover_flux_from_displacement
export houbolt_sdoof!, modal_sdoof!, build_modal_sdoof_ode

# ---------------------------------------------------------------------------
# Condensed free-DOF operators (thesis §4.1–4.2)
# ---------------------------------------------------------------------------

"""
    ModalSystem

Condensed free-displacement operators for the scalar wave BEM system

```
M̄ ü + K̄ u = f(t)
```

built from MECID/DIBEM matrices `(H, G, M)` after eliminating unknown fluxes
on Dirichlet nodes (thesis eqs. 4.7–4.12).
"""
struct ModalSystem
    M::Matrix{Float64}          # M̄ (nf × nf)
    K::Matrix{Float64}          # K̄ ≡ H̄ (nf × nf)
    f0::Vector{Float64}         # constant part of f from BCs (nf)
    free::Vector{Int}           # free u DOF indices in full nt vector
    diri::Vector{Int}           # Dirichlet boundary indices (known u, unknown q)
    neum::Vector{Int}           # Neumann boundary indices
    n::Int                      # boundary size
    nt::Int                     # collocation size
    BV::Vector{Float64}         # boundary values (copy)
    BC::Vector{Int}             # boundary codes (copy)
end

"""
    build_modal_system(dad) -> ModalSystem

Assemble condensed mass/stiffness for free displacements.

Requires `H,G` (`H_G_full_direct`) and DIBEM mass `M` (`DIBEM`).

Index sets (thesis):
- free `u`: Neumann boundary + internal collocation nodes
- unknown `q`: Dirichlet boundary nodes
- known `ū`: Dirichlet BV; known `q̄`: Neumann BV
"""
function build_modal_system(dad::BEMdata{<:Laplace})
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    has_cache(dad, :G) || error("call H_G_full_direct(dad) first")
    has_cache(dad, :M) || error("call DIBEM(dad) first")

    n, nt = dad.n, dad.nt
    H = Matrix{Float64}(dad.H)
    G = Matrix{Float64}(dad.G)
    Mass = Matrix{Float64}(dad.M)
    BC = Int.(dad.BC)
    BV = Float64.(dad.BV)

    diri = Int[i for i in 1:n if BC[i] == 0]
    neum = Int[i for i in 1:n if BC[i] == 1]
    free = vcat(neum, collect(n+1:nt))
    nf = length(free)
    nd = length(diri)
    nf >= 1 || error("ModalSystem: no free displacement DOFs")

    # Unknown flux columns = Dirichlet nodes; known flux = Neumann
    # Dynamic BIE: H u − G q = M ü   (static limit H u = G q)
    # Partition free / Dirichlet columns of H,M and G columns diri/neum.

    # Block rows: use all collocation equations, then condense q_d.
    # [H_f  -G_d] [u_f; q_d] = M_f ü_f + G_n q̄ − H_d ū   (ǖ=0 for fixed Dirichlet)
    Hf = H[:, free]                 # nt × nf
    Hd = isempty(diri) ? zeros(nt, 0) : H[:, diri]
    Gd = isempty(diri) ? zeros(nt, 0) : G[:, diri]
    Gn = isempty(neum) ? zeros(nt, 0) : G[:, neum]
    Mf = Mass[:, free]
    Md = isempty(diri) ? zeros(nt, 0) : Mass[:, diri]

    ū = isempty(diri) ? Float64[] : BV[diri]
    q̄ = isempty(neum) ? Float64[] : BV[neum]

    # Static known contribution (constant BCs, ǖ = 0)
    # rhs_known = G_n q̄ − H_d ū
    rhs_known = zeros(nt)
    if !isempty(neum)
        mul!(rhs_known, Gn, q̄, 1.0, 1.0)
    end
    if !isempty(diri)
        mul!(rhs_known, Hd, ū, -1.0, 1.0)
    end

    if nd == 0
        # pure Neumann + internals: square free system on free rows
        # Use free rows only for a square nf×nf system
        rows = free
        K = Hf[rows, :]
        M̄ = Mf[rows, :]
        f0 = rhs_known[rows]
        if tr(K) < 0
            K = -K
            f0 = -f0
        end
        return ModalSystem(M̄, K, f0, free, diri, neum, n, nt, BV, BC)
    end

    # Eliminate q_d using Dirichlet-node equations (rows = diri):
    # H_d_f u_f − G_dd q_d = M_d_f ü_f + (rhs_known)_d
    # ⇒ q_d = G_dd⁻¹ (H_d_f u_f − M_d_f ü_f − rhs_d)
    # Free rows (free):
    # H_f_f u_f − G_fd q_d − M_f_f ü_f = rhs_f
    # substitute q_d:
    # (H_ff − G_fd G_dd⁻¹ H_df) u_f − (M_ff − G_fd G_dd⁻¹ M_df) ü_f
    #   = rhs_f − G_fd G_dd⁻¹ rhs_d
    # ⇒ M̄ ü + K̄ u = f   with K̄ = H̄, M̄ as thesis (4.7)–(4.8) sign for motion (4.11)

    H_df = H[diri, free]
    H_ff = H[free, free]
    G_dd = G[diri, diri]
    G_fd = G[free, diri]
    M_df = Mass[diri, free]
    M_ff = Mass[free, free]
    rhs_d = rhs_known[diri]
    rhs_f = rhs_known[free]

    # Factor G_dd (may be ill-conditioned on coarse meshes — regularise lightly)
    FGdd = lu(G_dd + 1e-14 * I)
    # H̄ = H_ff − G_fd G_dd⁻¹ H_df
    # M̄ = M_ff − G_fd G_dd⁻¹ M_df
    T_H = FGdd \ H_df
    T_M = FGdd \ M_df
    T_r = FGdd \ rhs_d
    K̄ = H_ff - G_fd * T_H
    M̄ = M_ff - G_fd * T_M
    # From free rows after substitution:
    # H̄ u_f − M̄ ü_f = rhs_f − G_fd G_dd⁻¹ rhs_d
    # ⇒ M̄ ü_f = H̄ u_f − (rhs_f − G_fd T_r)   ... wrong sign vs M ü + K u = f
    #
    # Static: H u − G q = 0. Dynamic thesis (3.66)/(4.3): H u − G q = ω² M u
    # with ü = −ω² u ⇒ H u − G q = − M ü  ⇒  M ü + H_cond u = f_cond
    # Careful: (4.3) says H u − G q = ω² M u and ü = −ω² u so H u − G q = −M ü
    # ⇒ M ü + (condensed H) u = condensed f.
    #
    # Free row: H_ff u − G_fd q_d − … = − M_ff ü + …   if RHS is −M ü
    # Standard in this codebase Houbolt uses M ü + A u = b with A from H,
    # i.e. dynamic form M ü + H u − G q = known. That implies H u − G q = −M ü + known
    # which matches ω² form if M_code = M_thesis... 
    # Existing Houbolt: A = H+2M/dt², consistent with M ü + H u = rhs.
    # So: M̄ ü + K̄ u = f0  with K̄ = H̄, f0 = rhs_f − G_fd * T_r
    # (no sign flip on M̄ relative to Mass blocks)

    f0 = rhs_f - G_fd * T_r
    # BIE form in this codebase is H u − G q + M ü = known, which condenses to
    #   M̄ ü + H̄ u = f̂  with H̄ often *negative*-definite on free DOFs (diag ≈ −1
    # for internal collocation). The physical modal stiffness is K = −H̄ so that
    #   M̄ ü + K u = f ,  K φ = ω² M̄ φ ,  ω² > 0.
    if tr(K̄) < 0
        K̄ = -K̄
        f0 = -f0
    end
    return ModalSystem(M̄, K̄, f0, free, diri, neum, n, nt, BV, BC)
end

# ---------------------------------------------------------------------------
# Eigenanalysis: MMM and MMC
# ---------------------------------------------------------------------------

"""Result of a modal analysis (MMM or MMC)."""
struct ModalBasis
    method::Symbol                 # :mmm or :mmc
    ω²::Vector{Float64}            # eigenvalues (sorted ascending Re)
    ω::Vector{Float64}             # √max(ω²,0) natural frequencies
    Φ::Matrix{Float64}             # right modes (nf × nm) columns
    Φ̃::Matrix{Float64}             # left modes (nf × nm); MMC: Φ⁻ᵀ-like via inv
    Λ::Diagonal{Float64, Vector{Float64}}
    D::Matrix{Float64}             # dynamic matrix M̄⁻¹ K̄
    M::Matrix{Float64}
    K::Matrix{Float64}
    free::Vector{Int}
    f0::Vector{Float64}
    keep::Vector{Int}              # indices into full eigen spectrum used
end

"""
    modal_analysis_mmm(sys::ModalSystem; nmodes=nothing, sortby=:freq) -> ModalBasis

Método Modal Modificado (thesis §4.5):

1. `D = M̄ \\ K̄`
2. Right eigenpairs of `D`, left eigenpairs of `D'`
3. Match left↔right by eigenvalue, bi-normalise `φ̃ᵀ φ = 1`
4. `Φ̃ᵀ D Φ = Λ`, `Φ̃ᵀ Φ = I`
"""
function modal_analysis_mmm(sys::ModalSystem; nmodes=nothing, sortby::Symbol=:freq)
    M̄, K̄ = sys.M, sys.K
    nf = size(M̄, 1)
    # Dynamic matrix D = M̄⁻¹ K̄  (4.31)
    D = M̄ \ K̄

    # Full eigen-decompositions (non-symmetric)
    er = eigen(D)
    el = eigen(D')                          # left eigenvectors as right of Dᵀ

    λr = er.values
    λl = el.values
    Vr = er.vectors
    Vl = el.vectors

    # Keep nearly-real positive eigenvalues (discard large Im / negative Re)
    real_ok = findall(i -> abs(imag(λr[i])) < 1e-8 * max(1.0, abs(real(λr[i]))) &&
                          real(λr[i]) > -1e-10, eachindex(λr))
    isempty(real_ok) && error("MMM: no usable real eigenvalues")

    # sort by frequency
    perm = sort(real_ok; by = i -> real(λr[i]))
    nm_all = length(perm)
    nm = nmodes === nothing ? nm_all : min(Int(nmodes), nm_all)
    keep = perm[1:nm]

    Φ = zeros(nf, nm)
    Φ̃ = zeros(nf, nm)
    ω² = zeros(nm)

    for (k, ir) in enumerate(keep)
        λ = real(λr[ir])
        ω²[k] = λ
        φ = real.(Vr[:, ir])
        # match left eigenvector: closest eigenvalue on Dᵀ
        il = argmin(j -> abs(λl[j] - λr[ir]), eachindex(λl))
        φt = real.(Vl[:, il])
        # bi-normalise φ̃ᵀ φ = 1  (4.46)
        s = dot(φt, φ)
        if abs(s) < 1e-14
            # fallback: normalise φ and set φ̃ = φ / (φᵀφ)  (loses bi-orthogonality)
            nrm = norm(φ)
            φ ./= nrm + eps()
            φt .= φ
            s = 1.0
        else
            # scale left so φ̃ᵀ φ = 1; keep right with unit Euclidean as optional
            φt ./= s
        end
        Φ[:, k] = φ
        Φ̃[:, k] = φt
    end

    # Re-bi-orthogonalise via Φ̃ᵀ Φ scaling (column-wise already ~I)
    Gbi = Φ̃' * Φ
    for k in 1:nm
        d = Gbi[k, k]
        if abs(d) > 1e-14
            Φ̃[:, k] ./= d
        end
    end

    return ModalBasis(:mmm, ω², sqrt.(max.(ω², 0.0)), Φ, Φ̃,
        Diagonal(ω²), D, copy(M̄), copy(K̄), copy(sys.free), copy(sys.f0), keep)
end

"""
    modal_analysis_mmc(sys::ModalSystem; nmodes=nothing) -> ModalBasis

Método Modal Clássico for non-symmetric matrices (thesis §4.4):

`u = Φ y`, premultiply by `Φ⁻¹`:  `ÿ + Λ y = Φ⁻¹ M̄⁻¹ f`.

Stored as `Φ̃ := Φ⁻ᵀ` so that `f̄ = Φ̃ᵀ M̄⁻¹ f` matches the MMM residual API
(`Φ⁻¹ = Φ̃ᵀ` when `Φ̃ = Φ⁻ᵀ`).
"""
function modal_analysis_mmc(sys::ModalSystem; nmodes=nothing)
    M̄, K̄ = sys.M, sys.K
    nf = size(M̄, 1)
    D = M̄ \ K̄
    er = eigen(D)
    λ = er.values
    V = er.vectors

    real_ok = findall(i -> abs(imag(λ[i])) < 1e-8 * max(1.0, abs(real(λ[i]))) &&
                          real(λ[i]) > -1e-10, eachindex(λ))
    isempty(real_ok) && error("MMC: no usable real eigenvalues")
    perm = sort(real_ok; by = i -> real(λ[i]))
    nm_all = length(perm)
    nm = nmodes === nothing ? nm_all : min(Int(nmodes), nm_all)
    keep = perm[1:nm]

    Φc = real.(V[:, keep])
    ω² = real.(λ[keep])
    # Reduced modal basis: left inverse Φ⁺ such that Φ⁺ Φ = I (nmodes×nmodes).
    # Store Φ̃ with Φ̃ᵀ := Φ⁺ so f̄ = Φ̃ᵀ M̄⁻¹ f matches the MMM API.
    Φplus = pinv(Φc)             # nm × nf
    Φ̃ = Matrix(Φplus')          # nf × nm ,  Φ̃ᵀ = Φplus

    return ModalBasis(:mmc, ω², sqrt.(max.(ω², 0.0)), Φc, Φ̃,
        Diagonal(ω²), D, copy(M̄), copy(K̄), copy(sys.free), copy(sys.f0), keep)
end

# ---------------------------------------------------------------------------
# Mode selection (§4.6)
# ---------------------------------------------------------------------------

"""
    mode_amplitudes(basis::ModalBasis; f=basis.f0) -> Vector

Static modal amplitudes `Aᵢ = |f̄ᵢ / ωᵢ²|` (thesis 4.54), valid for
**constant** forcing. `f̄ = Φ̃ᵀ M̄⁻¹ f`.
"""
function mode_amplitudes(basis::ModalBasis; f::AbstractVector=basis.f0)
    length(f) == size(basis.M, 1) || throw(DimensionMismatch("f"))
    f̄ = basis.Φ̃' * (basis.M \ collect(Float64, f))
    A = similar(basis.ω²)
    @inbounds for i in eachindex(A)
        ω2 = basis.ω²[i]
        A[i] = abs(ω2) < 1e-30 ? abs(f̄[i]) : abs(f̄[i] / ω2)
    end
    return A
end

"""Relative amplitude `aᵢ = Aᵢ/A₁ × 100%` (4.55)."""
function mode_relative_amplitudes(A::AbstractVector{<:Real})
    isempty(A) && return Float64[]
    A1 = A[1]
    return A1 == 0 ? fill(0.0, length(A)) : 100 .* A ./ A1
end

"""
    select_modes_amplitude(basis; nkeep, f) -> Vector{Int}

Indices (into `basis` columns) of the `nkeep` largest-amplitude modes.
"""
function select_modes_amplitude(basis::ModalBasis; nkeep::Int=10,
                                f::AbstractVector=basis.f0)
    A = mode_amplitudes(basis; f=f)
    nkeep = min(nkeep, length(A))
    return partialsortperm(A, 1:nkeep; rev=true)
end

function _subset_basis(basis::ModalBasis, idx::AbstractVector{Int})
    Φ = basis.Φ[:, idx]
    Φ̃ = basis.Φ̃[:, idx]
    ω² = basis.ω²[idx]
    # re-bi-normalise subset
    for k in 1:length(idx)
        s = dot(Φ̃[:, k], Φ[:, k])
        abs(s) > 1e-14 && (Φ̃[:, k] ./= s)
    end
    return ModalBasis(basis.method, ω², sqrt.(max.(ω², 0.0)), Φ, Φ̃,
        Diagonal(ω²), basis.D, basis.M, basis.K, basis.free, basis.f0,
        basis.keep[idx])
end

# ---------------------------------------------------------------------------
# Modal SDOF via DifferentialEquations.jl  (ÿ + ω² y = f̄)
# ---------------------------------------------------------------------------

"""Evaluate modal force at continuous time `t` (seconds from start)."""
function _fbar_at_t(f̄::Real, t::Real, Δt::Real)
    return float(f̄)
end
function _fbar_at_t(f̄::AbstractVector, t::Real, Δt::Real)
    # piecewise-linear on the output grid tₖ = (k-1)Δt
    n = length(f̄)
    n == 0 && return 0.0
    n == 1 && return float(f̄[1])
    Δt > 0 || return float(f̄[1])
    ξ = t / Δt
    if ξ <= 0
        return float(f̄[1])
    elseif ξ >= n - 1
        return float(f̄[n])
    end
    i = floor(Int, ξ) + 1          # left sample index (1-based)
    α = ξ - (i - 1)
    return (1 - α) * float(f̄[i]) + α * float(f̄[i + 1])
end

"""In-place RHS for SecondOrderODEProblem: ÿ = f̄(t) − ω² y."""
function _modal_sdoof_rhs!(ddu, du, u, p, t)
    ddu[1] = _fbar_at_t(p.f̄, t, p.Δt) - p.ω2 * u[1]
    return nothing
end

"""
    build_modal_sdoof_ode(y0, dy0, ω², f̄, tspan; Δt=0.0) -> SecondOrderODEProblem

Build `ÿ + ω² y = f̄(t)` as a `SecondOrderODEProblem` (DifferentialEquations.jl).
`f̄` is a scalar (constant load) or a vector sampled every `Δt` on the save grid.
"""
function build_modal_sdoof_ode(y0::Real, dy0::Real, ω²::Real, f̄, tspan;
        Δt::Real=0.0)
    p = (ω2=float(ω²), f̄=f̄, Δt=float(Δt))
    u0 = [float(y0)]
    du0 = [float(dy0)]
    return SecondOrderODEProblem(_modal_sdoof_rhs!, du0, u0, tspan, p)
end

"""
    modal_sdoof!(y, f̄, ω², Δt; dy0=0, alg=Tsit5(), abstol, reltol)

Integrate the modal oscillator `ÿ + ω² y = f̄(t)` with **DifferentialEquations.jl**
(`SecondOrderODEProblem`).

`y` (length `nT`) is filled at times `0:Δt:(nT-1)Δt`; `y[1]` is the IC for `y`,
`dy0` is the IC for `ẏ`.  `f̄` is constant or length-`nT` samples.

Returns `y`.  Also available as [`houbolt_sdoof!`](@ref) (legacy name).
"""
function modal_sdoof!(y::AbstractVector{<:Real}, f̄, ω²::Real, Δt::Real;
        dy0::Real=0.0,
        alg=Tsit5(),
        abstol=1e-8,
        reltol=1e-8,
        kwargs...)
    nT = length(y)
    nT >= 2 || throw(ArgumentError("need nT ≥ 2"))
    Δt > 0 || throw(ArgumentError("Δt > 0"))
    tf = (nT - 1) * Δt
    tspan = (0.0, tf)
    tsave = range(0.0, tf; length=nT)

    y0 = float(y[1])
    prob = build_modal_sdoof_ode(y0, dy0, ω², f̄, tspan; Δt=Δt)
    sol = DifferentialEquations.solve(prob, alg;
        abstol=abstol, reltol=reltol,
        saveat=tsave,
        dense=false,
        kwargs...)

    # SecondOrderODESolution: sol(t) is ArrayPartition (dy, y) or similar
    @inbounds for i in 1:nT
        yi = _modal_position_at(sol, tsave[i])
        y[i] = yi
    end
    return y
end

"""Legacy name — now integrates with DifferentialEquations (see [`modal_sdoof!`](@ref))."""
const houbolt_sdoof! = modal_sdoof!

function _modal_position_at(sol, t)
    val = sol(t)
    # SecondOrderODEProblem → ArrayPartition(du, u) with u = [y]
    if hasproperty(val, :x)
        return float(val.x[2][1])
    elseif val isa Tuple
        return float(val[2][1])
    elseif val isa AbstractVector
        return float(length(val) >= 2 ? val[end] : val[1])
    else
        return float(val)
    end
end

"""
    solve_mmm!(dad, Δt, tf; kwargs...) -> (U, t, basis)

Transient response by **Método Modal Modificado** (thesis §4.5).

# Keywords
- `nmodes`: number of modes after sorting by frequency (default: all usable)
- `select=:amplitude` or `:freq` — mode selection (§4.6); `:amplitude` keeps
  `nkeep` largest `Aᵢ` among computed modes
- `nkeep`: modes retained if `select=:amplitude` (default `nmodes`)
- `u0`, `du0`: full-field ICs (length `nt`); default zero
- `f`: optional free-DOF force override (default condensed BC force `sys.f0`)
- `method=:mmm` or `:mmc`
- `basis`: precomputed `ModalBasis` (skip eigenanalysis)
"""
function solve_mmm!(dad::BEMdata{<:Laplace}, Δt::Real, tf::Real;
        nmodes=nothing, select::Symbol=:freq, nkeep=nothing,
        u0=nothing, du0=nothing, f=nothing,
        method::Symbol=:mmm, basis::Union{Nothing,ModalBasis}=nothing,
        verbose::Bool=false,
        alg=Tsit5(), abstol=1e-8, reltol=1e-8)

    sys = build_modal_system(dad)
    if basis === nothing
        basis = method === :mmc ? modal_analysis_mmc(sys; nmodes=nmodes) :
                                  modal_analysis_mmm(sys; nmodes=nmodes)
    end
    if select === :amplitude
        nk = nkeep === nothing ? size(basis.Φ, 2) : Int(nkeep)
        idx = select_modes_amplitude(basis; nkeep=nk, f = f === nothing ? sys.f0 : f)
        basis = _subset_basis(basis, idx)
    elseif select !== :freq
        throw(ArgumentError("select must be :freq or :amplitude"))
    end

    nf = size(basis.M, 1)
    nm = size(basis.Φ, 2)
    t = collect(0:Δt:tf)
    nT = length(t)

    # force on free DOFs
    ff = f === nothing ? copy(sys.f0) : collect(Float64, f)
    length(ff) == nf || throw(DimensionMismatch("f free-DOF length"))

    # f̄ = Φ̃ᵀ M̄⁻¹ f   (4.52)
    Minvf = basis.M \ ff
    f̄ = basis.Φ̃' * Minvf          # nm

    # ICs → modal  (4.39): y0 = Φ̃ᵀ u0  (MMM bi-ortho) / Φ⁻¹ u0 (MMC via Φ̃ᵀ)
    u0_full = u0 === nothing ? zeros(dad.nt) : collect(Float64, u0)
    du0_full = du0 === nothing ? zeros(dad.nt) : collect(Float64, du0)
    length(u0_full) == dad.nt || throw(DimensionMismatch("u0"))
    # pin Dirichlet in IC
    @inbounds for i in 1:dad.n
        if dad.BC[i] == 0
            u0_full[i] = dad.BV[i]
            du0_full[i] = 0.0
        end
    end
    u0f = u0_full[sys.free]
    du0f = du0_full[sys.free]
    y0 = basis.Φ̃' * u0f
    dy0v = basis.Φ̃' * du0f

    # integrate each mode with DifferentialEquations.jl
    Y = zeros(nm, nT)
    @inbounds for k in 1:nm
        yk = view(Y, k, :)
        yk[1] = y0[k]
        modal_sdoof!(yk, f̄[k], basis.ω²[k], Δt;
            dy0=dy0v[k], alg=alg, abstol=abstol, reltol=reltol)
    end

    # recover free displacements u_f = Φ y
    Uf = basis.Φ * Y                 # nf × nT
    U = zeros(dad.nt, nT)
    @inbounds for j in 1:nT
        U[sys.free, j] = Uf[:, j]
        for i in 1:dad.n
            if dad.BC[i] == 0
                U[i, j] = dad.BV[i]
            end
        end
    end

    # store on dad
    q = zeros(dad.n, nT)
    for j in 1:nT
        # q unknown on Dirichlet recovered optionally
        split_sol!(dad, view(U, :, j), view(q, :, j))
    end
    set_cache!(dad; T=U, q=q, time=t, modal_basis=basis)

    verbose && @info "solve_mmm!" method=basis.method nm nT Δt tf ω1=basis.ω[1]

    return U, t, basis
end

"""`solve_mmc!` — classical modal (§4.4) via `method=:mmc`."""
solve_mmc!(dad, Δt, tf; kwargs...) = solve_mmm!(dad, Δt, tf; method=:mmc, kwargs...)

# ---------------------------------------------------------------------------
# Optional: recover unknown fluxes from displacements (thesis §4.7 sketch)
# ---------------------------------------------------------------------------

"""
    recover_flux_from_displacement(dad, u_full; ü_full=zeros(...)) -> q_full

Given a full collocation displacement vector (Dirichlet entries = prescribed `ū`),
recover boundary flux `q` from the dynamic BIE

```
H u − G q = M ü
```

by solving the columns associated with unknown `q` (Dirichlet nodes) and
inserting known Neumann fluxes.
"""
function recover_flux_from_displacement(dad::BEMdata{<:Laplace}, u_full::AbstractVector;
        ü_full=nothing)
    has_cache(dad, :H) && has_cache(dad, :G) && has_cache(dad, :M) ||
        error("need H,G,M on dad")
    n, nt = dad.n, dad.nt
    length(u_full) == nt || throw(DimensionMismatch("u_full"))
    ü = ü_full === nothing ? zeros(nt) : collect(Float64, ü_full)

    H = Matrix{Float64}(dad.H)
    G = Matrix{Float64}(dad.G)
    Mass = Matrix{Float64}(dad.M)
    diri = Int[i for i in 1:n if dad.BC[i] == 0]
    neum = Int[i for i in 1:n if dad.BC[i] == 1]

    # H u − G_d q_d − G_n q̄ = M ü  ⇒  G_d q_d = H u − G_n q̄ − M ü
    rhs = H * collect(Float64, u_full) - Mass * ü
    if !isempty(neum)
        rhs .-= G[:, neum] * dad.BV[neum]
    end
    q = zeros(n)
    q[neum] .= dad.BV[neum]
    if !isempty(diri)
        # least-squares / overdetermined: use all rows
        qd = G[:, diri] \ rhs
        q[diri] .= qd
    end
    return q
end
