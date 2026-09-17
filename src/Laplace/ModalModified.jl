# =============================================================================
# Método Modal Modificado (MMM) — Prodonoff & Zepka; Santos Ch.4 §4.5
#
#   M̄ ü + K̄ u = f
#   D = M̄⁻¹ K̄
#   one geev: right and left vectors, bi-normalise φ̃ᵀ φ = 1
#   keep real ω² > 0;  u = Φ y,  ÿ + Λ y = Φ̃ᵀ M̄⁻¹ f
# =============================================================================

export ModalSystem, build_modal_system
export modal_analysis_mmm
export solve_mmm!, solve_mmm_heat!

export select_modes_amplitude, mode_amplitudes, mode_relative_amplitudes
export filter_mmm_ghosts
export recover_flux_from_displacement
export modal_sdoof!, houbolt_sdoof!, build_modal_sdoof_ode
export CondensedPencil, condensed_pencil, mul_Kbar!, mul_Mbar!
export modal_shiftinvert

# ---------------------------------------------------------------------------
# Condensed free-DOF operators
# ---------------------------------------------------------------------------

_neq(dad::BEMdata{<:LaplaceLike}) = (dad.nt, dad.n)
_neq(dad::BEMdata{<:Elasticity}) = (dad.dimension * dad.nt, dad.dimension * dad.n)
_neq(dad::BEMdata{<:AnisotropicElasticity3D}) = (3 * dad.nt, 3 * dad.n)

function _modal_layout(dad::BEMdata{<:LaplaceLike})
    n, nt = dad.n, dad.nt
    BC = Int.(dad.BC)
    diri = Int[i for i in 1:n if BC[i] == 0]
    neum = Int[i for i in 1:n if BC[i] == 1]
    free = vcat(neum, collect((n + 1):nt))
    return n, nt, diri, neum, free
end

function _modal_layout(dad::BEMdata{<:Elasticity})
    nb, ndof = 2 * dad.n, 2 * dad.nt
    BC = Int.(dad.BC)
    diri = Int[i for i in 1:nb if BC[i] == 0]
    neum = Int[i for i in 1:nb if BC[i] == 1]
    free = vcat(neum, collect((nb + 1):ndof))
    return nb, ndof, diri, neum, free
end

function _pin_known!(Ucol, dad)
    @inbounds for i in eachindex(dad.BC)
        dad.BC[i] == 0 && i <= length(Ucol) && (Ucol[i] = dad.BV[i])
    end
    return Ucol
end

_scatter_step!(dad::BEMdata{<:LaplaceLike}, Ucol, qcol) =
    split_sol!(dad, Ucol, qcol)

function _scatter_step!(dad::BEMdata{<:Elasticity}, Ucol, qcol)
    split_sol!(dad, Ucol, view(Ucol, 1:2 * dad.n), qcol)
    return nothing
end



"""Condensed `M̄ ü + K̄ u = f` on free displacements."""
struct ModalSystem
    M::Matrix{Float64}
    K::Matrix{Float64}
    f0::Vector{Float64}
    free::Vector{Int}
    diri::Vector{Int}
    neum::Vector{Int}
    fu::Matrix{Float64}
    fq::Matrix{Float64}
end


"""
    build_modal_system(dad) -> ModalSystem

Condense unknown Dirichlet fluxes/tractions from `H u − G q = M ü`.
Free `u`: Neumann boundary DOFs + internals.
Laplace and 2D elasticity (`2 n_t` DOFs).
"""
function build_modal_system(dad::BEMdata{<:Union{Laplace,Elasticity}})
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    has_cache(dad, :G) || error("call H_G_full_direct(dad) first")
    has_cache(dad, :M) || error("call DIBEM(dad) first")

    nb, ndof, diri, neum, free = _modal_layout(dad)
    H = Matrix{Float64}(dad.H)
    G = Matrix{Float64}(dad.G)
    Mass = Matrix{Float64}(dad.M)
    BV = Float64.(dad.BV)

    nf = length(free)
    nd = length(diri)
    nn = length(neum)
    nf >= 1 || error("ModalSystem: no free displacement DOFs")
    size(H, 1) == ndof || error("H size $(size(H)) ≠ ndof=$ndof")
    size(G, 2) == nb || error("G columns $(size(G, 2)) ≠ nb=$nb")

    function C_mat(A)
        Y = A[free, :]
        if nd > 0
            Y = Y - G[free, diri] * ((G[diri, diri] + 1e-14 * I) \ A[diri, :])
        end
        return Y
    end

    HD = nd == 0 ? zeros(ndof, 0) : H[:, diri]
    GN = nn == 0 ? zeros(ndof, 0) : G[:, neum]
    fu = nd == 0 ? zeros(nf, 0) : C_mat(-HD)
    fq = nn == 0 ? zeros(nf, 0) : C_mat(GN)

    if nd == 0
        K = H[free, free]
        M̄ = Mass[free, free]
    else
        Gdd = G[diri, diri] + 1e-14 * I
        G_fd = G[free, diri]
        K = H[free, free] - G_fd * (Gdd \ H[diri, free])
        M̄ = Mass[free, free] - G_fd * (Gdd \ Mass[diri, free])
    end
    ū = nd == 0 ? Float64[] : BV[diri]
    q̄ = nn == 0 ? Float64[] : BV[neum]
    f0 = fu * ū + fq * q̄
    if tr(K) < 0
        K = -K
        f0 = -f0
        fu = -fu
        fq = -fq
    end
    return ModalSystem(M̄, K, f0, free, diri, neum, fu, fq)
end



# ---------------------------------------------------------------------------
# Eigenanalysis
# ---------------------------------------------------------------------------

"""MMM basis: real positive `ω²`, bi-orthogonal `Φ̃ᵀ Φ = I`."""
struct ModalBasis
    ω²::Vector{Float64}
    ω::Vector{Float64}
    Φ::Matrix{Float64}
    Φ̃::Matrix{Float64}
    M::Matrix{Float64}
    K::Matrix{Float64}
    free::Vector{Int}
    f0::Vector{Float64}
end

"""One LAPACK geev: right and left eigenvectors, already paired."""
function _geev_paired(D::Matrix{Float64})
    wr, wi, VL, VR = LinearAlgebra.LAPACK.geev!('V', 'V', copy(D))
    return wr, wi, VL, VR
end

"""
    modal_analysis_mmm(sys; nmodes=nothing, ωmax=Inf, Amax_ratio=1e3)

Keep nearly-real `ω² > 0` with `ω ≤ ωmax`. Drop DIBEM ghosts whose
static (or IC) amplitude exceeds `Amax_ratio` times the **median** of
the positive amplitudes. A P90 reference left almost every ghost in.
"""

function modal_analysis_mmm(sys::ModalSystem; nmodes=nothing, ωmax::Real=Inf,
        Amax_ratio::Real=1e3)

    D = sys.M \ sys.K
    wr, wi, VL, VR = _geev_paired(D)
    ok = findall(i -> abs(wi[i]) < 1e-8 * max(1.0, abs(wr[i])) && wr[i] > 1e-12,
        eachindex(wr))
    isempty(ok) && error("MMM: no usable real eigenvalues")
    perm = sort(ok; by = i -> wr[i])
    if isfinite(ωmax)
        perm = [i for i in perm if sqrt(max(wr[i], 0.0)) <= float(ωmax)]
        isempty(perm) && error("MMM: no eigenvalues with ω ≤ $ωmax")
    end
    keep = perm
    nf = size(D, 1)
    nm = length(keep)
    Φ = zeros(nf, nm)
    Φ̃ = zeros(nf, nm)
    ω² = zeros(nm)
    @inbounds for (k, i) in enumerate(keep)
        ω²[k] = wr[i]
        φ = VR[:, i]
        φt = VL[:, i]
        nrm = norm(φ, Inf) + eps()
        φ ./= nrm
        φt .*= nrm
        s = dot(φt, φ)
        if abs(s) < 1e-14
            φt .= φ
            s = dot(φt, φ)
        end
        abs(s) > 1e-14 && (φt ./= s)
        Φ[:, k] = φ
        Φ̃[:, k] = φt
    end
    Gbi = Φ̃' * Φ
    @inbounds for k in 1:nm
        d = Gbi[k, k]
        abs(d) > 1e-14 && (Φ̃[:, k] ./= d)
    end
    basis = ModalBasis(ω², sqrt.(max.(ω², 0.0)), Φ, Φ̃,
        copy(sys.M), copy(sys.K), copy(sys.free), copy(sys.f0))
    basis = filter_mmm_ghosts(basis; Amax_ratio=Amax_ratio)

    if nmodes !== nothing
        nm2 = min(Int(nmodes), size(basis.Φ, 2))
        basis = _subset_basis(basis, 1:nm2)
    end
    return basis
end

"""Drop tiny-ω, exploded-left-vector, and huge-amplitude DIBEM ghosts.

`A` is the static ranking `|F̄/Ω²|`, optionally combined with IC
modal amplitudes `|y0|` and `|ẏ0|/Ω` (needed when the load is an
initial velocity, not a traction). The cut is `A ≤ Amax_ratio × median(A)`.

Left eigenvectors of the DIBEM pencil `M⁻¹K` can have `‖Φ̃‖_∞ ∼ 10⁸`
while the first physical modes sit at `∼10⁻²`. Those ghosts look like
high-`k` sines and dominate `Φ̃ᵀ f` for a compact source.
"""
function filter_mmm_ghosts(basis::ModalBasis; f=basis.f0,
        y0=nothing, dy0=nothing, Amax_ratio::Real=1e3)
    nm = length(basis.ω)
    nm == 0 && return basis
    nL = [norm(view(basis.Φ̃, :, i), Inf) for i in 1:nm]
    nLref = median(nL[1:min(8, nm)])
    idxL = findall(i -> nL[i] <= 1e6 * (nLref + eps()), eachindex(nL))
    if length(idxL) < nm
        isempty(idxL) && error("MMM: all left eigenvectors exploded")
        basis = _subset_basis(basis, idxL)
        nm = length(basis.ω)
        nm == 0 && return basis
    end
    A = mode_amplitudes(basis; f=f)
    if y0 !== nothing
        length(y0) == nm || throw(DimensionMismatch("y0"))
        A = max.(A, abs.(y0))
    end
    if dy0 !== nothing
        length(dy0) == nm || throw(DimensionMismatch("dy0"))
        A = max.(A, abs.(dy0) ./ (basis.ω .+ eps()))
    end
    ω = basis.ω
    ωpos = [w for w in ω if w > 0]
    if !isempty(ωpos)
        ωcut = 1e-3 * median(ωpos)
        idxω = findall(i -> ω[i] > ωcut, eachindex(ω))
        isempty(idxω) || (basis = _subset_basis(basis, idxω); A = A[idxω]; ω = basis.ω)
    end
    aok = [a for a in A if isfinite(a) && a > 0]
    if !isempty(aok) && isfinite(Amax_ratio) && Amax_ratio > 0
        nref = min(8, length(A))
        Aref = maximum(A[i] for i in 1:nref if isfinite(A[i]); init=eps())
        idx = findall(i -> isfinite(A[i]) && A[i] <= float(Amax_ratio) * (Aref + eps()),
            eachindex(A))
        isempty(idx) && error("MMM: all modes exceed Amax_ratio=$(Amax_ratio)")
        basis = _subset_basis(basis, idx)
    end
    return basis
end

# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

"""
    mode_amplitudes(basis; f=basis.f0, ω=0)

``A_i = |F̄_i / (Ω_i² - ω²)|``. `ω=0` is the static ranking.
Near-resonant denominators are floored so those modes rank first.
"""
function mode_amplitudes(basis::ModalBasis; f::AbstractVector=basis.f0, ω::Real=0.0)
    length(f) == size(basis.M, 1) || throw(DimensionMismatch("f"))
    f̄ = basis.Φ̃' * (basis.M \ collect(Float64, f))
    ω2 = float(ω)^2
    A = similar(basis.ω²)
    @inbounds for i in eachindex(A)
        den = basis.ω²[i] - ω2
        if abs(den) < 1e-30 * max(1.0, abs(basis.ω²[i]))
            A[i] = abs(f̄[i]) / 1e-30
        else
            A[i] = abs(f̄[i] / den)
        end
    end
    return A
end

function mode_relative_amplitudes(A::AbstractVector{<:Real})
    isempty(A) && return Float64[]
    A1 = A[1]
    return A1 == 0 ? fill(0.0, length(A)) : 100 .* A ./ A1
end

function select_modes_amplitude(basis::ModalBasis; nkeep::Int=10,
                                f::AbstractVector=basis.f0, ω::Real=0.0)
    A = mode_amplitudes(basis; f=f, ω=ω)
    nkeep = min(nkeep, length(A))
    return partialsortperm(A, 1:nkeep; rev=true)
end

function _subset_basis(basis::ModalBasis, idx::AbstractVector{Int})
    Φ = basis.Φ[:, idx]
    Φ̃ = basis.Φ̃[:, idx]
    ω² = basis.ω²[idx]
    for k in 1:length(idx)
        s = dot(Φ̃[:, k], Φ[:, k])
        abs(s) > 1e-14 && (Φ̃[:, k] ./= s)
    end
    return ModalBasis(ω², sqrt.(max.(ω², 0.0)), Φ, Φ̃,
        basis.M, basis.K, basis.free, basis.f0)
end

# ---------------------------------------------------------------------------
# Modal SDOF: ÿ + ω² y = f̄(t)
# ---------------------------------------------------------------------------

function _fbar_at_t(f̄::Real, t::Real, Δt::Real)
    return float(f̄)
end
function _fbar_at_t(f̄::AbstractVector, t::Real, Δt::Real)
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
    i = floor(Int, ξ) + 1
    α = ξ - (i - 1)
    return (1 - α) * float(f̄[i]) + α * float(f̄[i + 1])
end

function _modal_sdoof_rhs!(dy, y, p, t)
    v = y[1]
    u = y[2]
    dy[1] = _fbar_at_t(p.f̄, t, p.Δt) - p.ω2 * u
    dy[2] = v
    return nothing
end

function _modal_sdoof_jac!(J, y, p, t)
    fill!(J, 0)
    J[1, 2] = -p.ω2
    J[2, 1] = 1
    return J
end
_modal_sdoof_jac!(J::AbstractMatrix{<:Number}, p, t) = _modal_sdoof_jac!(J, nothing, p, t)

"""First-order form of `ÿ + ω² y = f̄` for DiffEq (`y = (ẏ, y)`)."""
function build_modal_sdoof_ode(y0::Real, dy0::Real, ω²::Real, f̄, tspan;
        Δt::Real=0.0)
    p = (ω2=float(ω²), f̄=f̄, Δt=float(Δt))
    u0 = [float(dy0), float(y0)]
    ff = ODEFunction{true}(_modal_sdoof_rhs!; jac=_modal_sdoof_jac!)
    return ODEProblem{true}(ff, u0, tspan, p)
end

"""
    houbolt_sdoof!(y, f̄, ω², Δt; dy0=0)

Same fill-in-place contract as [`modal_sdoof!`](@ref): `y[1]` is `y(0)`.
Steps 2–3 are Taylor; from 4 onward classical Houbolt
`(2/Δt² + ω²) y^{n+1} = f^{n+1} + (5 y^n − 4 y^{n−1} + y^{n−2})/Δt²`.
"""
function houbolt_sdoof!(y::AbstractVector{<:Real}, f̄, ω²::Real, Δt::Real;
        dy0::Real=0.0)
    nT = length(y)
    nT >= 2 || throw(ArgumentError("need nT ≥ 2"))
    Δt > 0 || throw(ArgumentError("Δt > 0"))
    ω2 = float(ω²)
    y1 = float(y[1])
    f1 = _fbar_at_t(f̄, 0.0, Δt)
    ÿ1 = f1 - ω2 * y1
    if nT >= 2
        y[2] = y1 + Δt * dy0 + 0.5 * Δt^2 * ÿ1
        ẏ2 = dy0 + Δt * ÿ1
        f2 = _fbar_at_t(f̄, Δt, Δt)
        ÿ2 = f2 - ω2 * y[2]
        if nT >= 3
            y[3] = y[2] + Δt * ẏ2 + 0.5 * Δt^2 * ÿ2
        end
    end
    c = 2 / Δt^2 + ω2
    invdt2 = 1 / Δt^2
    @inbounds for i in 4:nT
        fi = _fbar_at_t(f̄, (i - 1) * Δt, Δt)
        y[i] = (fi + (5 * y[i - 1] - 4 * y[i - 2] + y[i - 3]) * invdt2) / c
    end
    return y
end

"""
    modal_sdoof!(y, f̄, ω², Δt; dy0=0, alg=Rodas5P())

Fill `y` on `0:Δt:(nT-1)Δt`. `alg=:houbolt` uses [`houbolt_sdoof!`](@ref);
any DiffEq algorithm uses the first-order form (default `Rodas5P`).
"""
function modal_sdoof!(y::AbstractVector{<:Real}, f̄, ω²::Real, Δt::Real;
        dy0::Real=0.0,
        alg=Rodas5P(),
        abstol=1e-8,
        reltol=1e-8,
        kwargs...)
    if alg === :houbolt
        return houbolt_sdoof!(y, f̄, ω², Δt; dy0=dy0)
    end
    nT = length(y)
    nT >= 2 || throw(ArgumentError("need nT ≥ 2"))
    Δt > 0 || throw(ArgumentError("Δt > 0"))
    tf = (nT - 1) * Δt
    tsave = range(0.0, tf; length=nT)
    prob = build_modal_sdoof_ode(float(y[1]), dy0, ω², f̄, (0.0, tf); Δt=Δt)
    sol = OrdinaryDiffEq.solve(prob, alg; abstol=abstol, reltol=reltol,
        saveat=tsave, dense=false, kwargs...)
    @inbounds for i in 1:nT
        y[i] = float(sol(tsave[i])[2])
    end
    return y
end

# ---------------------------------------------------------------------------

function _scale_bc(fun, t, pattern)
    fun === nothing && return pattern
    v = fun(t)
    v isa Number && return float(v) .* pattern
    length(v) == length(pattern) || throw(DimensionMismatch("BC function length"))
    return collect(Float64, v)
end

function _modal_load_bar(basis::ModalBasis, ff::AbstractVector, Mfac)
    length(ff) == size(basis.M, 1) || throw(DimensionMismatch("f free-DOF length"))
    return basis.Φ̃' * (Mfac \ collect(Float64, ff))
end

function _modal_force_history(sys::ModalSystem, basis::ModalBasis, dad, t;
        f=nothing, load=nothing, u_load=nothing)
    nf = size(basis.M, 1)
    nm = size(basis.Φ, 2)
    nT = length(t)
    if f !== nothing
        Mfac = factorize(basis.M)
        if f isa Function
            Fhist = zeros(nm, nT)
            @inbounds for i in 1:nT
                Fhist[:, i] .= _modal_load_bar(basis, f(t[i]), Mfac)
            end
            return Fhist
        elseif f isa AbstractMatrix
            size(f, 1) == nf || throw(DimensionMismatch("f rows"))
            size(f, 2) == nT || throw(DimensionMismatch("f columns vs time"))
            Fhist = zeros(nm, nT)
            @inbounds for i in 1:nT
                Fhist[:, i] .= _modal_load_bar(basis, view(f, :, i), Mfac)
            end
            return Fhist
        else
            f̄ = _modal_load_bar(basis, f, Mfac)
            return repeat(f̄, 1, nT)
        end
    end
    Wu = isempty(sys.diri) ? zeros(nm, 0) : basis.Φ̃' * (basis.M \ sys.fu)
    Wq = isempty(sys.neum) ? zeros(nm, 0) : basis.Φ̃' * (basis.M \ sys.fq)
    ū0 = isempty(sys.diri) ? Float64[] : Float64.(dad.BV[sys.diri])
    q̄0 = isempty(sys.neum) ? Float64[] : Float64.(dad.BV[sys.neum])
    Fhist = zeros(nm, nT)
    @inbounds for i in 1:nT
        ū = _scale_bc(u_load, t[i], ū0)
        q̄ = _scale_bc(load, t[i], q̄0)
        if !isempty(ū)
            Fhist[:, i] .+= Wu * ū
        end
        if !isempty(q̄)
            Fhist[:, i] .+= Wq * q̄
        end
    end
    return Fhist
end

"""
    solve_mmm!(dad, Δt, tf; nmodes=nothing, select=:freq, ...) -> (U, t, basis)

Modified modal method (Prodonoff–Zepka): eigenpairs of the condensed
DIBEM pencil, uncoupled oscillators ``ÿ_i + ω_i² y_i = f̄_i``, reconstruct
full-field `U` on `0:Δt:tf`. `select=:freq` keeps the lowest `nmodes`;
`:amplitude` ranks by modal force. Call [`dibem!`](@ref) first.

`f` is the free-DOF load in ``\\bar M \\ddot u + \\bar K u = f``: a vector
(constant), a matrix `(n_free, nT)`, or `f(t)` returning a free-DOF vector.
"""
function solve_mmm!(dad::BEMdata{<:Union{Laplace,Elasticity}}, Δt::Real, tf::Real;

        nmodes=nothing, select::Symbol=:freq, nkeep=nothing,
        u0=nothing, du0=nothing, f=nothing, load=nothing, u_load=nothing,
        basis::Union{Nothing,ModalBasis}=nothing,
        verbose::Bool=false, ωmax::Real=Inf, ω::Real=0.0,
        alg=Rodas5P(), abstol=1e-8, reltol=1e-8)
    sys = build_modal_system(dad)
    if basis === nothing
        basis = modal_analysis_mmm(sys; nmodes=nmodes, ωmax=ωmax)
    elseif nmodes !== nothing && Int(nmodes) < size(basis.Φ, 2)
        basis = _subset_basis(basis, 1:Int(nmodes))
    end
    if select === :amplitude
        nk = nkeep === nothing ? size(basis.Φ, 2) : Int(nkeep)
        f_sel = if f === nothing
            sys.f0
        elseif f isa Function
            f(0.0)
        elseif f isa AbstractMatrix
            view(f, :, 1)
        else
            f
        end
        idx = select_modes_amplitude(basis; nkeep=nk, f=f_sel, ω=ω)
        basis = _subset_basis(basis, idx)
    elseif select !== :freq
        throw(ArgumentError("select must be :freq or :amplitude"))
    end


    t = collect(0:Δt:tf)
    nT = length(t)

    ndof, nb = _neq(dad)
    u0_full = u0 === nothing ? zeros(ndof) : collect(Float64, u0)
    du0_full = du0 === nothing ? zeros(ndof) : collect(Float64, du0)
    length(u0_full) == ndof || throw(DimensionMismatch("u0"))
    _pin_known!(u0_full, dad)
    @inbounds for i in eachindex(dad.BC)
        dad.BC[i] == 0 && i <= ndof && (du0_full[i] = 0.0)
    end
    y0 = basis.Φ̃' * u0_full[sys.free]
    dy0v = basis.Φ̃' * du0_full[sys.free]
    basis = filter_mmm_ghosts(basis; f=sys.f0, y0=y0, dy0=dy0v)
    nm = size(basis.Φ, 2)
    y0 = basis.Φ̃' * u0_full[sys.free]
    dy0v = basis.Φ̃' * du0_full[sys.free]
    Fhist = _modal_force_history(sys, basis, dad, t; f=f, load=load, u_load=u_load)

    Y = zeros(nm, nT)
    @inbounds for k in 1:nm
        yk = view(Y, k, :)
        yk[1] = y0[k]
        modal_sdoof!(yk, view(Fhist, k, :), basis.ω²[k], Δt;
            dy0=dy0v[k], alg=alg, abstol=abstol, reltol=reltol)
    end

    Uf = basis.Φ * Y
    U = zeros(ndof, nT)
    @inbounds for j in 1:nT
        U[sys.free, j] = Uf[:, j]
        _pin_known!(view(U, :, j), dad)
    end
    q = zeros(nb, nT)
    for j in 1:nT
        _scatter_step!(dad, view(U, :, j), view(q, :, j))
    end
    set_cache!(dad; T=U, q=q, u=U, traction=q, time=t, modal_basis=basis)
    verbose && @info "solve_mmm!" nm nT Δt tf ω1=basis.ω[1]
    return U, t, basis
end


function _modal_heat_rhs!(dy, y, p, t)
    dy[1] = _fbar_at_t(p.f̄, t, p.Δt) - p.ω2 * y[1]
    return nothing
end
function _modal_heat_jac!(J, y, p, t)
    J[1, 1] = -p.ω2
    return J
end
_modal_heat_jac!(J::AbstractMatrix{<:Number}, p, t) = _modal_heat_jac!(J, nothing, p, t)

"""Implicit Euler for `ẏ + λ y = f̄` (Houbolt-style first-order)."""
function modal_heat_sdoof!(y::AbstractVector{<:Real}, f̄, λ::Real, Δt::Real;
        alg=Rodas5P(), abstol=1e-8, reltol=1e-8, kwargs...)
    nT = length(y)
    nT >= 2 || throw(ArgumentError("need nT ≥ 2"))
    Δt > 0 || throw(ArgumentError("Δt > 0"))
    if alg === :houbolt || alg === :euler
        den = 1 + float(λ) * Δt
        @inbounds for i in 2:nT
            fi = _fbar_at_t(f̄, (i - 1) * Δt, Δt)
            y[i] = (y[i - 1] + Δt * fi) / den
        end
        return y
    end
    tf = (nT - 1) * Δt
    tsave = range(0.0, tf; length=nT)
    p = (ω2=float(λ), f̄=f̄, Δt=float(Δt))
    ff = ODEFunction{true}(_modal_heat_rhs!; jac=_modal_heat_jac!)
    prob = ODEProblem{true}(ff, [float(y[1])], (0.0, tf), p)
    sol = OrdinaryDiffEq.solve(prob, alg; abstol=abstol, reltol=reltol,
        saveat=tsave, dense=false, kwargs...)
    @inbounds for i in 1:nT
        y[i] = float(sol(tsave[i])[1])
    end
    return y
end

"""
    solve_mmm_heat!(dad, Δt, tf; kwargs...)

Diffusion on the MMM basis: `ẏ_i + ω_i² y_i = f̄_i`.
Spurious `ω² ≤ 0` modes are already dropped by [`modal_analysis_mmm`](@ref).
Same keywords as [`solve_mmm!`](@ref) except `du0`.
"""
function solve_mmm_heat!(dad::BEMdata{<:Laplace}, Δt::Real, tf::Real;
        nmodes=nothing, select::Symbol=:freq, nkeep=nothing,
        u0=nothing, f=nothing, load=nothing, u_load=nothing,
        basis::Union{Nothing,ModalBasis}=nothing,
        verbose::Bool=false, ωmax::Real=Inf, ω::Real=0.0,
        alg=:houbolt, abstol=1e-8, reltol=1e-8)
    sys = build_modal_system(dad)
    if basis === nothing
        basis = modal_analysis_mmm(sys; nmodes=nmodes, ωmax=ωmax)
    elseif nmodes !== nothing && Int(nmodes) < size(basis.Φ, 2)
        basis = _subset_basis(basis, 1:Int(nmodes))
    end
    if select === :amplitude
        nk = nkeep === nothing ? size(basis.Φ, 2) : Int(nkeep)
        idx = select_modes_amplitude(basis; nkeep=nk,
            f = f === nothing ? sys.f0 : f, ω=ω)

        basis = _subset_basis(basis, idx)
    elseif select !== :freq
        throw(ArgumentError("select must be :freq or :amplitude"))
    end
    nf = size(basis.M, 1)
    nm = size(basis.Φ, 2)
    t = collect(0:Δt:tf)
    nT = length(t)
    Fhist = _modal_force_history(sys, basis, dad, t; f=f, load=load, u_load=u_load)
    u0_full = u0 === nothing ? zeros(dad.nt) : collect(Float64, u0)
    @inbounds for i in 1:dad.n
        dad.BC[i] == 0 && (u0_full[i] = dad.BV[i])
    end
    y0 = basis.Φ̃' * u0_full[sys.free]
    Y = zeros(nm, nT)
    @inbounds for k in 1:nm
        yk = view(Y, k, :)
        yk[1] = y0[k]
        modal_heat_sdoof!(yk, view(Fhist, k, :), basis.ω²[k], Δt;
            alg=alg, abstol=abstol, reltol=reltol)
    end
    Uf = basis.Φ * Y
    U = zeros(dad.nt, nT)
    @inbounds for j in 1:nT
        U[sys.free, j] = Uf[:, j]
        for i in 1:dad.n
            dad.BC[i] == 0 && (U[i, j] = dad.BV[i])
        end
    end
    q = zeros(dad.n, nT)
    for j in 1:nT
        split_sol!(dad, view(U, :, j), view(q, :, j))
    end
    set_cache!(dad; T=U, q=q, time=t, modal_basis=basis)
    verbose && @info "solve_mmm_heat!" nm nT Δt tf ω1=basis.ω[1]
    return U, t, basis
end


# ---------------------------------------------------------------------------
# Flux recovery
# ---------------------------------------------------------------------------

function recover_flux_from_displacement(dad::BEMdata{<:Laplace}, u_full;
        ü_full=nothing)
    has_cache(dad, :H) && has_cache(dad, :G) && has_cache(dad, :M) ||
        error("need H, G, M")
    n = dad.n
    ü = ü_full === nothing ? zeros(dad.nt) : collect(Float64, ü_full)
    H = dad.H
    G = dad.G
    Mass = dad.M
    diri = Int[i for i in 1:n if dad.BC[i] == 0]
    neum = Int[i for i in 1:n if dad.BC[i] == 1]
    rhs = H * collect(Float64, u_full) - Mass * ü
    if !isempty(neum)
        rhs .-= G[:, neum] * dad.BV[neum]
    end
    q = zeros(n)
    q[neum] .= dad.BV[neum]
    if !isempty(diri)
        qd = G[:, diri] \ rhs
        q[diri] .= qd
    end
    return q
end

# ---------------------------------------------------------------------------
# Apply-only condensed pencil (H-matrix compatible)
# ---------------------------------------------------------------------------

struct CondensedPencil{TH,TG,TM,TF}
    H::TH
    G::TG
    M::TM
    free::Vector{Int}
    diri::Vector{Int}
    GddF::TF
    n::Int
    nt::Int
    xfull::Vector{Float64}
    yH::Vector{Float64}
    yG::Vector{Float64}
    gcol::Vector{Float64}
    tmpD::Vector{Float64}
end

function _extract_Gdd(G, diri::Vector{Int}, n::Int)
    nd = length(diri)
    Gdd = zeros(nd, nd)
    e = zeros(n)
    col = zeros(size(G, 1))
    @inbounds for (k, j) in enumerate(diri)
        e[j] = 1.0
        mul!(col, G, e)
        e[j] = 0.0
        for (p, i) in enumerate(diri)
            Gdd[p, k] = col[i]
        end
    end
    return Gdd
end

function condensed_pencil(dad::BEMdata{<:Laplace})
    has_cache(dad, :H) || error("call H_G_full_direct or H_G_Hmat first")
    has_cache(dad, :G) || error("call H_G_full_direct or H_G_Hmat first")
    has_cache(dad, :M) || error("call DIBEM first")
    n, nt = dad.n, dad.nt
    diri = Int[i for i in 1:n if dad.BC[i] == 0]
    neum = Int[i for i in 1:n if dad.BC[i] == 1]
    free = vcat(neum, collect((n + 1):nt))
    isempty(diri) && error("condensed_pencil: need Dirichlet nodes for G_DD")
    Gdd = _extract_Gdd(dad.G, diri, n)
    return CondensedPencil(dad.H, dad.G, dad.M, free, diri, factorize(Gdd),
        n, nt, zeros(nt), zeros(nt), zeros(nt), zeros(n), zeros(length(diri)))
end

function _schur_apply!(y, P::CondensedPencil, Op, x)
    fill!(P.xfull, 0.0)
    @inbounds for (k, i) in enumerate(P.free)
        P.xfull[i] = x[k]
    end
    mul!(P.yH, Op, P.xfull)
    @inbounds for (p, i) in enumerate(P.diri)
        P.tmpD[p] = P.yH[i]
    end
    pD = P.GddF \ P.tmpD
    fill!(P.gcol, 0.0)
    @inbounds for (p, i) in enumerate(P.diri)
        P.gcol[i] = pD[p]
    end
    mul!(P.yG, P.G, P.gcol)
    @inbounds for (k, i) in enumerate(P.free)
        y[k] = P.yH[i] - P.yG[i]
    end
    return y
end

mul_Kbar!(y, P::CondensedPencil, x) = _schur_apply!(y, P, P.H, x)
mul_Mbar!(y, P::CondensedPencil, x) = _schur_apply!(y, P, P.M, x)

function _maybe_flip_stiffness!(P::CondensedPencil)
    nf = length(P.free)
    x = zeros(nf)
    y = zeros(nf)
    trK = 0.0
    @inbounds for j in 1:nf
        x[j] = 1.0
        mul_Kbar!(y, P, x)
        trK += y[j]
        x[j] = 0.0
    end
    return trK < 0
end

struct ShiftInvertCondensed{TP,TF}
    P::TP
    Kf::TF
    flip::Bool
    tmp::Vector{Float64}
end

function ShiftInvertCondensed(P::CondensedPencil)
    nf = length(P.free)
    flip = _maybe_flip_stiffness!(P)
    Kbar = zeros(nf, nf)
    x = zeros(nf)
    @inbounds for j in 1:nf
        x[j] = 1.0
        mul_Kbar!(view(Kbar, :, j), P, x)
        x[j] = 0.0
    end
    flip && (Kbar .*= -1)
    return ShiftInvertCondensed(P, factorize(Kbar), flip, zeros(nf))
end

function (S::ShiftInvertCondensed)(y, x)
    mul_Mbar!(S.tmp, S.P, x)
    ldiv!(y, S.Kf, S.tmp)
    return y
end

function modal_shiftinvert(dad::BEMdata{<:Laplace}; nev::Integer=12, tol::Real=1e-10,
        restarts::Integer=200)
    AM = Base.require(Base.PkgId(
        Base.UUID("ec485272-7323-5ecc-a04f-4719b315124d"), "ArnoldiMethod"))
    P = condensed_pencil(dad)
    S = ShiftInvertCondensed(P)
    nf = length(P.free)
    L = LinearMaps.LinearMap{Float64}(S, nf; ismutating=true)
    decomp, hist = AM.partialschur(L; nev=nev, which=:LM, tol=tol, restarts=restarts)
    θ, X = AM.partialeigen(decomp)
    λ = 1 ./ θ
    ord = sortperm(real.(λ))
    λ = λ[ord]
    X = X[:, ord]
    ω = sqrt.(max.(real.(λ), 0.0))
    return (; ω, λ, X, history=hist, free=P.free, flip=S.flip)
end
