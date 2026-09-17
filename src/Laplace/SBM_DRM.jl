# SBM + Dual Reciprocity — 2D transient diffusion and wave
# Kovářík et al., Eng. Anal. Bound. Elem. (2017)
#
# Heat (∂t u = κ Δu): Euler / Houbolt as in the paper.
# Wave (ü = c² Δu): same (α,β) split; interior residual uses
#   Houbolt  2u^{n+1} − Δt² c² φβ = 5u^n − 4u^{n−1} + u^{n−2}
#   central  u^{n+1} − Δt² c² φβ = 2u^n − u^{n−1}
#
export SBMDRMState, sbm_drm_setup, solve_sbm_drm, sbm_drm_step!, sbm_drm_project_ic!
export solve_sbm_wave


const _DEFAULT_SBM_DRM_RBF = PHS(3; poly_deg=-1)
const _PHS2_BASIS = PHS(2; poly_deg=-1)

# Optional PHS2 closed forms (2D) — not the default (Gibbs on disc. IC)
@inline function _φ_phs2(r::Float64)
    r < 1e-30 && return 0.0
    return r * r * log(r)
end
@inline function _Φ_phs2(r::Float64)
    return laplace_particular(_PHS2_BASIS, r; dim=2)
end
@inline function _dΦ_phs2_dr(r::Float64)
    r < 1e-30 && return 0.0
    return 0.25 * r * r * r * log(r)
end


mutable struct SBMDRMState
    nodes::Vector{Point2D}
    normals::Vector{Point2D}
    internal::Vector{Point2D}
    BC::Vector{Int}
    BV::Vector{Float64}
    robin_H::Vector{Float64}
    robin_uf::Vector{Float64}
    k_cond::Float64
    κ::Float64
    Δt::Float64
    scheme::Symbol
    physics::Symbol
    c::Float64

    Gbb::Matrix{Float64}
    Hbb::Matrix{Float64}
    Gib::Matrix{Float64}
    Φb::Matrix{Float64}
    dΦb::Matrix{Float64}
    Φi::Matrix{Float64}
    φi::Matrix{Float64}
    A::Matrix{Float64}
    AF::Any
    α::Vector{Float64}
    β::Vector{Float64}
    name::String
end

Base.show(io::IO, s::SBMDRMState) =
    print(io, "SBMDRMState(N=$(length(s.nodes)), M=$(length(s.internal)), ",
          "$(s.physics)/$(s.scheme), Δt=$(s.Δt))")


function _rbf_pair(basis::AbstractRadialBasis)
    if basis isa PHS2
        return (_φ_phs2, _Φ_phs2, _dΦ_phs2_dr)
    end
    φ_of = r -> float(basis(r))   # PHS/MQ/Gaussian take Euclidean r, not r²
    Φ_of = r -> laplace_particular(basis, r; dim=2)
    function dΦdr_of(r)
        r < 1e-30 && return 0.0
        ε = 1e-7 * max(r, 1.0)
        return (Φ_of(r + ε) - Φ_of(max(r - ε, 0.0))) / (2ε)
    end
    return (φ_of, Φ_of, dΦdr_of)
end

"""
    sbm_drm_setup(dad; κ, Δt, scheme=:houbolt, basis=PHS(3),
                  robin_H=nothing, robin_uf=nothing)

`robin_H`, `robin_uf`: length-`n` vectors (or scalars) for BC==2 nodes.
Paper convection: ∂u/∂n = (H_r/k)(u_f − u)  [eq. 46]; q_pkg = H_r (u − u_f).
"""
function sbm_drm_setup(dad::BEMdata{<:Laplace};
                       κ::Real=1.25,
                       Δt::Real,
                       scheme::Symbol=:houbolt,
                       physics::Symbol=:heat,
                       c::Real=1.0,
                       basis::AbstractRadialBasis=_DEFAULT_SBM_DRM_RBF,
                       robin_H=nothing,
                       robin_uf=nothing)
    physics in (:heat, :wave) || throw(ArgumentError("physics ∈ (:heat,:wave)"))
    if physics === :heat
        scheme in (:euler, :houbolt) || throw(ArgumentError("heat scheme ∈ (:euler,:houbolt)"))
    else
        scheme in (:houbolt, :central) || throw(ArgumentError("wave scheme ∈ (:houbolt,:central)"))
    end

    dad.ni > 0 || error("SBM-DRM needs internal points as RBF centres")
    N, M = dad.n, dad.ni
    nodes = Point2D[Point2D(p) for p in dad.Nodes]
    normals = Point2D[Point2D(nn) for nn in dad.Normal]
    internal = Point2D[Point2D(p) for p in dad.internalNodes]
    BC = collect(Int, dad.BC[1:N])
    BV = collect(Float64, dad.BV[1:N])
    k_cond = float(dad.properties.k)
    κ = float(κ)
    Δt = float(Δt)

    rH = zeros(N)
    rUf = zeros(N)
    if robin_H !== nothing
        if robin_H isa Number
            @inbounds for i in 1:N
                BC[i] == 2 && (rH[i] = float(robin_H))
            end
        else
            length(robin_H) == N || throw(DimensionMismatch("robin_H"))
            rH .= float.(robin_H)
        end
    end
    if robin_uf !== nothing
        if robin_uf isa Number
            @inbounds for i in 1:N
                BC[i] == 2 && (rUf[i] = float(robin_uf))
            end
        else
            length(robin_uf) == N || throw(DimensionMismatch("robin_uf"))
            rUf .= float.(robin_uf)
        end
    end

    sd = sbm_from_bemdata(dad; internal=false)
    assemble_sbm!(sd)
    Gbb = Matrix(sd.G)
    Hbb = Matrix(sd.H)

    φ_of, Φ_of, dΦdr_of = _rbf_pair(basis)

    Φb = zeros(N, M)
    dΦb = zeros(N, M)
    Φi = zeros(M, M)
    φi = zeros(M, M)
    Gib = zeros(M, N)
    @inbounds for j in 1:M
        cj = internal[j]
        for i in 1:N
            rvec = nodes[i] - cj
            rr = hypot(rvec[1], rvec[2])
            Φb[i, j] = Φ_of(rr)
            if rr > 1e-30
                dΦb[i, j] = dΦdr_of(rr) *
                    (rvec[1] * normals[i][1] + rvec[2] * normals[i][2]) / rr
            end
        end
        for i in 1:M
            rr = norm(internal[i] - cj)
            φi[i, j] = φ_of(rr)
            Φi[i, j] = Φ_of(rr)
        end
    end
    @inbounds for i in 1:M
        φi[i, i] += 1e-14 * (1 + abs(φi[i, i]))
    end
    @inbounds for j in 1:N, i in 1:M
        Gib[i, j] = _sbm_U(internal[i] - nodes[j], k_cond)
    end

    invγ = if physics === :wave
        c2dt = float(c)^2 * Δt^2
        scheme === :houbolt ? (c2dt / 2) : c2dt
    else
        scheme === :euler ? (κ * Δt) : (6 * κ * Δt / 11)
    end


    A = zeros(N + M, N + M)
    @inbounds for i in 1:N
        if BC[i] == 0
            # Dirichlet: G α + Φ β = ū
            for j in 1:N
                A[i, j] = Gbb[i, j]
            end
            for j in 1:M
                A[i, N + j] = Φb[i, j]
            end
        elseif BC[i] == 1
            # Neumann package: H_pkg α − k ∂nΦ β = q_pkg
            for j in 1:N
                A[i, j] = Hbb[i, j]
            end
            for j in 1:M
                A[i, N + j] = -k_cond * dΦb[i, j]
            end
        else
            # Robin (paper 47): ∂u/∂n + (H_r/k) u = (H_r/k) u_f
            # q_pkg = H_r (u − u_f)
            #   H_pkg α − k ∂nΦ β − H_r (G α + Φ β) = −H_r u_f
            Hr = rH[i]
            for j in 1:N
                A[i, j] = Hbb[i, j] - Hr * Gbb[i, j]
            end
            for j in 1:M
                A[i, N + j] = -k_cond * dΦb[i, j] - Hr * Φb[i, j]
            end
        end
    end
    @inbounds for i in 1:M
        row = N + i
        for j in 1:N
            A[row, j] = Gib[i, j]
        end
        for j in 1:M
            A[row, N + j] = Φi[i, j] - invγ * φi[i, j]
        end
    end

    return SBMDRMState(
        nodes, normals, internal, BC, BV, rH, rUf, k_cond, κ, Δt, scheme,
        physics, float(c),
        Gbb, Hbb, Gib, Φb, dΦb, Φi, φi,
        A, factorize(A), zeros(N), zeros(M), string(dad.name),
    )
end

function _recover(st::SBMDRMState)
    N = length(st.nodes)
    u_b = st.Gbb * st.α .+ st.Φb * st.β
    q_b = st.Hbb * st.α .+ (-st.k_cond) .* (st.dΦb * st.β)
    @inbounds for i in 1:N
        if st.BC[i] == 0
            u_b[i] = st.BV[i]
        elseif st.BC[i] == 1
            q_b[i] = st.BV[i]
        elseif st.BC[i] == 2
            # q_pkg = H_r (u − u_f)
            q_b[i] = st.robin_H[i] * (u_b[i] - st.robin_uf[i])
        end
    end
    u_i = st.Gib * st.α .+ st.Φi * st.β
    return u_b, q_b, u_i
end

function _fill_rhs!(rhs, st::SBMDRMState, u_n, u_nm1, u_nm2)
    N, M = length(st.nodes), length(st.internal)
    @inbounds for i in 1:N
        if st.BC[i] == 0 || st.BC[i] == 1
            rhs[i] = st.BV[i]
        else
            rhs[i] = -st.robin_H[i] * st.robin_uf[i]
        end
    end
    if st.physics === :wave
        if st.scheme === :central
            @inbounds for i in 1:M
                un = u_n[N + i]
                un1 = u_nm1 === nothing ? un : u_nm1[N + i]
                rhs[N + i] = 2 * un - un1
            end
        else
            @inbounds for i in 1:M
                un = u_n[N + i]
                un1 = u_nm1 === nothing ? un : u_nm1[N + i]
                un2 = u_nm2 === nothing ? un : u_nm2[N + i]
                rhs[N + i] = (5 * un - 4 * un1 + un2) / 2
            end
        end
    elseif st.scheme === :euler
        @inbounds for i in 1:M
            rhs[N + i] = u_n[N + i]
        end
    else
        @inbounds for i in 1:M
            un = u_n[N + i]
            un1 = u_nm1 === nothing ? un : u_nm1[N + i]
            un2 = u_nm2 === nothing ? un : u_nm2[N + i]
            rhs[N + i] = (18 * un - 9 * un1 + 2 * un2) / 11
        end
    end

    return rhs
end

function sbm_drm_step!(st::SBMDRMState, u_n; u_nm1=nothing, u_nm2=nothing)
    N, M = length(st.nodes), length(st.internal)
    rhs = zeros(N + M)
    _fill_rhs!(rhs, st, u_n, u_nm1, u_nm2)
    x = st.AF \ rhs
    copyto!(st.α, 1, x, 1, N)
    copyto!(st.β, 1, x, N + 1, M)
    u_b, _, u_i = _recover(st)
    return vcat(u_b, u_i)
end

"""
    sbm_drm_project_ic!(st, u0) -> u_proj

Project nodal data `u0` onto the SBM–DRM trial space
``u = Gα + Φβ`` while enforcing the spatial BCs exactly.

Needed for discontinuous ICs (e.g. interior 30, boundary 0): marching the
raw nodal vector is not in the column space of `(G,Φ)`, which produces a
Gibbs-type overshoot (``max u > max u_0``) even though mean energy decays.
"""
function sbm_drm_project_ic!(st::SBMDRMState, u0::AbstractVector{<:Real})
    N, M = length(st.nodes), length(st.internal)
    length(u0) >= N + M || throw(ArgumentError("u0 short"))
    A = zeros(N + M, N + M)
    b = zeros(N + M)
    @inbounds for i in 1:N
        if st.BC[i] == 0
            for j in 1:N
                A[i, j] = st.Gbb[i, j]
            end
            for j in 1:M
                A[i, N + j] = st.Φb[i, j]
            end
            b[i] = st.BV[i]
        elseif st.BC[i] == 1
            for j in 1:N
                A[i, j] = st.Hbb[i, j]
            end
            for j in 1:M
                A[i, N + j] = -st.k_cond * st.dΦb[i, j]
            end
            b[i] = st.BV[i]
        else
            Hr = st.robin_H[i]
            for j in 1:N
                A[i, j] = st.Hbb[i, j] - Hr * st.Gbb[i, j]
            end
            for j in 1:M
                A[i, N + j] = -st.k_cond * st.dΦb[i, j] - Hr * st.Φb[i, j]
            end
            b[i] = -Hr * st.robin_uf[i]
        end
    end
    @inbounds for i in 1:M
        row = N + i
        for j in 1:N
            A[row, j] = st.Gib[i, j]
        end
        for j in 1:M
            A[row, N + j] = st.Φi[i, j]
        end
        b[row] = u0[N + i]
    end
    x = A \ b
    copyto!(st.α, 1, x, 1, N)
    copyto!(st.β, 1, x, N + 1, M)
    u_b, _, u_i = _recover(st)
    return vcat(u_b, u_i)
end

function solve_sbm_drm(dad::BEMdata{<:Laplace};
                       κ::Real=1.25,
                       Δt::Real,
                       tf::Real,
                       u0::AbstractVector{<:Real},
                       scheme::Symbol=:houbolt,
                       basis::AbstractRadialBasis=_DEFAULT_SBM_DRM_RBF,
                       robin_H=nothing,
                       robin_uf=nothing,
                       project_ic::Bool=true)
    N, M = dad.n, dad.ni
    length(u0) >= N + M || throw(ArgumentError("u0 length $(length(u0)) < n+ni=$(N+M)"))
    t = collect(0.0:float(Δt):float(tf))
    nT = length(t)
    U = zeros(N + M, nT)
    U[:, 1] .= float.(u0[1:N+M])
    @inbounds for j in 1:N
        dad.BC[j] == 0 && (U[j, 1] = dad.BV[j])
    end

    kw = (; κ, Δt, basis, robin_H, robin_uf)
    stE = sbm_drm_setup(dad; scheme=:euler, kw...)
    st = scheme === :houbolt ? sbm_drm_setup(dad; scheme=:houbolt, kw...) : stE

    if project_ic
        U[:, 1] .= sbm_drm_project_ic!(stE, view(U, :, 1))
    end

    nT >= 2 && (U[:, 2] .= sbm_drm_step!(stE, view(U, :, 1)))
    nT >= 3 && (U[:, 3] .= sbm_drm_step!(stE, view(U, :, 2)))
    @inbounds for i in 4:nT
        if scheme === :houbolt
            U[:, i] .= sbm_drm_step!(st, view(U, :, i - 1);
                                     u_nm1=view(U, :, i - 2),
                                     u_nm2=view(U, :, i - 3))
        else
            U[:, i] .= sbm_drm_step!(st, view(U, :, i - 1))
        end
    end
    return (; t, U, st, N, M, scheme, project_ic)
end

function _sbm_interior_reconstruction(st::SBMDRMState)
    N, M = length(st.nodes), length(st.internal)
    C = hcat(st.Gib, st.Φi)
    E = zeros(N + M, M)
    rhs = zeros(N + M)
    @inbounds for j in 1:M
        fill!(rhs, 0)
        rhs[N + j] = 1
        E[:, j] = st.AF \ rhs
    end
    return C * E
end

function _sbm_wave_stable_basis(st::SBMDRMState)
    R = _sbm_interior_reconstruction(st)
    γ = st.scheme === :houbolt ? (st.c^2 * st.Δt^2 / 2) : (st.c^2 * st.Δt^2)
    γ = max(γ, 1e-16)
    L = (I - R) ./ γ
    return stable_wave_subspace(L)
end

function _sbm_filter_interior!(Ucol, st::SBMDRMState, V)
    N = length(st.nodes)
    ui = view(Ucol, N + 1:length(Ucol))
    z = zeros(length(Ucol))
    up = sbm_drm_step!(st, z; u_nm1=z, u_nm2=z)
    δ = ui .- view(up, N + 1:length(up))
    ui .= view(up, N + 1:length(up)) .+ V * (V \ δ)
    return Ucol
end

"""
    solve_sbm_wave(dad; Δt, tf, c=1, scheme=:houbolt, load=nothing)

SBM–DRM for ``\\ddot u = c^2 Δu``. Same ``(α,β)`` split as heat.
`load(t)` scales the stored Neumann `BV` pattern each step.
Growing interior modes of the discrete Laplacian (mixed BC) are
projected out when `filter_unstable=true`.
"""

function solve_sbm_wave(dad::BEMdata{<:Laplace};
                       Δt::Real, tf::Real, c::Real=1.0,
                       u0=nothing, scheme::Symbol=:houbolt,
                       basis::AbstractRadialBasis=_DEFAULT_SBM_DRM_RBF,
                       load=nothing, project_ic::Bool=true,
                       filter_unstable::Bool=true)

    N, M = dad.n, dad.ni
    dad.ni > 0 || error("SBM wave needs internal RBF centres")
    t = collect(0.0:float(Δt):float(tf))
    nT = length(t)
    U = zeros(N + M, nT)
    if u0 === nothing
        @inbounds for j in 1:N
            dad.BC[j] == 0 && (U[j, 1] = dad.BV[j])
        end
    else
        length(u0) >= N + M || throw(DimensionMismatch("u0"))
        U[:, 1] .= float.(u0[1:N + M])
        @inbounds for j in 1:N
            dad.BC[j] == 0 && (U[j, 1] = dad.BV[j])
        end
    end
    q0 = collect(Float64, dad.BV[1:N])
    function apply_load!(st, ti)
        load === nothing && return
        s = float(load(ti))
        @inbounds for i in 1:N
            st.BC[i] == 1 && (st.BV[i] = s * q0[i])
        end
        return
    end
    kw = (; Δt, basis, physics=:wave, c)
    stC = sbm_drm_setup(dad; scheme=:central, kw...)
    st = scheme === :houbolt ? sbm_drm_setup(dad; scheme=:houbolt, kw...) : stC
    V = nothing
    n_drop = 0
    if filter_unstable
        sub = _sbm_wave_stable_basis(st)
        V = sub.V
        n_drop = sub.n_drop
    end
    filt!(col, sti) = V === nothing ? col : _sbm_filter_interior!(col, sti, V)
    if project_ic
        U[:, 1] .= sbm_drm_project_ic!(stC, view(U, :, 1))
        filt!(view(U, :, 1), stC)
    end
    if nT >= 2
        apply_load!(stC, t[2])
        U[:, 2] .= sbm_drm_step!(stC, view(U, :, 1); u_nm1=view(U, :, 1))
        filt!(view(U, :, 2), stC)
    end
    if nT >= 3
        apply_load!(stC, t[3])
        U[:, 3] .= sbm_drm_step!(stC, view(U, :, 2); u_nm1=view(U, :, 1))
        filt!(view(U, :, 3), stC)
    end
    @inbounds for i in 4:nT
        sti = scheme === :houbolt ? st : stC
        apply_load!(sti, t[i])
        U[:, i] .= sbm_drm_step!(sti, view(U, :, i - 1);
                                 u_nm1=view(U, :, i - 2),
                                 u_nm2=view(U, :, i - 3))
        filt!(view(U, :, i), sti)
    end
    q = zeros(N, nT)
    @inbounds for j in 1:N
        dad.BC[j] == 1 && (q[j, :] .= dad.BV[j])
    end
    set_cache!(dad; T=U, q=q, time=t)
    return (; t, U, st, N, M, scheme, physics=:wave, c=float(c), n_drop)
end


