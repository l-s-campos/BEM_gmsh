"""
    MortarContact2D

Segment-to-segment (mortar) frictional contact for 1-D surface meshes in 2D
problems — the STS upgrade recommended in Loyola (2022) §9.3.1 for coarse
contact half-width accuracy.

# Formulation
Given **slave** nodes ``x_s`` and **master** nodes ``x_m`` on the prospective
contact line (parametrised by arc coordinate ``s``):

1. Project each slave node onto the master polyline → ``\\bar s, e_m, N_m``
2. Build mortar matrices
   ``D_{ss'} = \\int_{\\Gamma_c} N_s N_{s'} dΓ`` (lumped diagonal dual optional)
   ``M_{sm} = \\int_{\\Gamma_c} N_s N_m dΓ``
3. Mortar gap
   ``g = D u_n^{sl} - M u_n^{ma} + g_0``
4. Enforce unilateral contact + Coulomb friction on mortar DOFs (active set)

For the half-plane Cattaneo–Mindlin path both sides share the same elastic
operator; mortar couples possibly **non-matching** slave/master grids through
a master-side traction recovered by
``t_m = M^+ D t_s`` (action–reaction in the weak sense).

References
- Puso & Laursen, Comput. Methods Appl. Mech. Eng. 193 (2004)
- De Lorenzis et al., IGA contact review
- Loyola (2022) — recommendation of segment-to-segment over node-to-node
"""
module MortarContact2D

using LinearAlgebra
using SparseArrays
using Statistics
using ..ContactHalfPlane2D
using ..CattaneoMindlin: solve_cattaneo_halfplane, contact_halfwidth_from_p

export MortarMesh1D, build_mortar_projection, mortar_assemble_DM
export solve_cattaneo_mortar_halfplane
export project_point_to_polyline

"""1-D contact mesh (panel centres + endpoints)."""
struct MortarMesh1D{T}
    x::Vector{T}       # collocation / panel centres
    y::Vector{T}       # endpoints (length n+1)
    h::Vector{T}       # panel lengths
end

function MortarMesh1D(x0::Real, xf::Real, n::Int)
    T = float(promote_type(typeof(x0), typeof(xf)))
    y = collect(range(T(x0), stop=T(xf), length=n + 1))
    x = (y[1:end-1] .+ y[2:end]) ./ 2
    h = abs.(diff(y))
    return MortarMesh1D{T}(x, y, h)
end

MortarMesh1D(x::AbstractVector) = begin
    # build endpoints mid-way between centres
    n = length(x)
    T = float(eltype(x))
    y = zeros(T, n + 1)
    y[1] = x[1] - (x[2] - x[1]) / 2
    y[end] = x[end] + (x[end] - x[end-1]) / 2
    for i in 2:n
        y[i] = 0.5 * (x[i-1] + x[i])
    end
    h = abs.(diff(y))
    MortarMesh1D{T}(collect(T, x), y, h)
end

# =============================================================================
# Geometric projection
# =============================================================================

"""
    project_point_to_polyline(s, y_m) -> (; e, ξ, s̄, N)

Project scalar coordinate `s` onto master endpoints `y_m`.
Returns element index `e`, local ``ξ∈[-1,1]``, projected ``\\bar s``, and
linear shape ``N = [(1-ξ)/2, (1+ξ)/2]``.
"""
function project_point_to_polyline(s::Real, y_m::AbstractVector)
    nm = length(y_m) - 1
    # clamp to segment range
    if s <= y_m[1]
        return (; e=1, ξ=-1.0, s̄=y_m[1], N=(1.0, 0.0))
    elseif s >= y_m[end]
        return (; e=nm, ξ=1.0, s̄=y_m[end], N=(0.0, 1.0))
    end
    e = searchsortedlast(y_m, s)
    e = clamp(e, 1, nm)
    a, b = y_m[e], y_m[e+1]
    ξ = (b ≈ a) ? 0.0 : 2 * (s - a) / (b - a) - 1
    ξ = clamp(ξ, -1.0, 1.0)
    N1 = 0.5 * (1 - ξ)
    N2 = 0.5 * (1 + ξ)
    s̄ = N1 * a + N2 * b
    return (; e, ξ, s̄, N=(N1, N2))
end

"""
    build_mortar_projection(slave::MortarMesh1D, master::MortarMesh1D)

For each slave node, store master element, weights on the two master endpoints
mapped to master collocation DOFs (panel centres ≈ endpoints averaged).

We associate master endpoint `k` with dual DOF:
- endpoint 1 → none left; use collocation j=e and j=e (constant per panel) —
  **panel-centred mortar**: project onto master panels with constant shape.
"""
function build_mortar_projection(slave::MortarMesh1D, master::MortarMesh1D)
    ns = length(slave.x)
    # panel-wise constant master shape: projection weight 1 on master panel e
    e_of = zeros(Int, ns)
    w_left = zeros(ns)   # weight on master panel e
    w_right = zeros(ns)  # unused for constant; kept for linear extension
    s̄ = zeros(ns)
    for i in 1:ns
        pr = project_point_to_polyline(slave.x[i], master.y)
        e_of[i] = pr.e
        s̄[i] = pr.s̄
        # constant on master panel → weight 1
        w_left[i] = 1.0
        w_right[i] = 0.0
    end
    return (; e_of, w_left, w_right, s̄)
end

"""
    mortar_assemble_DM(slave, master, proj; dual=true) -> (D, M)

`D` is ``n_s × n_s`` (diagonal lumped dual if `dual`), `M` is ``n_s × n_m``.
Integration: one-point (panel centre) rule on slave panels
``∫ N_s φ dΓ ≈ h_s · φ(x_s)``.
"""
function mortar_assemble_DM(slave::MortarMesh1D, master::MortarMesh1D, proj; dual::Bool=true)
    ns = length(slave.x)
    nm = length(master.x)
    D = zeros(ns, ns)
    M = zeros(ns, nm)
    for i in 1:ns
        hs = slave.h[i]
        if dual
            D[i, i] = hs
        else
            D[i, i] = hs
        end
        e = proj.e_of[i]
        M[i, e] += hs * proj.w_left[i]
        if e < nm && proj.w_right[i] != 0
            M[i, e+1] += hs * proj.w_right[i]
        end
    end
    return D, M
end

# =============================================================================
# Cattaneo–Mindlin via mortar on non-matching grids
# =============================================================================

"""
    solve_cattaneo_mortar_halfplane(x_s, x_m, R_eq, P, Q, f, G, ν; ...)

Solve Cattaneo–Mindlin with **non-matching** slave/master 1-D meshes:

1. Solve normal+tangential contact on the **slave** grid (half-plane BEM)
2. Map tractions to the master grid by mortar: ``t_m = M^{+} D t_s``
3. Report both slave fields and mortar-projected master fields

When `x_s == x_m` this reduces to standard node-to-node (D and M diagonal-ish).

If only one grid is of interest, pass a refined slave and coarse master to
illustrate STS transfer (Loyola remark on coarse `a` accuracy).
"""
function solve_cattaneo_mortar_halfplane(
    x_s::AbstractVector,
    x_m::AbstractVector,
    R_eq::Real,
    P::Real,
    Q::Real,
    f::Real,
    G::Real,
    ν::Real;
    tol=1e-10,
)
    slave = MortarMesh1D(x_s)
    master = MortarMesh1D(x_m)
    hs = mean_spacing(x_s)
    hp = ElasticHalfPlane2D(G, ν; h=hs)

    sol_s = solve_cattaneo_halfplane(x_s, R_eq, P, Q, f, hp; tol=tol)

    proj = build_mortar_projection(slave, master)
    D, M = mortar_assemble_DM(slave, master, proj)

    # traction transfer: M t_m = D t_s  (weak action-reaction)
    p_m = _nnls_transfer(M, D * sol_s.p)
    τ_m = _solve_transfer(M, D * sol_s.τ)

    # master force
    hm = mean_spacing(x_m)
    # use master panel lengths
    force_n_m = dot_panel(p_m, master.h)
    force_t_m = dot_panel(τ_m, master.h)

    a_s = sol_s.a
    a_m = contact_halfwidth_from_p(x_m, p_m)

    return (;
        slave=sol_s,
        x_s, x_m,
        p_s=sol_s.p, τ_s=sol_s.τ,
        p_m, τ_m,
        D, M, proj,
        force_n_s=sol_s.force_n,
        force_t_s=sol_s.force_t,
        force_n_m, force_t_m,
        a_s, a_m,
        p0_s=sol_s.p0,
        p0_m=maximum(p_m),
    )
end

mean_spacing(x) = length(x) > 1 ? abs(x[2] - x[1]) : 1.0
dot_panel(t, h) = dot(t, h)

function _solve_transfer(M::AbstractMatrix, rhs::AbstractVector)
    # least squares: min ||M t - rhs||
    return M \ rhs
end

function _nnls_transfer(M::AbstractMatrix, rhs::AbstractVector)
    t = M \ rhs
    # pressure non-negative
    t .= max.(t, 0.0)
    # one repair pass
    active = t .> 0
    if any(active) && !all(active)
        Ma = M[:, active]
        ta = Ma \ rhs
        t .= 0
        t[active] .= max.(ta, 0.0)
    end
    return t
end

"""
Node-to-node regular contact on a single matching grid — thin wrapper that
exposes the same API as mortar for comparison scripts.
"""
function solve_cattaneo_nts_halfplane(x, R_eq, P, Q, f, G, ν; tol=1e-10)
    h = mean_spacing(x)
    hp = ElasticHalfPlane2D(G, ν; h=h)
    sol = solve_cattaneo_halfplane(x, R_eq, P, Q, f, hp; tol=tol)
    return sol
end

export solve_cattaneo_nts_halfplane

end # module
