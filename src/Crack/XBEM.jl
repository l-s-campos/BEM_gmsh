# eXtended Dual BEM (Andrade & Leonel, EABE 121:158–179, 2020)
# Shifted first-order tip enrichment + crack-tip tying.
# Isotropic: Williams. Anisotropic: Sih–Paris–Irwin in the tip frame.
# Extra DOFs are (KI, KII) per tip; SIFs are read from the solution vector.

# =============================================================================
# Williams first-order displacement (Appendix D)
# =============================================================================

"""
    williams_M_local(μ, κ, ρ, ω) -> SMatrix{2,2}

Columns are in-plane displacement for unit `KI` and `KII` in the tip frame.
`ω` is measured from the ligament (ahead of the tip); crack faces are `±π`.
"""
function williams_M_local(μ::Real, κ::Real, ρ::Real, ω::Real)
    ρ <= 0 && return @SMatrix zeros(2, 2)
    s = sqrt(ρ / (2π))
    c2, s2 = cos(ω / 2), sin(ω / 2)
    fat = s / (2μ)
    uI = fat * c2 * (κ - 1 + 2 * s2^2)
    vI = fat * s2 * (κ + 1 - 2 * c2^2)
    uII = fat * s2 * (κ + 1 + 2 * c2^2)
    vII = -fat * c2 * (κ - 1 - 2 * s2^2)
    return @SMatrix [uI uII; vI vII]
end

"""Crack-face polar angle: `+π` on the side with `n·e2 < 0` (upper for a right tip)."""
@inline function crack_omega(e2::SVector{2}, n::SVector{2})
    return dot(n, e2) < 0 ? π : -π
end

"""
Global Williams matrix `R * M_local` at `x`.

On a crack face pass `ω` (`±π`); otherwise `ω` is `atan2` in the tip frame.
Coincident twins have the same `x`, so the face angle cannot be inferred from `x`.
"""
function williams_global(μ, κ, tip::SVector{2}, e1::SVector{2}, e2::SVector{2}, x::SVector{2};
        ω::Union{Real,Nothing}=nothing)
    return tip_u_global(IsoTipField(float(μ), float(κ)), tip, e1, e2, x; ω)
end

# =============================================================================
# Sih–Paris–Irwin / Lekhnitskii first-order field (anisotropic)
# Sollero & Aliabadi, Int. J. Fract. 64 (1993); Sih, Paris, Irwin (1965).
# μ, p_k, q_k are in the *tip* frame (ligament = x₁).
# =============================================================================

"""
    sih_M_local(p::LekhnitskiiParams, ρ, ω) -> SMatrix{2,2}

Columns are in-plane displacement for unit `KI` and `KII` in the tip frame.
`p` must already be rotated into that frame (`lekhnitskii_rotate`).
Crack-face polar angle is ``±π``. Degenerate ``μ_1≈μ_2`` falls back to Williams.
"""
function sih_M_local(p::LekhnitskiiParams, ρ::Real, ω::Real)
    ρ <= 0 && return @SMatrix zeros(2, 2)
    μ1, μ2 = p.mi[1], p.mi[2]
    dμ = μ1 - μ2
    if abs(dμ) < 1e-10
        a = inv(p.C)
        G = 1 / real(a[3, 3])
        κ = 8 * real(a[1, 1]) * G - 1
        return williams_M_local(G, κ, ρ, ω)
    end
    p1, p2 = p.q[1, 1], p.q[1, 2]
    q1, q2 = p.q[2, 1], p.q[2, 2]
    γ1 = _lekhnitskii_gamma(μ1, ω)
    γ2 = _lekhnitskii_gamma(μ2, ω)
    s = sqrt(2ρ / π)
    A11 = (μ1 * p2 * γ2 - μ2 * p1 * γ1) / dμ
    A12 = (p2 * γ2 - p1 * γ1) / dμ
    A21 = (μ1 * q2 * γ2 - μ2 * q1 * γ1) / dμ
    A22 = (q2 * γ2 - q1 * γ1) / dμ
    return @SMatrix [s * real(A11) s * real(A12); s * real(A21) s * real(A22)]
end

"""`√(cosθ + μ sinθ)` continuous on `θ ∈ [-π, π]`, with `γ(0) = 1`.

Julia's principal `sqrt(-1) = i` maps both crack faces to the same value;
the lower face (`θ = -π`) must be `γ = -i` (Suo 1990 branch).
"""
function _lekhnitskii_gamma(μ::Complex, θ::Real)
    z = cos(θ) + μ * sin(θ)
    a = angle(z)
    if abs(sin(θ)) < 1e-14 && cos(θ) < 0
        a = θ >= 0 ? π : -π
    elseif θ < 0 && a > 0
        a -= 2π
    elseif θ > 0 && a < 0
        a += 2π
    end
    return sqrt(abs(z)) * cis(a / 2)
end

struct IsoTipField{T}
    μ::T
    κ::T
end
struct AnisoTipField{T}
    params::LekhnitskiiParams{T}
end

tip_M(f::IsoTipField, ρ, ω) = williams_M_local(f.μ, f.κ, ρ, ω)
tip_M(f::AnisoTipField, ρ, ω) = sih_M_local(f.params, ρ, ω)

function make_tip_field(props::Elasticity, e1, e2)
    return IsoTipField(props.mu, kappa(props.E, props.nu, props.plane_strain))
end
function make_tip_field(props::AnisotropicElasticity, e1, e2)
    B = parentmodule(@__MODULE__)
    α = atan(e1[2], e1[1])
    return AnisoTipField(B.lekhnitskii_rotate(props.params, α))
end

"""Global tip-field matrix `R * M_local` at `x` (Williams or Sih)."""
function tip_u_global(field, tip::SVector{2}, e1::SVector{2}, e2::SVector{2}, x::SVector{2};
        ω::Union{Real,Nothing}=nothing)
    r = x - tip
    ρ = norm(r)
    ρ < 1e-15 && return @SMatrix zeros(2, 2)
    ωv = ω === nothing ? atan(dot(e2, r), dot(e1, r)) : float(ω)
    return hcat(e1, e2) * tip_M(field, ρ, ωv)
end

"""Ligament frame: `e1` ahead of the tip, `e2` = 90° CCW."""
function tip_frame(dad, tip_pos::SVector{2})
    faceA = crack_face_nodes(dad; face=2)
    isempty(faceA) && error("tip_frame: no face-A nodes")
    sort!(faceA; by=i -> norm(dad.Nodes[i] - tip_pos))
    into = dad.Nodes[faceA[min(2, length(faceA))]] - tip_pos
    nrm = norm(into)
    nrm < 1e-14 && error("tip_frame: degenerate crack at $tip_pos")
    e1 = -into / nrm
    e2 = Point2D(-e1[2], e1[1])
    return e1, e2
end

# =============================================================================
# Geometric tips (element ends, not inset collocation)
# =============================================================================

"""Crack-element indices on faces 2 and 3."""
function _crack_elements(dad)
    return dad.crack_face_a, dad.crack_face_b
end

function _elem_end_points(dad, el)
    B = parentmodule(@__MODULE__)
    xj = dad.Nodes[el.index]
    poly = dad.element_type
    Nm, _ = B.shapefun(poly, -1.0)
    Np, _ = B.shapefun(poly, 1.0)
    p0 = zero(xj[1])
    p1 = zero(xj[1])
    @inbounds for k in eachindex(xj)
        p0 += Nm[1, k] * xj[k]
        p1 += Np[1, k] * xj[k]
    end
    return p0, p1
end

function _elem_point(dad, el, ξ)
    B = parentmodule(@__MODULE__)
    xj = dad.Nodes[el.index]
    N, dN = B.shapefun(dad.element_type, ξ)
    pg = zero(xj[1])
    dx = zero(xj[1])
    @inbounds for k in eachindex(xj)
        pg += N[1, k] * xj[k]
        dx += dN[1, k] * xj[k]
    end
    J = norm(dx)
    nrm = J > 0 ? B.tan2normal(dx / J) : zero(xj[1])
    return pg, nrm, J, view(N, 1, :)
end

"""Free ends of crack face-A polylines (valence-1 vertices).

Centre crack: two tips. Edge crack: mouth + tip. Double-edge: two mouths + two tips.
[`williams_tip_positions`](@ref) then drops ends that lie on the outer wall.
"""
function geometric_tips(dad)
    faceA = dad.crack_face_a
    isempty(faceA) && return SVector{2,Float64}[]
    ends = SVector{2,Float64}[]
    counts = Int[]
    for ie in faceA
        p0, p1 = _elem_end_points(dad, dad.elements[ie])
        for p in (p0, p1)
            idx = findfirst(q -> norm(p - q) < 1e-9, ends)
            if idx === nothing
                push!(ends, p)
                push!(counts, 1)
            else
                counts[idx] += 1
            end
        end
    end
    return [ends[i] for i in eachindex(ends) if counts[i] == 1]
end

"""Distance from `p` to the segment `a—b`."""
function _dist_to_segment(p, a, b)
    v = b - a
    L2 = dot(v, v)
    L2 < 1e-30 && return norm(p - a)
    t = clamp(dot(p - a, v) / L2, 0.0, 1.0)
    return norm(p - (a + t * v))
end

"""True if `p` lies on the outer boundary (within a fraction of a typical element)."""
function _on_outer_boundary(dad, p; tol=nothing)
    Ltyp = maximum(el.Length for el in dad.elements)
    tol === nothing && (tol = 0.05 * Ltyp)
    eq = dad.eq_type
    @inbounds for el in dad.elements
        eq[el.index[1]] == 1 || continue
        a, b = _elem_end_points(dad, el)
        _dist_to_segment(p, a, b) < tol && return true
    end
    return false
end

"""Williams tips: geometric crack ends that are not a mouth on the outer wall."""
function williams_tip_positions(dad)
    tips = geometric_tips(dad)
    return [p for p in tips if !_on_outer_boundary(dad, p)]
end

# =============================================================================
# Enriched elements
# =============================================================================

function _enriched_elements(dad, tip_pos; n_enr::Int=3)
    out = Int[]
    for face_els in (dad.crack_face_a, dad.crack_face_b)
        isempty(face_els) && continue
        d = [begin
            el = dad.elements[ie]
            m = mean(dad.Nodes[j] for j in el.index)
            norm(m - tip_pos)
        end for ie in face_els]
        perm = sortperm(d)
        ntake = min(n_enr, length(face_els))
        append!(out, face_els[perm[1:ntake]])
    end
    return unique!(out)
end

function _closest_end_ξ(dad, el, tip_pos)
    p0, p1 = _elem_end_points(dad, el)
    return norm(p0 - tip_pos) <= norm(p1 - tip_pos) ? -1.0 : 1.0
end

# =============================================================================
# Extra H columns (kernels × shifted Williams)
# =============================================================================

"""Shifted Williams at `x` given nodal Williams `ψ_nodes` and shape `N`."""
function _shifted_williams(ψx::SMatrix{2,2}, N, ψ_nodes)
    φ = ψx
    @inbounds for k in eachindex(ψ_nodes)
        φ -= N[k] * ψ_nodes[k]
    end
    return φ
end

"""
    assemble_xbem_columns!(dad, tips; n_enr=3) -> (Hε, frames, enriched)

`Hε` is `2n × 2 n_tips` with columns `(KI, KII)` per tip.
"""
function _kernel_Tφ(dad, f, pf, el, ξ, ωel, tip_pos, e1, e2, field, ψ_nodes)
    pg, nrm, J, Nrow = _elem_point(dad, el, ξ)
    J < 1e-30 && return nothing
    r = pg - pf
    norm(r) < 1e-30 && return nothing
    _, T = f(dad, r, nrm)
    ψx = tip_u_global(field, tip_pos, e1, e2, pg; ω=ωel)
    φ = _shifted_williams(ψx, Nrow, ψ_nodes)
    return T * φ * J
end

function assemble_xbem_columns!(dad, tips; n_enr::Int=3, npg::Int=24)
    B = parentmodule(@__MODULE__)
    B.has_cache(dad, :qsi) || B._init_quadrature!(dad, npg)
    props = dad.properties
    dim = 2
    n = dad.n
    n_tips = length(tips)
    Hε = zeros(dim * n, 2 * n_tips)
    eq = dad.eq_type
    twin = dad.twin
    frames = Tuple{SVector{2,Float64},SVector{2,Float64}}[]
    enriched_all = Vector{Int}[]
    poly = dad.element_type
    z22 = @SMatrix zeros(2, 2)

    for (it, tip_pos) in enumerate(tips)
        e1, e2 = tip_frame(dad, tip_pos)
        push!(frames, (e1, e2))
        field = make_tip_field(props, e1, e2)
        enr = _enriched_elements(dad, tip_pos; n_enr=n_enr)
        push!(enriched_all, enr)
        cols = (2(it - 1) + 1):(2it)

        for ie in enr
            el = dad.elements[ie]
            xj = dad.Nodes[el.index]
            nn = length(el)
            n̄ = mean(dad.Normal[j] for j in el.index)
            ωel = crack_omega(e2, n̄)
            ψ_nodes = [tip_u_global(field, tip_pos, e1, e2, xj[k]; ω=ωel) for k in 1:nn]

            for i in 1:n
                pf = dad.Nodes[i]
                nf = dad.Normal[i]
                ii = B.expand(i, dim)
                tipo = eq[i]
                f = if tipo == 3
                    (d, r, nrm) -> B.fundamental_hyper(d, r, nrm, nf)
                else
                    B.fundamental
                end
                on = B._source_on_element(el, i) || B._source_on_twin_element(el, i, twin)
                if on
                    # S φ ~ 1/r on HBIE self/twin (shifted φ ~ r); T φ is regular on CBIE.
                    # Richardson Guiggiani: SST tensors are for un-shifted S ~ 1/r².
                    a, _, _ = B.closest_point_1d(poly, xj, pf; ξ0=B._seed_1d(poly, xj, pf))
                    a = clamp(float(a), nextfloat(-1.0), prevfloat(1.0))
                    oh = tipo == 3 ? -1 : 0
                    _, Ih = B.guiggiani_GH(a; order_G=0, order_H=oh,
                        qsi=dad.qsi, w=dad.w) do ξ
                        val = _kernel_Tφ(dad, f, pf, el, ξ, ωel, tip_pos, e1, e2, field, ψ_nodes)
                        val === nothing && return z22, z22
                        return z22, val
                    end
                    Hε[ii, cols] .+= Ih
                else
                    η, ww = if B._near_element(pf, xj, el)
                        B.transform(dad, el, xj, pf)
                    else
                        dad.qsi, dad.w
                    end
                    Nmat, dN = B.shapefun(poly, η)
                    @inbounds for q in eachindex(η)
                        pg = zero(pf)
                        dx = zero(pf)
                        for k in 1:nn
                            pg += Nmat[q, k] * xj[k]
                            dx += dN[q, k] * xj[k]
                        end
                        J = norm(dx)
                        J < 1e-30 && continue
                        r = pg - pf
                        norm(r) < 1e-14 && continue
                        nrm = B.tan2normal(dx / J)
                        _, T = f(dad, r, nrm)
                        ψx = tip_u_global(field, tip_pos, e1, e2, pg; ω=ωel)
                        φ = _shifted_williams(ψx, view(Nmat, q, :), ψ_nodes)
                        Hε[ii, cols] .+= T * φ * (J * ww[q])
                    end
                end
            end
        end
    end
    return Hε, frames, enriched_all
end

# =============================================================================
# Tip tying  u⁺(tip) = u⁻(tip)
# =============================================================================

"""Lagrange basis at `x=0` for nodes at `s` (all `s>0`)."""
function _lagrange_at_0(s::AbstractVector)
    n = length(s)
    L = zeros(n)
    @inbounds for i in 1:n
        Li = 1.0
        for j in 1:n
            j == i && continue
            Li *= (0 - s[j]) / (s[i] - s[j])
        end
        L[i] = Li
    end
    return L
end

"""
    xbem_tying(dad, tips, frames; n_v=9) -> (Cu, Cε)

Andrade eqs. (43)–(45) / Fig. 4: a macro-element on each face, Lagrange in
arc length (``\\rho`` from the tip), evaluated at the geometric tip.

Shifted Williams vanishes at ``\\rho=0``, so ``C_\\varepsilon`` is
``-\\sum L_i\\psi_i`` — the interpolant of nodal Williams, not zero.
Interpolating in ``\\sqrt{\\rho}`` makes that sum vanish (Williams is linear
in ``\\sqrt{\\rho}``) and drops the SIF from the tying row.

`n_v` is the number of collocation nodes per face (paper: same three
quadratic elements as the enrichment, nine nodes; cap 10, Runge).
"""
function xbem_tying(dad, tips, frames; n_v::Int=9)
    n = dad.n
    n_tips = length(tips)
    nv = clamp(n_v, 2, 10)
    Cu = zeros(2 * n_tips, 2n)
    Cε = zeros(2 * n_tips, 2 * n_tips)
    props = dad.properties
    B = parentmodule(@__MODULE__)
    I2 = @SMatrix [1.0 0.0; 0.0 1.0]

    for (it, tip_pos) in enumerate(tips)
        e1, e2 = frames[it]
        field = make_tip_field(props, e1, e2)
        rows = (2(it - 1) + 1):(2it)
        contrib = zeros(2, 2n)
        c_block = zeros(2, 2 * n_tips)
        for (s, face) in zip((1.0, -1.0), (2, 3))
            idx = crack_face_nodes(dad; face=face)
            isempty(idx) && continue
            sort!(idx; by=i -> norm(dad.Nodes[i] - tip_pos))
            take = idx[1:min(nv, length(idx))]
            ρ = [norm(dad.Nodes[i] - tip_pos) for i in take]
            L = _lagrange_at_0(ρ)
            n̄ = mean(dad.Normal[i] for i in take)
            ωel = crack_omega(e2, n̄)
            φ = zero(I2)
            for (k, j) in enumerate(take)
                jj = B.expand(j, 2)
                contrib[:, jj] .+= s * L[k] * I2
                φ -= L[k] * tip_u_global(field, tip_pos, e1, e2, dad.Nodes[j]; ω=ωel)
            end
            c_block[:, (2(it - 1) + 1):(2it)] .+= s * φ
        end
        Cu[rows, :] .= contrib
        Cε[rows, :] .= c_block
    end
    return Cu, Cε
end

# =============================================================================
# Solve
# =============================================================================

"""
    assemble_xbem!(dad; n_enr=3, npg=24, n_v=9) -> (Hε, Cu, Cε, tips)

Requires dual `H,G` already on `dad`. Stores enrichment bookkeeping on the cache.
"""
function assemble_xbem!(dad; n_enr::Int=3, npg::Int=24, n_v::Int=9)
    B = parentmodule(@__MODULE__)
    B.has_cache(dad, :H) || assemble_dual_elasticity!(dad; npg=npg, threaded=false)
    tips = williams_tip_positions(dad)
    isempty(tips) && error("assemble_xbem!: no Williams tips (all ends on the outer wall?)")
    Hε, frames, enriched = assemble_xbem_columns!(dad, tips; n_enr=n_enr, npg=npg)
    Cu, Cε = xbem_tying(dad, tips, frames; n_v=n_v)
    B.set_cache!(dad; H_xbem=Hε, C_u=Cu, C_sif=Cε, xbem_tips=tips,
        xbem_frames=frames, xbem_enriched=enriched)
    return Hε, Cu, Cε, tips
end

"""
    solve_xbem!(dad; n_enr=3, npg=24, n_v=9) -> (u, KI, KII)

Dual assembly (if needed) → extra Williams columns + tip tying → mixed BC
solve. Caches `u`, `traction`, `KI`, `KII`.
"""
function solve_xbem!(dad; n_enr::Int=3, npg::Int=24, n_v::Int=9, threaded::Bool=false)
    B = parentmodule(@__MODULE__)
    B.has_cache(dad, :H) || assemble_dual_elasticity!(dad; npg=npg, threaded=threaded)
    B.has_cache(dad, :H_xbem) || assemble_xbem!(dad; n_enr=n_enr, npg=npg, n_v=n_v)
    B.applyBC(dad)
    A = dad.A
    b = dad.b
    Hε = dad.H_xbem
    Cu = dad.C_u
    Cε = dad.C_sif
    nd = size(A, 1)
    nc = size(Hε, 2)
    # tying acts on displacements: Neumann columns of the mixed unknown are u
    Cmix = copy(Cu)
    @inbounds for dof in 1:min(nd, length(dad.BC))
        if dad.BC[dof] == 0
            Cmix[:, dof] .= 0
        end
    end
    ntot = nd + nc
    Afull = zeros(ntot, ntot)
    Afull[1:nd, 1:nd] .= A
    Afull[1:nd, nd+1:ntot] .= Hε
    Afull[nd+1:ntot, 1:nd] .= Cmix
    Afull[nd+1:ntot, nd+1:ntot] .= Cε
    bfull = zeros(ntot)
    bfull[1:nd] .= b
    xfull = B.bem_linsolve(Afull, bfull)
    x = xfull[1:nd]
    c = xfull[nd+1:ntot]
    u = zeros(nd)
    traction = zeros(nd)
    B.split_sol!(dad, x, u, traction)
    n_tips = length(c) ÷ 2
    KI = [c[2k - 1] for k in 1:n_tips]
    KII = [c[2k] for k in 1:n_tips]
    B.set_cache!(dad; u=u, traction=traction, T=u, KI=KI, KII=KII, xbem_c=c)
    return u, KI, KII
end

# =============================================================================
# Edge-crack mesh (Civelek–Erdogan square plate)
# =============================================================================

"""Civelek & Erdogan (1982) as tabulated by Andrade & Leonel Table 1 (`KI / σ√(πa)`)."""
const _CIVELEK_F = Dict(
    0.2 => 1.488,
    0.3 => 1.848,
    0.4 => 2.324,
    0.5 => 3.010,
)
analytical_KI_edge_crack(σ, a, W) = σ * sqrt(π * a) * _CIVELEK_F[round(a / W; digits=1)]

"""
    mesh_edge_crack(; W=1, H=0.5, a=0.5, σ=1, ...) -> path

Square plate `[0,W]×[-H,H]` (`H=W/2` → height `W`) with an **edge crack**
from `(0,0)` to `(a,0)` as coincident type-5 twins.
"""
function mesh_edge_crack(; W=1.0, H=nothing, a=0.5, σ=1.0,
        ndiv_b=8, ndiv_h=8, ndiv_crack=8, ndiv_left=4,
        ordem=2, nome="edge_crack", show=false)
    H === nothing && (H = W / 2)
    a > 0 || error("mesh_edge_crack: a must be positive")
    a < W || error("mesh_edge_crack: a must be < W")
    B = parentmodule(@__MODULE__)
    gmsh = B.gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, 2H) / max(ndiv_b, 8)

    p1 = gmsh.model.geo.addPoint(0.0, -H, 0, lc)
    p2 = gmsh.model.geo.addPoint(W, -H, 0, lc)
    p3 = gmsh.model.geo.addPoint(W, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, H, 0, lc)
    pM = gmsh.model.geo.addPoint(0.0, 0.0, 0, lc / 2)
    pT = gmsh.model.geo.addPoint(a, 0.0, 0, lc / 2)

    lb = gmsh.model.geo.addLine(p1, p2)
    lr = gmsh.model.geo.addLine(p2, p3)
    lt = gmsh.model.geo.addLine(p3, p4)
    llu = gmsh.model.geo.addLine(p4, pM)
    lll = gmsh.model.geo.addLine(pM, p1)
    cA = gmsh.model.geo.addLine(pM, pT)
    cB = gmsh.model.geo.addLine(pT, pM)
    cl = gmsh.model.geo.addCurveLoop([lb, lr, lt, llu, lll])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [cA, cB], 2, s)

    gmsh.model.mesh.setTransfiniteCurve(lb, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lt, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lr, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(llu, ndiv_left)
    gmsh.model.mesh.setTransfiniteCurve(lll, ndiv_left)
    gmsh.model.mesh.setTransfiniteCurve(cA, ndiv_crack)
    gmsh.model.mesh.setTransfiniteCurve(cB, ndiv_crack)

    gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;1;$(-σ)")
    gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;$σ")
    gmsh.model.addPhysicalGroup(1, [lr, llu, lll], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [cA], -1, "5;2;5;2")
    gmsh.model.addPhysicalGroup(1, [cB], -1, "5;3;5;3")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(ordem)
    out = B.datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

function edge_crack_problem(; W=1.0, H=nothing, a=0.5, E=1.0, ν=0.3, σ=1.0,
        ndiv_b=8, ndiv_h=8, ndiv_crack=8, ndiv_left=4,
        plane_strain=true, ordem=2, nome="edge_crack")
    H === nothing && (H = W / 2)
    B = parentmodule(@__MODULE__)
    msh = mesh_edge_crack(; W, H, a, σ, ndiv_b, ndiv_h, ndiv_crack, ndiv_left,
        ordem, nome, show=false)
    props = B.Elasticity(E, ν, 1.0; plane_strain=plane_strain)
    dad = B.format2d(msh, props; tipo=ordem, pontointerno=false)
    prepare_crack!(dad)
    _pin_edge_plate!(dad; W=W, H=H)
    return dad
end

function _pin_edge_plate!(dad; W=1.0, H=0.5)
    eq = dad.eq_type
    function nearest(pred)
        best, bd = 0, Inf
        @inbounds for i in 1:dad.n
            eq[i] == 1 || continue
            p = dad.Nodes[i]
            pred(p) || continue
            d = abs(p[1]) + abs(p[2])
            d < bd && ((bd, best) = (d, i))
        end
        return best
    end
    i_bot = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && p[1] > 0.3W)
    i_right = nearest(p -> abs(p[1] - W) < 1e-6 * max(W, 1) && abs(p[2]) < 0.4H)
    i_bot2 = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && p[1] > 0.1W && p[1] < 0.4W)
    function set_dir!(inode, dir)
        inode == 0 && return
        dad.BC[2 * (inode - 1) + dir] = 0
        dad.BV[2 * (inode - 1) + dir] = 0.0
        return nothing
    end
    set_dir!(i_right, 1)
    set_dir!(i_bot, 2)
    if i_bot2 != 0 && i_bot2 != i_bot
        set_dir!(i_bot2, 1)
    elseif i_right != 0
        set_dir!(i_right, 2)
    end
    return dad
end

# =============================================================================
# Double-edge crack (Hattori, Alatawi & Trevelyan 2016, §5.3)
# =============================================================================

"""
    mesh_double_edge_crack(; W=1, H=1, a=0.5, σ=1, ...) -> path

Square plate `[-W,W]×[-H,H]` with **two edge cracks** on `y=0`:
left `[-W,-W+a]`, right `[W-a,W]`, each a coincident type-5 twin pair.
Hattori §5.3: `h/w=1`, `a/w=0.5` → `W=H=1`, `a=0.5`.
"""
function mesh_double_edge_crack(; W=1.0, H=nothing, a=0.5, σ=1.0,
        ndiv_b=6, ndiv_side=3, ndiv_crack=4,
        ordem=2, nome="double_edge_crack", show=false)
    H === nothing && (H = W)
    0 < a < W || error("mesh_double_edge_crack: need 0 < a < W")
    B = parentmodule(@__MODULE__)
    gmsh = B.gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, H) / max(ndiv_b, 8)

    p1 = gmsh.model.geo.addPoint(-W, -H, 0, lc)
    p2 = gmsh.model.geo.addPoint(W, -H, 0, lc)
    p3 = gmsh.model.geo.addPoint(W, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(-W, H, 0, lc)
    pLM = gmsh.model.geo.addPoint(-W, 0, 0, lc / 2)
    pLT = gmsh.model.geo.addPoint(-W + a, 0, 0, lc / 2)
    pRM = gmsh.model.geo.addPoint(W, 0, 0, lc / 2)
    pRT = gmsh.model.geo.addPoint(W - a, 0, 0, lc / 2)

    lb = gmsh.model.geo.addLine(p1, p2)
    lrl = gmsh.model.geo.addLine(p2, pRM)
    lru = gmsh.model.geo.addLine(pRM, p3)
    lt = gmsh.model.geo.addLine(p3, p4)
    llu = gmsh.model.geo.addLine(p4, pLM)
    lll = gmsh.model.geo.addLine(pLM, p1)
    # 180° symmetry (needed for off-axis fibres): left A is mouth→tip (+x, n down);
    # right A is mouth→tip (−x, n up), the rotate of left A, not a mirror.
    cLA = gmsh.model.geo.addLine(pLM, pLT)
    cLB = gmsh.model.geo.addLine(pLT, pLM)
    cRA = gmsh.model.geo.addLine(pRM, pRT)
    cRB = gmsh.model.geo.addLine(pRT, pRM)
    cl = gmsh.model.geo.addCurveLoop([lb, lrl, lru, lt, llu, lll])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [cLA, cLB, cRA, cRB], 2, s)

    gmsh.model.mesh.setTransfiniteCurve(lb, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lt, ndiv_b)
    for c in (lrl, lru, llu, lll)
        gmsh.model.mesh.setTransfiniteCurve(c, ndiv_side)
    end
    for c in (cLA, cLB, cRA, cRB)
        gmsh.model.mesh.setTransfiniteCurve(c, ndiv_crack)
    end

    gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;1;$(-σ)")
    gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;$σ")
    gmsh.model.addPhysicalGroup(1, [lrl, lru, llu, lll], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [cLA, cRA], -1, "5;2;5;2")
    gmsh.model.addPhysicalGroup(1, [cLB, cRB], -1, "5;3;5;3")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(ordem)
    out = B.datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

function double_edge_crack_problem(; W=1.0, H=nothing, a=0.5, σ=1.0,
        ndiv_b=6, ndiv_side=3, ndiv_crack=4,
        ordem=2, nome="double_edge_crack", props=nothing)
    H === nothing && (H = W)
    B = parentmodule(@__MODULE__)
    msh = mesh_double_edge_crack(; W, H, a, σ, ndiv_b, ndiv_side, ndiv_crack,
        ordem, nome, show=false)
    mat = props === nothing ? B.Elasticity(1.0, 0.3, 1.0; plane_strain=false) : props
    dad = B.format2d(msh, mat; tipo=ordem, pontointerno=false)
    prepare_crack!(dad)
    _pin_double_edge!(dad; W=W, H=H)
    return dad
end

"""Dirichlet pins on the bottom edge, left/right symmetric, away from the mouths.

`u_x` at mid-bottom (not on one side — that pollutes an off-axis crack);
`u_y` at a symmetric pair so rotation dies without a side bias.
"""
function _pin_double_edge!(dad; W=1.0, H=1.0)
    eq = dad.eq_type
    tol = 1e-6 * max(H, 1)
    function nearest(pred)
        best, bd = 0, Inf
        @inbounds for i in 1:dad.n
            eq[i] == 1 || continue
            p = dad.Nodes[i]
            pred(p) || continue
            d = abs(p[1]) + abs(p[2] + H)
            d < bd && ((bd, best) = (d, i))
        end
        return best
    end
    i_bc = nearest(p -> abs(p[2] + H) < tol && abs(p[1]) < 0.2W)
    i_br = nearest(p -> abs(p[2] + H) < tol && p[1] > 0.3W)
    i_bl = nearest(p -> abs(p[2] + H) < tol && p[1] < -0.3W)
    function set_dir!(inode, dir)
        inode == 0 && return
        dad.BC[2 * (inode - 1) + dir] = 0
        dad.BV[2 * (inode - 1) + dir] = 0.0
        return nothing
    end
    set_dir!(i_bc, 1)
    set_dir!(i_bl, 2)
    set_dir!(i_br, 2)
    if i_bl == 0 || i_br == 0
        set_dir!(i_bc, 2)
    end
    return dad
end
