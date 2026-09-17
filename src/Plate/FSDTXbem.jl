# Reissner plate XBEM: Andrade–Leonel extra columns + tip tying, with the
# Hui–Zehnder / Sosa–Eischen near-tip field (Dolbow–Moës–Belytschko 2000
# eqs. 36). Extra DOFs are Dolbow (K1, K2, K3) per tip:
#   K1 = lim √(2r) M22,  K2 = lim √(2r) M12,  K3 = lim √(2r) Q2.
# Dirgantara K1b = √π K1, so Table 10.1  F = K1b/(Mo√(πa)) = K1/(Mo√a).

# =============================================================================
# Hui–Zehnder / Dolbow (36) in the tip frame
# =============================================================================

"""
    hui_zehnder_M_local(E, ν, h, ρ, ω) -> SMatrix{3,3}

Columns are `(ψ₁, ψ₂, w)` in the tip frame for unit Dolbow `(K1, K2, K3)`.
`ω` from the ligament; crack faces `±π`. Rotations `ψ ~ √ρ`, bending
deflection `w ~ ρ^{3/2}`, shear `w ~ √ρ` (Knowles–Wang / Hui–Zehnder).
"""
function hui_zehnder_M_local(E::Real, ν::Real, h::Real, ρ::Real, ω::Real)
    ρ <= 0 && return @SMatrix zeros(3, 3)
    μ = E / (2 * (1 + ν))
    s = sqrt(2 * ρ)
    c2, s2 = cos(ω / 2), sin(ω / 2)
    c, sθ = cos(ω), sin(ω)
    c32, s32 = cos(3ω / 2), sin(3ω / 2)
    kψ = 6 * s / (E * h^3)
    kw3 = 6 * s / (5 * h * μ)
    kwb = 6 * s * ρ / (E * h^3)
    kψ3 = kwb * 8 / 15
    ψ1_I = kψ * c2 * (4 - (1 + ν) * (1 + c))
    ψ2_I = kψ * (4 * s2 - (1 + ν) * c2 * sθ)
    w_I = kwb * ((7 + ν) / 3 * c32 - (1 - ν) * c2)
    ψ1_II = kψ * s2 * (4 + (1 + ν) * (1 + c))
    ψ2_II = kψ * (-2 * c2 * (1 - ν) + (1 + ν) * s2 * sθ)
    w_II = kwb * (-(5 + 3ν) / 3 * s32 + (1 - ν) * s2)
    ψ1_III = kψ3 * (-s2 - (1 + 3ν) * c2 * sθ)
    ψ2_III = kψ3 * c2 * (1 + (1 + 3ν) * c)
    w_III = kw3 * s2
    return @SMatrix [ψ1_I ψ1_II ψ1_III; ψ2_I ψ2_II ψ2_III; w_I w_II w_III]
end

"""Global `(ψx, ψy, w)` for unit `(K1, K2, K3)`. Pass `ω` on a crack face."""
function hui_zehnder_global(E, ν, h, tip::Point2D, e1::Point2D, e2::Point2D, x::Point2D;
        ω::Union{Real,Nothing}=nothing)
    r = x - tip
    ρ = norm(r)
    ρ < 1e-15 && return @SMatrix zeros(3, 3)
    ωv = ω === nothing ? atan(dot(e2, r), dot(e1, r)) : float(ω)
    Mloc = hui_zehnder_M_local(E, ν, h, ρ, ωv)
    R = @SMatrix [e1[1] e2[1] 0; e1[2] e2[2] 0; 0 0 1]
    return R * Mloc
end

@inline function _crack_omega_fsdt(e2::Point2D, n::Point2D)
    return dot(n, e2) < 0 ? π : -π
end

function _fsdt_tip_frame(mesh::FSDTMesh, tip_pos::Point2D)
    eq = mesh.eq_type
    faceA = Int[]
    for i in eachindex(mesh.nodes)
        eq[i] == 2 && push!(faceA, i)
    end
    isempty(faceA) && error("_fsdt_tip_frame: no face-A nodes")
    sort!(faceA; by=i -> norm(mesh.nodes[i] - tip_pos))
    into = mesh.nodes[faceA[min(2, length(faceA))]] - tip_pos
    nrm = norm(into)
    nrm < 1e-14 && error("_fsdt_tip_frame: degenerate tip $tip_pos")
    e1 = -into / nrm
    e2 = Point2D(-e1[2], e1[1])
    return e1, e2
end

function _fsdt_crack_el_ids(mesh::FSDTMesh)
    eq = mesh.eq_type
    a = Int[]
    b = Int[]
    for (ie, el) in enumerate(mesh.elements)
        t = eq[el.index[1]]
        t == 2 && push!(a, ie)
        t == 3 && push!(b, ie)
    end
    return a, b
end

function _fsdt_enriched_els(mesh::FSDTMesh, tip_pos; n_enr::Int=3)
    a, b = _fsdt_crack_el_ids(mesh)
    out = Int[]
    for face_els in (a, b)
        isempty(face_els) && continue
        d = [begin
            el = mesh.elements[ie]
            m = sum(mesh.nodes[j] for j in el.index) / length(el.index)
            norm(m - tip_pos)
        end for ie in face_els]
        perm = sortperm(d)
        append!(out, face_els[perm[1:min(n_enr, length(face_els))]])
    end
    return unique!(out)
end

function _shifted_hz(ψx::SMatrix{3,3}, N, ψ_nodes)
    φ = ψx
    @inbounds for k in eachindex(ψ_nodes)
        φ -= N[k] * ψ_nodes[k]
    end
    return φ
end

function _lagrange0(s::AbstractVector)
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

# =============================================================================
# Extra H columns: T * φ  (shifted Hui–Zehnder)
# =============================================================================

function _Tφ_el!(Hε, rows, cols, el, pf, nξ, tipo, props, poly, qs, ws,
        tip, e1, e2, ωel, ψ_nodes; nsub=8)
    D = bending_stiffness(props)
    ν = props.ν
    λ = reissner_lambda(props)
    E, h = props.E, props.h
    x1, x3 = el.geo[1], el.geo[end]
    ndiv = nsub
    dξ = 2 / ndiv
    for k in 1:ndiv
        ξa = -1 + (k - 1) * dξ
        ξb = -1 + k * dξ
        eet = 0.0
        xa = (1 - ξa) / 2 * x1[1] + (1 + ξa) / 2 * x3[1]
        xb = (1 - ξb) / 2 * x1[1] + (1 + ξb) / 2 * x3[1]
        den = xa - xb
        abs(den) > 1e-14 && (eet = (xa + xb - 2 * pf[1]) / den)
        if abs(x3[2] - x1[2]) > abs(x3[1] - x1[1])
            ya = (1 - ξa) / 2 * x1[2] + (1 + ξa) / 2 * x3[2]
            yb = (1 - ξb) / 2 * x1[2] + (1 + ξb) / 2 * x3[2]
            den = ya - yb
            abs(den) > 1e-14 && (eet = (ya + yb - 2 * pf[2]) / den)
        end
        eet = clamp(eet, -0.999, 0.999)
        Jsub = 0.5 * (ξb - ξa)
        for ig in eachindex(qs)
            ξt, Jt = _telles(qs[ig], eet)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt
            abs(ξ) > 1 + 1e-12 && continue
            pg, J, n̂ = elem_geom(el, ξ)
            R = norm(pg - pf)
            R < 1e-14 && continue
            Nf, _ = shapefun(poly, ξ)
            if tipo == 3
                _, P = reissner_hbie_kernels(pg, pf, n̂, nξ, D, ν, λ)
            else
                _, P, _ = fsdt_kernels(pg, pf, n̂, D, ν, λ)
            end
            ψx = hui_zehnder_global(E, ν, h, tip, e1, e2, pg; ω=ωel)
            φ = _shifted_hz(ψx, view(Nf, 1, :), ψ_nodes)
            Hε[rows, cols] .+= P * φ .* (J * ws[ig] * Jt * Jsub)
        end
    end
    return nothing
end

"""
    assemble_fsdt_xbem_columns!(mesh, tips; n_enr=3, npg=12, nsub=8)
        -> (Hε, frames, enriched)

`Hε` is `3n × 3 n_tips` with columns `(K1, K2, K3)` per tip (Dolbow).
"""
function assemble_fsdt_xbem_columns!(mesh::FSDTMesh, tips; n_enr::Int=3,
        npg::Int=12, nsub::Int=8)
    n = _n(mesh)
    n_tips = length(tips)
    Hε = zeros(3n, 3 * n_tips)
    eq = mesh.eq_type
    twin = mesh.twin
    props = mesh.props
    poly = mesh.element_type
    qs, ws = gausslegendre(npg)
    frames = Tuple{Point2D,Point2D}[]
    enriched_all = Vector{Int}[]
    for (it, tip_pos) in enumerate(tips)
        e1, e2 = _fsdt_tip_frame(mesh, tip_pos)
        push!(frames, (e1, e2))
        enr = _fsdt_enriched_els(mesh, tip_pos; n_enr=n_enr)
        push!(enriched_all, enr)
        cols = (3(it - 1) + 1):(3it)
        E, ν, h = props.E, props.ν, props.h
        for ie in enr
            el = mesh.elements[ie]
            nn = length(el.index)
            n̄ = sum(mesh.Normal[j] for j in el.index) / nn
            ωel = _crack_omega_fsdt(e2, n̄)
            ψ_nodes = [hui_zehnder_global(E, ν, h, tip_pos, e1, e2, mesh.nodes[el.index[k]];
                ω=ωel) for k in 1:nn]
            for i in 1:n
                # Self/twin: shifted φ=0 at the source; Dual already has the
                # ½I free term on u. Elasticity uses Guiggiani here; Telles of
                # raw HBIE T*φ pollutes extra columns (F ~ 8× Sih). Skip.
                (_on_el(el, i) || _on_twin_el(el, i, twin)) && continue
                pf = mesh.nodes[i]
                nξ = mesh.Normal[i]
                rows = 3i-2:3i
                tipo = eq[i]
                _Tφ_el!(Hε, rows, cols, el, pf, nξ, tipo, props, poly, qs, ws,
                    tip_pos, e1, e2, ωel, ψ_nodes; nsub=nsub)
            end
        end
    end
    return Hε, frames, enriched_all
end

"""Tip tying `u⁺(0)=u⁻(0)` (Andrade eqs. 43–45). 3 rows per tip.

Same as in-plane `xbem_tying`: Lagrange in arc length `ρ`, **not** `√ρ`.
Hui–Zehnder `ψ~√ρ` is not linear in `ρ`, so `Cε=−∑L_i ψ_i` is the
extrapolated Williams opening at the geometric tip and the extra DOFs
are the SIFs (`K = interpolant(Δu) / interpolant(Δψ)`). Lagrange in `√ρ`
makes `Cε=0` and drops `K` from the tying row.
"""
function xbem_tying_fsdt(mesh::FSDTMesh, tips, frames; n_v::Int=3)
    n = _n(mesh)
    n_tips = length(tips)
    nv = clamp(n_v, 2, 10)
    Cu = zeros(3 * n_tips, 3n)
    Cε = zeros(3 * n_tips, 3 * n_tips)
    eq = mesh.eq_type
    props = mesh.props
    E, ν, h = props.E, props.ν, props.h
    I3 = @SMatrix [1.0 0 0; 0 1.0 0; 0 0 1.0]
    for (it, tip_pos) in enumerate(tips)
        e1, e2 = frames[it]
        rows = (3(it - 1) + 1):(3it)
        contrib = zeros(3, 3n)
        c_block = zeros(3, 3 * n_tips)
        for (s, face) in zip((1.0, -1.0), (2, 3))
            idx = [i for i in 1:n if eq[i] == face]
            isempty(idx) && continue
            sort!(idx; by=i -> norm(mesh.nodes[i] - tip_pos))
            take = idx[1:min(nv, length(idx))]
            ρ = [norm(mesh.nodes[i] - tip_pos) for i in take]
            L = _lagrange0(ρ)
            n̄ = sum(mesh.Normal[i] for i in take) / length(take)
            ωel = _crack_omega_fsdt(e2, n̄)
            φ = zero(I3)
            for (k, j) in enumerate(take)
                contrib[:, 3j-2:3j] .+= s * L[k] * I3
                φ -= L[k] * hui_zehnder_global(E, ν, h, tip_pos, e1, e2, mesh.nodes[j]; ω=ωel)
            end
            c_block[:, (3(it - 1) + 1):(3it)] .+= s * φ
        end
        Cu[rows, :] .= contrib
        Cε[rows, :] .= c_block
    end
    return Cu, Cε
end

# =============================================================================
# Assemble / solve
# =============================================================================

"""
    assemble_fsdt_xbem!(mesh; n_enr=3, npg=12, nsub=8, n_v=3)
        -> (Hε, Cu, Cε, tips)

Requires dual `H,G` already on `mesh`. `n_v=3` is one quadratic tip
element (in-plane XBEM uses 9; plate tying mixes `ψ~√ρ` and `w~ρ^{3/2}`
so more nodes Runge-oscillate `Cε`).
"""
function assemble_fsdt_xbem!(mesh::FSDTMesh; n_enr::Int=3, npg::Int=12,
        nsub::Int=8, n_v::Int=3)
    isempty(mesh.H) && assemble_fsdt_dual!(mesh; npg=npg, nsub=nsub)
    tips = _fsdt_geometric_tips(mesh)
    isempty(tips) && error("assemble_fsdt_xbem!: no geometric crack tips")
    Hε, frames, _ = assemble_fsdt_xbem_columns!(mesh, tips; n_enr=n_enr, npg=npg, nsub=nsub)
    Cu, Cε = xbem_tying_fsdt(mesh, tips, frames; n_v=n_v)
    return Hε, Cu, Cε, tips, frames
end

"""
    solve_fsdt_xbem!(mesh; n_enr=3, npg=12, nsub=8, n_v=9)
        -> (u, K1, K2, K3)

Dolbow `(K1,K2,K3)` per tip (right tip first if `x` increases). Table 10.1
`F = K1 / (Mo √a)` because Dirgantara `K1b = √π K1`.
"""
function solve_fsdt_xbem!(mesh::FSDTMesh; n_enr::Int=3, npg::Int=12,
        nsub::Int=8, n_v::Int=3)
    isempty(mesh.H) && assemble_fsdt_dual!(mesh; npg=npg, nsub=nsub)
    Hε, Cu, Cε, tips, frames = assemble_fsdt_xbem!(mesh; n_enr=n_enr, npg=npg,
        nsub=nsub, n_v=n_v)
    A, b, is_kin, known = apply_bc_fsdt(mesh)
    nd = size(A, 1)
    nc = size(Hε, 2)
    Cmix = copy(Cu)
    Cεs = copy(Cε)
    @inbounds for dof in 1:min(nd, length(mesh.BC))
        if mesh.BC[dof] == 0
            Cmix[:, dof] .= 0
        end
    end
    # Hui–Zehnder ~ 1/(E h³); tying rows are ~1e-11 of ||H|| and vanish in
    # float64. Row-equilibrate (same constraint). In-plane Williams/μ is larger.
    @inbounds for r in 1:size(Cmix, 1)
        nrm = max(maximum(abs, view(Cmix, r, :)), maximum(abs, view(Cεs, r, :)), 1e-30)
        Cmix[r, :] ./= nrm
        Cεs[r, :] ./= nrm
    end
    ntot = nd + nc
    Afull = zeros(ntot, ntot)
    Afull[1:nd, 1:nd] .= A
    Afull[1:nd, nd+1:ntot] .= Hε
    Afull[nd+1:ntot, 1:nd] .= Cmix
    Afull[nd+1:ntot, nd+1:ntot] .= Cεs
    bfull = zeros(ntot)
    bfull[1:nd] .= b
    xfull = Afull \ bfull
    x = xfull[1:nd]
    c = xfull[nd+1:ntot]
    u = zeros(nd)
    t = zeros(_nb(mesh))
    for dof in 1:nd
        if is_kin[dof]
            u[dof] = known[dof]
            dof <= _nb(mesh) && (t[dof] = x[dof])
        else
            u[dof] = x[dof]
            if dof <= _nb(mesh)
                t[dof] = known[dof]
            end
        end
    end
    mesh.u = u
    mesh.t = t
    n_tips = length(c) ÷ 3
    K1 = [c[3k - 2] for k in 1:n_tips]
    K2 = [c[3k - 1] for k in 1:n_tips]
    K3 = [c[3k] for k in 1:n_tips]
    return u, K1, K2, K3, tips, frames
end

function assemble_fsdt_xbem!(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    m = FSDTMesh(dad)
    ret = assemble_fsdt_xbem!(m; kwargs...)
    _sync_fsdt_mesh!(dad, m)
    return ret
end

function solve_fsdt_xbem!(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    m = FSDTMesh(dad)
    ret = solve_fsdt_xbem!(m; kwargs...)
    _sync_fsdt_mesh!(dad, m)
    return ret
end
