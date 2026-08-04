# Dual interpolation Boundary Face Method with Hermite-type MLS (DiBFM-HMLS)
# Zhang, He, Chi, Lin — Appl. Math. Modelling 81 (2020) 457–472
#
# Extends DLIM (Zhang et al. 2017): second-layer uses Hermite MLS in Cartesian
# coordinates so influence domains can span adjacent edges (short edges / chamfers).
# Incomplete quadratic basis p = [1, x̃, ỹ, x̃ỹ, x̃²+ỹ²] avoids singular moment
# matrices on axis-aligned edges.
#
# Virtual DOFs couple both u and q:
#   u_v = Φuu u_s + Φuq q_s
#   q_v = Φqu u_s + Φqq q_s

export DiBFMData, dibfm_from_bemdata, assemble_dibfm!, solve_dibfm!
export dibfm_rel_error, solve_dibfm_laplace
export compare_dlim_dibfm, build_dibfm_condensation!

"""
DiBFM data: DLIM mesh topology plus second-layer condensation
(`:hmls` Hermite-MLS or `:rbf` / `:rbf_hermite` 2D RBF).

Blocks `Φuu, Φuq, Φqu, Φqq` implement
``u_v = Φuu u_s + Φuq q_s``, ``q_v = Φqu u_s + Φqq q_s``.
"""
mutable struct DiBFMData
    dlim::DLIMData
    Φuu::Matrix{Float64}
    Φuq::Matrix{Float64}
    Φqu::Matrix{Float64}
    Φqq::Matrix{Float64}
    A_u::Matrix{Float64}
    A_q::Matrix{Float64}
    T::Vector{Float64}
    q::Vector{Float64}
    second_layer::Symbol           # :hmls | :rbf | :rbf_hermite
end

# =============================================================================
# Build
# =============================================================================

function dibfm_from_bemdata(dad::BEMdata{<:Laplace})
    d = dlim_from_bemdata(dad)
    n_s = length(d.source_pos)
    n_v = length(d.virt_global)
    return DiBFMData(d,
        zeros(n_v, n_s), zeros(n_v, n_s), zeros(n_v, n_s), zeros(n_v, n_s),
        zeros(0, 0), zeros(0, 0), zeros(n_s), zeros(n_s), :hmls)
end

# =============================================================================
# Incomplete quadratic basis + normal derivative (paper Eqs. 10–11)
# =============================================================================

"""
    _hmls_basis(x, y, xe, ye, h, nx, ny) -> (p, pn)

Incomplete quadratic monomials and ``∂p/∂n`` at ``(x,y)`` with fixed point
``(xe,ye)``, scale ``h``, and outward normal ``(nx,ny)``.
"""
function _hmls_basis(x, y, xe, ye, h, nx, ny)
    h = max(h, 1e-14)
    x̃ = (x - xe) / h
    ỹ = (y - ye) / h
    # p = [1, x̃, ỹ, x̃ ỹ, x̃² + ỹ²]
    p = SVector{5,Float64}(1.0, x̃, ỹ, x̃ * ỹ, x̃^2 + ỹ^2)
    # ∂/∂n with chain rule (nx/h, ny/h)
    cx, cy = nx / h, ny / h
    pn = SVector{5,Float64}(
        0.0,
        cx,
        cy,
        x̃ * cy + ỹ * cx,
        2x̃ * cx + 2ỹ * cy,
    )
    return p, pn
end

"""Gaussian weight with rectangular support (paper Eq. 9, simplified isotropic)."""
function _hmls_weight(dx, dy, dx_max, dy_max; c=0.3)
    dx_max = max(dx_max, 1e-14)
    dy_max = max(dy_max, 1e-14)
    if abs(dx) > dx_max || abs(dy) > dy_max
        return 0.0
    end
    cx = c * dx_max
    cy = c * dy_max
    # separable Gaussian bump vanishing at support boundary
    wx = (exp(-(dx / cx)^2) - exp(-(dx_max / cx)^2)) / (1 - exp(-(dx_max / cx)^2) + 1e-30)
    wy = (exp(-(dy / cy)^2) - exp(-(dy_max / cy)^2)) / (1 - exp(-(dy_max / cy)^2) + 1e-30)
    return max(wx * wy, 0.0)
end

# =============================================================================
# HMLS condensation for one virtual node
# =============================================================================

"""
Hermite MLS shape rows at virtual node `iv` using neighbour source indices `ids`.

Returns 4 vectors of length `n_s` (sparse via ids): Φuu, Φuq, Φqu, Φqq rows.
"""
function _hmls_rows(d::DLIMData, iv::Int, ids::Vector{Int}, n_v_src::Vector{Point2D})
    n_s = length(d.source_pos)
    Φuu = zeros(n_s)
    Φuq = zeros(n_s)
    Φqu = zeros(n_s)
    Φqq = zeros(n_s)
    isempty(ids) && return Φuu, Φuq, Φqu, Φqq

    pv = d.all_pos[iv]
    nv = n_v_src[iv - n_s]   # virtual normal
    # scale h from neighbour spacing
    if length(ids) >= 2
        h = 1e-3
        for i in 1:length(ids), j in (i + 1):length(ids)
            h = max(h, norm(d.source_pos[ids[i]] - d.source_pos[ids[j]]))
        end
    else
        h = 1e-3
    end
    h = max(h, 1e-6)
    # support box
    dx_max = 2.5 * h
    dy_max = 2.5 * h

    M = length(ids)
    # build C (5×5), and Bf, Bfn columns
    C = zeros(5, 5)
    # store weighted p_I, pn_I for each neighbour
    w_list = Float64[]
    p_list = SVector{5,Float64}[]
    pn_list = SVector{5,Float64}[]
    id_ok = Int[]
    for j in ids
        xs = d.source_pos[j]
        ns = d.normals_src[j]
        dx, dy = pv[1] - xs[1], pv[2] - xs[2]
        w = _hmls_weight(dx, dy, dx_max, dy_max)
        w < 1e-14 && continue
        pI, pnI = _hmls_basis(xs[1], xs[2], pv[1], pv[2], h, ns[1], ns[2])
        C .+= w * (pI * pI' + pnI * pnI')
        push!(w_list, w)
        push!(p_list, pI)
        push!(pn_list, pnI)
        push!(id_ok, j)
    end
    isempty(id_ok) && return Φuu, Φuq, Φqu, Φqq

    # regularize C
    C .+= 1e-12 * tr(C) / 5 * I(5)
    Cinv = try
        inv(C)
    catch
        return Φuu, Φuq, Φqu, Φqq
    end

    pv_p, pv_pn = _hmls_basis(pv[1], pv[2], pv[1], pv[2], h, nv[1], nv[2])
    # α_u = C^{-1} p_v,  α_q = C^{-1} pn_v
    αu = Cinv * pv_p
    αq = Cinv * pv_pn

    for (k, j) in enumerate(id_ok)
        w = w_list[k]
        pI = p_list[k]
        pnI = pn_list[k]
        # u_v contrib
        Φuu[j] = w * dot(αu, pI)
        Φuq[j] = w * dot(αu, pnI)
        # q_v = ∂u/∂n at v
        Φqu[j] = w * dot(αq, pI)
        Φqq[j] = w * dot(αq, pnI)
    end
    return Φuu, Φuq, Φqu, Φqq
end

"""Normals at virtual nodes (average of adjacent element end normals)."""
function _virtual_normals(d::DLIMData)
    n_s = length(d.source_pos)
    n_v = length(d.virt_global)
    acc = [Point2D(0.0, 0.0) for _ in 1:n_v]
    cnt = zeros(Int, n_v)
    for el in d.elements
        # left virtual
        vL = el[1] - n_s
        vR = el[end] - n_s
        # element tangent from left to right
        t = d.all_pos[el[end]] - d.all_pos[el[1]]
        L = norm(t)
        L < 1e-14 && continue
        n̂ = Point2D(t[2] / L, -t[1] / L)
        acc[vL] += n̂
        acc[vR] += n̂
        cnt[vL] += 1
        cnt[vR] += 1
    end
    nrm = Vector{Point2D}(undef, n_v)
    for v in 1:n_v
        if cnt[v] == 0
            nrm[v] = Point2D(1.0, 0.0)
        else
            a = acc[v] / cnt[v]
            nrm[v] = a / (norm(a) + eps())
        end
    end
    return nrm
end

"""Neighbour source indices within Cartesian radius `R` (min `nmin` points)."""
function _dibfm_neighbours(d::DLIMData, pv, R; nmin=6)
    n_s = length(d.source_pos)
    ids = Int[]
    for j in 1:n_s
        if norm(d.source_pos[j] - pv) < R
            push!(ids, j)
        end
    end
    if length(ids) < nmin
        dists = sort([(norm(d.source_pos[j] - pv), j) for j in 1:n_s]; by=first)
        for (_, j) in dists
            j in ids || push!(ids, j)
            length(ids) >= min(nmin + 2, n_s) && break
        end
    end
    return ids
end

"""
    build_dibfm_condensation!(dib; method=:hmls, radius_factor=2.5, rbf=PHS(3; poly_deg=0))

Second-layer options (Cartesian support, can span adjacent edges):

| `method` | Description |
|----------|-------------|
| `:hmls` | Hermite MLS, incomplete quadratic basis (Zhang et al. 2020) |
| `:rbf` | 2D RBF on ``u`` and ``q`` separately (``Φuq=Φqu=0``) |
| `:rbf_hermite` | 2D Hermite RBF coupling ``u`` and ``∂u/∂n=q`` |
"""
function build_dibfm_condensation!(dib::DiBFMData; method=:hmls, radius_factor=2.5,
    rbf=PHS(3; poly_deg=0))

    method = Symbol(lowercase(string(method)))
    method in (:hmls, :rbf, :rbf_hermite, :rbfh) ||
        error("unknown DiBFM second-layer $method (use :hmls, :rbf, :rbf_hermite)")
    method == :rbfh && (method = :rbf_hermite)
    dib.second_layer = method

    d = dib.dlim
    n_s = length(d.source_pos)
    n_v = length(d.virt_global)
    Φuu = zeros(n_v, n_s)
    Φuq = zeros(n_v, n_s)
    Φqu = zeros(n_v, n_s)
    Φqq = zeros(n_v, n_s)

    nrm_v = _virtual_normals(d)
    hs = [norm(d.all_pos[el[end]] - d.all_pos[el[1]]) for el in d.elements]
    hmed = median(hs)
    R = radius_factor * hmed

    for v in 1:n_v
        iv = n_s + v
        pv = d.all_pos[iv]
        ids = _dibfm_neighbours(d, pv, R; nmin=method == :rbf ? 4 : 6)

        if method == :hmls
            uu, uq, qu, qq = _hmls_rows(d, iv, ids, nrm_v)
        elseif method == :rbf
            uu, uq, qu, qq = _rbf2d_rows(d, iv, ids, nrm_v, rbf; hermite=false)
        else
            uu, uq, qu, qq = _rbf2d_rows(d, iv, ids, nrm_v, rbf; hermite=true)
        end
        Φuu[v, :] .= uu
        Φuq[v, :] .= uq
        Φqu[v, :] .= qu
        Φqq[v, :] .= qq
    end
    dib.Φuu = Φuu
    dib.Φuq = Φuq
    dib.Φqu = Φqu
    dib.Φqq = Φqq
    return dib
end

# keep old name as alias
build_hmls_condensation!(dib::DiBFMData; kwargs...) =
    build_dibfm_condensation!(dib; method=:hmls, kwargs...)

# -----------------------------------------------------------------------------
# 2D RBF second layer (decoupled or Hermite-coupled)
# -----------------------------------------------------------------------------

"""φ = r³ PHS and radial derivative φ' = 3 r² (used if `rbf` is PHS3-like)."""
function _phi_dphi(rbf, r2)
    φ = rbf(r2)
    r = sqrt(max(r2, 0.0))
    # generic finite-diff φ' if not PHS; for r³: φ'=3r²
    dφ = 3 * r  # d(r³)/dr / r * r wait: ∇φ = φ'(r) x̂, |∇φ|=|φ'|
    # For φ=r^3, φ'=3r^2, ∂φ/∂x_k = 3r (x_k)
    return φ, r, 3 * r   # returns φ, r, coeff such that ∇φ = coeff * (x-xj)
end

function _rbf2d_rows(d::DLIMData, iv::Int, ids::Vector{Int}, nrm_v, rbf; hermite::Bool=false)
    n_s = length(d.source_pos)
    Φuu = zeros(n_s)
    Φuq = zeros(n_s)
    Φqu = zeros(n_s)
    Φqq = zeros(n_s)
    isempty(ids) && return Φuu, Φuq, Φqu, Φqq

    pv = d.all_pos[iv]
    n_s_loc = length(d.source_pos)
    nv = nrm_v[iv - n_s_loc]
    M = length(ids)
    pts = [d.source_pos[j] for j in ids]
    ns = [d.normals_src[j] for j in ids]

    if !hermite
        # decoupled 2D RBF via shared rbf_cardinal
        Φuu .= rbf_cardinal(pv, d.source_pos, ids; basis=rbf, ridge=1e-12)
        Φqq .= rbf_cardinal(pv, d.source_pos, ids; basis=rbf, ridge=1e-12)
        return Φuu, Φuq, Φqu, Φqq
    end

    # --- Hermite RBF: s(x) = Σ α_j φ_j + Σ β_j ψ_j + γ·p
    # φ_j = r^3, ψ_j = ∇_{x_j} φ · n_j = 3r (x_j - x)·n_j
    # collocation: s(x_i)=u_i, ∂s/∂n_i = q_i
    npoly = 3
    Nsys = 2M + npoly
    K = zeros(Nsys, Nsys)
    # blocks among α, β
    @inbounds for j in 1:M, i in 1:M
        rij = pts[i] - pts[j]
        r2 = dot(rij, rij)
        r = sqrt(max(r2, 0.0))
        φ = rbf(r2)
        # ∂φ_j/∂n_i at x_i: ∇_x φ · n_i = 3r (x_i-x_j)·n_i
        coeff = 3 * r   # ∇φ = coeff * (x - xj)
        dφ_dni = coeff * dot(rij, ns[i])          # at xi, x-xj = rij
        # ψ_j(x_i) = 3 r (xj - xi)·n_j = -3r (xi-xj)·n_j
        ψ = -coeff * dot(rij, ns[j])
        # ∂ψ_j/∂n_i at xi — FD-like via derivative of ψ
        # ψ_j(x) = 3|x-xj| (xj-x)·n_j
        # Use automatic structure for r³ Hermite (Fasshauer):
        # ∂/∂n_i ψ_j = ∂/∂n_i [3 r (xj-x)·n_j]
        # = 3[(xi-xj)/r · n_i] (xj-xi)·n_j + 3r (-n_i·n_j)
        dψ_dni = if r < 1e-14
            # same point: limit
            0.0
        else
            3 * (dot(rij, ns[i]) / r) * dot(-rij, ns[j]) + 3 * r * (-dot(ns[i], ns[j]))
        end
        # when i==j, r=0: φ=0, use ridge later
        K[i, j] = φ                    # s from α
        K[i, M+j] = ψ                  # s from β
        K[M+i, j] = dφ_dni             # ∂s/∂n from α
        K[M+i, M+j] = dψ_dni           # ∂s/∂n from β
    end
    # poly p=[1,x,y] and ∂p/∂n = [0,nx,ny]
    for i in 1:M
        K[i, 2M+1] = 1.0
        K[i, 2M+2] = pts[i][1]
        K[i, 2M+3] = pts[i][2]
        K[M+i, 2M+1] = 0.0
        K[M+i, 2M+2] = ns[i][1]
        K[M+i, 2M+3] = ns[i][2]
        # symmetry for poly constraints
        K[2M+1, i] = 1.0
        K[2M+2, i] = pts[i][1]
        K[2M+3, i] = pts[i][2]
    end
    # ridge on kernel block
    ε = 1e-10 * (tr(view(K, 1:2M, 1:2M)) / max(2M, 1) + 1)
    for i in 1:2M
        K[i, i] += ε
    end

    # Evaluate basis at pv for each unit data → build Φ rows
    # s(pv) = [φ(pv-xj), ψ_j(pv), p(pv)] · coef
    # For unit u_k: rhs = e_k in first M, zeros else
    # For unit q_k: rhs = e_k in second M

    function eval_basis_row(coef)
        α = coef[1:M]
        β = coef[M+1:2M]
        γ = coef[2M+1:2M+3]
        # u at pv
        uval = γ[1] + γ[2]*pv[1] + γ[3]*pv[2]
        qval = γ[2]*nv[1] + γ[3]*nv[2]
        for j in 1:M
            rij = pv - pts[j]
            r2 = dot(rij, rij)
            r = sqrt(max(r2, 0.0))
            φ = rbf(r2)
            coeff = 3 * r
            ψ = -coeff * dot(rij, ns[j])   # 3r (xj-pv)·nj = -3r (pv-xj)·nj
            uval += α[j] * φ + β[j] * ψ
            # ∂/∂n_v of φ_j: coeff * rij · nv
            dφ_dnv = coeff * dot(rij, nv)
            # ∂/∂n_v of ψ_j
            dψ_dnv = if r < 1e-14
                0.0
            else
                3 * (dot(rij, nv) / r) * dot(-rij, ns[j]) + 3 * r * (-dot(nv, ns[j]))
            end
            qval += α[j] * dφ_dnv + β[j] * dψ_dnv
        end
        return uval, qval
    end

    try
        F = lu(K)
        rhs = zeros(Nsys)
        for k in 1:M
            fill!(rhs, 0)
            rhs[k] = 1.0
            coef = F \ rhs
            uu, qu = eval_basis_row(coef)
            Φuu[ids[k]] = uu
            Φqu[ids[k]] = qu
            fill!(rhs, 0)
            rhs[M+k] = 1.0
            coef = F \ rhs
            uq, qq = eval_basis_row(coef)
            Φuq[ids[k]] = uq
            Φqq[ids[k]] = qq
        end
    catch
        Φuu .= rbf_cardinal(pv, d.source_pos, ids; basis=rbf, ridge=1e-12)
        Φqq .= rbf_cardinal(pv, d.source_pos, ids; basis=rbf, ridge=1e-12)
    end
    return Φuu, Φuq, Φqu, Φqq
end

# =============================================================================
# Assembly + coupled condensation
# =============================================================================

"""
    assemble_dibfm!(dib; npg=12, method=:hmls, ...)

First-layer continuous integration, then second-layer condensation
(`:hmls`, `:rbf`, or `:rbf_hermite`).
"""
function assemble_dibfm!(dib::DiBFMData; npg=12, method=:hmls, radius_factor=2.5,
    rbf=PHS(3; poly_deg=0))
    d = dib.dlim
    # assemble H,G (source × all) without DLIM Shepard condensation
    n_s = length(d.source_pos)
    n_all = length(d.all_pos)
    H = zeros(n_s, n_all)
    G = zeros(n_s, n_all)
    qsi_g, w_g = gausslegendre(npg)
    k = d.k

    @showprogress "DiBFM-HMLS assemble" for i in 1:n_s
        pf = d.source_pos[i]
        for el in d.elements
            X = [d.all_pos[j] for j in el]
            n_loc = length(el)
            ξnodes = _elem_xi(n_loc - 2)
            for (ig, ξ) in enumerate(qsi_g)
                N = _lagrange_N(ξ, ξnodes)
                dN = _lagrange_dN(ξ, ξnodes)
                x = sum(N[a] * X[a] for a in 1:n_loc)
                dx = sum(dN[a] * X[a] for a in 1:n_loc)
                J = norm(dx)
                J < 1e-16 && continue
                n̂ = Point2D(dx[2] / J, -dx[1] / J)
                rvec = x - pf
                R = norm(rvec)
                R < 1e-14 && continue
                Gker = -log(R) / (2π * k)
                Hker = dot(rvec, n̂) / (R^2 * 2π)
                wJ = J * w_g[ig]
                for a in 1:n_loc
                    ja = el[a]
                    H[i, ja] += Hker * N[a] * wJ
                    G[i, ja] += Gker * N[a] * wJ
                end
            end
        end
    end
    d.H = H
    d.G = G

    build_dibfm_condensation!(dib; method=method, radius_factor=radius_factor, rbf=rbf)

    # Split H = [Hss Hsv], G = [Gss Gsv]
    # all_pos = [sources | virtuals]
    Hss = H[:, 1:n_s]
    Hsv = H[:, n_s+1:end]
    Gss = G[:, 1:n_s]
    Gsv = G[:, n_s+1:end]
    Φuu, Φuq, Φqu, Φqq = dib.Φuu, dib.Φuq, dib.Φqu, dib.Φqq

    # Hss us + Hsv (Φuu us + Φuq qs) + 0.5 us
    #   = Gss qs + Gsv (Φqu us + Φqq qs)
    # (Hss + Hsv Φuu - Gsv Φqu + 0.5 I) us = (Gss + Gsv Φqq - Hsv Φuq) qs
    A_u = Hss + Hsv * Φuu - Gsv * Φqu
    @inbounds for i in 1:n_s
        A_u[i, i] += 0.5
    end
    A_q = Gss + Gsv * Φqq - Hsv * Φuq
    dib.A_u = A_u
    dib.A_q = A_q
    return dib
end

# =============================================================================
# Solve
# =============================================================================

function solve_dibfm!(dib::DiBFMData)
    d = dib.dlim
    n = length(d.source_pos)
    A = copy(dib.A_u)
    B = copy(dib.A_q)
    b = zeros(n)
    BC, BV = d.BC, d.BV
    @inbounds for j in 1:n
        if BC[j] == 0
            colA = A[:, j]
            colB = B[:, j]
            A[:, j] = -colB
            b .-= colA .* BV[j]
        else
            b .+= B[:, j] .* BV[j]
        end
    end
    x = A \ b
    T = zeros(n)
    q = zeros(n)
    @inbounds for j in 1:n
        if BC[j] == 0
            T[j] = BV[j]
            q[j] = x[j]
        else
            q[j] = BV[j]
            T[j] = x[j]
        end
    end
    dib.T = T
    dib.q = q
    d.T = T
    d.q = q
    return T
end

function dibfm_rel_error(dib::DiBFMData, Tana::Function)
    return dlim_rel_error(dib.dlim, Tana)
end

function solve_dibfm_laplace(dad::BEMdata{<:Laplace}; npg=12, method=:hmls,
    radius_factor=2.5, rbf=PHS(3; poly_deg=0))
    dib = dibfm_from_bemdata(dad)
    assemble_dibfm!(dib; npg=npg, method=method, radius_factor=radius_factor, rbf=rbf)
    solve_dibfm!(dib)
    return dib
end

"""
    compare_dlim_dibfm(dad, Tana) -> NamedTuple

Compare standard BEM, DLIM-MLS/RBF, and DiBFM-HMLS / DiBFM-RBF / DiBFM-RBF-Hermite.
"""
function compare_dlim_dibfm(dad::BEMdata{<:Laplace}, Tana::Function; npg=12)
    t0 = @elapsed begin
        H_G_full_direct(dad; npg=npg, threaded=false)
        solve(dad)
    end
    err0 = rel_error(dad)

    t_mls = @elapsed (d_mls = solve_dlim_laplace(dad; npg=npg, method=:mls))
    err_mls = dlim_rel_error(d_mls, Tana)

    t_rbf1 = @elapsed (d_rbf = solve_dlim_laplace(dad; npg=npg, method=:rbf))
    err_rbf1 = dlim_rel_error(d_rbf, Tana)

    t_h = @elapsed (dib_h = solve_dibfm_laplace(dad; npg=npg, method=:hmls))
    err_h = dibfm_rel_error(dib_h, Tana)

    t_r2 = @elapsed (dib_r = solve_dibfm_laplace(dad; npg=npg, method=:rbf))
    err_r2 = dibfm_rel_error(dib_r, Tana)

    t_rh = @elapsed (dib_rh = solve_dibfm_laplace(dad; npg=npg, method=:rbf_hermite))
    err_rh = dibfm_rel_error(dib_rh, Tana)

    return (
        err_std=err0,
        err_dlim_mls=err_mls,
        err_dlim_rbf=err_rbf1,
        err_dibfm_hmls=err_h,
        err_dibfm_rbf=err_r2,
        err_dibfm_rbfh=err_rh,
        # aliases
        err_dibfm=err_h,
        time_std=t0,
        time_mls=t_mls,
        time_rbf=t_rbf1,
        time_dibfm=t_h,
        time_dibfm_rbf=t_r2,
        time_dibfm_rbfh=t_rh,
        n_s=length(d_mls.source_pos),
        n_v=length(d_mls.virt_global),
    )
end
