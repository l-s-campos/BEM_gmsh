# Particular-solution heat: global Kansa RBF for u_p + homogeneous BEM for u_h
#
# Linear Euler / Houbolt coupling (no Picard). Interior centres only.
# Default basis: PHS(3) without polynomial null-space (poly_deg=-1).
#
# Algebra (Dirichlet; Neumann/Robin via bem_solve):
#   up = Φ α
#   uh = Uh_g + Uh α     (Uh columns: harmonic fields with BC shift −trace(φ_j))
#   u  = Uh_g + (Uh + Φ) α
#   L up = LΦ α = γ (u − u_hist)
#   (LΦ − γ Φ − γ Uh) α = γ (Uh_g − u_hist)
#
# BEM A depends only on BC type: factor once, backsolve M+1 RHS.

export solve_kansa_bem_heat

function solve_kansa_bem_heat(dad::BEMdata{<:Laplace};
                              κ::Real=1.25,
                              Δt::Real,
                              tf::Real,
                              u0::AbstractVector{<:Real},
                              scheme::Symbol=:houbolt,
                              basis_order::Int=3,
                              npg::Int=12,
                              ridge::Real=1e-10,
                              robin_H=nothing,
                              robin_uf=nothing)
    scheme in (:euler, :houbolt) || throw(ArgumentError("scheme"))
    N, nt = dad.n, dad.nt
    M = dad.ni
    M > 0 || error("need internal nodes")
    length(u0) >= nt || throw(ArgumentError("u0 short"))
    κ = float(κ); Δt = float(Δt)
    kcond = float(dad.properties.k)

    BC0 = Int.(dad.BC[1:N])
    BV0 = Float64.(dad.BV[1:N])
    rH = zeros(N); rUf = zeros(N)
    has_robin = any(==(2), BC0)
    if has_robin && robin_H !== nothing
        @inbounds for i in 1:N
            if BC0[i] == 2
                rH[i] = robin_H isa Number ? float(robin_H) : float(robin_H[i])
                rUf[i] = robin_uf isa Number ? float(robin_uf) : float(robin_uf[i])
            end
        end
    end

    nodes = Point2D[Point2D(p) for p in dad.Nodes]
    normals = Point2D[Point2D(nn) for nn in dad.Normal]
    interior = Point2D[Point2D(p) for p in dad.internalNodes]
    centres = copy(interior)

    H_G_full_direct(dad; npg=npg, threaded=false)
    H0 = Matrix{Float64}(dad.H)
    G0 = Matrix{Float64}(dad.G)
    Hm = copy(H0)
    if has_robin
        @inbounds for i in 1:N
            if BC0[i] == 2 && rH[i] != 0
                @views Hm[:, i] .-= rH[i] .* G0[:, i]
            end
        end
    end
    BC_work = copy(BC0)
    @inbounds for i in 1:N
        BC0[i] == 2 && (BC_work[i] = 1)
    end

    # A is independent of BV. Factor once.
    dad.BC[1:N] .= BC_work
    dad.BV[1:N] .= 0
    set_cache!(dad; H=Hm, G=G0)
    has_cache(dad, :A) && (dad.cache.A = nothing)
    applyBC(dad)
    AF = factorize(Matrix(dad.A))
    Hm_mat = Matrix{Float64}(Hm)
    G0_mat = Matrix{Float64}(G0)

    function fill_b!(b, g, qN, rboost)
        fill!(b, 0)
        @inbounds for j in 1:N
            if BC0[j] == 0
                @views b .-= view(Hm_mat, :, j) .* g[j]
            elseif BC0[j] == 1
                @views b .+= view(G0_mat, :, j) .* qN[j]
            else
                @views b .+= view(G0_mat, :, j) .* rboost[j]
            end
        end
        return b
    end

    function recover!(Th, qh, x, g, qN)
        Th[1:length(x)] .= x
        if length(x) < nt
            Th[(length(x) + 1):nt] .= 0
        end
        @inbounds for bc in 1:N
            if BC0[bc] == 0
                qh[bc] = Th[bc]
                Th[bc] = g[bc]
            elseif BC0[bc] == 1
                qh[bc] = qN[bc]
            else
                qh[bc] = 0
            end
        end
        return Th
    end

    # ---- RBF tables ----
    b = PHS(basis_order; poly_deg=-1)
    hh = max(rbf_length_scale(centres), 1e-14)
    Φi = zeros(M, M)
    LΦi = zeros(M, M)
    Φb = zeros(N, M)
    dΦn = zeros(N, M)
    @inbounds for j in 1:M
        cj = centres[j]
        for i in 1:M
            rij = norm(interior[i] - cj) / hh
            Φi[i, j] = b(_scale_r(norm(interior[i] - cj), hh))
            LΦi[i, j] = laplacian_phi(b, rij; dim=2) / hh^2
        end
        for i in 1:N
            Φb[i, j] = b(_scale_r(norm(nodes[i] - cj), hh))
            ε = 1e-7
            ni = normals[i]
            xp = Point2D(nodes[i][1] + ε * ni[1], nodes[i][2] + ε * ni[2])
            xm = Point2D(nodes[i][1] - ε * ni[1], nodes[i][2] - ε * ni[2])
            dΦn[i, j] = (b(_scale_r(norm(xp - cj), hh)) -
                         b(_scale_r(norm(xm - cj), hh))) / (2ε)
        end
    end
    qΦb = -kcond .* dΦn

    # ---- M+1 harmonic BEM solves: one factorization ----
    nrows = size(Hm_mat, 1)
    Bcols = zeros(nrows, M + 1)
    gcol = zeros(N, M + 1)
    qcol = zeros(N, M + 1)
    rbcol = zeros(N, M + 1)
    @inbounds for j in 1:M
        for i in 1:N
            if BC0[i] == 0
                gcol[i, j] = -Φb[i, j]
            elseif BC0[i] == 1
                qcol[i, j] = -qΦb[i, j]
            else
                rbcol[i, j] = -(qΦb[i, j] - rH[i] * Φb[i, j])
            end
        end
        fill_b!(view(Bcols, :, j), view(gcol, :, j), view(qcol, :, j), view(rbcol, :, j))
    end
    @inbounds for i in 1:N
        if BC0[i] == 0
            gcol[i, M + 1] = BV0[i]
        elseif BC0[i] == 1
            qcol[i, M + 1] = BV0[i]
        else
            rbcol[i, M + 1] = -rH[i] * rUf[i]
        end
    end
    fill_b!(view(Bcols, :, M + 1), view(gcol, :, M + 1), view(qcol, :, M + 1),
            view(rbcol, :, M + 1))

    X = AF \ Bcols
    Uh = zeros(nt, M)
    Uh_g = zeros(nt)
    qh = zeros(N)
    @inbounds for j in 1:M
        recover!(view(Uh, :, j), qh, view(X, :, j), view(gcol, :, j), view(qcol, :, j))
    end
    recover!(Uh_g, qh, view(X, :, M + 1), view(gcol, :, M + 1), view(qcol, :, M + 1))

    function build_A(γ)
        A = zeros(M, M)
        @inbounds for j in 1:M, i in 1:M
            A[i, j] = LΦi[i, j] - γ * Φi[i, j] - γ * Uh[N + i, j]
        end
        ε = float(ridge) * (sum(abs, A) / (M * M) + 1)
        @inbounds for i in 1:M
            A[i, i] += ε
        end
        return A
    end

    function reconstruct(α)
        u = copy(Uh_g)
        @inbounds for j in 1:M
            aj = α[j]
            for i in 1:nt
                u[i] += Uh[i, j] * aj
            end
            for i in 1:N
                u[i] += Φb[i, j] * aj
            end
            for i in 1:M
                u[N + i] += Φi[i, j] * aj
            end
        end
        @inbounds for i in 1:N
            BC0[i] == 0 && (u[i] = BV0[i])
        end
        return u
    end

    t = collect(0.0:Δt:float(tf))
    nT = length(t)
    U = zeros(nt, nT)
    U[:, 1] .= float.(u0[1:nt])
    @inbounds for j in 1:N
        BC0[j] == 0 && (U[j, 1] = BV0[j])
    end

    γE = 1 / (κ * Δt)
    γH = 11 / (6 * κ * Δt)
    AE = factorize(build_A(γE))
    AH = scheme === :houbolt ? factorize(build_A(γH)) : AE

    for k in 2:nT
        if scheme === :houbolt && k >= 4
            un, un1, un2 = view(U, :, k - 1), view(U, :, k - 2), view(U, :, k - 3)
            rhs = zeros(M)
            c = 1 / (6 * κ * Δt)
            @inbounds for i in 1:M
                hist = -18 * un[N + i] + 9 * un1[N + i] - 2 * un2[N + i]
                rhs[i] = γH * Uh_g[N + i] + c * hist
            end
            α = AH \ rhs
        else
            un = view(U, :, k - 1)
            rhs = zeros(M)
            @inbounds for i in 1:M
                rhs[i] = γE * (Uh_g[N + i] - un[N + i])
            end
            α = AE \ rhs
        end
        U[:, k] .= reconstruct(α)
    end

    set_cache!(dad; H=H0, G=G0, T=U[:, end])
    dad.BC[1:N] .= BC0
    dad.BV[1:N] .= BV0
    return (; t, U, N, M, scheme, method=:kansa_bem, backend=:bem_kansa)
end
