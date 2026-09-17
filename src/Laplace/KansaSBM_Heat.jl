# Kansa particular field u_p = φ α plus homogeneous SBM for u_h.
#
#   up = φ α                         (RBF itself, not DRM particular Φ)
#   uh = Uh_g + Uh α                 (SBM Laplace, BC = prescribed − trace(φ_j))
#   u  = Uh_g + (Uh + φ) α
#   Δ up = (Δφ) α = γ (u − u_hist)   at interior centres
#   (Δφ − γ φ − γ Uh_i) α = γ (Uh_g_i − u_hist)

export solve_kansa_sbm_heat

function solve_kansa_sbm_heat(dad::BEMdata{<:Laplace};
        κ::Real=1.25,
        Δt::Real,
        tf::Real,
        u0::AbstractVector{<:Real},
        scheme::Symbol=:houbolt,
        basis::AbstractRadialBasis=PHS(3; poly_deg=-1),
        ridge::Real=1e-10,
        robin_H=nothing,
        robin_uf=nothing)
    scheme in (:euler, :houbolt) || throw(ArgumentError("scheme ∈ (:euler,:houbolt)"))
    N, M = dad.n, dad.ni
    M > 0 || error("Kansa-SBM needs internal RBF centres")
    nt = N + M
    length(u0) >= nt || throw(ArgumentError("u0 short"))
    κ = float(κ)
    Δt = float(Δt)
    kcond = float(dad.properties.k)

    BC = collect(Int, dad.BC[1:N])
    BV = collect(Float64, dad.BV[1:N])
    rH = zeros(N)
    rUf = zeros(N)
    @inbounds for i in 1:N
        BC[i] == 2 || continue
        robin_H === nothing && continue
        rH[i] = robin_H isa Number ? float(robin_H) : float(robin_H[i])
        rUf[i] = robin_uf isa Number ? float(robin_uf) :
                 (robin_uf === nothing ? 0.0 : float(robin_uf[i]))
    end

    sd = sbm_from_bemdata(dad)
    assemble_sbm!(sd)
    nodes = sd.nodes
    normals = sd.normals
    interior = sd.internal
    length(interior) == M || (interior = Point2D[Point2D(p) for p in dad.internalNodes])

    # Mixed SBM operator (BC type only) — factor once, M+1 backsolves.
    As = zeros(N, N)
    n_dir = 0
    @inbounds for i in 1:N
        if BC[i] == 0
            n_dir += 1
            As[i, :] .= view(sd.G, i, :)
        elseif BC[i] == 1
            As[i, :] .= view(sd.H, i, :)
        else
            As[i, :] .= view(sd.H, i, :) .- rH[i] .* view(sd.G, i, :)
        end
    end
    if n_dir == 0
        As[N, :] .= 1.0
    end
    AF = factorize(As)

    Gib = zeros(M, N)
    @inbounds for j in 1:N, i in 1:M
        Gib[i, j] = _sbm_U(interior[i] - nodes[j], kcond)
    end

    hh = max(rbf_length_scale(interior), 1e-14)
    φi = zeros(M, M)
    Lφi = zeros(M, M)
    φb = zeros(N, M)
    qφ = zeros(N, M)
    @inbounds for j in 1:M
        cj = interior[j]
        for i in 1:M
            rij = norm(interior[i] - cj) / hh
            φi[i, j] = basis(_scale_r(norm(interior[i] - cj), hh))
            Lφi[i, j] = laplacian_phi(basis, rij; dim=2) / hh^2
        end
        for i in 1:N
            φb[i, j] = basis(_scale_r(norm(nodes[i] - cj), hh))
            ε = 1e-7
            ni = normals[i]
            xp = Point2D(nodes[i][1] + ε * ni[1], nodes[i][2] + ε * ni[2])
            xm = Point2D(nodes[i][1] - ε * ni[1], nodes[i][2] - ε * ni[2])
            dφn = (basis(_scale_r(norm(xp - cj), hh)) -
                   basis(_scale_r(norm(xm - cj), hh))) / (2ε)
            qφ[i, j] = -kcond * dφn
        end
    end

    B = zeros(N, M + 1)
    @inbounds for j in 1:M
        for i in 1:N
            if BC[i] == 0
                B[i, j] = -φb[i, j]
            elseif BC[i] == 1
                B[i, j] = -qφ[i, j]
            else
                B[i, j] = -(qφ[i, j] - rH[i] * φb[i, j])
            end
        end
    end
    @inbounds for i in 1:N
        if BC[i] == 0 || BC[i] == 1
            B[i, M + 1] = BV[i]
        else
            B[i, M + 1] = -rH[i] * rUf[i]
        end
    end
    if n_dir == 0
        B[N, :] .= 0.0
    end
    Alphas = AF \ B

    Uh_b = sd.G * Alphas
    @inbounds for j in 1:(M + 1), i in 1:N
        BC[i] == 0 && (Uh_b[i, j] = B[i, j])
    end
    Uh_i = Gib * Alphas
    Uh = Uh_i[:, 1:M]
    Uh_g = Uh_i[:, M + 1]

    function build_A(γ)
        A = zeros(M, M)
        @inbounds for j in 1:M, i in 1:M
            A[i, j] = Lφi[i, j] - γ * φi[i, j] - γ * Uh[i, j]
        end
        ε = float(ridge) * (sum(abs, A) / (M * M) + 1)
        @inbounds for i in 1:M
            A[i, i] += ε
        end
        return A
    end

    function reconstruct(α)
        u = zeros(nt)
        @inbounds for i in 1:N
            s = Uh_b[i, M + 1]
            for j in 1:M
                s += (Uh_b[i, j] + φb[i, j]) * α[j]
            end
            u[i] = BC[i] == 0 ? BV[i] : s
        end
        @inbounds for i in 1:M
            s = Uh_g[i]
            for j in 1:M
                s += (Uh[i, j] + φi[i, j]) * α[j]
            end
            u[N + i] = s
        end
        return u
    end

    t = collect(0.0:Δt:float(tf))
    nT = length(t)
    U = zeros(nt, nT)
    U[:, 1] .= float.(u0[1:nt])
    @inbounds for j in 1:N
        BC[j] == 0 && (U[j, 1] = BV[j])
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
                rhs[i] = γH * Uh_g[i] + c * hist
            end
            α = AH \ rhs
        else
            un = view(U, :, k - 1)
            rhs = zeros(M)
            @inbounds for i in 1:M
                rhs[i] = γE * (Uh_g[i] - un[N + i])
            end
            α = AE \ rhs
        end
        U[:, k] .= reconstruct(α)
    end

    dad.cache.extras[:kansa_sbm] = (; U, t)
    return (; t, U, N, M, scheme, method=:kansa_sbm, basis)
end
