# Transient heat: Kansa / DRM / DIBEM on the same BEMdata, PHS3, Houbolt.
# Interior RMSE is the primary figure.
#
# julia --project=. scripts/kansa_sbm_vs_drm.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Statistics
using Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(joinpath(@__DIR__, "sbm_drm_vs_dibem.jl"))

const BASIS = PHS(3; poly_deg=-1)

function _metrics(Uend, uex, N)
    nt = length(uex)
    length(Uend) >= nt || return (; r_all=NaN, r_int=NaN, r_bnd=NaN, umax=NaN, rel_int=NaN)
    ii = (N + 1):nt
    r_all = sqrt(mean(abs2, Uend[1:nt] .- uex))
    r_int = sqrt(mean(abs2, Uend[ii] .- uex[ii]))
    r_bnd = sqrt(mean(abs2, Uend[1:N] .- uex[1:N]))
    return (; r_all, r_int, r_bnd, umax=maximum(abs, Uend[1:nt]),
            rel_int=r_int / max(maximum(abs, uex[ii]), eps()))
end

const _NAN_MET = (; r_all=NaN, r_int=NaN, r_bnd=NaN, umax=NaN, rel_int=NaN)

function _print_row(name, m; t=NaN)
    @printf("  %-11s  int=%.3e  all=%.3e  bnd=%.3e  max|u|=%.3e  t=%.2fs\n",
            name, m.r_int, m.r_all, m.r_bnd, m.umax, t)
end

"""Pure Kansa heat: u = Σ α_j φ(|x−x_j|) on all collocation centres (no BEM/SBM)."""
function kansa_heat(dad::BEMdata{<:Laplace}, u0; κ, Δt, tf, scheme=:houbolt,
                    basis=BASIS, ridge=1e-10)
    N, M = dad.n, dad.ni
    nt = N + M
    M > 0 || error("pure Kansa needs interior collocation")
    nodes = Point2D[Point2D(p) for p in dad.Nodes]
    normals = Point2D[Point2D(nn) for nn in dad.Normal]
    interior = Point2D[Point2D(p) for p in dad.internalNodes]
    pts = vcat(nodes, interior)
    BC = collect(Int, dad.BC[1:N])
    BV = collect(Float64, dad.BV[1:N])
    kcond = float(dad.properties.k)
    hh = max(rbf_length_scale(pts), 1e-14)

    Φ = zeros(nt, nt)
    L = zeros(M, nt)
    Q = zeros(N, nt)
    @inbounds for j in 1:nt
        cj = pts[j]
        for i in 1:nt
            Φ[i, j] = basis(BEM._scale_r(norm(pts[i] - cj), hh))
        end
        for i in 1:M
            rij = norm(interior[i] - cj) / hh
            L[i, j] = laplacian_phi(basis, rij; dim=2) / hh^2
        end
        for i in 1:N
            ε = 1e-7
            ni = normals[i]
            xp = Point2D(nodes[i][1] + ε * ni[1], nodes[i][2] + ε * ni[2])
            xm = Point2D(nodes[i][1] - ε * ni[1], nodes[i][2] - ε * ni[2])
            dφn = (basis(BEM._scale_r(norm(xp - cj), hh)) -
                   basis(BEM._scale_r(norm(xm - cj), hh))) / (2ε)
            Q[i, j] = -kcond * dφn
        end
    end
    Φi = view(Φ, (N + 1):nt, :)

    function build_A(γ)
        A = zeros(nt, nt)
        @inbounds for i in 1:N
            if BC[i] == 0
                A[i, :] .= view(Φ, i, :)
            else
                A[i, :] .= view(Q, i, :)
            end
        end
        @inbounds for i in 1:M, j in 1:nt
            A[N + i, j] = L[i, j] - γ * Φi[i, j]
        end
        ε = float(ridge) * (sum(abs, A) / (nt * nt) + 1)
        @inbounds for i in 1:nt
            A[i, i] += ε
        end
        return A
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
    rhs = zeros(nt)

    for k in 2:nT
        @inbounds for i in 1:N
            rhs[i] = BV[i]
        end
        if scheme === :houbolt && k >= 4
            c = 1 / (6 * κ * Δt)
            @inbounds for i in 1:M
                hist = 18 * U[N + i, k - 1] - 9 * U[N + i, k - 2] + 2 * U[N + i, k - 3]
                rhs[N + i] = -c * hist
            end
            α = AH \ rhs
        else
            @inbounds for i in 1:M
                rhs[N + i] = -γE * U[N + i, k - 1]
            end
            α = AE \ rhs
        end
        mul!(view(U, :, k), Φ, α)
        @inbounds for i in 1:N
            BC[i] == 0 && (U[i, k] = BV[i])
        end
    end
    return (; t, U, N, M, scheme, method=:kansa)
end

function _try_metrics(label, thunk, uex, N)
    try
        t = @elapsed sol = thunk()
        Uend = sol.U[:, end]
        return _metrics(Uend, uex, N), t
    catch e
        @warn "failed" label exception = e
        return _NAN_MET, NaN
    end
end

function _run6(dad, u0, uex; κ, Δt, tf, scheme)
    N = dad.n
    m_sd, t_s = _try_metrics("SBM-DRM",
        () -> solve_sbm_drm(deepcopy(dad); κ=κ, Δt=Δt, tf=tf, u0=u0,
                            scheme=scheme, basis=BASIS), uex, N)
    m_ks, t_k = _try_metrics("Kansa-SBM",
        () -> solve_kansa_sbm_heat(deepcopy(dad); κ=κ, Δt=Δt, tf=tf, u0=u0,
                                   scheme=scheme, basis=BASIS), uex, N)
    m_kb, t_kb = _try_metrics("BEM-Kansa",
        () -> solve_kansa_bem_heat(deepcopy(dad); κ=κ, Δt=Δt, tf=tf, u0=u0,
                                   scheme=scheme, basis_order=3), uex, N)
    m_pk, t_pk = _try_metrics("Pure Kansa",
        () -> kansa_heat(deepcopy(dad), u0; κ=κ, Δt=Δt, tf=tf, scheme=scheme),
        uex, N)
    m_bd, t_b = _try_metrics("BEM-DRM",
        () -> bem_drm_heat(deepcopy(dad), u0; κ=κ, Δt=Δt, tf=tf,
                           scheme=scheme, basis=BASIS), uex, N)
    m_di, t_di = _try_metrics("DIBEM",
        () -> bem_dibem_heat(deepcopy(dad), u0; κ=κ, Δt=Δt, tf=tf,
                             scheme=scheme, rbf=BASIS), uex, N)
    return (; n=N, ni=dad.ni, Δt, tf, scheme,
            sbm=m_sd, t_s, kansa=m_ks, t_k, kansa_bem=m_kb, t_kb,
            kansa_pure=m_pk, t_pk, bem=m_bd, t_b, dibem=m_di, t_di)
end

function compare_sine(; ndiv=10, Δt=0.005, tf=0.05, scheme=:houbolt)
    msh = quadrado(ndiv=ndiv, show=false, nome="kansa_vs_drm_sin", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    dad.BC .= 0
    dad.BV .= 0
    u0 = [sin(pi * point(dad, i)[1]) * sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    u0[1:dad.n] .= 0
    uex = [exp(-2 * pi^2 * tf) * sin(pi * point(dad, i)[1]) *
           sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    r = _run6(dad, u0, uex; κ=1.0, Δt=Δt, tf=tf, scheme=scheme)
    return merge(r, (; case="sine unit square"))
end

function compare_kovarik(ex::Int; nb=12, nsteps=40, scheme=:houbolt)
    Lx, Ly, κ, tf = 3.0, 3.0, 1.25, 1.2
    Δt = tf / nsteps
    msh = rect_diffusion_mesh(; Lx, Ly, nb_side=nb, left_neumann=(ex == 2),
                              nome="kansa_ex$(ex)_n$(nb)")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    N, ni = dad.n, dad.ni
    u0 = fill(30.0, N + ni)
    u0[1:N] .= 0.0
    exact = ex == 1 ? exact_ex1 : exact_ex2
    pts = vcat([Point2D(p) for p in dad.Nodes],
               [Point2D(p) for p in dad.internalNodes])
    uex = [exact(p[1], p[2], tf; κ=κ, Lx=Lx, Ly=Ly) for p in pts]
    r = _run6(dad, u0, uex; κ=κ, Δt=Δt, tf=tf, scheme=scheme)
    return merge(r, (; case="Kovarik ex$ex", nsteps))
end

function _dump(r)
    extra = hasproperty(r, :nsteps) ? "  nsteps=$(r.nsteps)" : "  Δt=$(r.Δt) tf=$(r.tf)"
    println()
    println("$(r.case)  n=$(r.n) ni=$(r.ni)$extra  $(r.scheme)")
    _print_row("SBM-DRM", r.sbm; t=r.t_s)
    _print_row("Kansa-SBM", r.kansa; t=r.t_k)
    _print_row("BEM-Kansa", r.kansa_bem; t=r.t_kb)
    _print_row("Pure Kansa", r.kansa_pure; t=r.t_pk)
    _print_row("BEM-DRM", r.bem; t=r.t_b)
    _print_row("DIBEM", r.dibem; t=r.t_di)
end

function main()
    println("Kansa-SBM    u = u_h(SBM) + φ α      (Δφ at interiors)")
    println("BEM-Kansa    u = u_h(BEM) + φ α      (same particular, dense BEM for u_h)")
    println("Pure Kansa   u = φ α                 (all collocation centres, no BEM/SBM)")
    println("SBM-DRM      u = Gα + Φ β            (ΔΦ = φ)")
    println("BEM-DRM      Hu − Gq = Mú            (classical dual reciprocity)")
    println("DIBEM        Hu − Gq = Mú            (direct interpolation, same PHS3)")
    println("All six share format2d BEMdata + PHS(3) + Houbolt.")
    println("Internals = Gmsh cell centroids from format2d (pontointerno=true).")
    println("="^78)
    rows = Any[]
    push!(rows, compare_sine())
    push!(rows, compare_kovarik(1; nb=12, nsteps=40))
    push!(rows, compare_kovarik(2; nb=12, nsteps=40))
    push!(rows, compare_kovarik(1; nb=20, nsteps=60))
    push!(rows, compare_kovarik(2; nb=20, nsteps=60))
    for r in rows
        _dump(r)
    end
    println()
    println("summary  interior RMSE")
    @printf("%-22s %10s %10s %10s %10s %10s %10s\n",
            "case", "SBM-DRM", "Kansa-SBM", "BEM-Kansa", "PureKansa", "BEM-DRM", "DIBEM")
    for r in rows
        label = hasproperty(r, :nsteps) ? "$(r.case) n=$(r.n)" : r.case
        @printf("%-22s %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n",
                label, r.sbm.r_int, r.kansa.r_int, r.kansa_bem.r_int,
                r.kansa_pure.r_int, r.bem.r_int, r.dibem.r_int)
    end
    return rows
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "kansa_sbm_vs_drm.jl")
    main()
end
