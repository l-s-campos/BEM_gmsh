# SBM-DRM (Kovářík et al. 2017) vs BEM-DRM vs BEM-DIBEM — transient diffusion
# RBF: PHS(3) = r³, no polynomial (same kernel on all three paths)
#
# julia --project=. scripts/sbm_drm_vs_dibem.jl
# ENV: SDR_EX=1|2|all  SDR_NSTEPS=60  SDR_SCHEME=houbolt|euler  SDR_OUT=path.tsv
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays
using Printf
using Statistics

include(datadir("Laplace", "Laplace_dad.jl"))

const SDR_EX = get(ENV, "SDR_EX", "all")
const SDR_NSTEPS = parse(Int, get(ENV, "SDR_NSTEPS", "60"))
const SDR_SCHEME = Symbol(get(ENV, "SDR_SCHEME", "houbolt"))
const SDR_OUT = get(ENV, "SDR_OUT", "")
const SDR_RBF = PHS(3; poly_deg=-1)

# ---------------------------------------------------------------------------
# Mesh: rectangle (0,Lx)×(0,Ly)
# ---------------------------------------------------------------------------

function rect_diffusion_mesh(; Lx=3.0, Ly=3.0, nb_side=20,
                             left_neumann::Bool=false, nome="rect_diff")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / nb_side
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, Ly, 0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for ll in (l1, l2, l3, l4)
        gmsh.model.mesh.setTransfiniteCurve(ll, nb_side + 1)
    end
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    if left_neumann
        gmsh.model.addPhysicalGroup(1, [l1, l2, l3], -1, "0;0")
        gmsh.model.addPhysicalGroup(1, [l4], -1, "1;0")
    else
        gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], -1, "0;0")
    end
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

interior_grid(Lx, Ly, ns) =
    [Point2D(Lx * ix / (ns + 1), Ly * iy / (ns + 1)) for iy = 1:ns for ix = 1:ns]

# ---------------------------------------------------------------------------
# Exact solutions (paper Examples 1–2)
# ---------------------------------------------------------------------------

function exact_ex1(x, y, t; u0=30.0, κ=1.25, Lx=3.0, Ly=3.0, nterms=40)
    s = 0.0
    for i = 1:nterms, j = 1:nterms
        A = 4u0 * ((-1)^i - 1) * ((-1)^j - 1) / (i * j * pi^2)
        s += A * sin(i * pi * x / Lx) * sin(j * pi * y / Ly) *
             exp(-κ * ((i * pi / Lx)^2 + (j * pi / Ly)^2) * t)
    end
    return s
end

function exact_ex2(x, y, t; u0=30.0, κ=1.25, Lx=3.0, Ly=3.0, nterms=60)
    s = 0.0
    for n = 0:nterms, m = 1:nterms
        λx = (n + 0.5) * pi / Lx
        λy = m * pi / Ly
        Ix = ((-1)^n) / λx
        Iy = (1 - (-1)^m) / λy
        A = u0 * Ix * Iy / ((Lx / 2) * (Ly / 2))
        s += A * cos(λx * x) * sin(λy * y) *
             exp(-κ * (λx^2 + λy^2) * t)
    end
    return s
end

rmse_rinf(a, b) = (sqrt(mean(abs2, a .- b)), maximum(abs, a .- b))

# ---------------------------------------------------------------------------
# Shared BE / Houbolt march: H u − G q = M ú
# ---------------------------------------------------------------------------

function _heat_march!(dad, H0, G0, M, u0; Δt, tf, scheme=:houbolt)
    N, nt = dad.n, dad.nt
    t = collect(0.0:Δt:tf)
    nT = length(t)
    T = zeros(nt, nT)
    T[:, 1] .= u0[1:nt]
    @inbounds for j in 1:N
        dad.BC[j] == 0 && (T[j, 1] = dad.BV[j])
    end

    function factor(Heff)
        set_cache!(dad; H=Heff, G=G0)
        has_cache(dad, :A) && (dad.cache.A = nothing)
        applyBC(dad)
        return factorize(Matrix(dad.A)), copy(dad.b)
    end

    try
        Fe, be = factor(H0 - M / Δt)
        n_euler = scheme === :euler ? nT : min(3, nT)
        for i in 2:n_euler
            rhs = be .- (M / Δt) * T[:, i - 1]
            x = Fe \ rhs
            Tf = zeros(nt)
            qf = zeros(N)
            Tf[1:length(x)] .= x
            split_sol!(dad, Tf, qf)
            T[:, i] .= Tf
        end
        if scheme === :houbolt && nT >= 4
            Fh, bh = factor(H0 - 11 * M / (6 * Δt))
            for i in 4:nT
                rhs = bh .+ M * (-18 .* T[:, i-1] .+ 9 .* T[:, i-2] .-
                                 2 .* T[:, i-3]) / (6 * Δt)
                x = Fh \ rhs
                Tf = zeros(nt)
                qf = zeros(N)
                Tf[1:length(x)] .= x
                split_sol!(dad, Tf, qf)
                T[:, i] .= Tf
            end
        end
    finally
        set_cache!(dad; H=H0, G=G0)
        has_cache(dad, :A) && (dad.cache.A = nothing)
    end
    return (; t, U=T, N, M=dad.ni)
end

"""BEM + dual-reciprocity mass (`build_drm_matrices`)."""
function bem_drm_heat(dad, u0; κ=1.25, Δt=0.02, tf=1.2, scheme=:houbolt,
                      basis=SDR_RBF)
    H_G_full_direct(dad; npg=12, threaded=false)
    drm = build_drm_matrices(dad, basis; npg=12)
    return _heat_march!(dad, drm.H, drm.G, drm.M ./ κ, u0; Δt, tf, scheme)
end

"""BEM + DIBEM mass (`DIBEM`)."""
function bem_dibem_heat(dad, u0; κ=1.25, Δt=0.02, tf=1.2, scheme=:houbolt,
                        rbf=SDR_RBF)
    H_G_full_direct(dad; npg=12, threaded=false)
    DIBEM(dad; method=:dense, rbf=rbf)
    M = Matrix(dad.M) ./ κ
    H0 = Matrix(dad.H)
    G0 = Matrix(dad.G)
    return _heat_march!(dad, H0, G0, M, u0; Δt, tf, scheme)
end

# ---------------------------------------------------------------------------
# One example — three methods
# ---------------------------------------------------------------------------

function run_example(ex::Int; nsteps=SDR_NSTEPS, scheme=SDR_SCHEME,
                     nb=20, nint=9, rbf=SDR_RBF)
    Lx, Ly = 3.0, 3.0
    κ = 1.25
    tf = 1.2
    Δt = tf / nsteps
    left_neu = ex == 2
    nome = "sdr_ex$(ex)_n$(nb)"

    msh = rect_diffusion_mesh(; Lx, Ly, nb_side=nb, left_neumann=left_neu, nome)
    dad0 = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    set_internal_nodes!(dad0, interior_grid(Lx, Ly, nint))
    dad_s = deepcopy(dad0)
    dad_drm = deepcopy(dad0)
    dad_dib = deepcopy(dad0)

    N, ni = dad0.n, dad0.ni
    u0 = fill(30.0, N + ni)
    u0[1:N] .= 0.0

    exact = ex == 1 ? exact_ex1 : exact_ex2
    pts = vcat([Point2D(p) for p in dad0.Nodes],
               [Point2D(p) for p in dad0.internalNodes])
    uex = [exact(p[1], p[2], tf; κ=κ, Lx=Lx, Ly=Ly) for p in pts]

    t_s = @elapsed sol_s = solve_sbm_drm(dad_s; κ=κ, Δt=Δt, tf=tf, u0=u0,
                                         scheme=scheme, basis=rbf)
    rmse_s, rinf_s = rmse_rinf(sol_s.U[:, end], uex)

    t_drm = @elapsed sol_drm = bem_drm_heat(dad_drm, u0; κ=κ, Δt=Δt, tf=tf,
                                            scheme=scheme, basis=rbf)
    u_drm = sol_drm.U[:, end]
    length(u_drm) == length(uex) || (u_drm = u_drm[1:length(uex)])
    rmse_drm, rinf_drm = rmse_rinf(u_drm, uex)

    t_dib = @elapsed sol_dib = bem_dibem_heat(dad_dib, u0; κ=κ, Δt=Δt, tf=tf,
                                              scheme=scheme, rbf=rbf)
    u_dib = sol_dib.U[:, end]
    length(u_dib) == length(uex) || (u_dib = u_dib[1:length(uex)])
    rmse_dib, rinf_dib = rmse_rinf(u_dib, uex)

    return (
        ex=ex, nsteps=nsteps, scheme=scheme, n=N, ni=ni,
        r_sbm=rmse_s, e_sbm=rinf_s, t_sbm=t_s, max_sbm=maximum(abs, sol_s.U[:, end]),
        r_drm=rmse_drm, e_drm=rinf_drm, t_drm=t_drm, max_drm=maximum(abs, u_drm),
        r_dib=rmse_dib, e_dib=rinf_dib, t_dib=t_dib, max_dib=maximum(abs, u_dib),
        r_bem=rmse_dib, e_bem=rinf_dib, t_bem=t_dib, max_bem=maximum(abs, u_dib),
        mean_ex=mean(uex),
    )
end

function main()
    println("SBM-DRM vs BEM-DRM vs BEM-DIBEM | RBF=PHS(3) scheme=$SDR_SCHEME nsteps=$SDR_NSTEPS")
    println("-"^110)
    @printf("%-4s %6s %5s %5s  %10s %8s  %10s %8s  %10s %8s\n",
            "ex", "steps", "n", "ni",
            "RMSE_SBM", "t_SBM",
            "RMSE_DRM", "t_DRM",
            "RMSE_DIB", "t_DIB")
    rows = []
    exs = SDR_EX == "all" ? [1, 2] : [parse(Int, SDR_EX)]
    for ex in exs
        for ns in unique([SDR_NSTEPS, 2 * SDR_NSTEPS])
            r = run_example(ex; nsteps=ns, scheme=SDR_SCHEME)
            push!(rows, r)
            @printf("%-4d %6d %5d %5d  %10.3e %8.3f  %10.3e %8.3f  %10.3e %8.3f\n",
                    r.ex, r.nsteps, r.n, r.ni,
                    r.r_sbm, r.t_sbm,
                    r.r_drm, r.t_drm,
                    r.r_dib, r.t_dib)
        end
    end

    if !isempty(SDR_OUT)
        open(SDR_OUT, "w") do io
            println(io, "ex\tnsteps\tn\tni\trmse_sbm\tt_sbm\trmse_drm\tt_drm\trmse_dib\tt_dib")
            for r in rows
                @printf(io, "%d\t%d\t%d\t%d\t%.6e\t%.6f\t%.6e\t%.6f\t%.6e\t%.6f\n",
                        r.ex, r.nsteps, r.n, r.ni,
                        r.r_sbm, r.t_sbm, r.r_drm, r.t_drm, r.r_dib, r.t_dib)
            end
        end
        println("wrote $SDR_OUT")
    end
    return rows
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "sbm_drm_vs_dibem.jl")
    main()
end
