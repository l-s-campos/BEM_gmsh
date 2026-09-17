# Full study: SBM-DRM vs BEM-DRM vs BEM-DIBEM + DiffEq steppers
# Paper examples 1–3 (Kovářík et al. 2017). Same RBF on all methods:
# PHS3, no polynomial (PHS2 Gibbs-overshoots the discontinuous IC).
#
# julia --project=. scripts/sbm_transient_study.jl
# ENV: STUDY_OUT=...  STUDY_QUICK=1
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays
using Printf
using Statistics
using OrdinaryDiffEq
const DEsolve = OrdinaryDiffEq.solve

include(datadir("Laplace", "Laplace_dad.jl"))

const STUDY_OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const QUICK = get(ENV, "STUDY_QUICK", "0") == "1"
const RBF_BEM = PHS(3; poly_deg=-1)
const RBF_SBM = PHS(3; poly_deg=-1)
const κ0 = 1.25
const Lx0 = 3.0
const Ly0 = 3.0
const tf0 = 1.2
const u_init = 30.0

# Example 4 (all-Dirichlet unit square, α=1, IC=0):
#   u=100 on x=0 and y=0
#   u=100(1+sin(π y/2)) on x=1
#   u=100(1+sin(π x/2)) on y=1
const Lx4 = 1.0
const Ly4 = 1.0
const κ4 = 1.0
const tf4 = 0.1
const u_init4 = 0.0

function case_params(ex::Int)
    if ex <= 3
        return (; Lx=Lx0, Ly=Ly0, κ=κ0, tf=tf0, u0=u_init)
    elseif ex == 4
        return (; Lx=Lx4, Ly=Ly4, κ=κ4, tf=tf4, u0=u_init4)
    else
        throw(ArgumentError("unknown example $ex"))
    end
end

mkpath(STUDY_OUT)
mkpath(joinpath(STUDY_OUT, "figures"))

# ---------------------------------------------------------------------------
# I/O helpers (no CSV.jl dependency)
# ---------------------------------------------------------------------------

function write_csv(path, rows::Vector{<:NamedTuple})
    isempty(rows) && return
    keys_ = keys(rows[1])
    open(path, "w") do io
        println(io, join(keys_, ","))
        for r in rows
            vals = map(k -> begin
                v = r[k]
                v isa AbstractString ? v : (v isa Bool ? (v ? "true" : "false") :
                    (v isa Number ? @sprintf("%.10g", v) : string(v)))
            end, keys_)
            println(io, join(vals, ","))
        end
    end
    println("wrote $path  ($(length(rows)) rows)")
end

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

function rect_mesh(; Lx=Lx0, Ly=Ly0, nb=20, left_bc::Symbol=:dirichlet, nome="rect")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / nb
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
        gmsh.model.mesh.setTransfiniteCurve(ll, nb + 1)
    end
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3], -1, "0;0")
    tag = left_bc === :dirichlet ? "0;0" : "1;0"
    gmsh.model.addPhysicalGroup(1, [l4], -1, tag)
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

function make_dad(ex::Int; nb=20, nint=9)
    p = case_params(ex)
    left = (ex == 1 || ex == 4) ? :dirichlet : (ex == 2 ? :neumann : :robin)
    dad = format2d(rect_mesh(; Lx=p.Lx, Ly=p.Ly, nb=nb, left_bc=left,
                             nome="study_ex$(ex)_n$nb"),
                   Laplace(1.0); tipo=1, pontointerno=true)
    set_internal_nodes!(dad, interior_grid(p.Lx, p.Ly, nint))
    @inbounds for i in 1:dad.n
        x, y = dad.Nodes[i]
        on_left = abs(x) < 1e-9 * max(p.Lx, 1.0)
        on_right = abs(x - p.Lx) < 1e-9 * max(p.Lx, 1.0)
        on_bottom = abs(y) < 1e-9 * max(p.Ly, 1.0)
        on_top = abs(y - p.Ly) < 1e-9 * max(p.Ly, 1.0)
        if ex == 4
            dad.BC[i] = 0
            dad.BV[i] = exact_ex4(x, y, 0.0)
        elseif ex == 1 || (ex >= 2 && !on_left)
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        elseif ex == 2 && on_left
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        else
            dad.BC[i] = 2
            dad.BV[i] = 0.0
        end
    end
    return dad
end

# ---------------------------------------------------------------------------
# Exact solutions (paper Examples 1–2)
# ---------------------------------------------------------------------------

function exact_ex1(x, y, t; u0=u_init, κ=κ0, Lx=Lx0, Ly=Ly0, nterms=40)
    s = 0.0
    for i = 1:nterms, j = 1:nterms
        A = 4u0 * ((-1)^i - 1) * ((-1)^j - 1) / (i * j * pi^2)
        s += A * sin(i * pi * x / Lx) * sin(j * pi * y / Ly) *
             exp(-κ * ((i * pi / Lx)^2 + (j * pi / Ly)^2) * t)
    end
    return s
end

function exact_ex2(x, y, t; u0=u_init, κ=κ0, Lx=Lx0, Ly=Ly0, nterms=40)
    # Neumann ∂u/∂x=0 at x=0, Dirichlet 0 elsewhere; IC=u0
    # X=cos(λx x), λx=(2i-1)π/(2 Lx); Y=sin(jπ y/Ly)
    s = 0.0
    for i = 1:nterms, j = 1:nterms
        λx = (2i - 1) * pi / (2 * Lx)
        λy = j * pi / Ly
        Ix = ((-1)^(i - 1)) / λx
        Iy = (1 - (-1)^j) / λy
        A = u0 * Ix * Iy / ((Lx / 2) * (Ly / 2))
        s += A * cos(λx * x) * sin(λy * y) * exp(-κ * (λx^2 + λy^2) * t)
    end
    return s
end

"""sinh(a z)/sinh(a) for z∈[0,1], overflow-safe."""
function _sinh_ratio(a::Float64, z::Float64)
    a == 0 && return z
    return exp(a * (z - 1)) * (1 - exp(-2 * a * z)) / (1 - exp(-2 * a))
end

"""
Example 4: unit square, α=1, IC=0 interior, Dirichlet on all sides.

Steady piece (image eq. 26):
  100 − ∑_n [800 n cos(nπ)/(π(4n²−1))]
        [sinh(nπx)/sinh(nπ) sin(nπy) + sinh(nπy)/sinh(nπ) sin(nπx)]

Transient sine coefficients enforce u(·,0)=0 in the interior
(the printed 1600m/n formula does not).
"""
function _ex4_uss(x, y, nterms::Int)
    s = 100.0
    @inbounds for n in 1:nterms
        an = 800.0 * n * cos(n * pi) / (pi * (4n^2 - 1))
        s -= an * (_sinh_ratio(n * pi, float(x)) * sin(n * pi * y) +
                   _sinh_ratio(n * pi, float(y)) * sin(n * pi * x))
    end
    return s
end

const _EX4_C = Dict{Int,Matrix{Float64}}()

function _ex4_coeffs(nterms::Int)
    get!(_EX4_C, nterms) do
        nq = max(80, 4nterms)
        ξ, w = gausslegendre(nq)
        x = (ξ .+ 1) ./ 2
        ww = w ./ 2
        uss = [_ex4_uss(x[i], x[j], nterms) for i in 1:nq, j in 1:nq]
        C = zeros(nterms, nterms)
        @inbounds for n in 1:nterms, m in 1:nterms
            acc = 0.0
            for j in 1:nq, i in 1:nq
                acc += ww[i] * ww[j] * (-uss[i, j]) *
                       sin(m * pi * x[i]) * sin(n * pi * x[j])
            end
            C[m, n] = 4acc
        end
        C
    end
end

function exact_ex4(x, y, t; α=κ4, nterms=40, transient=true)
    s = _ex4_uss(x, y, nterms)
    (t <= 0 && transient === false) && return s
    C = _ex4_coeffs(nterms)
    @inbounds for n in 1:nterms, m in 1:nterms
        s += C[m, n] * sin(m * pi * x) * sin(n * pi * y) *
             exp(-α * (m^2 + n^2) * pi^2 * t)
    end
    return s
end

function exact_vec(dad, ex, t)
    pts = vcat([Point2D(p) for p in dad.Nodes],
               [Point2D(p) for p in dad.internalNodes])
    ex == 1 && return [exact_ex1(p[1], p[2], t) for p in pts]
    ex == 2 && return [exact_ex2(p[1], p[2], t) for p in pts]
    ex == 4 && return [exact_ex4(p[1], p[2], t) for p in pts]
    return fill(NaN, length(pts))
end

rmse_rinf(a, b) = (sqrt(mean(abs2, a .- b)), maximum(abs, a .- b))

# Robin BEM heat: eliminate q_i = H_r (u_i − u_f) into H (Kovářík eq. 46).
# Mass M is used as assembled. Do not zero blocks or clip eigenvalues.

function _robin_arrays(dad, robin_H, robin_uf)
    N = dad.n
    rH = zeros(N)
    rUf = zeros(N)
    robin_H === nothing && return rH, rUf
    @inbounds for i in 1:N
        if dad.BC[i] == 2
            rH[i] = robin_H isa Number ? float(robin_H) : float(robin_H[i])
            rUf[i] = robin_uf isa Number ? float(robin_uf) : float(robin_uf[i])
        end
    end
    return rH, rUf
end

"""H ← H − H_r G e_i e_iᵀ on Robin nodes (paper: q_pkg = H_r (u − u_f))."""
function apply_robin_to_H!(H::AbstractMatrix, G::AbstractMatrix, BC, rH)
    n = size(G, 2)
    @inbounds for i in 1:n
        if BC[i] == 2 && rH[i] != 0
            @views H[:, i] .-= rH[i] .* G[:, i]
        end
    end
    return H
end

function robin_rhs_boost!(b, G, BC, rH, rUf)
    n = size(G, 2)
    @inbounds for i in 1:n
        if BC[i] == 2 && rH[i] != 0
            @views b .-= (rH[i] * rUf[i]) .* G[:, i]
        end
    end
    return b
end

function bem_heat_march!(dad, H0, G0, M0, u0; Δt, tf, scheme=:houbolt,
                         robin_H=nothing, robin_uf=nothing)
    N, nt = dad.n, dad.nt
    BC0 = Int.(dad.BC[1:N])
    BV0 = Float64.(dad.BV[1:N])
    rH, rUf = _robin_arrays(dad, robin_H, robin_uf)
    has_robin = any(==(2), BC0)

    Hm = Matrix{Float64}(H0)
    Gm = Matrix{Float64}(G0)
    M = Matrix{Float64}(M0)
    if has_robin
        apply_robin_to_H!(Hm, Gm, BC0, rH)
        @inbounds for i in 1:N
            if BC0[i] == 2
                dad.BC[i] = 1
                dad.BV[i] = 0.0
            end
        end
    end

    t = collect(0.0:Δt:tf)
    nT = length(t)
    T = zeros(nt, nT)
    T[:, 1] .= u0[1:nt]
    @inbounds for j in 1:N
        BC0[j] == 0 && (T[j, 1] = BV0[j])
    end

    function factor(Heff)
        set_cache!(dad; H=Heff, G=Gm)
        has_cache(dad, :A) && (dad.cache.A = nothing)
        applyBC(dad)
        b = copy(dad.b)
        has_robin && robin_rhs_boost!(b, Gm, BC0, rH, rUf)
        return factorize(Matrix(dad.A)), b
    end

    try
        Fe, be = factor(Hm - M / Δt)
        n_euler = scheme === :euler ? nT : min(3, nT)
        for i in 2:n_euler
            x = Fe \ (be .- (M / Δt) * T[:, i - 1])
            Tf = zeros(nt); qf = zeros(N)
            Tf[1:length(x)] .= x
            split_sol!(dad, Tf, qf)
            T[:, i] .= Tf
        end
        if scheme === :houbolt && nT >= 4
            Fh, bh = factor(Hm - 11 * M / (6 * Δt))
            for i in 4:nT
                rhs = bh .+ M * (-18 .* T[:, i-1] .+ 9 .* T[:, i-2] .-
                                 2 .* T[:, i-3]) / (6 * Δt)
                x = Fh \ rhs
                Tf = zeros(nt); qf = zeros(N)
                Tf[1:length(x)] .= x
                split_sol!(dad, Tf, qf)
                T[:, i] .= Tf
            end
        end
    finally
        dad.BC[1:N] .= BC0
        dad.BV[1:N] .= BV0
        set_cache!(dad; H=Hm, G=Gm)
        has_cache(dad, :A) && (dad.cache.A = nothing)
    end
    return (; t, U=T, M_int=has_robin)
end

function bem_drm(dad, u0; κ=κ0, Δt, tf, scheme=:houbolt, robin_H=nothing, robin_uf=nothing,
                 basis=RBF_BEM)
    H_G_full_direct(dad; npg=12, threaded=false)
    drm = build_drm_matrices(dad, basis; npg=12)
    return bem_heat_march!(dad, drm.H, drm.G, drm.M ./ κ, u0;
                           Δt, tf, scheme, robin_H=robin_H, robin_uf=robin_uf)
end

function bem_dibem(dad, u0; κ=κ0, Δt, tf, scheme=:houbolt, robin_H=nothing, robin_uf=nothing,
                   rbf=RBF_BEM)
    H_G_full_direct(dad; npg=12, threaded=false)
    DIBEM(dad; method=:dense, rbf=rbf)
    return bem_heat_march!(dad, Matrix(dad.H), Matrix(dad.G), Matrix(dad.M) ./ κ, u0;
                           Δt, tf, scheme, robin_H=robin_H, robin_uf=robin_uf)
end


# ---------------------------------------------------------------------------
# DiffEq on BEM-DIBEM reduced system
# ---------------------------------------------------------------------------

function bem_diffeq(dad, u0; κ=κ0, tf=tf0, alg=Rodas5P(),
                    robin_H=nothing, robin_uf=nothing,
                    dt=nothing, adaptive=true)
    H_G_full_direct(dad; npg=12, threaded=false)
    DIBEM(dad; method=:dense, rbf=RBF_BEM)
    H0 = Matrix(dad.H)
    G0 = Matrix(dad.G)
    M = Matrix(dad.M) ./ κ
    N = dad.n
    BC0 = Int.(dad.BC[1:N])
    BV0 = Float64.(dad.BV[1:N])
    rH, rUf = _robin_arrays(dad, robin_H, robin_uf)
    has_robin = any(==(2), BC0)
    if has_robin
        apply_robin_to_H!(H0, G0, BC0, rH)
        @inbounds for i in 1:N
            if BC0[i] == 2
                dad.BC[i] = 1
                dad.BV[i] = 0.0
            end
        end
    end
    set_cache!(dad; H=H0, G=G0, M=M)
    has_cache(dad, :A) && (dad.cache.A = nothing)
    applyBC(dad)
    has_robin && robin_rhs_boost!(dad.b, G0, BC0, rH, rUf)

    sys = reduced_heat_system(dad.A, M, dad.b, dad.BC, dad.ni)
    ufull = float.(u0[1:dad.nt])
    @inbounds for j in 1:N
        BC0[j] == 0 && (ufull[j] = BV0[j])
    end
    u0_red = ufull[sys.unknown]

    par = (B = sys.B, f = sys.f)
    prob = ODEProblem{false}(heat_rhs, u0_red, (0.0, tf), par)
    t_cpu = @elapsed sol = if dt === nothing
        DEsolve(prob, alg; abstol=1e-6, reltol=1e-6,
                save_everystep=false, save_start=true, save_end=true)
    else
        DEsolve(prob, alg; dt=float(dt), adaptive=adaptive,
                saveat=float(tf), save_start=true, save_end=true)
    end
    u_end = copy(ufull)
    u_end[sys.unknown] .= sol.u[end]
    @inbounds for j in 1:N
        BC0[j] == 0 && (u_end[j] = BV0[j])
    end
    dad.BC[1:N] .= BC0
    dad.BV[1:N] .= BV0
    return (; t=[0.0, tf], U=hcat(ufull, u_end), t_cpu,
            alg=string(nameof(typeof(alg))), retcode=sol.retcode)
end


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

function run_case(; ex, method, scheme, nsteps, nb=20, nint=9, basis=nothing)
    par = case_params(ex)
    dad = make_dad(ex; nb=nb, nint=nint)
    N, ni = dad.n, dad.ni
    u0 = fill(par.u0, N + ni)
    @inbounds for i in 1:N
        dad.BC[i] == 0 && (u0[i] = dad.BV[i])
    end
    Δt = nsteps > 0 ? par.tf / nsteps : NaN
    rH = ex == 3 ? 10.0 : nothing
    rUf = ex == 3 ? 50.0 : nothing
    rbf_s = basis === nothing ? RBF_SBM : basis
    rbf_b = basis === nothing ? RBF_BEM : basis
    kord = rbf_s isa PHS1 ? 1 : (rbf_s isa PHS3 ? 3 : 3)

    label = ""
    t_cpu = NaN
    Uend = fill(NaN, N + ni)
    extra = ""
    try
        if method === :sbm
            label = "SBM-DRM/$(scheme)"
            t_cpu = @elapsed sol = solve_sbm_drm(dad; κ=par.κ, Δt=Δt, tf=par.tf, u0=u0,
                                                 scheme=scheme, basis=rbf_s,
                                                 robin_H=rH, robin_uf=rUf)
            Uend = sol.U[:, end]
        elseif method === :drm
            label = "BEM-DRM/$(scheme)"
            t_cpu = @elapsed sol = bem_drm(dad, u0; κ=par.κ, Δt=Δt, tf=par.tf, scheme=scheme,
                                           robin_H=rH, robin_uf=rUf, basis=rbf_b)
            Uend = sol.U[1:N+ni, end]
        elseif method === :dibem
            label = "BEM-DIBEM/$(scheme)"
            t_cpu = @elapsed sol = bem_dibem(dad, u0; κ=par.κ, Δt=Δt, tf=par.tf, scheme=scheme,
                                             robin_H=rH, robin_uf=rUf, rbf=rbf_b)
            Uend = sol.U[1:N+ni, end]
        elseif method === :kansa
            label = "Kansa-BEM/$(scheme)"
            t_cpu = @elapsed sol = solve_kansa_bem_heat(dad; κ=par.κ, Δt=Δt, tf=par.tf, u0=u0,
                scheme=scheme, basis_order=kord, robin_H=rH, robin_uf=rUf)
            Uend = sol.U[1:N+ni, end]
        elseif method === :diffeq
            alg = scheme isa Symbol ? eval(scheme)() : scheme
            label = "BEM-DIBEM/$(nameof(typeof(alg)))"
            sol = bem_diffeq(dad, u0; κ=par.κ, tf=par.tf, alg=alg, robin_H=rH, robin_uf=rUf)
            t_cpu = sol.t_cpu
            Uend = sol.U[1:N+ni, end]
            extra = string(sol.retcode)
        else
            error("unknown method")
        end
    catch e
        @warn "failed" ex method scheme exception=(e, catch_backtrace())
        return (ex=ex, method=string(method), scheme=string(scheme), label=label,
                nsteps=nsteps, n=N, ni=ni, rmse=NaN, rinf=NaN, t_cpu=NaN,
                maxu=NaN, meanu=NaN, ucenter=NaN, ok=false, note=string(e))
    end

    rmse = NaN
    rinf = NaN
    if (ex == 1 || ex == 2 || ex == 4) && all(isfinite, Uend)
        uex = exact_vec(dad, ex, par.tf)
        rmse, rinf = rmse_rinf(Uend, uex)
    end
    pts = vcat(dad.Nodes, dad.internalNodes)
    mid = Point2D(par.Lx / 2, par.Ly / 2)
    ic = argmin(norm(Point2D(pt) - mid) for pt in pts)
    ok = all(isfinite, Uend) && maximum(abs, Uend) < 1e6
    return (ex=ex, method=string(method), scheme=string(scheme), label=label,
            nsteps=nsteps, n=N, ni=ni, rmse=rmse, rinf=rinf, t_cpu=t_cpu,
            maxu=maximum(abs, Uend), meanu=mean(Uend), ucenter=Uend[ic],
            ok=ok, note=extra)
end

function main()
    println("="^72)
    println(" SBM transient study → $STUDY_OUT")
    println("="^72)

    nsteps_list = QUICK ? [60] : [60, 120, 240]
    nb = QUICK ? 12 : 20
    nint = QUICK ? 5 : 9

    rows = NamedTuple[]

    for ex in 1:4, meth in (:sbm, :drm, :dibem), sch in (:euler, :houbolt), ns in nsteps_list
        r = run_case(; ex, method=meth, scheme=sch, nsteps=ns, nb=nb, nint=nint)
        push!(rows, r)
        @printf("ex%d %-24s ns=%3d RMSE=%9.2e R∞=%9.2e t=%6.3fs max=%.3e %s\n",
                r.ex, r.label, r.nsteps, r.rmse, r.rinf, r.t_cpu, r.maxu,
                r.ok ? "ok" : "FAIL")
    end

    algs = Any[Tsit5(), Rosenbrock23(), Rodas5P(), FBDF()]
    for ex in (1, 2, 4), alg in algs
        r = run_case(; ex, method=:diffeq, scheme=alg, nsteps=0, nb=nb, nint=nint)
        push!(rows, r)
        @printf("ex%d %-24s      RMSE=%9.2e R∞=%9.2e t=%6.3fs %s %s\n",
                r.ex, r.label, r.rmse, r.rinf, r.t_cpu, r.ok ? "ok" : "FAIL", r.note)
    end

    write_csv(joinpath(STUDY_OUT, "results_all.csv"), rows)
    write_csv(joinpath(STUDY_OUT, "table_ex1.csv"), filter(r -> r.ex == 1 && r.method != "diffeq", rows))
    write_csv(joinpath(STUDY_OUT, "table_ex2.csv"), filter(r -> r.ex == 2 && r.method != "diffeq", rows))
    write_csv(joinpath(STUDY_OUT, "table_ex3.csv"), filter(r -> r.ex == 3 && r.method != "diffeq", rows))
    write_csv(joinpath(STUDY_OUT, "table_ex4.csv"), filter(r -> r.ex == 4 && r.method != "diffeq", rows))
    write_csv(joinpath(STUDY_OUT, "table_diffeq.csv"), filter(r -> r.method == "diffeq", rows))

    dad = make_dad(1; nb=nb, nint=nint)
    N, ni = dad.n, dad.ni
    u0 = fill(u_init, N + ni); u0[1:N] .= 0
    ns = QUICK ? 60 : 120
    sol = solve_sbm_drm(dad; κ=κ0, Δt=tf0/ns, tf=tf0, u0=u0, scheme=:houbolt, basis=RBF_SBM)
    pts = vcat(dad.Nodes, dad.internalNodes)
    mid = Point2D(Lx0 / 2, Ly0 / 2)
    ic = argmin(norm(Point2D(pt) - mid) for pt in pts)
    open(joinpath(STUDY_OUT, "history_ex1_center.csv"), "w") do io
        println(io, "t,u_sbm,u_exact")
        for (k, tt) in enumerate(sol.t)
            pt = Point2D(pts[ic])
            @printf(io, "%.6f,%.8e,%.8e\n", tt, sol.U[ic, k], exact_ex1(pt[1], pt[2], tt))
        end
    end
    println("wrote history_ex1_center.csv")

    return rows
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "sbm_transient_study.jl")
    main()
end
