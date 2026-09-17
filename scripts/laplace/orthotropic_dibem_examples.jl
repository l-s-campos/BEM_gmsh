# Orthotropic DIBEM paper examples (square mixed / Dirichlet / plate-with-hole).
#
#   julia --project=. scripts/laplace/orthotropic_dibem_examples.jl
#   julia --project=. scripts/laplace/orthotropic_dibem_examples.jl quick
#   julia --project=. scripts/laplace/orthotropic_dibem_examples.jl ex1 ex2a
#
# Two fundamental-solution paths:
#   :iso   — isotropic Laplace FS + DIBEM residual (`solve_anisotropic_dibem!`)
#   :aniso — anisotropic Laplace FS, homogeneous BEM (`AnisotropicLaplace`)
#
# Results JSON + Typst figures go to
#   /home/lsc/OneDrive/artigos/escritos/2026/DIBEM orto
using BEM
using LinearAlgebra
using StaticArrays
using Statistics
using Printf
using Dates
using Gmsh: gmsh
using DrWatson: datadir

include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))

const OUTDIR = get(ENV, "DIBEM_ORTO_OUT",
    "/home/lsc/OneDrive/artigos/escritos/2026/DIBEM orto")
const RESULTDIR = joinpath(OUTDIR, "results")
const FIGDIR = joinpath(OUTDIR, "figs")
const RBF = PHS(3; poly_deg=1)
const RBF_QUAD = PHS(3; poly_deg=2)
const NPG = 12
const CORNER_TOL = 0.02

mkpath(RESULTDIR)
mkpath(FIGDIR)

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
json_escape(s::AbstractString) = replace(replace(s, '\\' => "\\\\"), '"' => "\\\"")
writejson(io::IO, x::Integer) = print(io, x)
writejson(io::IO, x::Bool) = print(io, x ? "true" : "false")
writejson(io::IO, ::Nothing) = print(io, "null")
writejson(io::IO, x::Symbol) = writejson(io, String(x))
writejson(io::IO, x::AbstractString) = (print(io, '"', json_escape(x), '"'); nothing)
function writejson(io::IO, x::AbstractFloat)
    isfinite(x) ? @printf(io, "%.10e", x) : print(io, "null")
    return nothing
end
function writejson(io::IO, x::AbstractVector)
    print(io, '[')
    @inbounds for i in eachindex(x)
        i == firstindex(x) || print(io, ',')
        writejson(io, x[i])
    end
    print(io, ']')
    return nothing
end
function writejson(io::IO, x::AbstractDict)
    print(io, '{')
    first = true
    for (k, v) in x
        first || print(io, ',')
        first = false
        print(io, '"', k, "\":")
        writejson(io, v)
    end
    print(io, '}')
    return nothing
end
function save_json(path, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        writejson(io, obj)
    end
    println("  wrote ", path, "  (", @sprintf("%.1f", filesize(path) / 1024), " KiB)")
    return path
end

# ---------------------------------------------------------------------------
# Analytical solutions (stable Fourier sums)
# ---------------------------------------------------------------------------
"""Orthotropic mixed-BC square, two-series solution.

The manuscript Eq. (32) as transcribed (`u=x₁ + ∑ sin(nπx₁/2L){…}`) does
**not** satisfy `u(x,0)=0` or `∂u/∂x(1,y)=1`. The field used here is the
standard eigenfunction split on `[0,a]×[0,b]`:

- `u₁`: quarter-wave sine in `x` (homogeneous Neumann at `x=a`), top flux 1
- `u₂`: quarter-wave sine in `y` (homogeneous Neumann at `y=b`), right flux 1

Both vanish on `x=0` and `y=0`. Unit square: `a=b=1`.
"""
function _sinh_over_cosh(z, ztop)
    em = exp(-2 * clamp(ztop, -400.0, 400.0))
    return (exp(clamp(z - ztop, -700.0, 700.0)) -
            exp(clamp(-z - ztop, -700.0, 700.0))) / (1 + em)
end
function _cosh_over_cosh(z, ztop)
    em = exp(-2 * clamp(ztop, -400.0, 400.0))
    return (exp(clamp(z - ztop, -700.0, 700.0)) +
            exp(clamp(-z - ztop, -700.0, 700.0))) / (1 + em)
end

function example1_u(x, y; k1=2.0, k2=0.5, a=1.0, b=1.0, N=250)
    α = sqrt(k1 / k2)
    u = 0.0
    @inbounds for n in 1:N
        λ = (n - 0.5) * π / a
        u += (2 / (λ^2 * α)) * sin(λ * x) * _sinh_over_cosh(λ * α * y, λ * α * b)
        μ = (n - 0.5) * π / b
        γ = μ / α
        u += (2 / (μ * γ)) * sin(μ * y) * _sinh_over_cosh(γ * x, γ * a)
    end
    return u
end

function example1_grad(x, y; k1=2.0, k2=0.5, a=1.0, b=1.0, N=250)
    α = sqrt(k1 / k2)
    ux = 0.0
    uy = 0.0
    @inbounds for n in 1:N
        λ = (n - 0.5) * π / a
        sxc = _sinh_over_cosh(λ * α * y, λ * α * b)
        cxc = _cosh_over_cosh(λ * α * y, λ * α * b)
        ux += (2 / (λ * α)) * cos(λ * x) * sxc
        uy += 2 / λ * sin(λ * x) * cxc
        μ = (n - 0.5) * π / b
        γ = μ / α
        syc = _sinh_over_cosh(γ * x, γ * a)
        cyc = _cosh_over_cosh(γ * x, γ * a)
        ux += 2 / μ * sin(μ * y) * cyc
        uy += (2 / γ) * cos(μ * y) * syc
    end
    return SVector(ux, uy)
end

"""Eq. (33)."""
function example2_sinusoidal(x, y; kx=2.0, ky=0.5)
    r = sqrt(kx / ky)
    return sin(π * x) * sinh(π * y * r) / sinh(π * r)
end
function example2_sinusoidal_grad(x, y; kx=2.0, ky=0.5)
    r = sqrt(kx / ky)
    den = sinh(π * r)
    ux = π * cos(π * x) * sinh(π * y * r) / den
    uy = π * r * sin(π * x) * cosh(π * y * r) / den
    return SVector(ux, uy)
end

"""Eq. (35). Degenerate anisotropic: ∇·(K∇u)=0 with K=[1 1;1 1]."""
example2_anisotropic(X1, X2) = (X1 - X2)^2
example2_anisotropic_grad(X1, X2) = SVector(2 * (X1 - X2), -2 * (X1 - X2))

"""Eq. (36). Unit square L=W=1. Odd terms only."""
function example2_discontinuous(x1, x2; k1=2.0, k2=0.5, L=1.0, W=1.0, N=500)
    r = sqrt(k1 / k2)
    u = 0.0
    @inbounds for m in 0:N-1
        n = 2m + 1
        arg = n * π * r / L
        # sinh(n π x2 r / L) / sinh(n π W r / L)
        ratio = (exp(clamp(arg * (x2 - W), -700.0, 700.0)) *
                 (1 - exp(clamp(-2 * arg * x2, -700.0, 700.0)))) /
                (1 - exp(clamp(-2 * arg * W, -700.0, 700.0)))
        u += (2 / n) * sin(n * π * x1 / L) * ratio
    end
    return (2 / π) * u
end
function example2_discontinuous_grad(x1, x2; kwargs...)
    h = 1e-6
    ux = (example2_discontinuous(x1 + h, x2; kwargs...) -
          example2_discontinuous(x1 - h, x2; kwargs...)) / (2h)
    uy = (example2_discontinuous(x1, x2 + h; kwargs...) -
          example2_discontinuous(x1, x2 - h; kwargs...)) / (2h)
    return SVector(ux, uy)
end

avg_rel(num, ana) = begin
    m = maximum(abs, ana)
    m == 0 && return mean(abs.(num .- ana))
    mean(abs.(num .- ana)) / m
end

# ---------------------------------------------------------------------------
# Meshes: linear 1-D edges, unstructured interior (cell centroids → DIBEM poles)
# ---------------------------------------------------------------------------
function _count_surf_elems()
    etypes, etags, _ = gmsh.model.mesh.getElements(2, -1)
    isempty(etags) && return 0
    return sum(length, etags)
end

"""Unit square. `nel_edge` linear elements per side (total 4*nel_edge). `lc` sets interior size."""
function mesh_unit_square(; nome, nel_edge::Int, lc::Float64, ordem::Int=1)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    ni = with_gmsh(; terminal=0) do
        gmsh.clear()
        gmsh.model.add(nome)
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
        p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        s1 = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        npts = nel_edge + 1
        for l in (l1, l2, l3, l4)
            gmsh.model.mesh.setTransfiniteCurve(l, npts)
        end
        gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], -1, "1;0")
        gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", string(lc))
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(ordem)
        ncell = _count_surf_elems()
        gmsh.write(out)
        return ncell
    end
    return out, ni
end

"""Unit square minus disk R=0.25 at (0.5,0.5)."""
function mesh_square_hole(; nome, lc::Float64, lc_hole::Union{Nothing,Float64}=nothing,
        ordem::Int=1, L=1.0, r=0.25, cx=0.5, cy=0.5)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    with_gmsh(; terminal=0) do
        gmsh.clear()
        gmsh.model.add(nome)
        plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
        hole = gmsh.model.occ.addDisk(cx, cy, 0.0, r, r)
        gmsh.model.occ.cut([(2, plate)], [(2, hole)])
        gmsh.model.occ.synchronize()
        g = _plate_hole_curve_groups(; L=L)
        gmsh.model.addPhysicalGroup(1, g[:left], -1, "0;0")
        gmsh.model.addPhysicalGroup(1, g[:right], -1, "1;0")
        gmsh.model.addPhysicalGroup(1, vcat(g[:bottom], g[:top], g[:hole]), -1, "1;0")
        surfs = [t for (_d, t) in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surfs, -1, "plate")
        hhole = lc_hole === nothing ? lc : lc_hole
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(lc, hhole))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(lc, hhole))
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(ordem)
        try
            gmsh.model.mesh.reverse([(1, t) for t in g[:hole]])
        catch
        end
        gmsh.write(out)
    end
    return out
end

function mesh_square_target(nel_edge, ni_target; nome, tol=0.35, maxiter=8)
    lc = sqrt(2 / max(ni_target, 4))
    lo, hi = max(lc / 6, 0.01), min(lc * 6, 0.9)
    best_path, best_ni, best_lc = "", typemax(Int), lc
    for it in 1:maxiter
        path, ni = mesh_unit_square(; nome=nome * "_try$(it)", nel_edge=nel_edge, lc=lc)
        if abs(ni - ni_target) <= abs(best_ni - ni_target)
            best_path, best_ni, best_lc = path, ni, lc
        end
        rel = abs(ni - ni_target) / max(ni_target, 1)
        rel <= tol && return path, ni, lc
        if ni > ni_target
            lo = lc
            lc = sqrt(lc * hi)
        else
            hi = lc
            lc = sqrt(lc * lo)
        end
    end
    best_path == "" && error("mesh_square_target failed for nel_edge=$nel_edge ni=$ni_target")
    @printf("  mesh nel_edge=%d target_ni=%d -> ni=%d lc=%.3f\n",
        nel_edge, ni_target, best_ni, best_lc)
    return best_path, best_ni, best_lc
end

# ---------------------------------------------------------------------------
# Geometry helpers / BCs
# ---------------------------------------------------------------------------
function classify_edge(p; L=1.0, tol=1e-7)
    x, y = p[1], p[2]
    x <= tol && return :left
    x >= L - tol && return :right
    y <= tol && return :bottom
    y >= L - tol && return :top
    return :other
end

near_corner(p; L=1.0, tol=CORNER_TOL) =
    any(norm(SVector(p[1], p[2]) - c) < tol for c in
        (SVector(0.0, 0.0), SVector(L, 0.0), SVector(L, L), SVector(0.0, L)))

function apply_mixed_ex1!(dad; k1=2.0, k2=0.5)
    K = @SMatrix [k1 0.0; 0.0 k2]
    for i in 1:dad.n
        e = classify_edge(dad.Nodes[i])
        n = dad.Normal[i]
        if e === :left || e === :bottom
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        elseif e === :right
            dad.BC[i] = 1
            dad.BV[i] = -dot(n, K * SVector(1.0, 0.0))
        elseif e === :top
            dad.BC[i] = 1
            dad.BV[i] = -dot(n, K * SVector(0.0, 1.0))
        end
    end
    return dad
end

function apply_dirichlet!(dad, ufun)
    for i in 1:dad.n
        p = dad.Nodes[i]
        dad.BC[i] = 0
        dad.BV[i] = float(ufun(p[1], p[2]))
    end
    return dad
end

function apply_ex3!(dad; k1, k2)
    K = @SMatrix [k1 0.0; 0.0 k2]
    for i in 1:dad.n
        e = classify_edge(dad.Nodes[i])
        n = dad.Normal[i]
        if e === :left
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        elseif e === :right
            dad.BC[i] = 1
            # paper: ∂u/∂n = 1 on x=1
            dad.BV[i] = -dot(n, K * SVector(1.0, 0.0))
        else
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        end
    end
    return dad
end

pkg_flux(n, K, grad) = -dot(n, K * grad)

# ---------------------------------------------------------------------------
# Solve paths
# ---------------------------------------------------------------------------
function _nlocal(dad; hole=false)
    # IBP/RBF-FD: 21 neighbours. A larger stencil (41) on a coarse Cartesian
    # internal_grid is ill-conditioned (Example 1, 5×5 → 50–3000 % MRPE).
    nt = Int(dad.nt)
    return min(21, max(nt - 1, 4))
end

function solve_iso_dibem!(dad, K; rbf=RBF, hole=false, kiso=nothing)
    assemble!(dad; npg=NPG, threaded=true)
    DIBEM(dad; rbf=rbf, threaded=true)
    nl = _nlocal(dad; hole=hole)
    # S3 IBP: no Hess(u). S1 Hessian stalls on Cartesian internal_grid
    # (and on holes). kiso override (singular K, Example 2C) stays S1.
    if kiso === nothing
        solve_anisotropic_ibp!(dad, K; rbf=rbf, npg=NPG, nlocal=nl)
    else
        solve_anisotropic_dibem!(dad, K; rbf=rbf, npg=NPG, nlocal=nl, kiso=kiso)
    end
    return dad
end

function solve_aniso_fs!(dad)
    assemble!(dad; npg=NPG, threaded=true)
    solve(dad)
    return dad
end

function make_dad(path, fs, K; rbf_needed=true)
    if fs === :iso
        return format2d(path, Laplace(1.0); pontointerno=true, tipo=1)
    elseif fs === :aniso
        return format2d(path, AnisotropicLaplace(K); pontointerno=true, tipo=1)
    else
        error("fs must be :iso or :aniso")
    end
end

function run_case(path, fs, K; setbc, rbf=RBF, hole=false, kiso=nothing)
    dad = make_dad(path, fs, K)
    setbc(dad)
    t0 = time()
    if fs === :iso
        solve_iso_dibem!(dad, K; rbf=rbf, hole=hole, kiso=kiso)
    else
        solve_aniso_fs!(dad)
    end
    return dad, time() - t0
end

function errors_vs_ana(dad, ufun, gfun, K; skip_corners=true)
    n = dad.n
    T = dad.T
    q = dad.q
    # internals
    ni = dad.ni
    Ti = T[n+1:n+ni]
    Tia = [ufun(p[1], p[2]) for p in dad.internalNodes]
    err_int = ni == 0 ? NaN : avg_rel(Ti, Tia)
    # boundary T and q
    keep = trues(n)
    if skip_corners
        @inbounds for i in 1:n
            keep[i] = !near_corner(dad.Nodes[i])
        end
    end
    idx = findall(keep)
    Tb = T[idx]
    Tba = [ufun(dad.Nodes[i][1], dad.Nodes[i][2]) for i in idx]
    qb = q[idx]
    qba = [pkg_flux(dad.Normal[i], K, gfun(dad.Nodes[i][1], dad.Nodes[i][2])) for i in idx]
    err_Tb = avg_rel(Tb, Tba)
    err_q = avg_rel(qb, qba)
    Tmax = maximum(abs, vcat(Tia, Tba))
    return (; err_int, err_Tb, err_q, ni, n, nt=dad.nt, Tmax,
        T_int=Ti, T_int_ana=Tia, xy_int=[[p[1], p[2]] for p in dad.internalNodes],
        T_bnd=Tb, T_bnd_ana=Tba, q_bnd=qb, q_bnd_ana=qba,
        xy_bnd=[[dad.Nodes[i][1], dad.Nodes[i][2]] for i in idx])
end

function summarize(err; store_field=false)
    d = Dict{String,Any}(
        "err_int" => err.err_int,
        "err_T_bnd" => err.err_Tb,
        "err_q_bnd" => err.err_q,
        "ni" => err.ni,
        "n" => err.n,
        "nt" => err.nt,
        "Tmax_ana" => err.Tmax,
    )
    if store_field
        d["T_int"] = collect(Float64, err.T_int)
        d["T_int_ana"] = collect(Float64, err.T_int_ana)
        d["xy_int"] = err.xy_int
        d["T_bnd"] = collect(Float64, err.T_bnd)
        d["T_bnd_ana"] = collect(Float64, err.T_bnd_ana)
        d["q_bnd"] = collect(Float64, err.q_bnd)
        d["q_bnd_ana"] = collect(Float64, err.q_bnd_ana)
        d["xy_bnd"] = err.xy_bnd
    end
    return d
end

function right_edge(dad)
    ys = Float64[]; Ts = Float64[]; qs = Float64[]
    for i in 1:dad.n
        p = dad.Nodes[i]
        abs(p[1] - 1) < 0.03 || continue
        near_corner(p) && continue
        push!(ys, p[2]); push!(Ts, dad.T[i]); push!(qs, dad.q[i])
    end
    perm = sortperm(ys)
    return ys[perm], Ts[perm], qs[perm]
end

# ---------------------------------------------------------------------------
# Example drivers
# ---------------------------------------------------------------------------
const FS_BOTH = (:iso, :aniso)

function _print_run(tag, fs, nel, ni, dt, err)
    @printf("  %-18s fs=%-5s  nel=%4d  n=%4d  ni=%5d  t=%6.2fs  e_int=%.3e  e_T=%.3e  e_q=%.3e\n",
        tag, fs, nel, err.n, ni, dt, err.err_int, err.err_Tb, err.err_q)
    flush(stdout)
end

function run_example1(; nels=[80, 160, 320], ni_targets=[16, 21, 57, 161, 584],
        store_every=false)
    println("\n=== Example 1  mixed BC  k1=2 k2=0.5 ===")
    k1, k2 = 2.0, 0.5
    K = @SMatrix [k1 0.0; 0.0 k2]
    ufun = (x, y) -> example1_u(x, y; k1=k1, k2=k2)
    gfun = (x, y) -> example1_grad(x, y; k1=k1, k2=k2)
    rows = Dict{String,Any}[]
    for nel in nels
        nel_edge = nel ÷ 4
        for (it, ni_t) in enumerate(ni_targets)
            nome = @sprintf("orto_e1_n%d_i%d", nel, ni_t)
            path, ni, lc = mesh_square_target(nel_edge, ni_t; nome=nome)
            for fs in FS_BOTH
                dad, dt = run_case(path, fs, K; setbc=d -> apply_mixed_ex1!(d; k1=k1, k2=k2))
                err = errors_vs_ana(dad, ufun, gfun, K)
                _print_run("ex1", fs, nel, err.ni, dt, err)
                rec = merge(summarize(err; store_field=store_every && it == length(ni_targets) && nel == nels[end]),
                    Dict("example" => "1", "fs" => String(fs), "nel" => nel,
                        "ni_target" => ni_t, "lc" => lc, "k1" => k1, "k2" => k2,
                        "time" => dt))
                if store_every && nel == nels[end] && it == min(length(ni_targets), 4) && fs === :iso
                    ys, Ts, _qs = right_edge(dad)
                    rec["profile"] = Dict("y" => ys, "T" => Ts,
                        "Tana" => [ufun(1.0, y) for y in ys])
                end
                push!(rows, rec)
            end
        end
    end
    return rows
end

function run_example2a(; nels=[160, 320], ni_targets=[57, 161, 332, 584])
    println("\n=== Example 2A  sinusoidal Dirichlet  k1=2 k2=0.5 ===")
    k1, k2 = 2.0, 0.5
    K = @SMatrix [k1 0.0; 0.0 k2]
    ufun = (x, y) -> example2_sinusoidal(x, y; kx=k1, ky=k2)
    gfun = (x, y) -> example2_sinusoidal_grad(x, y; kx=k1, ky=k2)
    rows = Dict{String,Any}[]
    for nel in nels
        nel_edge = nel ÷ 4
        for ni_t in ni_targets
            nome = @sprintf("orto_e2a_n%d_i%d", nel, ni_t)
            path, ni, lc = mesh_square_target(nel_edge, ni_t; nome=nome)
            for fs in FS_BOTH
                dad, dt = run_case(path, fs, K;
                    setbc=d -> apply_dirichlet!(d, ufun))
                err = errors_vs_ana(dad, ufun, gfun, K)
                _print_run("ex2a", fs, nel, err.ni, dt, err)
                push!(rows, merge(summarize(err),
                    Dict("example" => "2A", "fs" => String(fs), "nel" => nel,
                        "ni_target" => ni_t, "lc" => lc, "k1" => k1, "k2" => k2,
                        "time" => dt)))
            end
        end
    end
    return rows
end

function run_example2b(; pairs=[(160, 161), (320, 332)], k1s=1.0:1.0:5.0, k2=0.5)
    println("\n=== Example 2B  diffusivity contrast ===")
    rows = Dict{String,Any}[]
    for (nel, ni_t) in pairs
        nel_edge = nel ÷ 4
        nome = @sprintf("orto_e2b_n%d_i%d", nel, ni_t)
        path, ni, lc = mesh_square_target(nel_edge, ni_t; nome=nome)
        for k1 in k1s
            K = @SMatrix [k1 0.0; 0.0 k2]
            ufun = (x, y) -> example2_sinusoidal(x, y; kx=k1, ky=k2)
            gfun = (x, y) -> example2_sinusoidal_grad(x, y; kx=k1, ky=k2)
            for fs in FS_BOTH
                dad, dt = run_case(path, fs, K;
                    setbc=d -> apply_dirichlet!(d, ufun))
                err = errors_vs_ana(dad, ufun, gfun, K)
                _print_run("ex2b k1=$(k1)", fs, nel, err.ni, dt, err)
                push!(rows, merge(summarize(err),
                    Dict("example" => "2B", "fs" => String(fs), "nel" => nel,
                        "ni_target" => ni_t, "lc" => lc, "k1" => k1, "k2" => k2,
                        "time" => dt)))
            end
        end
    end
    return rows
end

function run_example2c(; nels=[160, 320], ni_targets=[57, 161, 332, 584])
    println("\n=== Example 2C  rotated / degenerate anisotropic ===")
    # K = [1 1; 1 1] is singular (det=0). Anisotropic FS needs SPD K, so
    # :aniso is skipped. Iso DIBEM uses kiso=1 so ΔK=[0 1;1 0].
    K = @SMatrix [1.0 1.0; 1.0 1.0]
    ufun = example2_anisotropic
    gfun = example2_anisotropic_grad
    rows = Dict{String,Any}[]
    for nel in nels
        nel_edge = nel ÷ 4
        for ni_t in ni_targets
            nome = @sprintf("orto_e2c_n%d_i%d", nel, ni_t)
            path, ni, lc = mesh_square_target(nel_edge, ni_t; nome=nome)
            dad, dt = run_case(path, :iso, K;
                setbc=d -> apply_dirichlet!(d, ufun),
                rbf=RBF_QUAD, kiso=1.0)
            err = errors_vs_ana(dad, ufun, gfun, K)
            _print_run("ex2c", :iso, nel, err.ni, dt, err)
            push!(rows, merge(summarize(err),
                Dict("example" => "2C", "fs" => "iso", "nel" => nel,
                    "ni_target" => ni_t, "lc" => lc, "k11" => 1.0, "k12" => 1.0,
                    "k22" => 1.0, "time" => dt, "note" =>
                    "K=[1 1;1 1] singular; anisotropic FS skipped; iso DIBEM kiso=1")))
        end
    end
    return rows
end

function run_example2d(; nel=160, ni_target=161)
    println("\n=== Example 2D  discontinuous Dirichlet  k1=2 k2=0.5 ===")
    k1, k2 = 2.0, 0.5
    K = @SMatrix [k1 0.0; 0.0 k2]
    ufun = (x, y) -> example2_discontinuous(x, y; k1=k1, k2=k2)
    gfun = (x, y) -> example2_discontinuous_grad(x, y; k1=k1, k2=k2)
    nel_edge = nel ÷ 4
    path, ni, lc = mesh_square_target(nel_edge, ni_target; nome="orto_e2d")
    rows = Dict{String,Any}[]
    for fs in FS_BOTH
        dad, dt = run_case(path, fs, K;
            setbc=d -> apply_dirichlet!(d, (x, y) -> begin
                y >= 1 - 1e-8 ? 1.0 : 0.0
            end))
        err = errors_vs_ana(dad, ufun, gfun, K; skip_corners=true)
        _print_run("ex2d", fs, nel, err.ni, dt, err)
        rec = merge(summarize(err; store_field=true),
            Dict("example" => "2D", "fs" => String(fs), "nel" => nel,
                "ni_target" => ni_target, "lc" => lc, "k1" => k1, "k2" => k2,
                "time" => dt, "note" => "flux near upper corners excluded"))
        push!(rows, rec)
    end
    return rows
end

function run_example3(; lcs=[0.10, 0.06, 0.04],
        kpairs=vcat([(k, 0.5) for k in (1.0, 3.0, 5.0)],
            [(0.5, k) for k in 1.0:1.0:5.0]))
    println("\n=== Example 3  square with hole ===")
    rows = Dict{String,Any}[]
    for lc in lcs
        nome = @sprintf("orto_e3_lc%.3f", lc)
        path = mesh_square_hole(; nome=nome, lc=lc)
        for (k1, k2) in kpairs
            K = @SMatrix [k1 0.0; 0.0 k2]
            for fs in FS_BOTH
                dad, dt = run_case(path, fs, K;
                    setbc=d -> apply_ex3!(d; k1=k1, k2=k2), hole=true)
                ys, Ts, qs = right_edge(dad)
                Tmid = isempty(Ts) ? NaN : Ts[argmin(abs.(ys .- 0.5))]
                @printf("  ex3 fs=%-5s  lc=%.3f  n=%4d ni=%5d  k1=%.1f k2=%.1f  t=%.2fs  Tmid=%.4f\n",
                    fs, lc, dad.n, dad.ni, k1, k2, dt, Tmid)
                flush(stdout)
                push!(rows, Dict{String,Any}(
                    "example" => "3", "fs" => String(fs), "lc" => lc,
                    "n" => dad.n, "ni" => dad.ni, "nel" => length(dad.elements),
                    "k1" => k1, "k2" => k2, "time" => dt, "T_mid" => Tmid,
                    "y_right" => ys, "T_right" => Ts, "q_right" => qs,
                ))
            end
        end
    end
    return rows
end

# ---------------------------------------------------------------------------
# Analytical self-check
# ---------------------------------------------------------------------------
function check_analyticals()
    println("=== analytical BC / PDE checks ===")
    # ex1
    xs = range(0, 1; length=21)
    uL = maximum(abs, example1_u(0.0, y) for y in xs)
    uB = maximum(abs, example1_u(x, 0.0) for x in xs)
    uxR = maximum(abs, example1_grad(1.0, y)[1] - 1 for y in xs[2:end-1])
    uyT = maximum(abs, example1_grad(x, 1.0)[2] - 1 for x in xs[2:end-1])
    @printf("  ex1  u(0,y) max=%.2e  u(x,0) max=%.2e  ux(1,y)-1 max=%.2e  uy(x,1)-1 max=%.2e\n",
        uL, uB, uxR, uyT)
    # ex2a
    r = 2.0
    u2 = example2_sinusoidal(0.5, 1.0)
    @printf("  ex2a u(0.5,1)=%.6f  (expect 1)  u(0.5,0)=%.2e\n", u2, example2_sinusoidal(0.5, 0.0))
    # ex2c PDE
    # Hess: u,11=2, u,12=-2, u,22=2 → u,11+2 u,12+u,22 = 0
    @printf("  ex2c u(0,0)=%.1f u(1,0)=%.1f u(1,1)=%.1f u(0,1)=%.1f\n",
        example2_anisotropic(0, 0), example2_anisotropic(1, 0),
        example2_anisotropic(1, 1), example2_anisotropic(0, 1))
    # ex2d
    @printf("  ex2d u(0.5,1)=%.4f  u(0.5,0)=%.2e  u(0,0.5)=%.2e\n",
        example2_discontinuous(0.5, 1.0), example2_discontinuous(0.5, 0.0),
        example2_discontinuous(0.0, 0.5))
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function _typ_arr(v)
    return "(" * join((@sprintf("%.8e", float(x)) for x in v), ", ") * ")"
end

function write_conv_typ(path; title, xlabel, ylabel, series, logx=true, logy=true)
    # series: Vector of NamedTuples (label, x, y)
    open(path, "w") do io
        println(io, "#import \"/lib.typ\": figure-page, lq")
        println(io, "#figure-page({")
        println(io, "  show: lq.layout")
        println(io, "  lq.diagram(")
        println(io, "    width: 8.6cm, height: 5.6cm,")
        println(io, "    xlabel: [", xlabel, "], ylabel: [", ylabel, "],")
        logx && println(io, "    xscale: \"log\",")
        logy && println(io, "    yscale: \"log\",")
        println(io, "    legend: (position: top + right),")
        marks = ["\"o\"", "\"s\"", "\"d\"", "\"*\""]
        for (i, s) in enumerate(series)
            mk = marks[mod1(i, length(marks))]
            println(io, "    lq.plot(", _typ_arr(s.x), ", ", _typ_arr(s.y),
                ", mark: ", mk, ", label: [", s.label, "], stroke: 0.9pt),")
        end
        println(io, "  )")
        println(io, "})")
    end
    return path
end

function write_xy_typ(path; title, xlabel, ylabel, series)
    open(path, "w") do io
        println(io, "#import \"/lib.typ\": figure-page, lq")
        println(io, "#figure-page({")
        println(io, "  show: lq.layout")
        println(io, "  lq.diagram(")
        println(io, "    width: 8.6cm, height: 5.6cm,")
        println(io, "    xlabel: [", xlabel, "], ylabel: [", ylabel, "],")
        println(io, "    legend: (position: top + left),")
        for (i, s) in enumerate(series)
            ls = get(s, :dash, false) ? "(dash: \"dashed\")" : "0.9pt"
            mk = get(s, :mark, "none")
            println(io, "    lq.plot(", _typ_arr(s.x), ", ", _typ_arr(s.y),
                ", mark: ", mk, ", label: [", s.label, "], stroke: ", ls, "),")
        end
        println(io, "  )")
        println(io, "})")
    end
    return path
end

function runs_of(rows, example)
    return filter(r -> string(get(r, "example", "")) == example, rows)
end

function write_all_figures(rows)
    mkpath(FIGDIR)
    # Example 1: err_int vs ni, one series per (nel, fs)
    e1 = runs_of(rows, "1")
    if !isempty(e1)
        series = []
        for nel in sort(unique(Int(r["nel"]) for r in e1))
            for fs in ("iso", "aniso")
                sub = filter(r -> Int(r["nel"]) == nel && r["fs"] == fs, e1)
                isempty(sub) && continue
                perm = sortperm([Int(r["ni"]) for r in sub])
                sub = sub[perm]
                push!(series, (label="n=$(nel) $(fs)",
                    x=[Int(r["ni"]) for r in sub],
                    y=[float(r["err_int"]) for r in sub]))
            end
        end
        write_conv_typ(joinpath(FIGDIR, "conv-ex1.typ");
            title="Example 1", xlabel="\$N_\"int\"\$", ylabel="avg. rel. error",
            series=series)
    end
    e2a = runs_of(rows, "2A")
    if !isempty(e2a)
        series = []
        for nel in sort(unique(Int(r["nel"]) for r in e2a))
            for fs in ("iso", "aniso")
                sub = filter(r -> Int(r["nel"]) == nel && r["fs"] == fs, e2a)
                isempty(sub) && continue
                perm = sortperm([Int(r["ni"]) for r in sub])
                sub = sub[perm]
                push!(series, (label="n=$(nel) $(fs)",
                    x=[Int(r["ni"]) for r in sub],
                    y=[float(r["err_int"]) for r in sub]))
            end
        end
        write_conv_typ(joinpath(FIGDIR, "conv-ex2a.typ");
            title="Example 2A", xlabel="\$N_\"int\"\$", ylabel="avg. rel. error",
            series=series)
    end
    e2b = runs_of(rows, "2B")
    if !isempty(e2b)
        series = []
        for nel in sort(unique(Int(r["nel"]) for r in e2b))
            for fs in ("iso", "aniso")
                sub = filter(r -> Int(r["nel"]) == nel && r["fs"] == fs, e2b)
                isempty(sub) && continue
                perm = sortperm([float(r["k1"]) for r in sub])
                sub = sub[perm]
                push!(series, (label="n=$(nel) $(fs)",
                    x=[float(r["k1"]) for r in sub],
                    y=[float(r["err_int"]) for r in sub]))
            end
        end
        write_conv_typ(joinpath(FIGDIR, "conv-ex2b.typ");
            title="Example 2B", xlabel="\$k_1\$", ylabel="avg. rel. error",
            series=series, logx=false)
    end
    e2c = runs_of(rows, "2C")
    if !isempty(e2c)
        series = []
        sub = sort(e2c; by=r -> Int(r["ni"]))
        push!(series, (label="iso DIBEM",
            x=[Int(r["ni"]) for r in sub],
            y=[float(r["err_int"]) for r in sub]))
        write_conv_typ(joinpath(FIGDIR, "conv-ex2c.typ");
            title="Example 2C", xlabel="\$N_\"int\"\$", ylabel="avg. rel. error",
            series=series)
    end
    e3 = runs_of(rows, "3")
    if !isempty(e3)
        series = []
        # finest lc, k2=0.5, vary k1
        lcs = sort(unique(float(r["lc"]) for r in e3))
        lc = lcs[1]  # finest (smallest) mesh
        for (k1, k2) in ((1.0, 0.5), (3.0, 0.5), (5.0, 0.5))
            for fs in ("iso", "aniso")
                sub = filter(r -> abs(float(r["k1"]) - k1) < 1e-12 &&
                                  abs(float(r["k2"]) - k2) < 1e-12 &&
                                  r["fs"] == fs &&
                                  abs(float(r["lc"]) - lc) < 1e-12, e3)
                isempty(sub) && continue
                r = sub[1]
                haskey(r, "y_right") || continue
                push!(series, (label="k1=$(k1) $(fs)",
                    x=float.(r["y_right"]), y=float.(r["T_right"])))
            end
        end
        isempty(series) || write_xy_typ(joinpath(FIGDIR, "ex3-right.typ");
            title="Example 3 right edge", xlabel="y", ylabel="T(x=1)", series=series)
    end
    println("wrote conv typst under ", FIGDIR)
end

function parse_tasks(args)
    args = lowercase.(String.(args))
    isempty(args) && return [:all]
    tasks = Symbol[]
    for a in args
        a in ("quick", "all", "ex1", "ex2a", "ex2b", "ex2c", "ex2d", "ex3",
            "check", "large") && push!(tasks, Symbol(a))
    end
    isempty(tasks) && push!(tasks, :all)
    return tasks
end

function main(args=ARGS)
    tasks = parse_tasks(args)
    println("orthotropic DIBEM examples  tasks=", tasks)
    println("out = ", OUTDIR)
    check_analyticals()
    :check in tasks && :all ∉ tasks && :quick ∉ tasks && return
    quick = :quick in tasks
    all = :all in tasks || quick
    large = :large in tasks
    bundle = Dict{String,Any}(
        "created" => string(now()),
        "rbf" => "PHS3 poly_deg=1 (2C: poly_deg=2)",
        "elements" => "linear discontinuous (Gauss–Legendre collocation)",
        "error" => "mean|num-ana| / max|ana|",
        "fs" => Dict(
            "iso" => "Laplace FS + DIBEM residual",
            "aniso" => "AnisotropicLaplace FS (homogeneous BEM)",
        ),
        "runs" => Dict{String,Any}[],
    )
    function take(tag, thunk)
        try
            rows = thunk()
            append!(bundle["runs"], rows)
            save_json(joinpath(RESULTDIR, "example_$(tag).json"),
                Dict("created" => bundle["created"], "runs" => rows))
        catch e
            @error "example $tag failed" exception=(e, catch_backtrace())
        end
    end
    if all || :ex1 in tasks
        nels = quick ? [80] : [80, 160, 320]
        ni = quick ? [57] : (large ? [16, 21, 57, 161, 584, 1000, 2500, 5000, 10000] :
                             [16, 21, 57, 161, 584])
        take("1", () -> run_example1(; nels=nels, ni_targets=ni, store_every=!quick))
    end
    if all || :ex2a in tasks
        nels = quick ? [160] : [160, 320]
        ni = quick ? [57] : [57, 161, 332, 584]
        take("2a", () -> run_example2a(; nels=nels, ni_targets=ni))
    end
    if all || :ex2b in tasks
        pairs = quick ? [(160, 161)] : [(160, 161), (320, 332)]
        k1s = quick ? (1.0:2.0:5.0) : (1.0:1.0:5.0)
        take("2b", () -> run_example2b(; pairs=pairs, k1s=k1s))
    end
    if all || :ex2c in tasks
        nels = quick ? [160] : [160, 320]
        ni = quick ? [57] : [57, 161, 332, 584]
        take("2c", () -> run_example2c(; nels=nels, ni_targets=ni))
    end
    if all || :ex2d in tasks
        take("2d", () -> run_example2d(; nel=quick ? 80 : 160, ni_target=quick ? 57 : 161))
    end
    if all || :ex3 in tasks
        lcs = quick ? [0.08] : [0.08, 0.05]
        kpairs = quick ? [(1.0, 0.5), (5.0, 0.5)] :
                 vcat([(k, 0.5) for k in (1.0, 3.0, 5.0)],
            [(0.5, k) for k in 1.0:1.0:5.0])
        take("3", () -> run_example3(; lcs=lcs, kpairs=kpairs))
    end
    save_json(joinpath(RESULTDIR, "all.json"), bundle)
    write_all_figures(bundle["runs"])
    println("\ndone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
