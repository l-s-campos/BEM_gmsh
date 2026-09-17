# Compare RBF-FD heat (RadialBasisFunctions.jl / Macchiato stack) against
# BEM-SBM / BEM-DIBEM / Kansa-BEM on Kovářík heat Ex1–3.
#
# RBF-FD path builds a sparse Laplacian on a WhatsThePoint cloud, eliminates
# Dirichlet DOFs, and integrates ∂t u = κ ∇²u with FBDF (same stack Macchiato
# uses under the hood, without Macchiato's soft du=g-u Dirichlet pinning).
#
# julia --project=scripts/meshless scripts/meshless/macchiato_heat_compare.jl
using DrWatson
@quickactivate :BEM

using LinearAlgebra
using SparseArrays
using StaticArrays
using Printf
using Statistics
using Unitful: m, °, ustrip


import WhatsThePoint as WTP
using WhatsThePoint:
    PointBoundary, split_surface!, discretize, ConstantSpacing,
    FornbergFlyer, points, coords, names

using OrdinaryDiffEq: FBDF, ODEProblem, solve
include(joinpath(@__DIR__, "..", "sbm_drm", "sbm_transient_study.jl"))

# Import RBF-FD after BEM so package PHS does not shadow BEM.PHS
import RadialBasisFunctions as RBF
using RadialBasisFunctions: laplacian, weights

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

function rectangle_pts(Lx, Ly; n::Int=50)
    dx, dy = Lx / n, Ly / n
    rx, ry = (dx:dx:(Lx - dx)), (dy:dy:(Ly - dy))
    pts = vcat(
        [WTP.Point(x, zero(Ly)) for x in rx],
        [WTP.Point(Lx, y) for y in ry],
        [WTP.Point(x, Ly) for x in reverse(rx)],
        [WTP.Point(zero(Lx), y) for y in reverse(ry)],
    )
    nrms = vcat(
        fill(WTP.Vec(0.0, -1.0), length(rx)),
        fill(WTP.Vec(1.0, 0.0), length(ry)),
        fill(WTP.Vec(0.0, 1.0), length(rx)),
        fill(WTP.Vec(-1.0, 0.0), length(ry)),
    )
    areas = fill(dx, length(pts))
    return pts, nrms, areas
end

function surface_side(cloud, name::Symbol; Lx=Lx0, Ly=Ly0)
    pts = points(cloud[name])
    xs = [float(ustrip(coords(p).x)) for p in pts]
    ys = [float(ustrip(coords(p).y)) for p in pts]
    cx, cy = mean(xs), mean(ys)
    tol = 1e-6 * max(Lx, Ly)
    cy < tol && return :bottom
    cy > Ly - tol && return :top
    cx < tol && return :left
    cx > Lx - tol && return :right
    error("surface $name centroid ($cx,$cy) not on boundary")
end

function make_cloud(; n_edge=32, Lx=Lx0, Ly=Ly0)
    part = PointBoundary(rectangle_pts(Lx * m, Ly * m; n=n_edge)...)
    split_surface!(part, 75°)
    dx = (Lx / n_edge) * m
    return discretize(part, ConstantSpacing(dx); alg=FornbergFlyer())
end

function cloud_xy(cloud)
    pts = points(cloud)
    return [SVector(float(ustrip(coords(p).x)), float(ustrip(coords(p).y))) for p in pts]
end

"""Return (xy, side_per_node) with side ∈ {:interior,:bottom,:right,:top,:left}."""
function classify_nodes(cloud; Lx=Lx0, Ly=Ly0)
    xy = cloud_xy(cloud)
    n = length(xy)
    side = fill(:interior, n)
    # boundary blocks come first in Macchiato/WhatsThePoint ordering
    offset = 0
    for name in names(cloud.boundary)
        N = length(cloud[name])
        s = surface_side(cloud, name; Lx=Lx, Ly=Ly)
        for i in 1:N
            side[offset + i] = s
        end
        offset += N
    end
    return xy, side
end

function exact_on_xy(ex, xy, t; κ=κ0)
    if ex == 1
        return [exact_ex1(p[1], p[2], t; κ=κ) for p in xy]
    elseif ex == 2
        return [exact_ex2(p[1], p[2], t; κ=κ) for p in xy]
    else
        return fill(NaN, length(xy))
    end
end

function ic_from_sides(ex, side, u0=u_init)
    map(side) do s
        if s === :interior
            float(u0)
        elseif s === :left
            ex == 1 ? 0.0 : float(u0)
        else
            0.0   # bottom/right/top Dirichlet 0
        end
    end
end

# ---------------------------------------------------------------------------
# RBF-FD heat (hard Dirichlet elimination)
# ---------------------------------------------------------------------------

"""Nearest nodes with x > x_i, sorted by distance (for left-side one-sided BC)."""
function rightward_neighbors(i, xy, side; nmax=8)
    xi, yi = xy[i]
    cands = Tuple{Float64,Int}[]
    @inbounds for j in eachindex(xy)
        j == i && continue
        side[j] === :left && continue
        dx = xy[j][1] - xi
        dx <= 0 && continue
        d2 = dx^2 + (xy[j][2] - yi)^2
        push!(cands, (d2, j))
    end
    isempty(cands) && error("no rightward neighbor for node $i")
    sort!(cands; by=first)
    return first(cands, min(nmax, length(cands)))
end

"""Zero-slope Neumann: u_i = wa · u_nbrs (even poly in dx)."""
function neumann_extrapolate_weights(i, xy, side; nmax=12)
    nbrs = rightward_neighbors(i, xy, side; nmax=nmax)
    js = [j for (_, j) in nbrs]
    m = length(js)
    use_quad = m >= 6
    nb = use_quad ? 4 : 2
    A = zeros(m, nb)
    xi, yi = xy[i]
    @inbounds for (row, j) in enumerate(js)
        dx = xy[j][1] - xi
        dy = xy[j][2] - yi
        A[row, 1] = 1
        A[row, 2] = dy
        if use_quad
            A[row, 3] = dx^2
            A[row, 4] = dx * dy
        end
    end
    wt = A \ I(m)
    return js, wt[1, :]
end

"""Robin row: paper ∂n u = (H/k)(u − u_f), n = −e_x ⇒ H a + k b = H T∞."""
function robin_left_row(i, xy, side; h=10.0, k_bc=1.0, nmax=12)
    nbrs = rightward_neighbors(i, xy, side; nmax=nmax)
    js = [i; [j for (_, j) in nbrs]]
    m = length(js)
    use_quad = m >= 8
    nb = use_quad ? 5 : 3
    A = zeros(m, nb)
    xi, yi = xy[i]
    @inbounds for (row, j) in enumerate(js)
        dx = xy[j][1] - xi
        dy = xy[j][2] - yi
        A[row, 1] = 1.0
        A[row, 2] = dx
        A[row, 3] = dy
        if use_quad
            A[row, 4] = dx^2
            A[row, 5] = dx * dy
        end
    end
    Wt = A \ I(m)
    wrow = h .* Wt[1, :] .+ k_bc .* Wt[2, :]
    return js, wrow
end

"""
RBF-FD heat: ∂t u = κ ∇²u on interior; hard Dirichlet elimination;
left Neumann/Robin via local polynomial ∂x.
"""
function solve_rbffd_heat(ex::Int; n_edge=32, Δt=0.01, tf=tf0, κ=κ0, u0=u_init,
                          stencil=25, basis=RBF.PHS(3; poly_deg=2))
    cloud = make_cloud(; n_edge=n_edge)
    xy, side = classify_nodes(cloud)
    n = length(xy)

    k = min(stencil, n - 1)
    lap = laplacian(xy; basis=basis, k=k)
    L = sparse(weights(lap))

    is_dir = falses(n)
    @inbounds for i in 1:n
        s = side[i]
        if s === :bottom || s === :right || s === :top
            is_dir[i] = true
        elseif s === :left && ex == 1
            is_dir[i] = true
        end
    end
    is_neu_left = [side[i] === :left && ex == 2 for i in 1:n]
    is_rob_left = [side[i] === :left && ex == 3 for i in 1:n]

    k_bc = 1.0
    h_rob, T∞ = 10.0, 50.0

    Lmod = spzeros(n, n)
    g = zeros(n)
    @inbounds for i in 1:n
        if is_neu_left[i]
            js, wa = neumann_extrapolate_weights(i, xy, side)
            Lmod[i, i] = 1.0
            for (α, j) in enumerate(js)
                Lmod[i, j] -= wa[α]
            end
        elseif is_rob_left[i]
            js, wrow = robin_left_row(i, xy, side; h=h_rob, k_bc=k_bc)
            for (α, j) in enumerate(js)
                Lmod[i, j] += wrow[α]
            end
            g[i] = h_rob * T∞
        end
    end

    free = findall(.!is_dir)
    nfree = length(free)
    # reduced: for free rows r: Lmod[r, free] u_free + Lmod[r, dir] g_dir = rhs_bc
    # time evolution only on truly dynamic nodes (interior + maybe left)
    # BC algebraic rows (neu/rob) → DAE. Simpler: treat neu/rob as algebraic via
    # mass matrix 0, or eliminate them too by solving BC at each step.
    #
    # Dynamic DOFs = interior only. Left Neumann/Robin values recovered from BC.
    dyn = findall(i -> side[i] === :interior, 1:n)
    nd = length(dyn)

    # At each stage, assemble effective operator on dyn:
    # For Neumann/Robin left nodes, solve local BC given current interior → u_left(u_int)
    # For Dirichlet, u_dir = 0.
    # Then (L u)_dyn = L_dd u_d + L_db u_b(u_d)
    #
    # Build affine map u_b = B u_d + c  for boundary non-dirichlet, and
    # fixed Dirichlet c_dir.

    # Collect boundary non-dirichlet (left neu/rob)
    bnd_free = findall(i -> is_neu_left[i] || is_rob_left[i], 1:n)
    # Dirichlet ids
    dir_ids = findall(is_dir)

    # For left BC nodes, Lmod rows are already the BC equations:
    # Lmod[b, :] u = g[b]
    # Split: Lbb u_b + Lbd u_d + Lb_dir u_dir = g_b
    # u_dir = 0 for our problems.
    if !isempty(bnd_free)
        Lbb = Matrix(Lmod[bnd_free, bnd_free])
        Lbd = Matrix(Lmod[bnd_free, dyn])
        gb = g[bnd_free]
        Fbb = factorize(Lbb)
        # u_b = Fbb \ (gb - Lbd u_d)
        Bmap = -(Fbb \ Lbd)          # n_b × n_d
        cmap = Fbb \ gb              # n_b
    else
        Bmap = zeros(0, nd)
        cmap = zeros(0)
    end

    # Dynamic residual: κ (L[dyn, dyn] u_d + L[dyn, bnd] u_b + L[dyn, dir] * 0)
    Ldd = Matrix(L[dyn, dyn])
    Ldb = isempty(bnd_free) ? zeros(nd, 0) : Matrix(L[dyn, bnd_free])
    # L_eff u_d + c_eff
    Leff = Ldd + Ldb * Bmap
    ceff = isempty(bnd_free) ? zeros(nd) : Ldb * cmap

    A = κ .* Leff
    c = κ .* ceff

    u0_full = ic_from_sides(ex, side, u0)
    u0d = u0_full[dyn]

    function f!(du, u, p, t)
        mul!(du, A, u)
        du .+= c
        return nothing
    end

    prob = ODEProblem(f!, u0d, (0.0, float(tf)))
    sol = solve(prob, FBDF(); dt=Δt, save_everystep=false, save_end=true,
                abstol=1e-8, reltol=1e-8)

    ud = sol.u[end]
    ufull = zeros(n)
    ufull[dyn] .= ud
    ufull[dir_ids] .= 0.0
    if !isempty(bnd_free)
        ufull[bnd_free] .= Bmap * ud .+ cmap
    end

    return (; t=[float(tf)], U=reshape(ufull, n, 1), xy, side, cloud,
            method=:rbffd, n=n, n_dyn=nd)
end

function center_value(xy, Ucol; Lx=Lx0, Ly=Ly0)
    ic = argmin(norm(p - SVector(Lx / 2, Ly / 2)) for p in xy)
    return Ucol[ic], ic
end

# ---------------------------------------------------------------------------
# BEM-family runners
# ---------------------------------------------------------------------------

function run_bem_family(ex, method; nb=16, nint=8, nsteps=60, scheme=:houbolt)
    return run_case(; ex=ex, method=method, scheme=scheme, nsteps=nsteps, nb=nb, nint=nint)
end

function run_kansa(ex; nb=16, nint=8, nsteps=60, scheme=:houbolt)
    dad = make_dad(ex; nb=nb, nint=nint)
    N, ni = dad.n, dad.ni
    u0v = fill(u_init, N + ni)
    @inbounds for i in 1:N
        dad.BC[i] == 0 && (u0v[i] = 0.0)
    end
    rH = ex == 3 ? 10.0 : nothing
    rUf = ex == 3 ? 50.0 : nothing
    t = @elapsed sol = solve_kansa_bem_heat(dad; κ=κ0, Δt=tf0 / nsteps, tf=tf0,
        u0=u0v, scheme=scheme, robin_H=rH, robin_uf=rUf)
    ok = all(isfinite, sol.U) && maximum(abs, sol.U) < 1e5
    rmse = NaN
    if ex <= 2 && ok
        rmse = sqrt(mean(abs2, sol.U[:, end] .- exact_vec(dad, ex, tf0)))
    end
    pts = Point2D[Point2D(p) for p in vcat(dad.Nodes, dad.internalNodes)]
    ic = argmin(norm(p - Point2D(Lx0 / 2, Ly0 / 2)) for p in pts)
    return (; rmse, maxu=maximum(abs, sol.U[:, end]), ctr=sol.U[ic, end], t, ok,
            method=:kansa_bem)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    println("="^72)
    println("RBF-FD (RadialBasisFunctions) vs BEM-SBM / DIBEM / Kansa-BEM")
    println("κ=$(κ0), domain=$(Lx0)×$(Ly0), tf=$(tf0), IC=$(u_init)")
    println("="^72)

    rows = NamedTuple[]

    for ex in (1, 2, 3)
        n_edge = 32
        Δt = 0.01
        println("\n[RBF-FD] ex$ex  n_edge=$n_edge Δt=$Δt")
        local sol
        t = @elapsed begin
            sol = try
                solve_rbffd_heat(ex; n_edge=n_edge, Δt=Δt, tf=tf0)
            catch e
                @error "RBF-FD failed" exception = (e, catch_backtrace())
                nothing
            end
        end
        if sol === nothing
            push!(rows, (; ex, method="rbffd", rmse=NaN, maxu=NaN, ctr=NaN,
                         time=t, ok=false, npts=0))
            continue
        end
        Tend = sol.U[:, end]
        ok = all(isfinite, Tend) && maximum(abs, Tend) < 1e5
        rmse = NaN
        if ex <= 2 && ok
            uex = exact_on_xy(ex, sol.xy, tf0)
            rmse = sqrt(mean(abs2, Tend .- uex))
        end
        ctr, _ = center_value(sol.xy, Tend)
        @printf("  n=%d dyn=%d  RMSE=%9.2e  max=%.3e  ctr=%.4f  t=%.2fs  %s\n",
            sol.n, sol.n_dyn, rmse, maximum(abs, Tend), ctr, t, ok ? "ok" : "FAIL")
        push!(rows, (; ex, method="rbffd", rmse, maxu=maximum(abs, Tend),
                     ctr, time=t, ok, npts=sol.n))
    end

    for ex in (1, 2, 3)
        for meth in (:sbm, :dibem)
            println("\n[BEM-$meth] ex$ex")
            r = try
                run_bem_family(ex, meth; nb=16, nint=8, nsteps=60)
            catch e
                @error "BEM $meth failed" exception = e
                nothing
            end
            r === nothing && continue
            @printf("  RMSE=%9.2e  max=%.3e  ctr=%.4f  ok=%s\n",
                r.rmse, r.maxu, r.ucenter, r.ok)
            push!(rows, (; ex, method=string(meth), rmse=r.rmse, maxu=r.maxu,
                         ctr=r.ucenter, time=r.t_cpu, ok=r.ok, npts=r.n + r.ni))
        end
        println("\n[Kansa-BEM] ex$ex")
        r = try
            run_kansa(ex; nb=16, nint=8, nsteps=60)
        catch e
            @error "Kansa failed" exception = e
            nothing
        end
        if r !== nothing
            @printf("  RMSE=%9.2e  max=%.3e  ctr=%.4f  t=%.2fs  %s\n",
                r.rmse, r.maxu, r.ctr, r.t, r.ok ? "ok" : "FAIL")
            push!(rows, (; ex, method="kansa_bem", rmse=r.rmse, maxu=r.maxu,
                         ctr=r.ctr, time=r.t, ok=r.ok, npts=0))
        end
    end

    println("\n" * "="^72)
    println("SUMMARY")
    @printf("%-4s %-14s %12s %12s %10s %8s\n", "ex", "method", "RMSE", "center", "max", "ok")
    for r in rows
        @printf("%-4d %-14s %12.3e %12.4f %10.3e %8s\n",
            r.ex, r.method, r.rmse, r.ctr, r.maxu, string(r.ok))
    end

    out = joinpath(STUDY_OUT, "macchiato_compare.csv")
    write_csv(out, rows)
    println("\nwrote $out")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
