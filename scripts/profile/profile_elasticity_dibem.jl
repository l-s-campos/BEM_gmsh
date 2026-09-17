#!/usr/bin/env julia
# =============================================================================
# Profile isotropic elasticity: H/G assembly + DIBEM mass + body-force solve
#   julia --project=. -t 8 scripts/profile/profile_elasticity_dibem.jl
#   julia --project=. -t 8 scripts/profile/profile_elasticity_dibem.jl --ndiv=16
# =============================================================================

using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))

using BEM
using LinearAlgebra
using StaticArrays
using Printf
using Profile
using InteractiveUtils: code_warntype

const PROJECT = dirname(dirname(@__DIR__))
include(joinpath(PROJECT, "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(PROJECT, "data", "Laplace", "cube_mesh.jl"))

function _parse_ndiv(args)
    ndiv = 12
    for a in args
        startswith(a, "--ndiv=") && (ndiv = parse(Int, split(a, "="; limit=2)[2]))
    end
    return ndiv
end

fmt_bytes(b) = begin
    b < 1024 && return @sprintf("%d B", b)
    b < 1024^2 && return @sprintf("%.2f KiB", b / 1024)
    b < 1024^3 && return @sprintf("%.2f MiB", b / 1024^2)
    return @sprintf("%.2f GiB", b / 1024^3)
end

function timed(label, f)
    f()
    GC.gc(false)
    t0 = time_ns()
    b0 = Base.gc_bytes()
    r = f()
    t1 = time_ns()
    b1 = Base.gc_bytes()
    dt = (t1 - t0) / 1e9
    db = Int(b1 - b0)
    @printf("  %-40s  %8.3f s   alloc %s\n", label, dt, fmt_bytes(db))
    return r, dt, db
end

section(title) = (println(); println("="^72); println(title); println("="^72))
_alloc(f) = (f(); GC.gc(false); @allocated f())

function elast2d(ndiv; pontointerno=true)
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = quadrado_elasticity(ndiv=ndiv, show=false, nome="prof_el2d_$ndiv")
    dad = format2d(msh, props; pontointerno=pontointerno)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
    apply_analytical_bc!(dad, ana)
    return dad
end

function elast3d(ndiv; internals=true)
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=ndiv, nome="prof_el3d_$ndiv", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    internals && set_internal_nodes!(dad, vec(cube_interior_grid(1.0, max(ndiv, 2))))
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01, dim=3)
    neumann = [i for i in 1:dad.n if abs(dad.Normal[i][1]) < 0.5]
    apply_analytical_bc!(dad, ana, neumann)
    return dad, props
end

function count_near_far(dad)
    n_near = 0
    n_far = 0
    @inbounds for i in 1:dad.nt
        x = point(dad, i)
        for el in dad.elements
            xj = dad.Nodes[el.index]
            if BEM._near_element(x, xj, el)
                n_near += 1
            else
                n_far += 1
            end
        end
    end
    return n_near, n_far
end

function elast_FD!(F, D, dad, rbf)
    n0 = dad.Normal[1]
    pts = all_points(dad)
    nt = dad.nt
    dim = dad.dimension
    @inbounds for j in 1:nt, i in 1:nt
        rvec = pts[j] - pts[i]
        R = norm(rvec)
        F[i, j] = rbf(R)
        R > 0 || continue
        U, _ = fundamental(dad, rvec, n0)
        D[BEM.expand(i, dim), BEM.expand(j, dim)] .= U
    end
    return F, D
end

function elast_FD_Uonly!(F, D, dad, rbf)
    pts = all_points(dad)
    nt = dad.nt
    dim = dad.dimension
    props = dad.properties
    n0 = dad.Normal[1]
    @inbounds for j in 1:nt
        xj = pts[j]
        for i in 1:nt
            rvec = xj - pts[i]
            R = norm(rvec)
            F[i, j] = rbf(R)
            R > 0 || continue
            U, _ = fundamental(props, rvec, n0)
            D[BEM.expand(i, dim), BEM.expand(j, dim)] .= BEM._to_smat(U)
        end
    end
    return F, D
end

function elast_FD_threaded!(F, D, dad, rbf)
    n0 = dad.Normal[1]
    pts = all_points(dad)
    nt = dad.nt
    dim = dad.dimension
    Threads.@threads for j in 1:nt
        xj = pts[j]
        @inbounds for i in 1:nt
            rvec = xj - pts[i]
            R = norm(rvec)
            F[i, j] = rbf(R)
            R > 0 || continue
            U, _ = fundamental(dad, rvec, n0)
            D[BEM.expand(i, dim), BEM.expand(j, dim)] .= U
        end
    end
    return F, D
end

"""Laplace-style near Gauss / far nodal RIM for elasticity ID (experiment)."""
function elast_IF_ID_lumped!(IF, ID, dad, rbf)
    geos = BEM._rim_build_elements(dad)
    dim = dad.dimension
    props = dad.properties
    fill!(IF, 0)
    fill!(ID, 0)
    @inbounds for i in 1:dad.nt
        x = point(dad, i)
        rows = BEM.expand(i, dim)
        accF = 0.0
        for g in geos
            if BEM._near_element(x, g.nodes, g.el)
                for q in eachindex(g.wJ)
                    wJ = g.wJ[q]
                    wJ == 0 && continue
                    y = g.y[q]
                    r = y - x
                    R = norm(r)
                    R < 1e-14 && continue
                    e = r / R
                    wJn = wJ * BEM._rim_factor(g.n[q], r, R, dim)
                    accF += BEM.int(rbf, x, y) * wJn
                    ID[rows, :] .+= BEM._galerkin_Ustar(props, R, e) * wJn
                end
            else
                for j in eachindex(g.xj)
                    xj = g.xj[j]
                    r = xj - x
                    R = norm(r)
                    R < 1e-10 && continue
                    e = r / R
                    wJn = g.wj[j] * BEM._rim_factor(g.nj[j], r, R, dim)
                    accF += BEM.int(rbf, x, xj) * wJn
                    ID[rows, :] .+= BEM._galerkin_Ustar(props, R, e) * wJn
                end
            end
        end
        IF[i] = accF
    end
    return IF, ID
end

function profile_dibem_breakdown(dad; rbf=PHS(3; poly_deg=1), npg=12)
    nt = dad.nt
    dim = dad.dimension
    ndof = dim * nt
    n_near, n_far = count_near_far(dad)
    println("  collocation nt = ", nt, "  boundary n = ", dad.n, "  ni = ", dad.ni)
    println("  elements       = ", length(dad.elements), "  dim = ", dim,
        "  ndof = ", ndof)
    println("  H/G pairs      = near $n_near  far $n_far  (",
        @sprintf("%.1f%% near", 100n_near / max(n_near + n_far, 1)), ")")
    println("  DIBEM RIM      = lumped near/far (rim=:full_gauss for comparison)")

    F = zeros(nt, nt)
    D = zeros(ndof, ndof)
    timed("F,D fill (serial, full Kelvin)", () -> elast_FD!(F, D, dad, rbf))
    fill!(F, 0); fill!(D, 0)
    timed("F,D fill (props U + _to_smat)", () -> elast_FD_Uonly!(F, D, dad, rbf))
    fill!(F, 0); fill!(D, 0)
    timed("F,D fill (threaded)", () -> elast_FD_threaded!(F, D, dad, rbf))

    timed("RIM IF,ID (lumped, threaded)", () -> BEM._dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=true, rim=:lumped))
    timed("RIM IF,ID (lumped, serial)", () -> BEM._dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=false, rim=:lumped))
    timed("RIM IF,ID (full Gauss, threaded)", () -> BEM._dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=true, rim=:full_gauss))
    IFl, IDl, _ = BEM._dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=true, rim=:lumped)
    IFg, IDg, _ = BEM._dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=true, rim=:full_gauss)
    @printf("  ‖ID_lumped − ID_gauss‖/‖ID‖ = %.3e\n",
        norm(IDl - IDg) / (norm(IDg) + 1e-14))

    timed("monomial IP (RIM)", () -> BEM._dibem_monomial_IP(dad, rbf))
    BEM._dibem_ridge_F!(F)
    IP = BEM._dibem_monomial_IP(dad, rbf)
    timed("CPD F\\IF (dense LU)", () -> BEM._dibem_poly_c(F, IFl, all_points(dad), rbf; IP=IP))
    c = BEM._dibem_poly_c(F, IFl, all_points(dad), rbf; IP=IP)

    timed("M = D diag(c) + block remainder", () -> begin
        M = zeros(ndof, ndof)
        @inbounds for j in 1:nt
            a = c[j]
            cols = BEM.expand(j, dim)
            for d in 1:dim
                M[:, cols[d]] .= a .* D[:, cols[d]]
            end
        end
        @inbounds for i in 1:nt
            rows = BEM.expand(i, dim)
            M[rows, rows] .= 0
            S = zeros(dim, dim)
            for d in 1:dim
                S[:, d] = vec(sum(view(M, rows, d:dim:ndof); dims=2))
            end
            M[rows, rows] .= .-S .+ IDg[rows, :]
        end
        M
    end)

    b = zeros(ndof)
    if dim == 2
        b[1:2:end] .= 1.0
    else
        b[1:3:end] .= 1.0
    end
    M = dad.M
    timed("M * b  (dense matvec)", () -> M * b)
    # constant body force along e_x: M (1⊗e_x) = ID[:,1] (block remainder)
    timed("ID * e_x (constant-b shortcut)", () -> IDg[:, 1])
    return nothing
end

function profile_pipeline_2d(ndiv)
    rbf = PHS(3; poly_deg=1)
    npg = 12
    section("2-D elasticity patch  ndiv=$ndiv  threads=$(Threads.nthreads())")
    dad = elast2d(ndiv; pontointerno=false)
    println("  n=$(dad.n)  ni=$(dad.ni)  nt=$(dad.nt)  nelem=$(length(dad.elements))",
        "  ndof=$(2 * dad.nt)")
    timed("assemble! H,G dense (threaded)", () -> assemble!(dad; npg=npg, threaded=true))
    timed("applyBC + dense \\  (patch)", () -> solve(dad))
    @printf("  rel error u = %.3e   traction = %.3e\n", rel_error(dad), rel_error_flux(dad))

    section("2-D elasticity + DIBEM body force  u=(x²,0)")
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    λ, μ = props.lambda, props.mu
    bval = -2 * (λ + 2μ)
    dadb = elast2d(ndiv; pontointerno=true)
    println("  n=$(dadb.n)  ni=$(dadb.ni)  nt=$(dadb.nt)  nelem=$(length(dadb.elements))",
        "  ndof=$(2 * dadb.nt)")
    timed("assemble! H,G dense", () -> assemble!(dadb; npg=npg, threaded=true))
    timed("DIBEM_dense (F,D,RIM,CPD,M)", () -> DIBEM(dadb; rbf=rbf, npg=npg, method=:dense))
    timed("solve_thermoelastic! bodyforce", () ->
        solve_thermoelastic!(dadb; bodyforce=(x, y) -> SVector(bval, 0.0), θ=0.0, npg_dibem=npg, rbf=rbf))

    section("2-D DIBEM kernel breakdown")
    dad2 = elast2d(ndiv; pontointerno=true)
    assemble!(dad2; npg=npg, threaded=true)
    DIBEM(dad2; rbf=rbf, npg=npg, method=:dense)  # for dad.M in matvec
    profile_dibem_breakdown(dad2; rbf=rbf, npg=npg)

    section("2-D compressed / GPU backends")
    dadh = elast2d(ndiv; pontointerno=true)
    assemble!(dadh; npg=npg, threaded=true)
    try
        timed("DIBEM method=:hmatrix", () -> DIBEM(dadh; rbf=rbf, method=:hmatrix, npg=npg,
            nmax=32, atol=1e-6, rtol=1e-6, threads=true))
    catch e
        println("  hmatrix skipped: ", e)
    end
    try
        dadg = elast2d(ndiv; pontointerno=true)
        assemble!(dadg; npg=npg, threaded=true)
        timed("DIBEM method=:gpu device=:cpu", () -> DIBEM(dadg; rbf=rbf, method=:gpu,
            device=:cpu, npg=npg, T=Float64, threaded=true))
    catch e
        println("  gpu-cpu skipped: ", e)
    end
    return dadb
end

function profile_pipeline_3d(ndiv)
    rbf = PHS(3; poly_deg=1)
    npg = 8
    section("3-D elasticity patch  ndiv=$ndiv")
    dad, props = elast3d(ndiv; internals=false)
    println("  n=$(dad.n)  ni=$(dad.ni)  nt=$(dad.nt)  nelem=$(length(dad.elements))",
        "  ndof=$(3 * dad.nt)")
    timed("assemble! H,G dense", () -> assemble!(dad; npg=npg, threaded=true))
    timed("applyBC + dense \\", () -> solve(dad))
    @printf("  rel error u = %.3e   traction = %.3e\n", rel_error(dad), rel_error_flux(dad))

    section("3-D elasticity DIBEM body force  u=(x²,0,0)")
    dadb, props = elast3d(ndiv; internals=true)
    λ, μ = props.lambda, props.mu
    bval = -2 * (λ + 2μ)
    println("  n=$(dadb.n)  ni=$(dadb.ni)  nt=$(dadb.nt)  nelem=$(length(dadb.elements))",
        "  ndof=$(3 * dadb.nt)")
    timed("assemble! H,G dense", () -> assemble!(dadb; npg=npg, threaded=true))
    timed("DIBEM_dense", () -> DIBEM(dadb; rbf=rbf, npg=npg, method=:dense))
    timed("solve_thermoelastic! (H,M cached)", () ->
        solve_thermoelastic!(dadb; bodyforce=p -> SVector(bval, 0.0, 0.0), θ=0.0, rbf=rbf))

    section("3-D DIBEM kernel breakdown")
    dad2, _ = elast3d(ndiv; internals=true)
    assemble!(dad2; npg=npg, threaded=true)
    DIBEM(dad2; rbf=rbf, npg=npg, method=:dense)
    profile_dibem_breakdown(dad2; rbf=rbf, npg=npg)
    return dadb
end

function micro_allocs(dad; rbf=PHS(3; poly_deg=1))
    section("Micro-allocations (hot kernels, post-compile)")
    BEM._init_quadrature!(dad, 12)
    el = dad.elements[1]
    nodes = dad.Nodes[el.index]
    x = point(dad, 1)
    props = dad.properties
    n0 = dad.Normal[1]
    rvec = point(dad, min(2, dad.nt)) - x
    R = norm(rvec)
    e = rvec / R
    dim = dad.dimension

    fundamental(dad, rvec, n0)
    fundamental(props, rvec, n0)
    BEM._galerkin_Ustar(props, R, e)
    BEM._near_element(x, nodes, el)
    dad.Nodes[el.index]

    println("  Kelvin fundamental(dad)     alloc = ", _alloc(() -> fundamental(dad, rvec, n0)), " B")
    println("  Kelvin fundamental(props)   alloc = ", _alloc(() -> fundamental(props, rvec, n0)), " B")
    U, T = fundamental(props, rvec, n0)
    println("  _to_smat(U)                 alloc = ", _alloc(() -> BEM._to_smat(U)), " B")
    println("  _galerkin_Ustar             alloc = ", _alloc(() -> BEM._galerkin_Ustar(props, R, e)), " B")
    println("  dad.Nodes[el.index]         alloc = ", _alloc(() -> dad.Nodes[el.index]), " B")
    println("  _near_element               alloc = ", _alloc(() -> BEM._near_element(x, nodes, el)), " B")
    println("  expand(i,dim)               alloc = ", _alloc(() -> BEM.expand(1, dim)), " B")
    nn = length(el)
    jj = BEM.expand(el.index, dim)
    println("  expand(el.index, dim)       alloc = ", _alloc(() -> BEM.expand(el.index, dim)), " B")
    println("  zeros hloc (dim × nnode*d)  alloc = ",
        _alloc(() -> zeros(Float64, dim, length(jj))), " B")
    if dad.dimension == 2
        N, dN = BEM.shapefun(dad.element_type, dad.qsi)
        println("  shapefun(poly, ηs)          alloc = ",
            _alloc(() -> BEM.shapefun(dad.element_type, dad.qsi)), " B")
    end
end

function warntype_hot(dad; rbf=PHS(3; poly_deg=1))
    section("Type stability (Any / Core.Box counts)")
    props = dad.properties
    n0 = dad.Normal[1]
    x = point(dad, 1)
    y = point(dad, min(2, dad.nt))
    function flags(label, f, args...)
        buf = IOBuffer()
        code_warntype(buf, f, typeof.(args))
        txt = String(take!(buf))
        n_any = length(collect(eachmatch(r"\bAny\b", txt)))
        n_box = length(collect(eachmatch(r"Core\.Box", txt)))
        println("    $label: Any≈$n_any  Core.Box≈$n_box")
    end
    flags("fundamental(props)", fundamental, props, y - x, n0)
    flags("fundamental(dad)", fundamental, dad, y - x, n0)
    flags("_galerkin_Ustar", BEM._galerkin_Ustar, props, 0.3, (y - x) / norm(y - x))
    flags("_dibem_elast_IF_ID", BEM._dibem_elast_IF_ID, dad, rbf)
end

function scaling_scan()
    section("Scaling  2-D elasticity (assemble + DIBEM_dense)")
    rbf = PHS(3; poly_deg=1)
    println("  ndiv    n   nt   ndof   t_HG[s]  t_DIBEM[s]  t_solve[s]  mem_HGM[MiB]")
    for nd in (6, 8, 12, 16)
        dad = elast2d(nd; pontointerno=true)
        assemble!(dad; npg=10, threaded=true)
        DIBEM(dad; rbf=rbf, npg=10, method=:dense)
        dad = elast2d(nd; pontointerno=true)
        GC.gc(false)
        t0 = time_ns(); assemble!(dad; npg=10, threaded=true); tHG = (time_ns() - t0) / 1e9
        t0 = time_ns(); DIBEM(dad; rbf=rbf, npg=10, method=:dense); tM = (time_ns() - t0) / 1e9
        t0 = time_ns(); solve(dad); tS = (time_ns() - t0) / 1e9
        mem = (Base.summarysize(dad.H) + Base.summarysize(dad.G) + Base.summarysize(dad.M)) / 1024^2
        @printf("  %4d %5d %5d %5d  %8.3f  %10.3f  %9.3f   %10.2f\n",
            nd, dad.n, dad.nt, 2 * dad.nt, tHG, tM, tS, mem)
    end
end

function dump_profile(dad; rbf=PHS(3; poly_deg=1), npg=12)
    section("CPU sample profile  (assemble! + DIBEM_dense)")
    dad2 = deepcopy(dad)
    has_cache(dad2, :H) && (dad2.cache.H = nothing)
    assemble!(dad2; npg=npg, threaded=true)
    DIBEM(dad2; rbf=rbf, npg=npg, method=:dense)

    dad3 = deepcopy(dad)
    has_cache(dad3, :H) && (dad3.cache.H = nothing)
    has_cache(dad3, :M) && (dad3.cache.M = nothing)
    Profile.clear()
    @profile begin
        assemble!(dad3; npg=npg, threaded=true)
        DIBEM(dad3; rbf=rbf, npg=npg, method=:dense)
        applyBC(dad3)
        BEM.bem_linsolve(dad3.A, dad3.b)
    end
    out = joinpath(@__DIR__, "elasticity_dibem_profile.txt")
    open(out, "w") do io
        println(io, "flat profile, sorted by count")
        Profile.print(io; format=:flat, sortedby=:count, mincount=8, C=false, maxdepth=60)
    end
    println("  wrote ", out)
    buf = IOBuffer()
    Profile.print(buf; format=:flat, sortedby=:count, mincount=15, C=false)
    lines = split(String(take!(buf)), '\n')
    println("  top samples:")
    for L in Iterators.take(lines, 22)
        println("    ", L)
    end
end

function main(args=ARGS)
    ndiv = _parse_ndiv(args)
    println("Elasticity + DIBEM profile  ndiv=$ndiv  threads=$(Threads.nthreads())  julia=$(VERSION)")
    dad = profile_pipeline_2d(ndiv)
    micro_allocs(dad)
    try
        warntype_hot(dad)
    catch e
        println("  warntype skipped: ", e)
    end
    try
        dump_profile(dad)
    catch e
        println("  CPU profile dump skipped: ", e)
    end
    scaling_scan()
    nd3 = ndiv <= 10 ? 2 : 3
    try
        profile_pipeline_3d(nd3)
    catch e
        println("  3-D pipeline skipped: ", e)
        showerror(stdout, e)
        println()
    end
    section("Done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
