#!/usr/bin/env julia
# =============================================================================
# ThinPlate vs Laplace assembly profile
#   julia --project=. scripts/profile/profile_thinplate.jl
#   julia --project=. scripts/profile/profile_thinplate.jl --nel=10
# =============================================================================

using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))

using BEM
using BEM.Plate
using LinearAlgebra
using StaticArrays
using FastGaussQuadrature
using Printf
using Statistics
using Profile
using InteractiveUtils: code_warntype

const TP = BEM.Plate.ThinPlate

const PROJECT = dirname(dirname(@__DIR__))
include(joinpath(PROJECT, "data", "Laplace", "Laplace_dad.jl"))

const Point2D = SVector{2,Float64}

function _parse_nel(args)
    nel = 8
    for a in args
        startswith(a, "--nel=") && (nel = parse(Int, split(a, "="; limit=2)[2]))
    end
    return nel
end

bytes(x) = Base.summarysize(x)

function fmt_bytes(b)
    b < 1024 && return @sprintf("%d B", b)
    b < 1024^2 && return @sprintf("%.2f KiB", b / 1024)
    b < 1024^3 && return @sprintf("%.2f MiB", b / 1024^2)
    return @sprintf("%.2f GiB", b / 1024^3)
end

function section(title)
    println()
    println("="^72)
    println(title)
    println("="^72)
end

function timed(label, f; n::Int=1)
    f()
    GC.gc(false)
    t0 = time_ns()
    b0 = Base.gc_bytes()
    r = nothing
    for _ in 1:n
        r = f()
    end
    t1 = time_ns()
    b1 = Base.gc_bytes()
    dt = (t1 - t0) / 1e9 / n
    db = (b1 - b0) / n
    @printf("  %-36s  %8.3f s   alloc %s\n", label, dt, fmt_bytes(db))
    return r, dt, db
end

function warntype_summary(label, f, args...)
    buf = IOBuffer()
    code_warntype(buf, f, typeof.(args))
    txt = String(take!(buf))
    n_any = length(collect(eachmatch(r"\bAny\b", txt)))
    n_union = length(collect(eachmatch(r"\bUnion\{", txt)))
    n_box = length(collect(eachmatch(r"Core\.Box", txt)))
    n_red = length(collect(eachmatch(r"::Any", txt)))
    println("  $label")
    println("    Any≈$n_any  ::Any≈$n_red  Union{≈$n_union  Core.Box≈$n_box")
    return (; n_any, n_union, n_box, n_red)
end

function quiet(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f()
        end
    end
end

function plate_mesh(; n_el=8, p=2, n_internal=1, q_c=1.0)
    props = ThinPlateProps(; E=1e5, ν=0.3, h=0.01, q_c=q_c)
    return build_square_plate(; a=1.0, n_el=n_el, bc="SSSS", props=props,
        corner_bc='F', n_internal=n_internal, p=p)
end

function laplace_dad(; ndiv=9, ordem=2)
    msh = quadrado(ndiv=ndiv, ordem=ordem, show=false, nome="prof_tp_lap_$(ndiv)_o$(ordem)")
    return format2d(msh, Laplace(1.0); pontointerno=false)
end

function kernel_microbench()
    section("1. Kernel micro-benchmarks (1e5 evals)")
    pg = Point2D(0.41, 0.27)
    pf = Point2D(0.02, 0.01)
    n = Point2D(0.0, 1.0)
    nf = Point2D(1.0, 0.0)
    props = ThinPlateProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0)
    props_abs = mesh_props_abs(props)
    lap = Laplace(1.0)
    r = pg - pf

    N = 100_000
    plate_kernels(pg, pf, n, nf, props)
    fundamental(lap, r, n)
    GC.gc(false)

    t0 = time_ns(); for _ in 1:N; plate_kernels(pg, pf, n, nf, props); end
    t_iso = (time_ns() - t0) / 1e9
    a_iso = @allocated plate_kernels(pg, pf, n, nf, props)

    t0 = time_ns(); for _ in 1:N; plate_kernels(pg, pf, n, nf, props_abs); end
    t_abs = (time_ns() - t0) / 1e9
    a_abs = @allocated plate_kernels(pg, pf, n, nf, props_abs)

    t0 = time_ns(); for _ in 1:N; fundamental(lap, r, n); end
    t_lap = (time_ns() - t0) / 1e9
    a_lap = @allocated fundamental(lap, r, n)

    # anisotropic (Useche-like, non-repeated μ)
    Ex, Ey, νxy, Gxy, h = 2.068e11, 2.068e11 / 15, 0.3, 6.055e8, 0.01
    den = 1 - νxy^2 * Ey / Ex
    aniso = aniso_thin_plate_props(;
        D11=Ex * h^3 / (12 * den), D22=Ey * h^3 / (12 * den),
        D12=νxy * Ey * h^3 / (12 * den), D66=Gxy * h^3 / 12)
    plate_kernels(pg, pf, n, nf, aniso)
    t0 = time_ns(); for _ in 1:N; plate_kernels(pg, pf, n, nf, aniso); end
    t_an = (time_ns() - t0) / 1e9
    a_an = @allocated plate_kernels(pg, pf, n, nf, aniso)

    el = plate_mesh(; n_el=2, n_internal=0).elements[1]
    TP.elem_geom(el, 0.1)
    t0 = time_ns(); for _ in 1:N; TP.elem_geom(el, 0.1); end
    t_geom = (time_ns() - t0) / 1e9
    a_geom = @allocated TP.elem_geom(el, 0.1)

    poly = BEM.Legendre(2)
    BEM.shapefun(poly, 0.1)
    t0 = time_ns(); for _ in 1:N; BEM.shapefun(poly, 0.1); end
    t_sf = (time_ns() - t0) / 1e9
    a_sf = @allocated BEM.shapefun(poly, 0.1)

    @printf("  %-28s  %8.2f ns/call   alloc %d B\n", "plate_kernels iso typed", 1e9 * t_iso / N, a_iso)
    @printf("  %-28s  %8.2f ns/call   alloc %d B\n", "plate_kernels via Abstract", 1e9 * t_abs / N, a_abs)
    @printf("  %-28s  %8.2f ns/call   alloc %d B\n", "plate_kernels aniso", 1e9 * t_an / N, a_an)
    @printf("  %-28s  %8.2f ns/call   alloc %d B\n", "fundamental Laplace", 1e9 * t_lap / N, a_lap)
    @printf("  %-28s  %8.2f ns/call   alloc %d B\n", "elem_geom", 1e9 * t_geom / N, a_geom)
    @printf("  %-28s  %8.2f ns/call   alloc %d B\n", "shapefun(Legendre(2), ξ)", 1e9 * t_sf / N, a_sf)
    println()
    println("  kernel cost ratio iso/Laplace = ", round(t_iso / t_lap; digits=2))
    println("  kernel cost ratio aniso/iso   = ", round(t_an / t_iso; digits=2))
    println("  abstract-props slowdown       = ", round(t_abs / t_iso; digits=2),
        "  (mesh.props is AbstractThinPlateProps)")
end

# Force abstract dispatch the way assemble_plate! sees mesh.props
mesh_props_abs(p::TP.AbstractThinPlateProps) = p

function hot_allocs()
    section("2. Hot-path allocations (single call)")
    mesh = plate_mesh(; n_el=4, n_internal=1)
    el = mesh.elements[1]
    pf = mesh.nodes[1]
    nf = mesh.Normal[1]
    poly = mesh.element_type
    qsi, w = gausslegendre(12)
    qsi_f, w_f = gausslegendre(20)
    h_el = zeros(2, 6)
    g_el = zeros(2, 6)
    TP._plate_singular_guiggiani!(h_el, g_el, el, poly, pf, nf, poly.nodes[1],
        mesh.props; qsi=qsi_f, w=w_f)
    a_g = @allocated TP._plate_singular_guiggiani!(h_el, g_el, el, poly, pf, nf,
        poly.nodes[1], mesh.props; qsi=qsi_f, w=w_f)
    TP.integraelemsing(el.geo[1], el.geo[end], mesh.props, poly.nodes[1], poly)
    a_a = @allocated TP.integraelemsing(el.geo[1], el.geo[end], mesh.props,
        poly.nodes[1], poly)
    TP.compute_q_el(pf, nf, mesh, el, qsi, w)
    a_q = @allocated TP.compute_q_el(pf, nf, mesh, el, qsi, w)
    TP.compute_Rw(pf, nf, mesh.corners, mesh.props)
    a_rw = @allocated TP.compute_Rw(pf, nf, mesh.corners, mesh.props)
    TP._poly_moments(poly, 0.0)
    a_m = @allocated TP._poly_moments(poly, 0.0)

    println("  _plate_singular_guiggiani!  ", fmt_bytes(a_g))
    println("  integraelemsing (analytic)  ", fmt_bytes(a_a))
    println("  compute_q_el                ", fmt_bytes(a_q))
    println("  compute_Rw                  ", fmt_bytes(a_rw))
    println("  _poly_moments               ", fmt_bytes(a_m))
    println("  PlateMesh.props type        ", typeof(mesh.props),
        "  declared=", fieldtype(typeof(mesh), :props))
    println("  element_type field          ", typeof(mesh.element_type),
        "  declared=", fieldtype(typeof(mesh), :element_type))
end

function compare_assembly(nel)
    section("3. Assembly: ThinPlate vs Laplace  (n_el/edge=$nel, p=2, npg=12)")
    mesh = plate_mesh(; n_el=nel, p=2, n_internal=1)
    n = length(mesh.nodes)
    ni = length(mesh.internal)
    nc = length(mesh.corners)
    ndof = 2n + ni + nc
    println("  Plate:  elements=$(length(mesh.elements))  nodes=$n  ndof=$ndof  (2n+ni+nc)")
    println("  PlateMesh size = ", fmt_bytes(bytes(mesh)))

    quiet() do
        assemble_plate!(plate_mesh(; n_el=nel); npg=12, singular=:guiggiani)
    end

    mesh_g = plate_mesh(; n_el=nel)
    (_, t_g, a_g) = timed("assemble_plate! :guiggiani", () ->
        quiet(() -> assemble_plate!(mesh_g; npg=12, singular=:guiggiani)))

    mesh_a = plate_mesh(; n_el=nel)
    (_, t_a, a_a) = timed("assemble_plate! :analytic", () ->
        quiet(() -> assemble_plate!(mesh_a; npg=12, singular=:analytic)))

    mesh_q0 = plate_mesh(; n_el=nel, q_c=0.0)
    (_, t_q0, _) = timed("assemble_plate! q_c=0 analytic", () ->
        quiet(() -> assemble_plate!(mesh_q0; npg=12, singular=:analytic)))

    (_, t_bc, a_bc) = timed("apply_bc_plate", () -> TP.apply_bc_plate(mesh_g))
    (_, t_sol, a_sol) = timed("solve_plate!", () -> solve_plate!(mesh_g))

    # Laplace: match element count. Gmsh transfinite n nodes → n-1 segments.
    ndiv = nel + 1
    dad = laplace_dad(; ndiv=ndiv, ordem=2)
    println()
    println("  Laplace: elements=$(length(dad.elements))  n=$(dad.n)  nt=$(dad.nt)")
    println("  BEMdata size = ", fmt_bytes(bytes(dad)))
    H_G_full_direct(deepcopy(dad); npg=12, threaded=false)

    dad1 = deepcopy(dad)
    (_, t_ld, a_ld) = timed("H_G_full_direct serial", () ->
        H_G_full_direct(dad1; npg=12, threaded=false))
    dad2 = deepcopy(dad)
    (_, t_lt, a_lt) = timed("H_G_full_direct threaded", () ->
        H_G_full_direct(dad2; npg=12, threaded=true))
    dad3 = deepcopy(dad)
    (_, t_lnf, _) = timed("H_G_full_direct near=Inf", () ->
        H_G_full_direct(dad3; npg=12, threaded=false, near_factor=Inf))
    timed("applyBC + solve Laplace", () -> (applyBC(dad1); solve(dad1)))

    println()
    @printf("  plate t / Laplace serial t     = %.2fx\n", t_g / t_ld)
    @printf("  plate analytic / Laplace       = %.2fx\n", t_a / t_ld)
    @printf("  plate ndof² / Laplace n²       = %.2f  (matrix work ratio)\n",
        ndof^2 / dad.n^2)
    @printf("  plate t / N_src / n_el         = %.2f μs  (src=nodes, all elems quad)\n",
        1e6 * t_g / n / length(mesh.elements))
    @printf("  Laplace t / n / n_el           = %.2f μs\n",
        1e6 * t_ld / dad.n / length(dad.elements))
    @printf("  plate H+G memory               = %s  (%d×%d × 2)\n",
        fmt_bytes(bytes(mesh_g.H) + bytes(mesh_g.G)), ndof, ndof)
    @printf("  Laplace H+G memory             = %s  (H %s, G %s)\n",
        fmt_bytes(bytes(dad1.H) + bytes(dad1.G)), size(dad1.H), size(dad1.G))
    @printf("  q_c=0 vs q_c=1 (analytic)      = %.0f ms extra for particular integral\n",
        1e3 * (t_a - t_q0))
    @printf("  guiggiani vs analytic          = %.2fx slower\n", t_g / t_a)
    @printf("  Laplace far-lumping speedup    = %.2fx  (near=1.5 vs Inf)\n", t_lnf / t_ld)
    @printf("  Laplace thread speedup         = %.2fx  (nthreads=%d)\n",
        t_ld / max(t_lt, 1e-12), Threads.nthreads())
    println("  apply_bc_plate copies H and G: ", fmt_bytes(a_bc))
    return (; mesh_g, dad1, t_g, t_a, t_ld, ndof, n)
end

function scaling()
    section("4. Scaling scan (analytic singular, npg=12, p=2)")
    println("  n_el   Nsrc   ndof    t[s]    alloc     t/ndof² [ns]   t/Nsrc/nel [μs]")
    for nel in (4, 6, 8, 12)
        mesh = plate_mesh(; n_el=nel)
        quiet(() -> assemble_plate!(plate_mesh(; n_el=nel); npg=12, singular=:analytic))
        GC.gc(false)
        t0 = time_ns()
        b0 = Base.gc_bytes()
        quiet(() -> assemble_plate!(mesh; npg=12, singular=:analytic))
        dt = (time_ns() - t0) / 1e9
        db = Base.gc_bytes() - b0
        n = length(mesh.nodes)
        ndof = size(mesh.H, 1)
        nel_tot = length(mesh.elements)
        @printf("  %4d  %5d  %5d  %7.3f  %8s  %12.1f  %14.2f\n",
            nel, n, ndof, dt, fmt_bytes(db), 1e9 * dt / ndof^2,
            1e6 * dt / n / nel_tot)
    end
    println()
    println("  Laplace dense (ordem=2, npg=12, serial, default near=1.5)")
    println("  ndiv    n     nel     t[s]    alloc     t/n² [ns]      t/n/nel [μs]")
    for nd in (5, 7, 9, 13)
        dad = laplace_dad(; ndiv=nd, ordem=2)
        H_G_full_direct(deepcopy(dad); npg=12, threaded=false)
        d = deepcopy(dad)
        GC.gc(false)
        t0 = time_ns()
        b0 = Base.gc_bytes()
        H_G_full_direct(d; npg=12, threaded=false)
        dt = (time_ns() - t0) / 1e9
        db = Base.gc_bytes() - b0
        @printf("  %4d  %5d  %5d  %7.3f  %8s  %12.1f  %14.2f\n",
            nd, d.n, length(d.elements), dt, fmt_bytes(db),
            1e9 * dt / d.n^2, 1e6 * dt / d.n / length(d.elements))
    end
end

function type_stability()
    section("5. Type stability")
    mesh = plate_mesh(; n_el=2, n_internal=0)
    pg, pf = mesh.nodes[2], mesh.nodes[1]
    n, nf = mesh.Normal[2], mesh.Normal[1]
    props = mesh.props
    props_t = ThinPlateProps(; E=1e5, ν=0.3, h=0.01)
    lap = Laplace(1.0)
    r = pg - pf
    el = mesh.elements[1]

    warntype_summary("plate_kernels(::ThinPlateProps)", plate_kernels, pg, pf, n, nf, props_t)
    warntype_summary("plate_kernels(mesh.props Abstract)", plate_kernels, pg, pf, n, nf, props)
    warntype_summary("fundamental(Laplace)", fundamental, lap, r, n)
    warntype_summary("elem_geom", TP.elem_geom, el, 0.1)
    warntype_summary("shapefun", BEM.shapefun, mesh.element_type, 0.1)
end

function cpu_profile(nel)
    section("6. CPU sample profile  assemble_plate! :guiggiani  n_el=$nel")
    mesh = plate_mesh(; n_el=nel)
    quiet(() -> assemble_plate!(plate_mesh(; n_el=nel); npg=12, singular=:guiggiani))
    Profile.init(n=10^8, delay=0.001)
    Profile.clear()
    @profile quiet(() -> assemble_plate!(mesh; npg=12, singular=:guiggiani))
    buf = IOBuffer()
    Profile.print(buf; format=:flat, sortedby=:count, mincount=8, C=false, maxdepth=20)
    txt = String(take!(buf))
    lines = split(txt, '\n')
    # keep the header + top ~40 interesting frames (skip empty)
    keep = String[]
    for ln in lines
        isempty(strip(ln)) && continue
        push!(keep, ln)
        length(keep) >= 45 && break
    end
    println(join(keep, '\n'))
end

function struct_dump()
    section("7. Structure comparison (what Laplace has that plates do not)")
    println("""
  Laplace (BEMdata + Assembly_full)
  ---------------------------------
  • H_G_full_direct: row-wise Threads.@threads, thread-local h/g buffers
  • far nodal lumping when r > 1.5 L  (skips Gauss on most pairs)
  • nearly-singular sinh via transform / closest_point_1d
  • on-element Guiggiani fused G+H (guiggiani_GH, one sample pass)
  • shapefun(poly, eta_vector) once per element (all qp together)
  • H-matrix (H_G_Hmat), GPU far field, method=:dense|:hmatrix|:gpu
  • typed Problem (Laplace) on BEMdata{<:Laplace} — specialized kernels
  • applyBC in-place on cache; solve via LinearSolve

  ThinPlate (PlateMesh + assemble_plate!)
  ---------------------------------------
  • serial @showprogress loops; no threaded= kwarg
  • always full Gauss on every source–element pair (no far lumping)
  • no sinh / closest-point nearly-singular map
  • on-element: 5 separate guiggiani_integral calls (G + 4 H entries),
    each allocating a closure + zeros; regular Gauss is run then overwritten
  • shapefun + elem_geom at every scalar ξ
  • no H-matrix / GPU path; cannot call assemble!(dad; method=...)
  • PlateMesh.props::AbstractThinPlateProps  (runtime dispatch)
  • PlateMesh.element_type::Any
  • nested assemble_w_row! closure over H,G,qsi,...
  • compute_q_el even when q≡0; I(2) allocated in the free-term loop
  • apply_bc_plate copies H and G
  • 2 DOFs/node + corners + internals  (Kirchhoff physics, not a bug)
""")
end

function main(args=ARGS)
    nel = _parse_nel(args)
    println("ThinPlate vs Laplace profile  n_el=$nel  threads=$(Threads.nthreads())")
    println("Julia ", VERSION)
    kernel_microbench()
    hot_allocs()
    compare_assembly(nel)
    scaling()
    type_stability()
    cpu_profile(nel)
    struct_dump()
    section("8. Done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
