#!/usr/bin/env julia
# =============================================================================
# Profile Laplace Poisson via DIBEM (assemble H/G + M + solve)
#   julia --project=. -t 8 scripts/profile/profile_poisson_dibem.jl
#   julia --project=. -t 8 scripts/profile/profile_poisson_dibem.jl --ndiv=24
# =============================================================================

using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))

using BEM
using LinearAlgebra
using StaticArrays
using Printf
using Statistics: mean
using Profile
using InteractiveUtils: code_warntype

const PROJECT = dirname(dirname(@__DIR__))
include(joinpath(PROJECT, "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(PROJECT, "data", "Laplace", "cube_mesh.jl"))

function _parse_ndiv(args)
    ndiv = 16
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
    @printf("  %-36s  %8.3f s   alloc %s\n", label, dt, fmt_bytes(db))
    return r, dt, db
end

section(title) = (println(); println("="^72); println(title); println("="^72))

function poisson2d(ndiv; pontointerno=true)
    msh = quadrado(ndiv=ndiv, show=false, nome="prof_po2d_$ndiv")
    dad = format2d(msh, Laplace(1.0); pontointerno=pontointerno)
    apply_analytical_bc!(dad, ana_poisson_r2(; k=1.0, dim=2))
    return dad
end

function poisson3d(ndiv)
    msh = mesh_unit_cube(; L=1.0, ndiv=ndiv, nome="prof_po3d_$ndiv")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, max(ndiv, 2))))
    apply_analytical_bc!(dad, ana_poisson_r2(; k=1.0, dim=3))
    return dad
end

# ---------------------------------------------------------------------------
# DIBEM internals (same math as DIBEM_dense, no ProgressMeter)
# ---------------------------------------------------------------------------

function dibem_FD!(F, D, dad, rbf)
    props = dad.properties
    n0 = dad.Normal[1]
    pts = all_points(dad)
    nt = dad.nt
    @inbounds for j in 1:nt, i in 1:nt
        rvec = pts[j] - pts[i]
        R = norm(rvec)
        if R > 0
            F[i, j] = rbf(R)
            D[i, j] = fundamental(props, rvec, n0).U
        end
    end
    return F, D
end

function dibem_FD_Uonly!(F, D, dad, rbf)
    props = dad.properties
    k = float(props.k)
    dim = dad.dimension
    pts = all_points(dad)
    nt = dad.nt
    inv2πk = 1 / (2π * k)
    inv4πk = 1 / (4π * k)
    @inbounds for j in 1:nt
        xj = pts[j]
        for i in 1:nt
            rvec = xj - pts[i]
            R = norm(rvec)
            if R > 0
                F[i, j] = rbf(R)
                D[i, j] = dim == 2 ? -log(R) * inv2πk : inv4πk / R
            end
        end
    end
    return F, D
end

function dibem_FD_threaded!(F, D, dad, rbf)
    props = dad.properties
    n0 = dad.Normal[1]
    pts = all_points(dad)
    nt = dad.nt
    Threads.@threads for j in 1:nt
        xj = pts[j]
        @inbounds for i in 1:nt
            rvec = xj - pts[i]
            R = norm(rvec)
            if R > 0
                F[i, j] = rbf(R)
                D[i, j] = fundamental(props, rvec, n0).U
            end
        end
    end
    return F, D
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

function profile_dibem_breakdown(dad; rbf=PHS(3; poly_deg=1), npg=12)
    nt = dad.nt
    BEM._init_quadrature!(dad, npg)
    F = zeros(nt, nt)
    D = zeros(nt, nt)
    IF = zeros(nt)
    ID = zeros(nt)

    n_near, n_far = count_near_far(dad)
    println("  collocation nt = ", nt, "  boundary n = ", dad.n, "  ni = ", dad.ni)
    println("  elements       = ", length(dad.elements), "  dim = ", dad.dimension)
    println("  RIM pairs      = near $n_near  far $n_far  (",
        @sprintf("%.1f%% near", 100n_near / max(n_near + n_far, 1)), ")")

    timed("F,D fill (serial, KernelPair)", () -> dibem_FD!(F, D, dad, rbf))
    fill!(F, 0); fill!(D, 0)
    timed("F,D fill (U-only kernel)", () -> dibem_FD_Uonly!(F, D, dad, rbf))
    fill!(F, 0); fill!(D, 0)
    timed("F,D fill (threaded)", () -> dibem_FD_threaded!(F, D, dad, rbf))
    BEM._dibem_ridge_F!(F)

    timed("RIM IF,ID (lumped near/far)", () -> BEM._dibem_accumulate_IF_ID!(IF, ID, dad, rbf))
    IFg, IDg = zeros(nt), zeros(nt)
    timed("RIM IF,ID (full Gauss)", () -> begin
        a, b = BEM._dibem_IF_ID_gauss(dad, rbf, all_points(dad), npg)
        IFg .= a; IDg .= b
        nothing
    end)

    timed("monomial IP (RIM)", () -> BEM._dibem_monomial_IP(dad, rbf))
    IP = BEM._dibem_monomial_IP(dad, rbf)
    timed("CPD F\\IF  (dense LU)", () -> BEM._dibem_poly_c(F, IF, all_points(dad), rbf; IP=IP))
    c = BEM._dibem_poly_c(F, IF, all_points(dad), rbf; IP=IP)

    timed("M = D .* c' + row-sum diag", () -> begin
        M = D .* c'
        @inbounds for i in 1:nt
            M[i, i] = 0
            M[i, i] = -sum(view(M, i, :)) + ID[i]
        end
        M
    end)
    M = D .* c'
    @inbounds for i in 1:nt
        M[i, i] = 0
        M[i, i] = -sum(view(M, i, :)) + ID[i]
    end
    fv = fill(dad.dimension == 2 ? 4.0 : 6.0, nt)
    timed("M * f  (dense matvec)", () -> M * fv)
    timed("f * ID (constant-f shortcut)", () -> fv[1] .* ID)

    onesv = ones(nt)
    rel = norm(M * onesv - ID) / (norm(ID) + 1e-14)
    @printf("  remainder identity  ‖M1−ID‖/‖ID‖ = %.3e\n", rel)
    return (; F, D, M, c, IF, ID)
end

function profile_pipeline_2d(ndiv)
    rbf = PHS(3; poly_deg=1)
    npg = 12
    fsrc = 4.0

    section("2-D Poisson DIBEM  ∇²u=4, u=|x|²   ndiv=$ndiv  threads=$(Threads.nthreads())")
    dad = poisson2d(ndiv)
    println("  n=$(dad.n)  ni=$(dad.ni)  nt=$(dad.nt)  nelem=$(length(dad.elements))")

    timed("assemble! H,G dense (threaded)", () -> assemble!(dad; npg=npg, threaded=true))
    timed("DIBEM_dense (F,D,RIM,CPD,M)", () -> DIBEM(dad; rbf=rbf, npg=npg, method=:dense))
    timed("applyBC", () -> applyBC(dad))
    timed("RHS += M*f + dense \\", () -> begin
        fv = BEM._eval_field(fsrc, all_points(dad))
        dad.b .+= dad.M * fv
        x = BEM.bem_linsolve(dad.A, dad.b)
        Tfull = zeros(eltype(x), dad.nt)
        qfull = zeros(eltype(x), dad.n)
        Tfull[1:length(x)] .= x
        BEM.split_sol!(dad, Tfull, qfull)
        set_cache!(dad; T=Tfull[1:dad.nt], q=qfull)
        dad.T
    end)
    @printf("  rel error T = %.3e   flux = %.3e\n", rel_error(dad), rel_error_flux(dad))

    section("2-D DIBEM kernel breakdown")
    dad2 = poisson2d(ndiv)
    assemble!(dad2; npg=npg, threaded=true)
    profile_dibem_breakdown(dad2; rbf=rbf, npg=npg)

    section("2-D compressed / GPU backends (same mesh)")
    dadh = poisson2d(ndiv)
    assemble!(dadh; npg=npg, threaded=true)
    try
        timed("DIBEM method=:hmatrix", () -> DIBEM(dadh; rbf=rbf, method=:hmatrix, nmax=32,
            atol=1e-6, rtol=1e-6, threads=true))
        timed("solve_poisson_dibem! hmatrix", () -> begin
            dadh.cache.A = nothing
            applyBC(dadh)
            solve_poisson_dibem!(dadh, fsrc; rbf=rbf, method=:hmatrix, npg=npg)
        end)
        @printf("  rel error T (hmat) = %.3e\n", rel_error(dadh))
    catch e
        println("  hmatrix skipped: ", e)
    end
    try
        dadg = poisson2d(ndiv)
        assemble!(dadg; npg=npg, threaded=true)
        timed("DIBEM method=:gpu device=:cpu", () -> DIBEM(dadg; rbf=rbf, method=:gpu,
            device=:cpu, npg=npg, T=Float64, threaded=true))
    catch e
        println("  gpu-cpu skipped: ", e)
    end

    return dad
end

function profile_pipeline_3d(ndiv)
    rbf = PHS(3; poly_deg=1)
    npg = 8
    fsrc = 6.0
    section("3-D Poisson DIBEM  ∇²u=6, u=|x|²   ndiv=$ndiv")
    dad = poisson3d(ndiv)
    println("  n=$(dad.n)  ni=$(dad.ni)  nt=$(dad.nt)  nelem=$(length(dad.elements))")
    timed("assemble! H,G dense", () -> assemble!(dad; npg=npg, threaded=true))
    timed("DIBEM_dense", () -> DIBEM(dad; rbf=rbf, npg=npg, method=:dense))
    timed("solve_poisson_dibem! (H,M cached)", () -> solve_poisson_dibem!(dad, fsrc; rbf=rbf, npg=npg))
    @printf("  rel error T = %.3e   flux = %.3e\n", rel_error(dad), rel_error_flux(dad))

    section("3-D DIBEM kernel breakdown")
    dad2 = poisson3d(ndiv)
    assemble!(dad2; npg=npg, threaded=true)
    profile_dibem_breakdown(dad2; rbf=rbf, npg=npg)
    return dad
end

_alloc(f) = (f(); GC.gc(false); @allocated f())

function _cached_rim_lumped!(IF, ID, dad, rbf)
    # Same near/far split as `_dibem_accumulate_IF_ID!`, but:
    #   * shapefun(N, dN) once
    #   * per-element Gauss y, n, wJ cached
    #   * no `Nodes[el.index]` per source
    BEM._init_quadrature!(dad, length(dad.qsi))
    ηs, ws = dad.qsi, dad.w
    N, dN = BEM.shapefun(dad.element_type, ηs)
    props = dad.properties
    fill!(IF, 0); fill!(ID, 0)
    geos = map(dad.elements) do el
        nodes = dad.Nodes[el.index]
        pg = N * nodes
        dx = dN * nodes
        nref = dad.Normal[el.index[1]]
        nq = length(ηs)
        y = Vector{eltype(nodes)}(undef, nq)
        nrm = Vector{typeof(nref)}(undef, nq)
        wJ = Vector{Float64}(undef, nq)
        @inbounds for q in 1:nq
            Jv = dx[q]
            J = norm(Jv)
            nn = J < 1e-16 ? nref : BEM.tan2normal(Jv / J)
            nn ⋅ nref < 0 && (nn = -nn)
            y[q] = pg[q]
            nrm[q] = nn
            wJ[q] = ws[q] * J
        end
        (; el, nodes, y, nrm, wJ)
    end
    @inbounds for i in 1:dad.nt
        x = point(dad, i)
        for g in geos
            if BEM._near_element(x, g.nodes, g.el)
                for q in eachindex(g.wJ)
                    r = g.y[q] - x
                    R = norm(r)
                    R < 1e-14 && continue
                    wJn = g.wJ[q] * BEM._rim_factor(g.nrm[q], r, R, 2)
                    IF[i] += BEM.int(rbf, x, g.y[q]) * wJn
                    ID[i] += BEM.radial_integral(props, R, 2) * wJn
                end
            else
                for j in eachindex(g.el.index)
                    ind = g.el.index[j]
                    xj = dad.Nodes[ind]
                    r = xj - x
                    R = norm(r)
                    R < 1e-10 && continue
                    wJn = dad.elem_weight[j] * g.el.Jacobian[j] * BEM._rim_factor(dad.Normal[ind], r, R, 2)
                    IF[i] += BEM.int(rbf, x, xj) * wJn
                    ID[i] += BEM.radial_integral(props, R, 2) * wJn
                end
            end
        end
    end
    return IF, ID
end

function _cached_rim_threaded!(IF, ID, dad, rbf)
    BEM._init_quadrature!(dad, length(dad.qsi))
    ηs, ws = dad.qsi, dad.w
    N, dN = BEM.shapefun(dad.element_type, ηs)
    props = dad.properties
    fill!(IF, 0); fill!(ID, 0)
    geos = map(dad.elements) do el
        nodes = dad.Nodes[el.index]
        pg = N * nodes
        dx = dN * nodes
        nref = dad.Normal[el.index[1]]
        nq = length(ηs)
        y = Vector{eltype(nodes)}(undef, nq)
        nrm = Vector{typeof(nref)}(undef, nq)
        wJ = Vector{Float64}(undef, nq)
        @inbounds for q in 1:nq
            Jv = dx[q]
            J = norm(Jv)
            nn = J < 1e-16 ? nref : BEM.tan2normal(Jv / J)
            nn ⋅ nref < 0 && (nn = -nn)
            y[q] = pg[q]
            nrm[q] = nn
            wJ[q] = ws[q] * J
        end
        (; el, nodes, y, nrm, wJ)
    end
    Threads.@threads for i in 1:dad.nt
        x = point(dad, i)
        accF = 0.0
        accD = 0.0
        @inbounds for g in geos
            if BEM._near_element(x, g.nodes, g.el)
                for q in eachindex(g.wJ)
                    r = g.y[q] - x
                    R = norm(r)
                    R < 1e-14 && continue
                    wJn = g.wJ[q] * BEM._rim_factor(g.nrm[q], r, R, 2)
                    accF += BEM.int(rbf, x, g.y[q]) * wJn
                    accD += BEM.radial_integral(props, R, 2) * wJn
                end
            else
                for j in eachindex(g.el.index)
                    ind = g.el.index[j]
                    xj = dad.Nodes[ind]
                    r = xj - x
                    R = norm(r)
                    R < 1e-10 && continue
                    wJn = dad.elem_weight[j] * g.el.Jacobian[j] * BEM._rim_factor(dad.Normal[ind], r, R, 2)
                    accF += BEM.int(rbf, x, xj) * wJn
                    accD += BEM.radial_integral(props, R, 2) * wJn
                end
            end
        end
        IF[i] = accF
        ID[i] = accD
    end
    return IF, ID
end

function micro_allocs(dad; rbf=PHS(3; poly_deg=1))
    section("Micro-allocations (hot kernels, post-compile)")
    BEM._init_quadrature!(dad, 12)
    el = dad.elements[1]
    nodes = dad.Nodes[el.index]
    x = point(dad, 1)
    ηs, ws = dad.qsi, dad.w
    props = dad.properties
    n0 = dad.Normal[1]
    rvec = point(dad, min(2, dad.nt)) - x
    R = norm(rvec)
    IF = zeros(dad.nt); ID = zeros(dad.nt)

    # compile
    rbf(R); fundamental(props, rvec, n0); BEM.shapefun(dad.element_type, ηs)
    BEM.int(rbf, x, nodes[1])
    BEM._rim_element!(dad, el, nodes, x, ηs, ws) do wJn, R_, e, y; nothing; end
    BEM._dibem_near_gauss!(IF, ID, dad, el, nodes, x, 1, rbf, ηs, ws, props, dad.dimension)
    BEM._dibem_far_lump!(IF, ID, dad, el, x, 1, rbf, props, dad.dimension)
    BEM._near_element(x, nodes, el)

    println("  PHS3(R)                    alloc = ", _alloc(() -> rbf(R)), " B")
    println("  fundamental → KernelPair   alloc = ", _alloc(() -> fundamental(props, rvec, n0)), " B")
    println("  fundamental(...).U         alloc = ", _alloc(() -> fundamental(props, rvec, n0).U), " B")
    println("  shapefun(poly, ηs)         alloc = ", _alloc(() -> BEM.shapefun(dad.element_type, ηs)), " B")
    println("  int(PHS3, x, y)            alloc = ", _alloc(() -> BEM.int(rbf, x, nodes[1])), " B")
    println("  _rim_element! (empty body) alloc = ",
        _alloc(() -> BEM._rim_element!(dad, el, nodes, x, ηs, ws) do wJn, R_, e, y; nothing; end), " B")
    println("  _dibem_near_gauss!         alloc = ",
        _alloc(() -> BEM._dibem_near_gauss!(IF, ID, dad, el, nodes, x, 1, rbf, ηs, ws, props, dad.dimension)), " B")
    println("  _dibem_far_lump!           alloc = ",
        _alloc(() -> BEM._dibem_far_lump!(IF, ID, dad, el, x, 1, rbf, props, dad.dimension)), " B")
    println("  dad.Nodes[el.index]        alloc = ", _alloc(() -> dad.Nodes[el.index]), " B")
    println("  _near_element              alloc = ", _alloc(() -> BEM._near_element(x, nodes, el)), " B")
    if dad.dimension == 3
        BEM.shapefun2D(dad.element_type, ηs); kron(ws, ws)
        println("  shapefun2D(poly, ηs)       alloc = ",
            _alloc(() -> BEM.shapefun2D(dad.element_type, ηs)), " B")
        println("  kron(ws,ws)                alloc = ", _alloc(() -> kron(ws, ws)), " B")
    end

    if dad.dimension == 2
        section("RIM experiment: cache Gauss geometry vs current")
        IFa = zeros(dad.nt); IDa = zeros(dad.nt)
        IFb = zeros(dad.nt); IDb = zeros(dad.nt)
        IFc = zeros(dad.nt); IDc = zeros(dad.nt)
        timed("current lumped RIM", () -> (fill!(IFa, 0); fill!(IDa, 0);
            BEM._dibem_accumulate_IF_ID!(IFa, IDa, dad, rbf)))
        timed("cached shapefun + nodes (serial)", () -> _cached_rim_lumped!(IFb, IDb, dad, rbf))
        timed("cached shapefun + nodes (threaded)", () -> _cached_rim_threaded!(IFc, IDc, dad, rbf))
        @printf("  ‖ID_cached − ID_current‖/‖ID‖ = %.3e\n",
            norm(IDb - IDa) / (norm(IDa) + 1e-14))
        @printf("  ‖ID_thread − ID_current‖/‖ID‖ = %.3e\n",
            norm(IDc - IDa) / (norm(IDa) + 1e-14))
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
        return n_any, n_box
    end
    println("  fundamental(Laplace):")
    flags("fundamental", fundamental, props, y - x, n0)
    println("  PHS3(r):")
    flags("PHS3", rbf, 0.3)
    println("  _near_element:")
    el = dad.elements[1]
    flags("_near_element", BEM._near_element, x, dad.Nodes[el.index], el)
    println("  radial_integral:")
    flags("radial_integral", BEM.radial_integral, props, 0.3, dad.dimension)
    println("  _dibem_accumulate_IF_ID!:")
    IF = zeros(dad.nt); ID = zeros(dad.nt)
    flags("_dibem_accumulate_IF_ID!", BEM._dibem_accumulate_IF_ID!, IF, ID, dad, rbf)
end

function scaling_scan()
    section("Scaling  2-D Poisson (assemble + DIBEM_dense + solve)")
    rbf = PHS(3; poly_deg=1)
    println("  ndiv    n   nt    t_HG[s]  t_DIBEM[s]  t_solve[s]   mem_HGM[MiB]  relT")
    for nd in (8, 12, 16, 24)
        dad = poisson2d(nd)
        assemble!(dad; npg=10, threaded=true)  # compile / warmup
        dad = poisson2d(nd)
        GC.gc(false)
        t0 = time_ns(); assemble!(dad; npg=10, threaded=true); tHG = (time_ns() - t0) / 1e9
        t0 = time_ns(); DIBEM(dad; rbf=rbf, npg=10, method=:dense); tM = (time_ns() - t0) / 1e9
        t0 = time_ns(); solve_poisson_dibem!(dad, 4.0; rbf=rbf, npg=10); tS = (time_ns() - t0) / 1e9
        mem = (Base.summarysize(dad.H) + Base.summarysize(dad.G) + Base.summarysize(dad.M)) / 1024^2
        @printf("  %4d %5d %5d  %8.3f  %10.3f  %9.3f   %10.2f  %.2e\n",
            nd, dad.n, dad.nt, tHG, tM, tS, mem, rel_error(dad))
    end
end

function dump_profile(dad; rbf=PHS(3; poly_deg=1), npg=12)
    section("CPU sample profile  (DIBEM_dense + assemble!)")
    dad2 = deepcopy(dad)
    has_cache(dad2, :H) && (dad2.cache.H = nothing)
    assemble!(dad2; npg=npg, threaded=true)
    DIBEM(dad2; rbf=rbf, npg=npg, method=:dense)  # warmup

    dad3 = deepcopy(dad)
    has_cache(dad3, :H) && (dad3.cache.H = nothing)
    has_cache(dad3, :M) && (dad3.cache.M = nothing)
    Profile.clear()
    @profile begin
        assemble!(dad3; npg=npg, threaded=true)
        DIBEM(dad3; rbf=rbf, npg=npg, method=:dense)
        applyBC(dad3)
        fv = BEM._eval_field(4.0, all_points(dad3))
        dad3.b .+= dad3.M * fv
        BEM.bem_linsolve(dad3.A, dad3.b)
    end
    out = joinpath(@__DIR__, "poisson_dibem_profile.txt")
    open(out, "w") do io
        println(io, "flat profile, sorted by count")
        Profile.print(io; format=:flat, sortedby=:count, mincount=8, C=false, maxdepth=60)
        println(io, "\n\n==== tree (noisefloor=2) ====\n")
        Profile.print(io; C=false, noisefloor=2, maxdepth=18, mincount=8)
    end
    println("  wrote ", out)
    # print a short top-of-flat to stdout
    buf = IOBuffer()
    Profile.print(buf; format=:flat, sortedby=:count, mincount=20, C=false)
    lines = split(String(take!(buf)), '\n')
    println("  top samples:")
    for L in Iterators.take(lines, 25)
        println("    ", L)
    end
end

function main(args=ARGS)
    ndiv = _parse_ndiv(args)
    println("Poisson + DIBEM profile  ndiv=$ndiv  threads=$(Threads.nthreads())  julia=$(VERSION)")
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
    nd3 = ndiv <= 12 ? 3 : 4
    profile_pipeline_3d(nd3)
    section("Done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
