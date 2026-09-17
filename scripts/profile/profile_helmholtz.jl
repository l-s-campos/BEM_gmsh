#!/usr/bin/env julia
# =============================================================================
# Profile Helmholtz: native Hankel collocation vs Laplace correlato (H+κ²M)
#   julia --project=. -t 8 scripts/profile/profile_helmholtz.jl
#   julia --project=. -t 8 scripts/profile/profile_helmholtz.jl --ndiv=16
# =============================================================================

using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))

using BEM
using LinearAlgebra
using StaticArrays
using Printf
using Profile
using InteractiveUtils: code_warntype
using SpecialFunctions: hankelh1

const PROJECT = dirname(dirname(@__DIR__))
include(joinpath(PROJECT, "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(PROJECT, "data", "Laplace", "potencial_problems.jl"))
include(joinpath(PROJECT, "data", "Laplace", "wave_propagation.jl"))
include(joinpath(PROJECT, "data", "Laplace", "helmholtz_problems.jl"))

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
    @printf("  %-40s  %8.3f s   alloc %s\n", label, dt, fmt_bytes(db))
    return r, dt, db
end

section(title) = (println(); println("="^72); println(title); println("="^72))
_alloc(f) = (f(); GC.gc(false); @allocated f())

function helm_dad(ndiv; ω=4.0, c=1.0, pontointerno=false)
    format2d(quadrado(ndiv=ndiv, show=false, nome="prof_helm_$ndiv"),
        Helmholtz(; ω=ω, c=c); pontointerno=pontointerno)
end

function lap_dad(ndiv; pontointerno=false)
    format2d(quadrado(ndiv=ndiv, show=false, nome="prof_hlap_$ndiv"),
        Laplace(1.0); pontointerno=pontointerno)
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

function profile_native(ndiv; ω=4.0, npg=12)
    section("Native Helmholtz (Hankel) vs Laplace  ndiv=$ndiv  ω=$ω  threads=$(Threads.nthreads())")
    dadH = helm_dad(ndiv; ω=ω)
    dadL = lap_dad(ndiv)
    n_near, n_far = count_near_far(dadH)
    println("  n=$(dadH.n)  ni=$(dadH.ni)  nt=$(dadH.nt)  nelem=$(length(dadH.elements))")
    println("  eltype Helmholtz H = ", kernel_eltype(dadH.properties),
        "  Laplace H = ", kernel_eltype(dadL.properties))
    println("  pairs near=$n_near  far=$n_far  (",
        @sprintf("%.1f%% near", 100n_near / max(n_near + n_far, 1)), ")")
    κ = wavenumber(dadH.properties)
    println("  κ = ω/c = ", κ)

    timed("Laplace assemble! H,G", () -> assemble!(dadL; npg=npg, threaded=true))
    timed("Helmholtz assemble! H,G", () -> assemble!(dadH; npg=npg, threaded=true))
    timed("Laplace applyBC + \\", () -> solve(dadL))
    timed("Helmholtz applyBC + complex \\", () -> solve(dadH))
    @printf("  Helmholtz T finite  real=%s  imag=%s\n",
        string(all(isfinite, real.(dadH.T))), string(all(isfinite, imag.(dadH.T))))
    @printf("  mem H+G Helmholtz = %s   Laplace = %s\n",
        fmt_bytes(Base.summarysize(dadH.H) + Base.summarysize(dadH.G)),
        fmt_bytes(Base.summarysize(dadL.H) + Base.summarysize(dadL.G)))
    return dadH, dadL
end

function profile_correlato(ndiv; κ=4.0, npg=12)
    section("Laplace correlato  (H + κ² M) T = G q   κ=$κ")
    rbf = PHS(3; poly_deg=1)
    msh = helm1d_mesh(; ndiv=ndiv, nome="prof_cor_$ndiv")
    dad = format2d(msh, Laplace(1.0); pontointerno=true)
    attach_analytical!(dad, ana_helm1d(κ))
    println("  n=$(dad.n)  ni=$(dad.ni)  nt=$(dad.nt)  nelem=$(length(dad.elements))")
    timed("assemble! Laplace H,G", () -> assemble!(dad; npg=npg, threaded=true))
    timed("DIBEM dense M", () -> DIBEM(dad; rbf=rbf, npg=npg, method=:dense))
    timed("solve blocks H+κ²M", () -> solve(dad; blocks=true, M=dad.M, κ2=κ^2))
    @printf("  rel error T = %.3e\n", rel_error(dad))

    section("Burton–Miller extras (Laplace HBIE + M′)")
    dad2 = format2d(helm1d_mesh(; ndiv=ndiv, nome="prof_bm_$ndiv"), Laplace(1.0);
        pontointerno=true)
    assemble!(dad2; npg=npg, threaded=true)
    DIBEM(dad2; rbf=rbf, npg=npg, method=:dense)
    timed("H_G_hyper (HBIE)", () -> H_G_hyper(dad2; npg=npg, threaded=true))
    timed("dibem_hyper_mass M′", () -> dibem_hyper_mass(dad2; rbf=rbf))
    timed("assemble_wave_burton_miller!", () ->
        assemble_wave_burton_miller!(dad2; mass=:dibem, α=1.0, npg=npg, rbf=rbf, threaded=true))
    return dad
end

function micro_kernels()
    section("Micro-allocations / kernel cost")
    r = Point2D(0.3, 0.4)
    n = Point2D(1.0, 0.0)
    nf = Point2D(0.0, 1.0)
    lap = Laplace(1.0)
    hel = Helmholtz(; ω=4.0, c=1.0)
    z = wavenumber(hel) * norm(r)
    fundamental(lap, r, n)
    fundamental(hel, r, n)
    fundamental_hyper(lap, r, n, nf)
    fundamental_hyper(hel, r, n, nf)
    hankelh1(0, z)
    hankelh1(1, z)

    println("  Laplace fundamental          alloc = ", _alloc(() -> fundamental(lap, r, n)), " B")
    println("  Helmholtz fundamental        alloc = ", _alloc(() -> fundamental(hel, r, n)), " B")
    println("  Laplace fundamental_hyper    alloc = ", _alloc(() -> fundamental_hyper(lap, r, n, nf)), " B")
    println("  Helmholtz fundamental_hyper  alloc = ", _alloc(() -> fundamental_hyper(hel, r, n, nf)), " B")
    println("  hankelh1(0, z)               alloc = ", _alloc(() -> hankelh1(0, z)), " B")
    println("  hankelh1(1, z)               alloc = ", _alloc(() -> hankelh1(1, z)), " B")

    nrep = 200_000
    function ns(f)
        f()
        t0 = time_ns()
        for _ in 1:nrep
            f()
        end
        return (time_ns() - t0) / nrep
    end
    @printf("  ns/call  Laplace FS=%6.0f  Helmholtz FS=%6.0f  Hankel H0=%6.0f  H1=%6.0f\n",
        ns(() -> fundamental(lap, r, n)),
        ns(() -> fundamental(hel, r, n)),
        ns(() -> hankelh1(0, z)),
        ns(() -> hankelh1(1, z)))
    @printf("  ns/call  Laplace hyper=%6.0f  Helmholtz hyper=%6.0f  (H1+H2)\n",
        ns(() -> fundamental_hyper(lap, r, n, nf)),
        ns(() -> fundamental_hyper(hel, r, n, nf)))
end

function warntype_hot()
    section("Type stability")
    r = Point2D(0.3, 0.4)
    n = Point2D(1.0, 0.0)
    hel = Helmholtz(; ω=4.0, c=1.0)
    lap = Laplace(1.0)
    function flags(label, f, args...)
        buf = IOBuffer()
        code_warntype(buf, f, typeof.(args))
        txt = String(take!(buf))
        n_any = length(collect(eachmatch(r"\bAny\b", txt)))
        n_box = length(collect(eachmatch(r"Core\.Box", txt)))
        println("    $label: Any≈$n_any  Core.Box≈$n_box")
    end
    flags("fundamental(Helmholtz)", fundamental, hel, r, n)
    flags("fundamental(Laplace)", fundamental, lap, r, n)
    flags("hankelh1", hankelh1, 0, 1.2)
end

function freq_scan(ndiv)
    section("Native Helmholtz assemble vs frequency  (ndiv=$ndiv)")
    println("  ω      κ      t_assemble[s]   alloc")
    npg = 12
    for ω in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
        dad = helm_dad(ndiv; ω=ω)
        assemble!(dad; npg=npg, threaded=true)
        dad = helm_dad(ndiv; ω=ω)
        GC.gc(false)
        t0 = time_ns(); b0 = Base.gc_bytes()
        assemble!(dad; npg=npg, threaded=true)
        dt = (time_ns() - t0) / 1e9
        db = Int(Base.gc_bytes() - b0)
        @printf("  %5.1f  %5.1f    %8.3f     %s\n",
            ω, wavenumber(dad.properties), dt, fmt_bytes(db))
    end
end

function scaling_scan()
    section("Scaling  native Helmholtz vs Laplace assemble")
    println("  ndiv    n     t_Lap[s]  t_Helm[s]  Helm/Lap   mem_H_helm")
    for nd in (8, 12, 16, 24)
        dL = lap_dad(nd)
        dH = helm_dad(nd; ω=4.0)
        assemble!(dL; npg=10, threaded=true)
        assemble!(dH; npg=10, threaded=true)
        dL = lap_dad(nd)
        dH = helm_dad(nd; ω=4.0)
        GC.gc(false)
        t0 = time_ns(); assemble!(dL; npg=10, threaded=true); tL = (time_ns() - t0) / 1e9
        t0 = time_ns(); assemble!(dH; npg=10, threaded=true); tH = (time_ns() - t0) / 1e9
        @printf("  %4d %5d   %8.3f  %8.3f   %6.2fx   %s\n",
            nd, dH.n, tL, tH, tH / max(tL, 1e-12),
            fmt_bytes(Base.summarysize(dH.H)))
    end
end

function dump_profile(ndiv)
    section("CPU sample  native Helmholtz assemble!")
    dad = helm_dad(ndiv; ω=4.0)
    assemble!(dad; npg=12, threaded=true)
    dad2 = helm_dad(ndiv; ω=4.0)
    Profile.clear()
    @profile assemble!(dad2; npg=12, threaded=true)
    out = joinpath(@__DIR__, "helmholtz_profile.txt")
    open(out, "w") do io
        Profile.print(io; format=:flat, sortedby=:count, mincount=8, C=false)
    end
    println("  wrote ", out)
    buf = IOBuffer()
    Profile.print(buf; format=:flat, sortedby=:count, mincount=12, C=false)
    for L in Iterators.take(split(String(take!(buf)), '\n'), 22)
        println("    ", L)
    end
end

function main(args=ARGS)
    ndiv = _parse_ndiv(args)
    println("Helmholtz profile  ndiv=$ndiv  threads=$(Threads.nthreads())  julia=$(VERSION)")
    profile_native(ndiv)
    micro_kernels()
    try
        warntype_hot()
    catch e
        println("  warntype skipped: ", e)
    end
    freq_scan(ndiv)
    scaling_scan()
    try
        dump_profile(ndiv)
    catch e
        println("  profile dump skipped: ", e)
    end
    try
        profile_correlato(ndiv)
    catch e
        println("  correlato skipped: ", e)
        showerror(stdout, e)
        println()
    end
    section("Done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
