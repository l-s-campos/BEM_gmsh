# =============================================================================
# Fast DIBEM — elasticity / thermoelasticity paper benchmarks
# Backends: dense | H-matrix | FMM
# Focus: assembly time, matvec time, memory, matvec accuracy
# Geometry: unit square (thermoelasticity paper examples)
# Output → artigos/escritos/2026/Fast_DIBEM_elastic
# =============================================================================
using DrWatson
@quickactivate :BEM
using BEM.HMatrices

using LinearAlgebra
using Statistics
using Printf
using Dates
using Random
using Plots
using LaTeXStrings

const OUTDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\Fast_DIBEM_elastic"
const FIGDIR = joinpath(OUTDIR, "figures")
mkpath(FIGDIR)

include(joinpath(projectdir(), "data", "Laplace", "Laplace_dad.jl"))

println("="^72)
println(" Fast DIBEM elasticity benchmarks")
println(" ", Dates.now())
println("="^72)

# -----------------------------------------------------------------------------
# Mesh helper (boundary + internal collocation, matching thermo paper style)
# -----------------------------------------------------------------------------
function make_elast_dad(ndiv; n_int_side = 0, nome = "fdibem",
        E = 210e9, ν = 0.3, α = 1.2e-5, plane_strain = true)
    msh = Base.invokelatest(quadrado_elasticity; ndiv = ndiv, show = false, nome = nome,
        Lx = 1.0, Ly = 1.0, ordem = 1)
    props = Elasticity(E, ν, 1.0; plane_strain = plane_strain, α = α)
    dad = format2d(msh, props; tipo = 1, pontointerno = false)
    if n_int_side > 0
        xs = range(0.15, 0.85; length = n_int_side)
        set_internal_nodes!(dad, [SVector(float(x), float(y)) for y in xs for x in xs])
    end
    return dad
end

function mem_MB(obj)
    return Base.summarysize(obj) / 1024^2
end

function mem_D_MB(dad, M)
    # Prefer compressed Kelvin operator when available
    if has_cache(dad, :dibem_D)
        return mem_MB(dad.dibem_D)
    end
    return mem_MB(M)
end

function compression_of(dad, M, n2)
    dense_bytes = n2 * n2 * sizeof(Float64)
    used = has_cache(dad, :dibem_D) ? Base.summarysize(dad.dibem_D) : Base.summarysize(M)
    return dense_bytes / max(used, 1)
end

function time_matvec(M, x; nwarm = 2, nrun = 8)
    y = M * x
    for _ in 1:nwarm
        y = M * x
    end
    t0 = time_ns()
    local acc
    for _ in 1:nrun
        acc = M * x
    end
    t1 = time_ns()
    return (t1 - t0) / nrun / 1e9, acc
end

# -----------------------------------------------------------------------------
# Backend specs
# -----------------------------------------------------------------------------
const METHODS = (
    (name = :dense,   kw = (;)),
    # threads=false: ACA on ExpandedClusterTree is not yet race-free
    (name = :hmatrix, kw = (; atol = 1e-8, rtol = 1e-8, nmax = 28, f_method = :dense, threads = false)),
    (name = :fmm,     kw = (; eps = 1e-6, nmax = 28, eta = 3.0, f_method = :dense)),
)

# Mesh ladder: boundary divisions; internals grow slowly
# ndiv → ~4*ndiv boundary nodes (linear discontinuous / continuous depending on mesh)
const NDIVS = (6, 10, 16, 24, 36, 48)

# -----------------------------------------------------------------------------
# Study 1 — operator assembly / matvec / memory scaling (body-force DIBEM M)
# -----------------------------------------------------------------------------
println("\n[1] DIBEM operator scaling (elasticity Kelvin)")
rows = NamedTuple[]
Random.seed!(1)

for ndiv in NDIVS
    n_int = max(2, ndiv ÷ 4)
    println("\n--- ndiv=$ndiv  n_int_side=$n_int ---")
    # dense reference
    dad_ref = make_elast_dad(ndiv; n_int_side = n_int, nome = "fd_ref_$ndiv")
    t0 = time()
    Md = DIBEM(dad_ref; method = :dense)
    t_dense = time() - t0
    n2 = size(Md, 1)
    x = randn(n2)
    t_mv_d, yd = time_matvec(Md, x)
    mem_d = mem_D_MB(dad_ref, Md)
    @printf("  dense    n=%4d  build=%8.3fs  mv=%8.4fs  mem=%8.2f MB\n",
        n2, t_dense, t_mv_d, mem_d)
    push!(rows, (; method = "dense", ndiv, n = n2, t_build = t_dense, t_mv = t_mv_d,
        mem_MB = mem_d, cr = 1.0, rel_err = 0.0))

    for m in METHODS
        m.name === :dense && continue
        try
            dad = make_elast_dad(ndiv; n_int_side = n_int, nome = "fd_$(m.name)_$ndiv")
            t0 = time()
            M = DIBEM(dad; method = m.name, m.kw...)
            t_b = time() - t0
            t_mv, y = time_matvec(M, x; nrun = m.name === :fmm ? 5 : 8)
            err = norm(y - yd) / max(norm(yd), eps())
            mem = mem_D_MB(dad, M)
            cr = compression_of(dad, M, n2)
            @printf("  %-8s n=%4d  build=%8.3fs  mv=%8.4fs  mem=%8.2f MB  cr=%6.2f  err=%9.2e\n",
                m.name, n2, t_b, t_mv, mem, cr, err)
            push!(rows, (; method = String(m.name), ndiv, n = n2, t_build = t_b, t_mv = t_mv,
                mem_MB = mem, cr = cr, rel_err = err))
        catch e
            @warn "method $(m.name) failed at ndiv=$ndiv" exception = (e, catch_backtrace())
        end
    end
end

# Write CSV manually (avoid CSV.jl dependency)
function write_csv(path, rows)
    open(path, "w") do io
        println(io, "method,ndiv,n,t_build,t_mv,mem_MB,compression,rel_err")
        for r in rows
            @printf(io, "%s,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                r.method, r.ndiv, r.n, r.t_build, r.t_mv, r.mem_MB, r.cr, r.rel_err)
        end
    end
end
write_csv(joinpath(OUTDIR, "results_scaling.csv"), rows)
println("Wrote results_scaling.csv")

# group helper
function by_method(rows)
    d = Dict{String,Vector{NamedTuple}}()
    for r in rows
        push!(get!(d, r.method, NamedTuple[]), r)
    end
    for k in keys(d)
        sort!(d[k]; by = r -> r.n)
    end
    return d
end
R = by_method(rows)
cols = Dict("dense" => :black, "hmatrix" => :crimson, "h2" => :dodgerblue, "fmm" => :seagreen)
marks = Dict("dense" => :circle, "hmatrix" => :utriangle, "h2" => :rect, "fmm" => :diamond)

# Fig: assembly time
fig = plot(; xlabel = L"n = 2 n_t", ylabel = "assembly time [s]",
    title = "Elasticity DIBEM assembly", xscale = :log2, yscale = :log10,
    legend = :topleft, legendfontsize = 9, size = (480, 360),
    framestyle = :box, background_color = :white)
for m in ("dense", "hmatrix", "h2", "fmm")
    haskey(R, m) || continue
    rs = R[m]
    plot!(fig, [r.n for r in rs], [r.t_build for r in rs]; color = cols[m],
        marker = marks[m], label = m)
end
savefig(fig, joinpath(FIGDIR, "fig_assembly_time.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_assembly_time.png"))

# Fig: matvec time
fig = plot(; xlabel = L"n = 2 n_t", ylabel = "matvec time [s]",
    title = "Elasticity DIBEM matvec", xscale = :log2, yscale = :log10,
    legend = :topleft, legendfontsize = 9, size = (480, 360),
    framestyle = :box, background_color = :white)
for m in ("dense", "hmatrix", "h2", "fmm")
    haskey(R, m) || continue
    rs = R[m]
    plot!(fig, [r.n for r in rs], [r.t_mv for r in rs]; color = cols[m],
        marker = marks[m], label = m)
end
if haskey(R, "dense") && length(R["dense"]) >= 2
    ns = [R["dense"][1].n, R["dense"][end].n]
    c2 = R["dense"][end].t_mv / (ns[2]^2)
    plot!(fig, ns, c2 .* (ns .^ 2); color = :gray, linestyle = :dash, label = L"O(n^2)")
    clog = R["dense"][end].t_mv / (ns[2] * log2(ns[2])) * 0.15
    plot!(fig, ns, clog .* (ns .* log2.(ns)); color = :gray, linestyle = :dot, label = L"O(n\log n)")
end
savefig(fig, joinpath(FIGDIR, "fig_matvec_time.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_matvec_time.png"))

# Fig: memory
fig = plot(; xlabel = L"n = 2 n_t", ylabel = "memory [MB]",
    title = "Kelvin operator storage", xscale = :log2, yscale = :log10,
    legend = :topleft, legendfontsize = 9, size = (480, 360),
    framestyle = :box, background_color = :white)
for m in ("dense", "hmatrix", "h2", "fmm")
    haskey(R, m) || continue
    rs = R[m]
    plot!(fig, [r.n for r in rs], [max(r.mem_MB, 1e-3) for r in rs]; color = cols[m],
        marker = marks[m], label = m)
end
savefig(fig, joinpath(FIGDIR, "fig_memory.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_memory.png"))

# Fig: compression ratio
fig = plot(; xlabel = L"n = 2 n_t", ylabel = "compression ratio",
    title = "Storage compression vs dense", xscale = :log2,
    legend = :topleft, legendfontsize = 9, size = (480, 360),
    framestyle = :box, background_color = :white)
for m in ("hmatrix", "h2", "fmm")
    haskey(R, m) || continue
    rs = R[m]
    plot!(fig, [r.n for r in rs], [r.cr for r in rs]; color = cols[m],
        marker = marks[m], label = m)
end
savefig(fig, joinpath(FIGDIR, "fig_compression.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_compression.png"))

# Fig: matvec error
fig = plot(; xlabel = L"n = 2 n_t", ylabel = "relative matvec error",
    title = "Accuracy vs dense DIBEM", xscale = :log2, yscale = :log10,
    legend = :topright, legendfontsize = 9, size = (480, 360),
    framestyle = :box, background_color = :white)
for m in ("hmatrix", "h2", "fmm")
    haskey(R, m) || continue
    rs = R[m]
    ys = [max(r.rel_err, 1e-16) for r in rs]
    plot!(fig, [r.n for r in rs], ys; color = cols[m], marker = marks[m], label = m)
end
savefig(fig, joinpath(FIGDIR, "fig_matvec_error.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_matvec_error.png"))

# -----------------------------------------------------------------------------
# Study 2 — thermoelastic examples (accuracy + DIBEM cost)
# Revisit thermo paper cases on the unit/scaled square with body force / θ
# -----------------------------------------------------------------------------
println("\n[2] Thermoelasticity-style examples with compressed DIBEM")

function solve_with_dibem_method!(dad; method = :dense, θ = 0.0, bodyforce = nothing, kw...)
    fill!(dad.BC, 0)          # all Dirichlet by default → overridden below
    # caller sets BC/BV
    H_G_full_direct(dad; npg = 10, threaded = false)
    # Build M with requested backend when domain term needed
    need_M = bodyforce !== nothing || !(θ isa Number)
    if need_M
        DIBEM(dad; method = method, kw...)
    end
    return solve_thermoelastic!(dad; θ = θ, bodyforce = bodyforce, npg_dibem = 8)
end

"""
Example A — constant gravity body force on a square (vertical beam / self-weight analogue).
Bottom fixed, top free, sides rollers. Analytical beam-like fields not required;
we compare backends against dense DIBEM solution.
"""
function example_selfweight(ndiv; method = :dense, kw...)
    ρg = 1.0
    dad = make_elast_dad(ndiv; n_int_side = max(3, ndiv ÷ 5), nome = "sw_$(method)_$ndiv",
        E = 1e3, ν = 0.3, plane_strain = true)
    # BC: bottom uy=0 + ux=0; left/right ux=0; top free traction 0
    # format2d defaults — set explicitly
    n = dad.n
    fill!(dad.BC, 0)
    fill!(dad.BV, 0.0)
    # Heuristic by coordinate
    for i in 1:n
        x, y = dad.Nodes[i]
        if y < 1e-9                 # bottom: fixed
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0
            dad.BC[2i]   = 0; dad.BV[2i]   = 0
        elseif abs(x) < 1e-9 || abs(x - 1) < 1e-9   # sides: ux=0, ty free
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0
            dad.BC[2i]   = 1; dad.BV[2i]   = 0
        else                        # top: free
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0
            dad.BC[2i]   = 1; dad.BV[2i]   = 0
        end
    end
    bf = (x, y) -> SVector(0.0, -ρg)
    t0 = time()
    H_G_full_direct(dad; npg = 10, threaded = false)
    t_hg = time() - t0
    t0 = time()
    DIBEM(dad; method = method, kw...)
    t_m = time() - t0
    t0 = time()
    u = solve_thermoelastic!(dad; bodyforce = bf, θ = 0.0)
    t_s = time() - t0
    return (; dad, u, t_hg, t_m, t_s, mem = mem_D_MB(dad, dad.M), n = 2dad.nt)
end

"""
Example B — square under uniform temperature (constrained) — no DIBEM needed,
but we still time H,G and report thermal stress.
"""
function example_uniform_theta(ndiv; Δθ = 1.0)
    dad = make_elast_dad(ndiv; n_int_side = 0, nome = "th_uni_$ndiv",
        E = 210e9, ν = 0.3, α = 1.2e-5, plane_strain = true)
    fill!(dad.BC, 0); fill!(dad.BV, 0.0)   # fully constrained
    t0 = time()
    H_G_full_direct(dad; npg = 10, threaded = false)
    t_hg = time() - t0
    t0 = time()
    u = solve_thermoelastic!(dad; θ = Δθ)
    t_s = time() - t0
    k̂ = thermal_modulus(dad.properties)
    σ_ana = -k̂ * Δθ
    # mean |traction| vs analytical
    err = 0.0
    for i in 1:dad.n
        t_ana = -k̂ * Δθ * dad.Normal[i]
        err = max(err, abs(dad.traction[2i-1] - t_ana[1]),
                       abs(dad.traction[2i]   - t_ana[2]))
    end
    return (; dad, u, t_hg, t_s, err_rel = err / abs(σ_ana), σ_ana, n = 2dad.n)
end

"""
Example C — quadratic-like thermal field θ = θ0 + β y^2  (non-uniform → DIBEM)
"""
function example_quadratic_theta(ndiv; method = :dense, θ0 = 10.0, β = 40.0, kw...)
    dad = make_elast_dad(ndiv; n_int_side = max(3, ndiv ÷ 5), nome = "th_q_$(method)_$ndiv",
        E = 1e4, ν = 0.3, α = 1e-5, plane_strain = true)
    fill!(dad.BC, 0); fill!(dad.BV, 0.0)
    # rollers: bottom uy=0, left ux=0, others free-ish with weak supports
    for i in 1:dad.n
        x, y = dad.Nodes[i]
        if y < 1e-9
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0
            dad.BC[2i]   = 0; dad.BV[2i]   = 0
        elseif abs(x) < 1e-9
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0
            dad.BC[2i]   = 1; dad.BV[2i]   = 0
        else
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0
            dad.BC[2i]   = 1; dad.BV[2i]   = 0
        end
    end
    θfun = (x, y) -> θ0 + β * y^2
    t0 = time()
    H_G_full_direct(dad; npg = 10, threaded = false)
    t_hg = time() - t0
    t0 = time()
    DIBEM(dad; method = method, kw...)
    t_m = time() - t0
    t0 = time()
    u = solve_thermoelastic!(dad; θ = θfun)
    t_s = time() - t0
    return (; dad, u, t_hg, t_m, t_s, mem = mem_D_MB(dad, dad.M), n = 2dad.nt)
end

# --- run examples on a medium mesh, all backends ---
const EX_NDIV = 16

function run_examples()
    ex_rows = NamedTuple[]
    refA = nothing
    refC = nothing

    println("\n  Example A — self weight")
    for m in METHODS
        try
            kw = m.name === :dense ? (;) : m.kw
            sol = example_selfweight(EX_NDIV; method = m.name, kw...)
            if refA === nothing
                refA = sol
                err = 0.0
            else
                ncmp = min(length(sol.u), length(refA.u))
                err = norm(sol.u[1:ncmp] - refA.u[1:ncmp]) / max(norm(refA.u[1:ncmp]), eps())
            end
            @printf("    %-8s  n=%d  t_M=%.3fs  t_solve=%.3fs  mem=%.2fMB  err=%.2e\n",
                m.name, sol.n, sol.t_m, sol.t_s, sol.mem, err)
            push!(ex_rows, (; example = "selfweight", method = String(m.name), n = sol.n,
                t_M = sol.t_m, t_solve = sol.t_s, mem_MB = sol.mem, rel_err = err))
        catch e
            @warn "selfweight $(m.name) failed" exception = e
        end
    end

    println("\n  Example B — uniform temperature (no DIBEM)")
    solB = example_uniform_theta(EX_NDIV)
    @printf("    dense-BC  n=%d  t_HG=%.3fs  t_solve=%.3fs  traction_err=%.2e\n",
        solB.n, solB.t_hg, solB.t_s, solB.err_rel)
    push!(ex_rows, (; example = "uniform_theta", method = "boundary_only", n = solB.n,
        t_M = 0.0, t_solve = solB.t_s, mem_MB = 0.0, rel_err = solB.err_rel))

    println("\n  Example C — quadratic temperature (DIBEM)")
    for m in METHODS
        try
            kw = m.name === :dense ? (;) : m.kw
            sol = example_quadratic_theta(EX_NDIV; method = m.name, kw...)
            if refC === nothing
                refC = sol
                err = 0.0
            else
                ncmp = min(length(sol.u), length(refC.u))
                err = norm(sol.u[1:ncmp] - refC.u[1:ncmp]) / max(norm(refC.u[1:ncmp]), eps())
            end
            @printf("    %-8s  n=%d  t_M=%.3fs  t_solve=%.3fs  mem=%.2fMB  err=%.2e\n",
                m.name, sol.n, sol.t_m, sol.t_s, sol.mem, err)
            push!(ex_rows, (; example = "quadratic_theta", method = String(m.name), n = sol.n,
                t_M = sol.t_m, t_solve = sol.t_s, mem_MB = sol.mem, rel_err = err))
        catch e
            @warn "quadratic_theta $(m.name) failed" exception = e
        end
    end

    open(joinpath(OUTDIR, "results_examples.csv"), "w") do io
        println(io, "example,method,n,t_M,t_solve,mem_MB,rel_err")
        for r in ex_rows
            @printf(io, "%s,%s,%d,%.6e,%.6e,%.6e,%.6e\n",
                r.example, r.method, r.n, r.t_M, r.t_solve, r.mem_MB, r.rel_err)
        end
    end

    # Bar charts for example A and C
    function bar_example(exname, fname)
        sub = filter(r -> r.example == exname, ex_rows)
        isempty(sub) && return
        xt = (1:length(sub), [r.method for r in sub])
        bcols = [get(cols, r.method, :gray) for r in sub]
        ax1 = bar(1:length(sub), [r.t_M for r in sub]; title = "$exname — DIBEM assembly",
            ylabel = "time [s]", xticks = xt, color = bcols, legend = false, framestyle = :box)
        ax2 = bar(1:length(sub), [max(r.mem_MB, 1e-6) for r in sub]; title = "memory",
            ylabel = "MB", xticks = xt, color = bcols, legend = false, framestyle = :box)
        fig = plot(ax1, ax2; layout = (1, 2), size = (520, 360), background_color = :white)
        savefig(fig, joinpath(FIGDIR, fname * ".pdf"))
        savefig(fig, joinpath(FIGDIR, fname * ".png"))
    end
    bar_example("selfweight", "fig_ex_selfweight")
    bar_example("quadratic_theta", "fig_ex_quadratic_theta")

    # Displacement magnitude for dense selfweight (illustration)
    try
        sol = refA === nothing ? example_selfweight(12; method = :dense) : refA
        dad = sol.dad
        xs = [p[1] for p in dad.Nodes]
        ys = [p[2] for p in dad.Nodes]
        umag = [hypot(sol.u[2i-1], sol.u[2i]) for i in 1:dad.n]
        fig = scatter(xs, ys; zcolor = umag, c = :viridis, markersize = 8,
            xlabel = L"x", ylabel = L"y", title = "Self-weight |u| (dense DIBEM)",
            aspect_ratio = :equal, colorbar_title = L"|u|", legend = false,
            size = (420, 400), framestyle = :box, background_color = :white)
        savefig(fig, joinpath(FIGDIR, "fig_selfweight_u.pdf"))
        savefig(fig, joinpath(FIGDIR, "fig_selfweight_u.png"))
    catch e
        @warn "selfweight plot failed" exception = e
    end
    return ex_rows
end

ex_rows = run_examples()

println("\nDone.")
println("Figures → $FIGDIR")
println("CSV      → $OUTDIR")
