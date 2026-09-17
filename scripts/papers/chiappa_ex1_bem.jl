# Chiappa Example 1 (unit square, roller walls, velocity patch) with BEM mass.
# Newmark: u(0)=0, ú(0)=F1 on D. Houbolt elasticity has no ú0, so it is not used.
#   julia --project=. scripts/chiappa_ex1_bem.jl
#   CHIAPPA_NDIV=21  CHIAPPA_NTS=200
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, JSON

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
include(datadir("elastico", "iso", "elastodynamics_problems.jl"))
include(raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\dibem elast transient\julia\ChiappaWave.jl")
using .ChiappaWave

const NDIV = parse(Int, get(ENV, "CHIAPPA_NDIV", "21"))
const NTS = parse(Int, get(ENV, "CHIAPPA_NTS", "200"))
const NPG = 12
const TF = 4.7e-4
const OUT = get(ENV, "CHIAPPA_OUT",
    joinpath(projectdir(), "results", "chiappa_ex1_bem_ndiv$(NDIV)_nts$(NTS).json"))

function _chiappa_mesh(ndiv)
    return mesh_short_beam(1.0, 1.0; ndivx=ndiv, ndivy=ndiv, nome="chiappa_ex1_$(ndiv)",
        leftbc="0;0;1;0",   # ux=0, ty=0
        rightbc="0;0;1;0",
        botbc="1;0;0;0",    # tx=0, uy=0
        topbc="1;0;0;0")
end

function _chiappa_props()
    mat = steel_chiappa()
    return Elasticity(E=mat.E, nu=mat.ν, rho=mat.ρ; plane_strain=true)
end

"""ú(0): F1 êx on the Chiappa patch D, 0 elsewhere (all collocation + internals)."""
function _patch_du0(dad, p::ChiappaWave.Impulse)
    pts = all_points(dad)
    v = zeros(2 * length(pts))
    x0, x1 = p.h3, p.h3 + p.h4
    y0, y1 = p.b - p.h1 - p.h2, p.b - p.h1
    @inbounds for i in eachindex(pts)
        x, y = pts[i]
        if x0 - 1e-12 <= x <= x1 + 1e-12 && y0 - 1e-12 <= y <= y1 + 1e-12
            v[2i - 1] = p.F1
        end
    end
    return v
end

function _probe_hist(U, dad, probe; comp=1)
    pts = all_points(dad)
    ip = argmin(norm(q - probe) for q in pts)
    row = 2 * (ip - 1) + comp
    return ip, pts[ip], collect(U[row, :])
end

function _run_mass(dad, du0, spec, dt, tf, probe, ux_ana)
    t0 = time()
    if spec.key === :cell
        build_cell_mass(dad; npg=NPG)
    else
        DIBEM(dad; method=:dense, rbf=spec.rbf, centers=:collocation, npg=NPG)
    end
    t_mass = time() - t0
    t0 = time()
    U = solve_Newmark(dad, dt, tf; du0=du0, β=1 / 4, γ=1 / 2)
    t_step = time() - t0
    t = collect(dad.time)
    ip, p, ux = _probe_hist(U, dad, probe; comp=1)
    _, _, uy = _probe_hist(U, dad, probe; comp=2)
    den = norm(ux_ana)
    rel = den > 0 ? norm(ux .- ux_ana) / den : NaN
    rec = Dict{String,Any}(
        "method" => string(spec.key),
        "mass_s" => t_mass,
        "step_s" => t_step,
        "probe_used" => [p[1], p[2]],
        "probe_id" => ip,
        "ux" => ux,
        "uy" => uy,
        "rel_l2_ux" => rel,
        "max_abs_ux" => maximum(abs, ux),
        "max_abs_uy" => maximum(abs, uy),
    )
    has_cache(dad, :M) && (dad.cache.M = nothing)
    has_cache(dad, :u) && (dad.cache.u = nothing)
    has_cache(dad, :T) && (dad.cache.T = nothing)
    U = nothing
    GC.gc()
    return rec
end

function main()
    imp = bulk_impulse()
    pr = build_problem(imp, 80, 80)
    mat = pr.mat
    probe = Point2D(0.25, 0.25)
    dt = TF / NTS
    @printf("Chiappa Ex1 BEM  ndiv=%d  nts=%d  dt=%.3e  tf=%.3e\n", NDIV, NTS, dt, TF)
    @printf("  E=%.4e  nu=%.4f  rho=%.0f  plane strain\n", mat.E, mat.ν, mat.ρ)

    msh = _chiappa_mesh(NDIV)
    dad = format2d(msh, _chiappa_props(); tipo=1, pontointerno=true)
    H_G_full_direct(dad; npg=NPG, threaded=false)
    du0 = _patch_du0(dad, imp)
    n_patch = count(!=(0), du0)
    @printf("  n=%d ni=%d nt=%d  patch DOFs=%d  (expect interiors in D)\n",
        dad.n, dad.ni, dad.nt, n_patch)

    t = collect(0.0:dt:TF)
    @printf("  series history at P=(0.25,0.25)  %d samples\n", length(t))
    ux_ana, uy_ana = time_history(pr, 0.25, 0.25, t)

    methods = (
        (key=:cell, rbf=nothing),
        (key=:dibem_phs3_poly, rbf=PHS(3; poly_deg=1)),
    )
    runs = Any[]
    for spec in methods
        println("\n--- ", spec.key, " ---")
        rec = try
            _run_mass(dad, du0, spec, dt, TF, probe, ux_ana)
        catch e
            @error "method failed" spec.key exception=(e, catch_backtrace())
            Dict{String,Any}("method" => string(spec.key),
                "status" => "error", "error_msg" => sprint(showerror, e))
        end
        haskey(rec, "rel_l2_ux") && @printf(
            "  mass=%.1fs  step=%.1fs  rel_L2 ux=%.3e  max|ux|=%.3e  ana=%.3e\n",
            rec["mass_s"], rec["step_s"], rec["rel_l2_ux"],
            rec["max_abs_ux"], maximum(abs, ux_ana))
        push!(runs, rec)
        payload_now = Dict{String,Any}(
            "problem" => "chiappa_ex1",
            "ndiv" => NDIV, "n" => dad.n, "ni" => dad.ni, "nt" => dad.nt,
            "nts" => NTS, "dt" => dt, "tf" => TF,
            "probe" => [0.25, 0.25],
            "t" => collect(t),
            "ux_ana" => collect(ux_ana),
            "uy_ana" => collect(uy_ana),
            "max_abs_ux_ana" => maximum(abs, ux_ana),
            "runs" => runs,
        )
        mkpath(dirname(OUT))
        open(OUT, "w") do io
            JSON.print(io, payload_now, 2)
        end
        println("  checkpoint ", OUT)
    end

    payload = Dict{String,Any}(
        "problem" => "chiappa_ex1",
        "ndiv" => NDIV,
        "n" => dad.n,
        "ni" => dad.ni,
        "nt" => dad.nt,
        "nts" => NTS,
        "dt" => dt,
        "tf" => TF,
        "probe" => [0.25, 0.25],
        "t" => collect(t),
        "ux_ana" => collect(ux_ana),
        "uy_ana" => collect(uy_ana),
        "max_abs_ux_ana" => maximum(abs, ux_ana),
        "runs" => runs,
    )
    mkpath(dirname(OUT))
    open(OUT, "w") do io
        JSON.print(io, payload, 2)
    end
    println("\nwrote ", OUT)
    return nothing
end

main()
