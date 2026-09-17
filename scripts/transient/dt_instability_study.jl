# DIBEM PHS3+poly then MMM: refine Δt until blow-up, 3 meshes, one JSON per problem.
#   julia --project=. scripts/dt_instability_study.jl
#   STUDY_DT_OUT=results/dt_study
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, JSON, Dates

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))
include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
include(datadir("elastico", "iso", "elastodynamics_problems.jl"))
include(raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\dibem elast transient\julia\ChiappaWave.jl")
using .ChiappaWave

const NPG = 12
const OUTDIR = get(ENV, "STUDY_DT_OUT", joinpath(projectdir(), "results", "dt_study"))
const MESH_TAGS = (200, 500, 1000)
const NTS_LIST = (20, 40, 80, 160, 320, 640, 1280, 2560)
const RBF = PHS(3; poly_deg=1)
function _parse_problems()
    s = strip(get(ENV, "STUDY_DT_PROBLEMS", ""))
    isempty(s) || return Tuple(Symbol(strip(p)) for p in split(s, ','; keepempty=false))
    return (:bar_sudden, :toe_cantilever, :toe_beam_uniform, :chiappa_ex1)
end
const PROBLEMS = _parse_problems()

# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------
function _chiappa_mesh(ndiv)
    return mesh_short_beam(1.0, 1.0; ndivx=ndiv, ndivy=ndiv, nome="chiappa_ex1_$(ndiv)",
        leftbc="0;0;1;0", rightbc="0;0;1;0", botbc="1;0;0;0", topbc="1;0;0;0")
end

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

function build_case(name::Symbol, mesh_tag::Int)
    if name === :chiappa_ex1
        ndiv = ELASTO_MESH_NDIV[mesh_tag]
        imp = bulk_impulse()
        pr = build_problem(imp, 80, 80)
        mat = pr.mat
        msh = _chiappa_mesh(ndiv)
        dad = format2d(msh, Elasticity(E=mat.E, nu=mat.ν, rho=mat.ρ; plane_strain=true);
            tipo=1, pontointerno=true)
        probe = Point2D(0.25, 0.25)
        tf = 4.7e-4
        ana_hist = ts -> collect(time_history(pr, probe[1], probe[2], ts)[1])
        du0 = _patch_du0(dad, imp)
        return dad, (; name, mesh_tag, T=imp.a / mat.cL, tf, probe, comp=1,
            ana_kind=:disp, δ=NaN, amp=NaN, ana_hist=ana_hist, du0=du0,
            notes="Chiappa Ex1 rollers + patch ú0")
    end
    dad, meta = elastodynamics_problem(name; mesh_tag=mesh_tag)
    du0 = zeros(2 * dad.nt)
    return dad, merge(meta, (; du0=du0))
end

function _probe_row(dad, probe, comp)
    pts = all_points(dad)
    ip = argmin(norm(q - probe) for q in pts)
    row = 2 * (ip - 1) + comp
    if row > 2 * dad.nt
        ip = argmin(norm(dad.Nodes[i] - probe) for i in 1:dad.n)
        row = 2 * (ip - 1) + comp
    end
    return ip, pts[min(ip, length(pts))], row
end

function _target(meta, t)
    if hasproperty(meta, :ana_hist)
        return Float64.(meta.ana_hist(t))
    end
    if meta.ana_kind === :disp && hasproperty(meta, :ana)
        return Float64[float(meta.ana(meta.probe, ti)) for ti in t]
    end
    hasproperty(meta, :δ) && isfinite(meta.δ) || return fill(NaN, length(t))
    return float(meta.δ) .* (1 .- cos.(2π .* t ./ float(meta.T)))
end

function _ref_amp(meta, ua)
    hasproperty(meta, :amp) && isfinite(meta.amp) && return abs(float(meta.amp))
    return isempty(ua) || !all(isfinite, ua) ? 0.0 : maximum(abs, ua)
end

function _is_blowup(un, ref)
    isempty(un) && return true
    !all(isfinite, un) && return true
    return maximum(abs, un) > 50 * (1 + ref)
end

function _metrics(meta, t, un, ua)
    rec = Dict{String,Any}(
        "rel_l2" => all(isfinite, ua) && !isempty(un) ? (norm(ua) > 0 ? norm(un .- ua) / norm(ua) : NaN) : NaN,
        "max_num" => isempty(un) ? NaN : maximum(abs, un),
        "max_ana" => isempty(ua) ? NaN : maximum(abs, ua),
    )
    if hasproperty(meta, :δ) && isfinite(meta.δ) && !isempty(un)
        s = abs(minimum(un)) > abs(maximum(un)) ? -1.0 : 1.0
        ap = beam_amp_period(t, s .* un; minfrac=0.25)
        rec["u_peak"] = ap.u_peak
        rec["T_num"] = ap.T_num
        rec["amp_ratio"] = ap.u_peak / (float(meta.amp) + eps())
        rec["T_ratio"] = ap.T_num / (float(meta.T) + eps())
    end
    return rec
end

function _step_newmark(dad0, meta, dt, tf)
    dad = deepcopy(dad0)
    U = solve_Newmark(dad, dt, tf; du0=meta.du0, β=1 / 4, γ=1 / 2)
    t = collect(dad.time)
    _, _, row = _probe_row(dad, meta.probe, meta.comp)
    un = collect(U[row, :])
    if !isempty(un) && abs(minimum(un)) > abs(maximum(un))
        un .*= -1
    end
    return t, un
end

function _step_mmm(dad0, meta, dt, tf, basis)
    dad = deepcopy(dad0)
    U, t, _ = solve_mmm!(dad, dt, tf; basis=basis, du0=meta.du0, alg=:houbolt)
    _, _, row = _probe_row(dad, meta.probe, meta.comp)
    un = collect(U[row, :])
    if !isempty(un) && abs(minimum(un)) > abs(maximum(un))
        un .*= -1
    end
    return collect(t), un
end

function _one_dt(stepper, dad0, meta, nts, basis)
    dt = meta.tf / nts
    t0 = time()
    rec = try
        t, un = stepper === :newmark ? _step_newmark(dad0, meta, dt, meta.tf) :
            _step_mmm(dad0, meta, dt, meta.tf, basis)
        ua = _target(meta, t)
        ref = _ref_amp(meta, ua)
        status = _is_blowup(un, ref) ? "blowup" : "ok"
        merge(Dict{String,Any}(
            "stepper" => string(stepper),
            "nts" => nts,
            "dt" => dt,
            "t" => t,
            "u_num" => un,
            "u_ana" => ua,
            "status" => status,
            "error_msg" => "",
            "runtime_s" => time() - t0,
        ), _metrics(meta, t, un, ua))
    catch e
        Dict{String,Any}(
            "stepper" => string(stepper),
            "nts" => nts,
            "dt" => dt,
            "t" => Float64[],
            "u_num" => Float64[],
            "u_ana" => Float64[],
            "status" => "error",
            "error_msg" => sprint(showerror, e),
            "runtime_s" => time() - t0,
            "rel_l2" => NaN,
            "max_num" => NaN,
        )
    end
    return rec
end

function sweep_stepper(stepper, dad0, meta, basis)
    runs = Any[]
    dt_last_ok = NaN
    dt_first_bad = NaN
    for nts in NTS_LIST
        rec = _one_dt(stepper, dad0, meta, nts, basis)
        push!(runs, rec)
        ar = get(rec, "amp_ratio", NaN)
        @printf("    %s  nts=%4d  dt=%.3e  status=%-7s  max=%.3e  rel=%.3e  amp/2δ=%s  %.1fs\n",
            stepper, nts, rec["dt"], rec["status"], rec["max_num"], rec["rel_l2"],
            isfinite(ar) ? @sprintf("%.3f", ar) : "—", rec["runtime_s"])
        if rec["status"] == "ok"
            dt_last_ok = rec["dt"]
        elseif !isfinite(dt_first_bad)
            dt_first_bad = rec["dt"]
            stepper === :newmark && break
        end
    end
    return Dict{String,Any}(
        "stepper" => string(stepper),
        "dt_last_ok" => dt_last_ok,
        "dt_first_bad" => dt_first_bad,
        "runs" => runs,
    )
end

function save_problem(name, payload)
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, string(name) * ".json")
    tmp = path * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, payload, 2)
    end
    mv(tmp, path; force=true)
    println("  wrote ", path)
    return path
end

function run_problem(name::Symbol)
    println("\n======== ", name, " ========")
    meshes = Any[]
    payload = Dict{String,Any}(
        "problem" => string(name),
        "written" => string(now()),
        "rbf" => "PHS(3; poly_deg=1)",
        "nts_list" => collect(NTS_LIST),
        "meshes" => meshes,
    )
    for tag in MESH_TAGS
        println("\n--- mesh ", tag, " ---")
        dad0, meta = try
            build_case(name, tag)
        catch e
            @error "build failed" name tag exception=(e, catch_backtrace())
            continue
        end
        @printf("  n=%d ni=%d nt=%d  T=%.4g tf=%.4g\n",
            dad0.n, dad0.ni, dad0.nt, meta.T, meta.tf)
        H_G_full_direct(dad0; npg=NPG, threaded=false)
        DIBEM(dad0; method=:dense, rbf=RBF, centers=:collocation, npg=NPG)
        basis = try
            sys = build_modal_system(dad0)
            ωmax = isfinite(meta.T) && meta.T > 0 ? 40 * 2π / meta.T : Inf
            modal_analysis_mmm(sys; ωmax=ωmax)
        catch e
            @error "MMM basis failed" exception=(e, catch_backtrace())
            nothing
        end
        nmmm = basis === nothing ? 0 : length(basis.ω)
        @printf("  MMM modes kept: %d\n", nmmm)
        mesh = Dict{String,Any}(
            "mesh_tag" => tag,
            "n" => dad0.n,
            "ni" => dad0.ni,
            "nt" => dad0.nt,
            "T" => meta.T,
            "tf" => meta.tf,
            "delta" => hasproperty(meta, :δ) ? meta.δ : NaN,
            "amp" => hasproperty(meta, :amp) ? meta.amp : NaN,
            "probe" => [meta.probe[1], meta.probe[2]],
            "notes" => meta.notes,
            "mmm_nmodes" => nmmm,
        )
        steppers = let s = strip(get(ENV, "STUDY_DT_STEPPERS", "newmark,mmm"))
            Tuple(Symbol(strip(p)) for p in split(s, ','; keepempty=false))
        end
        if :newmark in steppers
            mesh["newmark"] = sweep_stepper(:newmark, dad0, meta, basis)
        elseif isfile(joinpath(OUTDIR, string(name) * ".json"))
            old = JSON.parsefile(joinpath(OUTDIR, string(name) * ".json"))
            om = findfirst(x -> x["mesh_tag"] == tag, old["meshes"])
            om !== nothing && (mesh["newmark"] = old["meshes"][om]["newmark"])
        end
        if :mmm in steppers
            mesh["mmm"] = basis === nothing ? Dict("status" => "error", "runs" => []) :
                sweep_stepper(:mmm, dad0, meta, basis)
        end
        push!(meshes, mesh)
        save_problem(name, payload)
    end
    save_problem(name, payload)
    return payload
end

function main()
    mkpath(OUTDIR)
    println("dt instability study → ", OUTDIR)
    println("DIBEM ", RBF, "  then MMM (alg=:houbolt)")
    println("nts = ", NTS_LIST, "  meshes = ", MESH_TAGS)
    for name in PROBLEMS
        run_problem(name)
    end
    println("\ndone")
    return nothing
end

main()
