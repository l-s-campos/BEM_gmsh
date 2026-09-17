# Mesh × Δt elastodynamics study: cell / DRM / DIBEM.
#   julia --project=. scripts/elastodynamics_sweep.jl
#   STUDY_OUT=...  STUDY_SMOKE=1  (only bar_sudden, mesh 200, nts=40)
#   STUDY_PROBLEMS=toe_cantilever,toe_beam_uniform
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Dates
using JSON

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))
include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
include(datadir("elastico", "iso", "plate_with_hole.jl"))
include(datadir("elastico", "iso", "elastodynamics_problems.jl"))

const OUTDIR = get(ENV, "STUDY_OUT", joinpath(projectdir(), "results"))
mkpath(OUTDIR)
const JSONL = joinpath(OUTDIR, "elastodynamics_sweep.jsonl")
const JSONOUT = joinpath(OUTDIR, "elastodynamics_sweep.json")
const SMOKE = get(ENV, "STUDY_SMOKE", "0") == "1"
const NPG = 12
const NTS = SMOKE ? (40,) : (40, 80, 160, 320, 640, 1280)
const MESH_TAGS = SMOKE ? (200,) : (200, 500, 1000)
function _parse_problems()
    s = strip(get(ENV, "STUDY_PROBLEMS", ""))
    isempty(s) || return Tuple(Symbol(strip(p)) for p in split(s, ','; keepempty=false))
    return SMOKE ? (:bar_sudden,) : ELASTO_PROBLEMS
end
const PROBLEMS = _parse_problems()

const MASS_METHODS = (
    (key=:cell, rbf=nothing, poly_deg=nothing),
    (key=:drm, rbf=nothing, poly_deg=-1),
    (key=:drm_poly, rbf=nothing, poly_deg=1),
    (key=:dibem_phs1, rbf=PHS(1; poly_deg=-1), poly_deg=-1),
    (key=:dibem_phs1_poly, rbf=PHS(1; poly_deg=1), poly_deg=1),
    (key=:dibem_phs3, rbf=PHS(3; poly_deg=-1), poly_deg=-1),
    (key=:dibem_phs3_poly, rbf=PHS(3; poly_deg=1), poly_deg=1),
)

function run_key(problem, mesh_tag, method, nts)
    return "$(problem)|$(mesh_tag)|$(method)|$(nts)"
end

function load_done(path)
    done = Set{String}()
    isfile(path) || return done
    for line in eachline(path)
        isempty(strip(line)) && continue
        rec = JSON.parse(line)
        rec["status"] == "ok" || continue
        push!(done, run_key(rec["problem"], rec["mesh_tag"], rec["method"], rec["nts"]))
    end
    return done
end

function append_jsonl(path, rec)
    open(path, "a") do io
        println(io, JSON.json(rec))
    end
    return nothing
end

function fold_json(jsonl, jsonout)
    runs = Any[]
    isfile(jsonl) || return nothing
    for line in eachline(jsonl)
        isempty(strip(line)) && continue
        push!(runs, JSON.parse(line))
    end
    open(jsonout, "w") do io
        JSON.print(io, Dict("meta" => Dict(
                "written" => string(now()),
                "nts" => collect(NTS),
                "meshes" => collect(MESH_TAGS),
                "problems" => collect(PROBLEMS),
            ), "runs" => runs), 2)
    end
    return length(runs)
end

function apply_mass!(dad, spec)
    k = spec.key
    if k === :cell
        build_cell_mass(dad; npg=NPG)
    elseif k === :drm
        build_drm_matrices(dad; kernel=:r, poly_deg=-1, npg=NPG)
    elseif k === :drm_poly
        build_drm_matrices(dad; kernel=:r, poly_deg=1, npg=NPG)
    else
        DIBEM(dad; method=:dense, rbf=spec.rbf, centers=:collocation, npg=NPG)
    end
    return dad
end

function n_cells_of(dad)
    has_cache(dad, :cells) || return length(extract_domain_cells(dad))
    return length(dad.cells)
end

function rel_l2(un, ua)
    den = norm(ua)
    return den > 0 ? norm(un .- ua) / den : NaN
end

function _target_series(meta, t)
    if meta.ana_kind === :disp
        a0 = meta.ana(meta.probe, first(t))
        a0 isa Number && return Float64[float(meta.ana(meta.probe, ti)) for ti in t]
    end
    hasproperty(meta, :δ) || return fill(NaN, length(t))
    δ, T = float(meta.δ), float(meta.T)
    return δ .* (1 .- cos.(2π .* t ./ T))
end

function probe_series(U, dad, meta)
    ip, _ = _elasto_probe_id(dad, meta.probe)
    row = 2 * (ip - 1) + meta.comp
    if row > size(U, 1)
        ip = argmin(norm(dad.Nodes[i] - meta.probe) for i in 1:dad.n)
        row = 2 * (ip - 1) + meta.comp
    end
    un = collect(U[row, :])
    if !isempty(un) && abs(minimum(un)) > abs(maximum(un))
        un .*= -1
    end
    t = collect(dad.time)
    ua = _target_series(meta, t)
    return t, un, ua
end

function make_record(meta, dad, spec, nts, dt, t, un, ua; status="ok", err="", runtime=0.0)
    rbfn = spec.rbf === nothing ? nothing : string(typeof(spec.rbf).name.name)
    rec = Dict{String,Any}(
        "problem" => string(meta.name),
        "mesh_tag" => meta.mesh_tag,
        "n" => dad.n,
        "ni" => dad.ni,
        "nt" => dad.nt,
        "n_cells" => n_cells_of(dad),
        "method" => string(spec.key),
        "rbf" => rbfn,
        "poly_deg" => spec.poly_deg,
        "T" => meta.T,
        "tf" => meta.tf,
        "nts" => nts,
        "dt" => dt,
        "probe" => [meta.probe[1], meta.probe[2]],
        "comp" => meta.comp,
        "ana_kind" => string(meta.ana_kind),
        "t" => t,
        "u_num" => un,
        "u_ana" => ua,
        "rel_l2" => (meta.ana_kind === :disp || hasproperty(meta, :δ)) && !isempty(un) && all(isfinite, ua) ?
            rel_l2(un, ua) : NaN,
        "max_num" => isempty(un) ? NaN : maximum(abs, un),
        "max_ana" => isempty(ua) ? NaN : maximum(abs, ua),
        "status" => status,
        "error_msg" => err,
        "runtime_s" => runtime,
        "notes" => meta.notes,
    )
    if hasproperty(meta, :δ)
        rec["delta"] = float(meta.δ)
        rec["amp"] = float(meta.amp)
        if !isempty(t) && !isempty(un) && status == "ok"
            ap = beam_amp_period(t, un; minfrac=0.25)
            rec["u_peak"] = ap.u_peak
            rec["t_peak"] = ap.t_peak
            rec["T_num"] = ap.T_num
            rec["npeak"] = ap.npeak
            rec["amp_ratio"] = ap.u_peak / (float(meta.amp) + eps())
            rec["T_ratio"] = ap.T_num / (float(meta.T) + eps())
        else
            rec["u_peak"] = NaN
            rec["T_num"] = NaN
            rec["amp_ratio"] = NaN
            rec["T_ratio"] = NaN
        end
    end
    return rec
end

function run_time_step!(dad, meta, spec, nts, done)
    key = run_key(meta.name, meta.mesh_tag, spec.key, nts)
    key in done && return :skip
    dt = meta.tf / nts
    t0 = time()
    rec = try
        U = solve_Houbolt(dad, dt, meta.tf)
        t, un, ua = probe_series(U, dad, meta)
        status = "ok"
        ref = hasproperty(meta, :amp) ? abs(float(meta.amp)) :
            (isempty(ua) || !all(isfinite, ua) ? 0.0 : maximum(abs, ua))
        if !all(isfinite, un) || maximum(abs, un) > 50 * (1 + ref)
            status = "blowup"
        end
        make_record(meta, dad, spec, nts, dt, t, un, ua;
            status=status, runtime=time() - t0)
    catch e
        make_record(meta, dad, spec, nts, dt, Float64[], Float64[], Float64[];
            status="error", err=sprint(showerror, e), runtime=time() - t0)
    end
    append_jsonl(JSONL, rec)
    push!(done, key)
    ar = get(rec, "amp_ratio", NaN)
    tr = get(rec, "T_ratio", NaN)
    @printf("  %s  nts=%4d  status=%s  rel=%.3e  amp/2δ=%.3f  T/T1=%.3f  %.1fs\n",
        spec.key, nts, rec["status"], rec["rel_l2"], ar, tr, rec["runtime_s"])
    return rec["status"] === "ok" ? :ok : :fail
end

function main()
    mkpath(OUTDIR)
    done = load_done(JSONL)
    println("results → ", OUTDIR, "  already ok: ", length(done))
    for pname in PROBLEMS
        for tag in MESH_TAGS
            println("\n=== ", pname, "  mesh ", tag, " ===")
            dad0, meta = try
                elastodynamics_problem(pname; mesh_tag=tag)
            catch e
                @error "problem build failed" pname tag exception=(e, catch_backtrace())
                continue
            end
            @printf("  n=%d ni=%d nt=%d  T=%.4g tf=%.4g\n",
                dad0.n, dad0.ni, dad0.nt, meta.T, meta.tf)
            try
                H_G_full_direct(dad0; npg=NPG, threaded=false)
            catch e
                @error "H/G failed" pname tag exception=(e, catch_backtrace())
                continue
            end
            for spec in MASS_METHODS
                dad = deepcopy(dad0)
                try
                    apply_mass!(dad, spec)
                catch e
                    rec = make_record(meta, dad, spec, NTS[1], meta.tf / NTS[1],
                        Float64[], Float64[], Float64[];
                        status="error", err="mass: " * sprint(showerror, e))
                    append_jsonl(JSONL, rec)
                    @error "mass failed" spec.key exception=(e, catch_backtrace())
                    continue
                end
                for nts in NTS
                    run_time_step!(dad, meta, spec, nts, done)
                end
            end
            fold_json(JSONL, JSONOUT)
        end
    end
    n = fold_json(JSONL, JSONOUT)
    println("\nwrote ", JSONOUT, "  runs=", n)
    return nothing
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "elastodynamics_sweep.jl")
    main()
end
