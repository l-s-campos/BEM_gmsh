# Manchester A-2.2 (CONTACT mbench_a22_left/right) via our planar wheel–rail stack.
# Units: N, mm, MPa. Compare to CONTACT-main/examples/mbench_a22_*.ref_out.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra, Printf, Statistics, FFTW, Plots
gr()
default(size=(720, 480), linewidth=1.6, legendfontsize=8, guidefontsize=11,
        tickfontsize=9, titlefontsize=11, grid=false, framestyle=:box)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIG  = joinpath(ROOT, "plots", "julia_lerma", "wheel_rail")
const DATA = joinpath(ROOT, "data", "contact", "vollebregt")
const CEX  = get(ENV, "CONTACT_EXAMPLES",
                 raw"C:\Users\ufesl\Downloads\CONTACT-main\CONTACT-main\examples")
mkpath(FIG)

function _savefig(plt, name)
    path = joinpath(FIG, name)
    try
        isfile(path) && rm(path; force=true)
        savefig(plt, path)
        println("  wrote ", path)
    catch e
        @warn "savefig failed" name exception=e
    end
end

"""Parse CONTACT mbench .ref_out for per-case (Y, FN, FX, FS, pmax, YCP, Δ, ξx)."""
function parse_mbench_ref(path)
    recs = NamedTuple[]
    y = NaN; fn=NaN; fx=NaN; fs=NaN; pmax=NaN; ycp=NaN; delt=NaN; cksi=NaN
    lines = readlines(path)
    i = 1
    while i <= length(lines)
        ln = lines[i]
        if occursin("Y_WS", ln) && occursin("Z_WS", ln) && i < length(lines)
            nums = Float64[]
            for s in split(lines[i+1]); v = tryparse(Float64,s); v!==nothing && push!(nums,v); end
            length(nums) >= 2 && (y = nums[2])
        end
        if occursin("YCP(TR)", ln) && i < length(lines)
            nums = Float64[]
            for s in split(lines[i+1]); v = tryparse(Float64,s); v!==nothing && push!(nums,v); end
            length(nums) >= 4 && (ycp = nums[2]; delt = nums[4])
        end
        if occursin("CKSI", ln) && occursin("CPHI", ln) && i < length(lines)
            nums = Float64[]
            for s in split(lines[i+1]); v = tryparse(Float64,s); v!==nothing && push!(nums,v); end
            length(nums) >= 4 && (cksi = nums[end-2])
        end
        if occursin("FX/FSTAT/FN", ln) && occursin("PMAX", ln) && i < length(lines)
            nums = Float64[]
            for s in split(lines[i+1]); v = tryparse(Float64,s); v!==nothing && push!(nums,v); end
            length(nums) >= 5 && (pmax = nums[5])
        end
        if occursin("FN", ln) && occursin("FX", ln) && occursin("FS", ln) && i < length(lines) &&
           occursin("TOTAL FORCES, TORSIONAL", ln)
            nums = Float64[]
            for s in split(lines[i+1]); v = tryparse(Float64,s); v!==nothing && push!(nums,v); end
            if length(nums) >= 3
                fn, fx, fs = nums[1], nums[2], nums[3]
                push!(recs, (; y, fn, fx, fs, pmax, ycp, delt, cksi))
            end
        end
        i += 1
    end
    return recs
end

function run_mbench(; side=:left, ys=0.0:0.5:2.0, force=true, dx=0.25, ds=0.25)
    rail = read_rail_profile(joinpath(DATA, "MBench_UIC60_v3.prr"))
    wheel = read_wheel_profile(joinpath(DATA, "MBench_S1002_v3.prw"))
    trk = TrackGeom()
    out = NamedTuple[]
    yaw0 = 0.0
    vp0 = side === :left ? -4.34811810 : -4.34811810
    for (k, y) in enumerate(ys)
        yaw = yaw0 + (y / 0.5) * 0.0012   # A-2.2: 1.2 mrad per 0.5 mm
        ws = WheelsetGeom(y=y, z=0.1981, yaw=yaw, vs=2000.0, vpitch=vp0, fz=10000.0)
        print(@sprintf("  %s Y=%5.1f yaw=%.4f ... ", side, y, yaw))
        t0 = time()
        res = solve_wheel_rail(trk, ws, rail, wheel; side=side, force=force, dx=dx, ds=ds,
                               maxit=force ? 6 : 1, maxiter=400, rtol=0.02)
        dt = time() - t0
        p = isempty(res.patches) ? nothing : res.patches[1]
        if p === nothing
            println("NO CONTACT")
            continue
        end
        println(@sprintf("z=%.4f FN=%.0f pmax=%.1f YCP=%.2f (%.1fs)",
                         res.z_ws, p.FN, p.pmax, p.YCP_tr, dt))
        push!(out, (; y, yaw, z=res.z_ws, fn=p.FN, fx=p.FX, fs=p.FS, pmax=p.pmax,
                     ycp=p.YCP_tr, delt=p.DELT, cksi=p.ξx, fz=res.FZ_tr))
    end
    return out
end

println("=== Manchester A-2.2 left, Y=0 (N=1, FZ=10 kN) ===")
left0 = run_mbench(; side=:left, ys=[0.0], force=true, dx=0.2, ds=0.2)
if !isempty(left0)
    r = left0[1]
    println(@sprintf("  ours    FN=%.0f  FX=%.2f  FS=%.1f  pmax=%.1f  YCP=%.2f  Δ=%.4f  ξx=%.3e",
                     r.fn, r.fx, r.fs, r.pmax, r.ycp, r.delt, r.cksi))
    println("  CONTACT FN=9997  FX=0.28  FS=253.6  pmax=338.3  YCP=-751.87  Δ=0.029  ξx=-5.67e-5")
end

refp = joinpath(CEX, "mbench_a22_left.ref_out")
if isfile(refp) && get(ENV, "WR_SWEEP", "0") == "1"
    println("=== sweep vs CONTACT ref_out ===")
    ours = run_mbench(; side=:left, ys=0.0:0.5:2.0, force=true, dx=0.25, ds=0.25)
    ref = parse_mbench_ref(refp)
    if !isempty(ours) && !isempty(ref)
        n = min(length(ours), length(ref))
        plt = plot(xlabel="Y_ws [mm]", ylabel="FN [N]", title="mbench A-2.2 left FN")
        plot!(plt, [r.y for r in ref[1:n]], [r.fn for r in ref[1:n]]; label="CONTACT")
        plot!(plt, [r.y for r in ours], [r.fn for r in ours]; label="BEM", ls=:dash)
        _savefig(plt, "mbench_left_FN.png")
        plt2 = plot(xlabel="Y_ws [mm]", ylabel="pmax [MPa]", title="mbench A-2.2 left pmax")
        plot!(plt2, [r.y for r in ref[1:n]], [r.pmax for r in ref[1:n]]; label="CONTACT")
        plot!(plt2, [r.y for r in ours], [r.pmax for r in ours]; label="BEM", ls=:dash)
        _savefig(plt2, "mbench_left_pmax.png")
    end
end
