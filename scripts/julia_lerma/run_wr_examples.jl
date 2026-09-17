# Run CONTACT mbench_a22 left/right (Y=0:0.5:10) and wheelflat vs .ref_out.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra, Printf, Statistics, FFTW, Plots
gr()
default(size=(780, 460), linewidth=1.6, legendfontsize=8, guidefontsize=11,
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

function _nums(ln)
    out = Float64[]
    for s in split(ln)
        t = replace(s, r"[dD]$" => "")
        v = tryparse(Float64, t)
        v !== nothing && push!(out, v)
    end
    return out
end

"""Parse CONTACT module-1 pose lines: S,Y,FZ,ROLL,YAW,PITCH then VS,...,VPITCH."""
function parse_wr_inp(path)
    lines = readlines(path)
    recs = NamedTuple[]
    i = 1
    while i < length(lines)
        ln = lines[i]
        if occursin("S, Y, FZ", ln) || occursin("S, Y, Z", ln) || occursin("ROLL, YAW, PITCH", ln)
            a = _nums(ln)
            b = _nums(i < length(lines) ? lines[i+1] : "")
            if length(a) >= 6 && length(b) >= 6
                yaw = a[5]
                pitch = a[6]
                # CONTACT trailing `d` means degrees
                if occursin(r"-?[0-9.]+d", ln)
                    occursin(r"PITCH", ln) && (pitch = deg2rad(pitch))
                end
                push!(recs, (; s=a[1], y=a[2], fz=a[3], roll=a[4], yaw, pitch,
                               vs=b[1], vy=b[2], vz=b[3], vroll=b[4], vyaw=b[5], vpitch=b[6]))
            end
        end
        i += 1
    end
    return recs
end

function parse_ref_out(path)
    recs = NamedTuple[]
    y = NaN; z = NaN; pitch = NaN; fn=NaN; fx=NaN; fs=NaN; pmax=NaN; ycp=NaN; delt=NaN
    lines = readlines(path)
    i = 1
    while i <= length(lines)
        ln = lines[i]
        if occursin("S_WS", ln) && occursin("Y_WS", ln) && occursin("Z_WS", ln) &&
           !occursin("VX_WS", ln) && i < length(lines)
            n = _nums(lines[i+1])
            length(n) >= 6 && (y = n[2]; z = n[3]; pitch = n[6])
        end
        if occursin("YCP(TR)", ln) && i < length(lines)
            n = _nums(lines[i+1])
            length(n) >= 4 && (ycp = n[2]; delt = n[4])
        end
        if occursin("FX/FSTAT/FN", ln) && occursin("PMAX", ln) && i < length(lines)
            n = _nums(lines[i+1])
            length(n) >= 5 && (pmax = n[5])
        end
        if occursin("TOTAL FORCES, TORSIONAL", ln)
            # header line is i+1 ("FN FX FS ..."), numbers on i+2
            for j in (i+1, i+2)
                j > length(lines) && break
                n = _nums(lines[j])
                if length(n) >= 3 && abs(n[1]) > 1.0  # FN is thousands of N, not a label
                    fn, fx, fs = n[1], n[2], n[3]
                    push!(recs, (; y, z, pitch, fn, fx, fs, pmax, ycp, delt))
                    break
                end
            end
        end
        i += 1
    end
    return recs
end

function run_poses(rail, wheel, trk, poses; side=:left, dx=0.2, ds=0.2,
                   nom_radius=460.0, z0=0.198, label="")
    out = NamedTuple[]
    z = z0
    for (k, po) in enumerate(poses)
        ws = WheelsetGeom(nom_radius=nom_radius, y=po.y, z=z, roll=po.roll, yaw=po.yaw,
                          pitch=po.pitch, vs=po.vs, vy=po.vy, vz=po.vz, vroll=po.vroll,
                          vyaw=po.vyaw, vpitch=po.vpitch, fz=po.fz)
        print(@sprintf("  %s %2d/%d Y=%5.1f yaw=%6.2f mrad pitch=%6.1f° ... ",
                       label, k, length(poses), po.y, po.yaw*1e3, rad2deg(po.pitch)))
        t0 = time()
        res = try
            solve_wheel_rail(trk, ws, rail, wheel; side=side, force=true, dx=dx, ds=ds,
                             maxit=8, maxiter=350, rtol=0.025)
        catch e
            println("ERROR ", sprint(showerror, e))
            continue
        end
        dt = time() - t0
        if isempty(res.patches) || res.FZ_tr < 1
            println(@sprintf("NO CONTACT z=%.4f (%.1fs)", res.z_ws, dt))
            z = z0
            continue
        end
        p = res.patches[1]
        if p.FN > 8 * po.fz
            println(@sprintf("OVERLOAD z=%.4f FN=%.0f — reset z (%.1fs)", res.z_ws, p.FN, dt))
            z = z0
            continue
        end
        z = res.z_ws
        println(@sprintf("z=%.4f FN=%.0f FX=%.1f FS=%.1f pmax=%.0f YCP=%.2f (%.1fs)",
                         z, p.FN, p.FX, p.FS, p.pmax, p.YCP_tr, dt))
        push!(out, (; y=po.y, yaw=po.yaw, pitch=po.pitch, z, fn=p.FN, fx=p.FX, fs=p.FS,
                     pmax=p.pmax, ycp=p.YCP_tr, delt=p.DELT, fz=res.FZ_tr, ncon=p.ncon))
    end
    return out
end

function compare_table(name, ours, ref; x=:y, xlab="Y_ws [mm]")
    n = min(length(ours), length(ref))
    n == 0 && return
    println("\n  ", name, "  vs CONTACT  (n=", n, ")")
    println("    x        FN_o/FN_c     FX_o     FX_c    FS_o    FS_c   pmax_o/pmax_c")
    for i in 1:n
        o, r = ours[i], ref[i]
        xv = getfield(o, x)
        println(@sprintf("  %7.2f  %7.0f/%-7.0f  %7.1f %7.1f  %7.1f %7.1f  %6.0f/%-6.0f",
                         xv, o.fn, r.fn, o.fx, r.fx, o.fs, r.fs, o.pmax, r.pmax))
    end
    xo = [getfield(o, x) for o in ours]
    xr = x === :pitch ? [r.pitch for r in ref[1:n]] : [r.y for r in ref[1:n]]
    plt = plot(layout=(2,2), size=(900, 640),
               plot_title=name)
    plot!(plt[1], xr, [r.fn for r in ref[1:n]]; label="CONTACT", xlabel=xlab, ylabel="FN [N]")
    plot!(plt[1], xo, [o.fn for o in ours]; label="BEM", ls=:dash)
    plot!(plt[2], xr, [r.fx for r in ref[1:n]]; label="CONTACT", xlabel=xlab, ylabel="FX [N]")
    plot!(plt[2], xo, [o.fx for o in ours]; label="BEM", ls=:dash)
    plot!(plt[3], xr, [r.fs for r in ref[1:n]]; label="CONTACT", xlabel=xlab, ylabel="FS [N]")
    plot!(plt[3], xo, [o.fs for o in ours]; label="BEM", ls=:dash)
    plot!(plt[4], xr, [r.pmax for r in ref[1:n]]; label="CONTACT", xlabel=xlab, ylabel="pmax [MPa]")
    plot!(plt[4], xo, [o.pmax for o in ours]; label="BEM", ls=:dash)
    slug = replace(lowercase(name), r"[^a-z0-9]+" => "_")
    _savefig(plt, slug * ".png")
end

# ---------------------------------------------------------------------------
rail = read_rail_profile(joinpath(DATA, "MBench_UIC60_v3.prr"))
wheel = read_wheel_profile(joinpath(DATA, "MBench_S1002_v3.prw"))
trk = TrackGeom()

println("="^72)
println(" mbench_a22_left  Y=0:0.5:10 mm  (yaw 0:1.2:24 mrad)")
println("="^72)
left_poses = parse_wr_inp(joinpath(CEX, "mbench_a22_left.inp"))
println("  parsed ", length(left_poses), " poses")
left = run_poses(rail, wheel, trk, left_poses; side=:left, dx=0.2, ds=0.2,
                 z0=0.198, label="L")
left_ref = parse_ref_out(joinpath(CEX, "mbench_a22_left.ref_out"))
compare_table("mbench A-2.2 left", left, left_ref)

println("\n", "="^72)
println(" mbench_a22_right  Y=0:0.5:10 mm")
println("="^72)
right_poses = parse_wr_inp(joinpath(CEX, "mbench_a22_right.inp"))
println("  parsed ", length(right_poses), " poses")
right = run_poses(rail, wheel, trk, right_poses; side=:right, dx=0.2, ds=0.2,
                  z0=0.198, label="R")
right_ref = parse_ref_out(joinpath(CEX, "mbench_a22_right.ref_out"))
compare_table("mbench A-2.2 right", right, right_ref)

println("\n", "="^72)
println(" wheelflat.inp  pitch -25° : -1° : -50°   FZ=125 kN")
println("="^72)
rail_f = read_rail_profile(joinpath(DATA, "r300_wide.prr"))
wheel_f = read_wheel_profile(joinpath(DATA, "S1002_flat.slcw"))
println("  rail n=", length(rail_f.spl), "  wheel slices=", length(wheel_f.slices))
trk_f = TrackGeom(cant=0.020)
flat_poses = parse_wr_inp(joinpath(CEX, "wheelflat.inp"))
println("  parsed ", length(flat_poses), " poses")
# CONTACT writes -25.0d as degrees; parser may leave radians if the `d` was stripped
if !isempty(flat_poses) && abs(flat_poses[1].pitch) > 1
    flat_poses = [(; p..., pitch=deg2rad(p.pitch)) for p in flat_poses]
end
flat = run_poses(rail_f, wheel_f, trk_f, flat_poses; side=:right, dx=0.4, ds=0.4,
                 nom_radius=490.0, z0=0.37, label="F")
flat_ref = parse_ref_out(joinpath(CEX, "wheelflat.ref_out"))
compare_table("wheelflat", flat, flat_ref; x=:pitch, xlab="pitch [rad]")

println("\ndone. figures in ", FIG)
