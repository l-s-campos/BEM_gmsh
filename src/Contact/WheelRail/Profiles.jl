# Profile I/O and cubic splines. Lengths in mm (CONTACT working units).

"""Natural cubic spline of a planar (y,z) polyline, parameterised by arc length `s`."""
struct ProfileSpline
    s::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    ypp::Vector{Float64}
    zpp::Vector{Float64}
end

struct WRProfile
    name::String
    is_wheel::Bool
    spl::ProfileSpline
    # variable wheel: circumferential slices (θ, ProfileSpline)
    slices::Vector{Tuple{Float64,ProfileSpline}}
end

WRProfile(name, is_wheel, spl) = WRProfile(name, is_wheel, spl, Tuple{Float64,ProfileSpline}[])

Base.length(p::ProfileSpline) = length(p.s)

function _arc_length(y, z)
    n = length(y)
    s = zeros(n)
    @inbounds for i in 2:n
        s[i] = s[i - 1] + hypot(y[i] - y[i - 1], z[i] - z[i - 1])
    end
    return s
end

"""Second derivatives for a natural cubic spline of `u(s)`."""
function _natural_pp(s, u)
    n = length(s)
    n < 3 && return zeros(n)
    h = diff(s)
    α = zeros(n)
    @inbounds for i in 2:n-1
        h[i-1] == 0 && continue
        h[i] == 0 && continue
        α[i] = 3 * ((u[i+1] - u[i]) / h[i] - (u[i] - u[i-1]) / h[i-1])
    end
    l = ones(n); μ = zeros(n); z = zeros(n)
    @inbounds for i in 2:n-1
        l[i] = 2 * (s[i+1] - s[i-1]) - h[i-1] * μ[i-1]
        l[i] == 0 && (l[i] = 1)
        μ[i] = h[i] / l[i]
        z[i] = (α[i] - h[i-1] * z[i-1]) / l[i]
    end
    pp = zeros(n)
    @inbounds for j in n-1:-1:2
        pp[j] = z[j] - μ[j] * pp[j+1]
    end
    return pp
end

function make_profile_spline(y, z)
    s = _arc_length(y, z)
    # drop duplicate s
    keep = trues(length(s))
    @inbounds for i in 2:length(s)
        keep[i] = s[i] > s[i-1] + 1e-12
    end
    s, y, z = s[keep], y[keep], z[keep]
    return ProfileSpline(s, y, z, _natural_pp(s, y), _natural_pp(s, z))
end

function _spline_eval(s, u, upp, t)
    n = length(s)
    t <= s[1] && return u[1]
    t >= s[n] && return u[n]
    i = searchsortedlast(s, t)
    i = clamp(i, 1, n - 1)
    h = s[i+1] - s[i]
    h == 0 && return u[i]
    A = (s[i+1] - t) / h
    B = (t - s[i]) / h
    return A * u[i] + B * u[i+1] + ((A^3 - A) * upp[i] + (B^3 - B) * upp[i+1]) * h^2 / 6
end

function _spline_deriv(s, u, upp, t)
    n = length(s)
    i = searchsortedlast(s, t)
    i = clamp(i, 1, n - 1)
    h = s[i+1] - s[i]
    h == 0 && return 0.0
    A = (s[i+1] - t) / h
    B = (t - s[i]) / h
    return (u[i+1] - u[i]) / h - (3A^2 - 1) / 6 * h * upp[i] + (3B^2 - 1) / 6 * h * upp[i+1]
end

eval_y(spl::ProfileSpline, t) = _spline_eval(spl.s, spl.y, spl.ypp, t)
eval_z(spl::ProfileSpline, t) = _spline_eval(spl.s, spl.z, spl.zpp, t)
eval_dy(spl::ProfileSpline, t) = _spline_deriv(spl.s, spl.y, spl.ypp, t)
eval_dz(spl::ProfileSpline, t) = _spline_deriv(spl.s, spl.z, spl.zpp, t)
eval_yz(spl::ProfileSpline, t) = (eval_y(spl, t), eval_z(spl, t))

"""Inclination `atan2(dz, dy)` (CONTACT `spline_get_alpha_at_s`)."""
eval_alpha(spl::ProfileSpline, t) = atan(eval_dz(spl, t), eval_dy(spl, t))

function eval_dzdy(spl::ProfileSpline, t)
    dy = eval_dy(spl, t)
    dz = eval_dz(spl, t)
    return dz / min(-1e-6, dy)   # CONTACT locus_prismatic: dz/dy with dy clipped negative
end

"""Arc-length `s` at a given `y` (first crossing, linear then Newton)."""
function s_at_y(spl::ProfileSpline, yq)
    y = spl.y
    n = length(y)
    # scan for a bracketing segment
    best_i, best_d = 1, abs(y[1] - yq)
    @inbounds for i in 1:n-1
        yi, yj = y[i], y[i+1]
        d = abs(yi - yq)
        d < best_d && (best_i = i; best_d = d)
        if (yi - yq) * (yj - yq) <= 0
            t = abs(yj - yi) < 1e-16 ? 0.0 : (yq - yi) / (yj - yi)
            s0 = spl.s[i] + t * (spl.s[i+1] - spl.s[i])
            # Newton polish
            for _ in 1:8
                dy = eval_dy(spl, s0)
                abs(dy) < 1e-14 && break
                s0 -= (eval_y(spl, s0) - yq) / dy
                s0 = clamp(s0, spl.s[1], spl.s[end])
            end
            return s0
        end
    end
    return spl.s[best_i]
end

function z_at_y(spl::ProfileSpline, yq)
    return eval_z(spl, s_at_y(spl, yq))
end

function yz_at_minz(spl::ProfileSpline)
    k = argmin(spl.z)
    return spl.y[k], spl.z[k], spl.s[k]
end

# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

function _skip_comment(ln)
    s = strip(ln)
    return isempty(s) || startswith(s, "!") || startswith(s, "%") || startswith(s, "#")
end

function _parse_kv(ln)
    # "key = value ! comment"
    m = match(r"^\s*([A-Za-z0-9_.]+)\s*=\s*(\S+)", ln)
    m === nothing && return nothing
    key = lowercase(m.captures[1])
    raw = m.captures[2]
    v = tryparse(Float64, raw)
    return key, (v === nothing ? raw : v)
end

function _read_yz_block(io)
    y = Float64[]; z = Float64[]
    for ln in eachline(io)
        s = strip(ln)
        (isempty(s) || startswith(s, "!") || startswith(s, "%")) && continue
        occursin("point.end", lowercase(s)) && break
        occursin("spline.end", lowercase(s)) && break
        parts = split(s)
        length(parts) < 2 && continue
        yi = tryparse(Float64, parts[1])
        zi = tryparse(Float64, parts[2])
        (yi === nothing || zi === nothing) && continue
        push!(y, yi); push!(z, zi)
    end
    return y, z
end

"""Read a SIMPACK `.prr` / `.prw` into millimetres."""
function read_simpack_profile(path; is_wheel::Bool, inp_mirror_y=0, inp_mirror_z=0, scale=1.0)
    file_mirror_y = 0
    file_mirror_z = 0
    flip_data = 0          # inversion
    shift_y = 0.0
    shift_z = 0.0
    units_len_fac = 1000.0 # default: user unit = mm
    y = Float64[]; z = Float64[]
    in_point = false
    open(path) do io
        for ln in eachline(io)
            sl = lowercase(strip(ln))
            if occursin("point.begin", sl)
                in_point = true
                y, z = _read_yz_block(io)
                break
            end
            kv = _parse_kv(ln)
            kv === nothing && continue
            key, val = kv
            key == "mirror.y" && (file_mirror_y = Int(val))
            key == "mirror.z" && (file_mirror_z = Int(val))
            key == "inversion" && (flip_data = Int(val))
            key == "shift.y" && (shift_y = Float64(val))
            key == "shift.z" && (shift_z = Float64(val))
            key == "units.len.f" && (units_len_fac = Float64(val))
        end
    end
    isempty(y) && error("no profile points in $path")
    y = y .+ shift_y
    z = z .+ shift_z
    if inp_mirror_y + file_mirror_y == 1
        y = -y
    end
    if inp_mirror_z + file_mirror_z == 1
        z = -z
    end
    # SIMPACK user → m → mm, then inp SCALE
    fac = scale * 1000.0 / units_len_fac
    y .*= fac
    z .*= fac
    if flip_data == 1
        reverse!(y); reverse!(z)
    end
    # rails: y should be ascending
    if !is_wheel && y[1] > y[end]
        reverse!(y); reverse!(z)
    end
    spl = make_profile_spline(y, z)
    return WRProfile(basename(path), is_wheel, spl)
end

"""Plain two-column `y z` slice (wheelflat `Wheel_section_*.txt`)."""
function read_ascii_profile(path; is_wheel=true, scale=1.0, mirror_y=0, mirror_z=0)
    y = Float64[]; z = Float64[]
    open(path) do io
        for ln in eachline(io)
            _skip_comment(ln) && continue
            parts = split(strip(ln))
            length(parts) < 2 && continue
            yi = tryparse(Float64, parts[1])
            zi = tryparse(Float64, parts[2])
            (yi === nothing || zi === nothing) && continue
            push!(y, yi); push!(z, zi)
        end
    end
    isempty(y) && error("no points in $path")
    mirror_y == 1 && (y = -y)
    mirror_z == 1 && (z = -z)
    y .*= scale
    z .*= scale
    return make_profile_spline(y, z)
end

"""CONTACT `.slcw` variable-wheel catalogue."""
function read_slice_catalogue(path; is_wheel=true, scale=1.0, mirror_y=0, mirror_z=0, smooth=0.0)
    dir = dirname(path)
    slices = Tuple{Float64,ProfileSpline}[]
    th_off = 0.0
    th_scale = 1.0
    nslc = 0
    open(path) do io
        # header: TH_OFFSET TH_SCALE, then NSLC
        for ln in eachline(io)
            _skip_comment(ln) && continue
            parts = split(strip(ln))
            if length(parts) >= 2 && nslc == 0 && isempty(slices)
                a = tryparse(Float64, parts[1])
                b = tryparse(Float64, parts[2])
                if a !== nothing && b !== nothing
                    th_off, th_scale = a, b
                    continue
                end
            end
            if nslc == 0
                v = tryparse(Int, parts[1])
                v !== nothing && (nslc = v)
                continue
            end
            # skip NFEAT / TH_INTPOL lines until we see a filename
            if length(parts) >= 2 && (occursin(".txt", parts[end]) || occursin("'", parts[end]) ||
                                      occursin("/", parts[end]) || occursin("\\", parts[end]))
                th = parse(Float64, parts[1]) * th_scale + th_off
                fname = strip(parts[end], ['\'', '"'])
                fpath = isabspath(fname) ? fname : joinpath(dir, fname)
                if !isfile(fpath)
                    # CONTACT examples store slices next to the slcw
                    alt = joinpath(get(ENV, "CONTACT_EXAMPLES",
                                       raw"C:\Users\ufesl\Downloads\CONTACT-main\CONTACT-main\examples"),
                                   fname)
                    isfile(alt) && (fpath = alt)
                end
                spl = read_ascii_profile(fpath; is_wheel=is_wheel, scale=scale,
                                         mirror_y=mirror_y, mirror_z=mirror_z)
                push!(slices, (th, spl))
            end
        end
    end
    isempty(slices) && error("no slices in $path")
    sort!(slices; by=first)
    # representative spline: slice nearest θ=0 (or mid)
    _, k = findmin(abs(θ) for (θ, _) in slices)
    return WRProfile(basename(path), is_wheel, slices[k][2], slices)
end

function read_rail_profile(path; kwargs...)
    ext = lowercase(splitext(path)[2])
    ext in (".prr", ".prw") && return read_simpack_profile(path; is_wheel=false, kwargs...)
    return WRProfile(basename(path), false, read_ascii_profile(path; is_wheel=false, kwargs...))
end

function read_wheel_profile(path; kwargs...)
    ext = lowercase(splitext(path)[2])
    ext == ".slcw" && return read_slice_catalogue(path; is_wheel=true, kwargs...)
    ext in (".prr", ".prw") && return read_simpack_profile(path; is_wheel=true, kwargs...)
    return WRProfile(basename(path), true, read_ascii_profile(path; is_wheel=true, kwargs...))
end

is_varprof(p::WRProfile) = !isempty(p.slices)

"""Interpolate a variable wheel to pitch `θ` (nearest-slice, then linear in θ)."""
function profile_at_theta(p::WRProfile, θ)
    isempty(p.slices) && return p.spl
    ths = first.(p.slices)
    # wrap into the catalogue span (≈[−π, π])
    span = ths[end] - ths[1]
    if span > 1e-12
        while θ < ths[1]; θ += 2π; end
        while θ > ths[end]; θ -= 2π; end
        θ = clamp(θ, ths[1], ths[end])
    end
    if θ <= ths[1]
        return p.slices[1][2]
    elseif θ >= ths[end]
        return p.slices[end][2]
    end
    i = searchsortedlast(ths, θ)
    i = clamp(i, 1, length(ths) - 1)
    t0, s0 = p.slices[i]
    t1, s1 = p.slices[i + 1]
    w = (θ - t0) / (t1 - t0)
    # resample s1 onto s0's y-grid
    y = s0.y
    z = (1 - w) .* s0.z .+ w .* [z_at_y(s1, yi) for yi in y]
    return make_profile_spline(y, z)
end
