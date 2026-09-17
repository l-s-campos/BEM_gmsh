# FMM3D plane-wave M2L (`lpwrouts.f`) on equal-size cubic-octree pairs.
#
# Moments use FMM3D Y_n^m = sqrt((n-m)!/(n+m)!) P_n^m. M2X is
#   F_m(λ) = i^m Σ_n mpole(n,m) λ^n / sqrt((n-m)! (n+m)!)
# (stable polynomial in λ). Shift is diagonal in (λ,α). Mixed-size
# pairs stay equivalent-sphere.

struct Laplace3DPWQuad
    λ::Vector{Float64}
    wλ::Vector{Float64}
    nα::Vector{Int}
    nfour::Vector{Int}
    αoff::Vector{Int}
    nexp::Int
    zref::Float64
    cα::Vector{Float64}
    sα::Vector{Float64}
    eim::Matrix{ComplexF64}
    sfact::Vector{Float64}
    rlsc::Array{Float64,3}
    impow::Vector{ComplexF64}
    A::Vector{Vector{Matrix{Float64}}}
    Ainv::Vector{Vector{Factorization{Float64}}}
    shifts::Array{Vector{ComplexF64},4}
    p::Int
end

mutable struct Laplace3DPWWS
    acc::Vector{ComplexF64}
    mp_dir::Vector{Float64}
    loc_dir::Vector{Float64}
    F::Matrix{ComplexF64}
end
function Laplace3DPWWS(p::Int, nexp::Int, nλ::Int)
    K = _ncoeff_sph(p)
    return Laplace3DPWWS(
        zeros(ComplexF64, nexp), zeros(Float64, K), zeros(Float64, K),
        zeros(ComplexF64, p + 1, nλ),
    )
end

@inline function _box_side(node::ClusterTree{3})
    rec = container(node)
    d = high_corner(rec) - low_corner(rec)
    return maximum(abs, d)
end

@inline function _pw_rot(dir::Int, x::Float64, y::Float64, z::Float64)
    if dir == 1
        return x, y, z
    elseif dir == 2
        return x, y, -z
    elseif dir == 3
        return y, z, x
    elseif dir == 4
        return y, z, -x
    elseif dir == 5
        return z, x, y
    else
        return z, x, -y
    end
end

@inline function _pw_unrot(dir::Int, x::Float64, y::Float64, z::Float64)
    if dir == 1
        return x, y, z
    elseif dir == 2
        return x, y, -z
    elseif dir == 3
        return z, x, y
    elseif dir == 4
        return -z, x, y
    elseif dir == 5
        return y, z, x
    else
        return y, -z, x
    end
end

function _pw_dir(ix::Int, iy::Int, iz::Int)
    ax, ay, az = abs(ix), abs(iy), abs(iz)
    if az >= 2 && az >= ax && az >= ay
        return iz > 0 ? 1 : 2
    elseif ax >= 2 && ax >= ay
        return ix > 0 ? 3 : 4
    elseif ay >= 2
        return iy > 0 ? 5 : 6
    end
    return 0
end

function _pw_axis_maps(p::Int)
    A = Vector{Vector{Matrix{Float64}}}(undef, 6)
    Ainv = Vector{Vector{Factorization{Float64}}}(undef, 6)
    Ptab = zeros(Float64, p + 1, p + 1)
    mp = zeros(Float64, _ncoeff_sph(p))
    for dir in 1:6
        A[dir] = Vector{Matrix{Float64}}(undef, p + 1)
        Ainv[dir] = Vector{Factorization{Float64}}(undef, p + 1)
        for n in 0:p
            ncol = 2n + 1
            ns = ncol + 4
            pts = fibonacci_sphere(SVector(0.0, 0.0, 0.0), 1.0, ns)
            Yx = zeros(Float64, ns, ncol)
            Yd = zeros(Float64, ns, ncol)
            off = n * n
            @inbounds for k in 1:ncol
                fill!(mp, 0)
                mp[off + k] = 1.0
                for j in 1:ns
                    pt = pts[j]
                    Yx[j, k] = _mpole_series(pt, mp, p, Ptab)
                    xr, yr, zr = _pw_unrot(dir, pt[1], pt[2], pt[3])
                    Yd[j, k] = _mpole_series(SVector(xr, yr, zr), mp, p, Ptab)
                end
            end
            M = Yx \ Yd
            A[dir][n + 1] = M
            Ainv[dir][n + 1] = lu(M)
        end
    end
    return A, Ainv
end

function _sqrt_fact_table(nmax::Int)
    s = Vector{Float64}(undef, nmax + 1)
    s[1] = 1.0
    @inbounds for k in 1:nmax
        s[k + 1] = s[k] * sqrt(Float64(k))
    end
    return s
end

const _PWQUAD_CACHE = Dict{Int,Laplace3DPWQuad}()

function _pw_quad(p::Int)
    return get!(_PWQUAD_CACHE, p) do
        zref = 2.0
        nλ = clamp(p + 8, 16, 32)
        x, w = gausslaguerre(nλ)
        λ = x ./ zref
        wλ = w ./ zref
        nα = Vector{Int}(undef, nλ)
        nfour = Vector{Int}(undef, nλ)
        αoff = Vector{Int}(undef, nλ)
        off = 1
        @inbounds for k in 1:nλ
            # FMM3D `numthetafour`-style: ≥ 2(p+1), grows with λ
            na = max(2 * (p + 1), 4 * (round(Int, λ[k]) + 2))
            na = 2 * cld(na, 2)
            nα[k] = na
            nfour[k] = min(p + 1, max(2, na ÷ 2))
            αoff[k] = off
            off += na
        end
        nexp = off - 1
        cα = Vector{Float64}(undef, nexp)
        sα = Vector{Float64}(undef, nexp)
        eim = zeros(ComplexF64, nexp, p)
        @inbounds for il in 1:nλ
            na = nα[il]
            dα = 2π / na
            base = αoff[il]
            nf = nfour[il]
            for j in 0:(na - 1)
                α = j * dα
                c, s = cos(α), sin(α)
                idx = base + j
                cα[idx], sα[idx] = c, s
                e = complex(c, s)
                em = e
                for m in 1:min(p, nf - 1)
                    eim[idx, m] = em
                    em *= e
                end
            end
        end
        sfact = _sqrt_fact_table(2p)
        # FMM3D `rlscini`: λ^n / (√((n-m)!) √((n+m)!))
        rlsc = zeros(Float64, p + 1, p + 1, nλ)
        @inbounds for il in 1:nλ
            λpow = 1.0
            λi = λ[il]
            for n in 0:p
                for m in 0:n
                    rlsc[n + 1, m + 1, il] = λpow / (sfact[n - m + 1] * sfact[n + m + 1])
                end
                λpow *= λi
            end
        end
        impow = Vector{ComplexF64}(undef, p + 1)
        impow[1] = 1
        @inbounds for m in 1:p
            impow[m + 1] = impow[m] * im
        end
        A, Ainv = _pw_axis_maps(p)
        shifts = [_pw_make_shift_raw(λ, cα, sα, nα, αoff, nexp, zref, dir, ix, iy, iz)
                  for ix in -5:5, iy in -5:5, iz in -5:5, dir in 1:6]
        Laplace3DPWQuad(λ, wλ, nα, nfour, αoff, nexp, zref, cα, sα, eim,
            sfact, rlsc, impow, A, Ainv, shifts, p)
    end
end

function _pw_make_shift_raw(λ, cα, sα, nα, αoff, nexp, zref, dir, ix, iy, iz)
    xr, yr, zr = _pw_rot(dir, Float64(ix), Float64(iy), Float64(iz))
    zr <= 0 && return ComplexF64[]
    sh = Vector{ComplexF64}(undef, nexp)
    @inbounds for il in eachindex(λ)
        λi = λ[il]
        ez = exp(-λi * (zr - zref))
        na = nα[il]
        base = αoff[il]
        for t in 0:(na - 1)
            idx = base + t
            θ = λi * (xr * cα[idx] + yr * sα[idx])
            s, c = sincos(θ)
            sh[idx] = ez * complex(c, s)
        end
    end
    return sh
end

@inline function _pw_shift_vec(quad::Laplace3DPWQuad, dir::Int, ix::Int, iy::Int, iz::Int)
    return quad.shifts[ix + 6, iy + 6, iz + 6, dir]
end

function _split_m2l_planewave(m2l, quad::Union{Laplace3DPWQuad,Nothing}=nothing)
    pw = [Tuple{ClusterTree{3,Float64},Vector{ClusterTree{3,Float64}}}[] for _ in 1:6]
    es = similar(m2l, 0)
    tmap = [Dict{Int,Int}() for _ in 1:6]
    @inbounds for (tnode, snode) in m2l
        at = _box_side(tnode)
        as = _box_side(snode)
        (at < 1e-30 || as < 1e-30) && (push!(es, (tnode, snode)); continue)
        if abs(at - as) > 1e-8 * (at + as)
            push!(es, (tnode, snode))
            continue
        end
        a = 0.5 * (at + as)
        Δ = center(container(tnode)) - center(container(snode))
        ix = round(Int, Δ[1] / a)
        iy = round(Int, Δ[2] / a)
        iz = round(Int, Δ[3] / a)
        if hypot(Δ[1] - ix * a, Δ[2] - iy * a, Δ[3] - iz * a) > 0.05 * a
            push!(es, (tnode, snode))
            continue
        end
        linf = max(abs(ix), abs(iy), abs(iz))
        if linf < 2 || linf > 5
            push!(es, (tnode, snode))
            continue
        end
        dir = _pw_dir(ix, iy, iz)
        if dir == 0
            push!(es, (tnode, snode))
            continue
        end
        groups = pw[dir]
        map = tmap[dir]
        tid = node_id(tnode)
        gi = get(map, tid, 0)
        if gi == 0
            push!(groups, (tnode, ClusterTree{3,Float64}[snode]))
            map[tid] = length(groups)
        else
            push!(groups[gi][2], snode)
        end
    end
    return pw, es
end

function _pw_apply_A!(out, inp, A, p)
    @inbounds for n in 0:p
        r = (n * n + 1):((n + 1)^2)
        mul!(view(out, r), A[n + 1], view(inp, r))
    end
    return out
end

function _pw_apply_Ainv!(out, inp, Ainv, p)
    @inbounds for n in 0:p
        r = (n * n + 1):((n + 1)^2)
        copyto!(view(out, r), view(inp, r))
        ldiv!(Ainv[n + 1], view(out, r))
    end
    return out
end

"""FMM3D `mpoletoexp` for the +z frame. `mpole` is already in that frame."""
function _pw_mpole_to_F!(F::Matrix{ComplexF64}, mpole, a::Float64, p::Int,
        quad::Laplace3DPWQuad)
    fill!(F, 0)
    rlsc = quad.rlsc
    impow = quad.impow
    @inbounds for il in eachindex(quad.λ)
        nf = quad.nfour[il]
        apow = 1.0
        for n in 0:p
            off = n * n
            mmax = min(n, nf - 1)
            inva = 1 / apow
            F[1, il] += mpole[off + 1] * rlsc[n + 1, 1, il] * inva
            for m in 1:mmax
                Mc = complex(mpole[off + 2m], mpole[off + 2m + 1])
                F[m + 1, il] += impow[m + 1] * Mc * rlsc[n + 1, m + 1, il] * inva
            end
            apow *= a
        end
    end
    return F
end

function _pw_F_to_phys!(mexp, F, p, quad::Laplace3DPWQuad)
    eim = quad.eim
    @inbounds for il in eachindex(quad.λ)
        nα = quad.nα[il]
        nf = quad.nfour[il]
        base = quad.αoff[il]
        F0 = F[1, il]
        mmax = min(p, nf - 1)
        for j in 0:(nα - 1)
            idx = base + j
            s = F0
            for m in 1:mmax
                z = eim[idx, m] * F[m + 1, il]
                if iseven(m)
                    s += 2 * real(z)
                else
                    s += complex(0.0, 2 * imag(z))
                end
            end
            mexp[idx] = s
        end
    end
    return mexp
end

function _pw_m2x!(mexp, mpole, a, dir, p, ws::Laplace3DPWWS, quad::Laplace3DPWQuad)
    if dir == 1
        _pw_mpole_to_F!(ws.F, mpole, a, p, quad)
    else
        _pw_apply_A!(ws.mp_dir, mpole, quad.A[dir], p)
        _pw_mpole_to_F!(ws.F, ws.mp_dir, a, p, quad)
    end
    return _pw_F_to_phys!(mexp, ws.F, p, quad)
end

function _pw_shift_add!(acc::AbstractVector{ComplexF64}, src::AbstractVector{ComplexF64},
        ix::Int, iy::Int, iz::Int, dir::Int, quad::Laplace3DPWQuad)
    sh = _pw_shift_vec(quad, dir, ix, iy, iz)
    @inbounds for i in eachindex(acc)
        acc[i] = muladd(src[i], sh[i], acc[i])
    end
    return acc
end

function _pw_phys_to_F!(F::Matrix{ComplexF64}, acc, p, quad::Laplace3DPWQuad)
    fill!(F, 0)
    eim = quad.eim
    @inbounds for il in eachindex(quad.λ)
        nα = quad.nα[il]
        nf = quad.nfour[il]
        base = quad.αoff[il]
        invn = 1 / nα
        mmax = min(p, nf - 1)
        for j in 0:(nα - 1)
            idx = base + j
            v = acc[idx]
            F[1, il] += v
            for m in 1:mmax
                F[m + 1, il] += v * conj(eim[idx, m])
            end
        end
        for m in 0:mmax
            F[m + 1, il] *= invn
        end
    end
    return F
end

"""FMM3D `exptolocal` for +z (`lexp2 = 0`), then × i^m."""
function _pw_F_to_local!(loc, F, a, p, quad::Laplace3DPWQuad)
    fill!(loc, 0)
    rlsc = quad.rlsc
    impow = quad.impow
    @inbounds for il in eachindex(quad.λ)
        w0 = quad.wλ[il]
        nf = quad.nfour[il]
        apow = a
        for n in 0:p
            off = n * n
            w = iseven(n) ? w0 : -w0
            inva = 1 / apow
            mmax = min(n, nf - 1)
            loc[off + 1] += w * rlsc[n + 1, 1, il] * inva * real(F[1, il])
            for m in 1:mmax
                coef = w * rlsc[n + 1, m + 1, il] * inva * impow[m + 1] * F[m + 1, il]
                loc[off + 2m] += real(coef)
                loc[off + 2m + 1] += imag(coef)
            end
            apow *= a
        end
    end
    return loc
end

function _pw_x2l!(localexp, a, dir, p, acc, ws::Laplace3DPWWS, quad::Laplace3DPWQuad)
    _pw_phys_to_F!(ws.F, acc, p, quad)
    _pw_F_to_local!(ws.loc_dir, ws.F, a, p, quad)
    if dir == 1
        localexp .+= ws.loc_dir
    else
        _pw_apply_Ainv!(ws.mp_dir, ws.loc_dir, quad.Ainv[dir], p)
        localexp .+= ws.mp_dir
    end
    return localexp
end
